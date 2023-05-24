from pyscf import gto, scf
import numpy as np
import attr
from pyscf import ao2mo
from functools import reduce
from pyscf.data import nist
from pyscf.dft import numint
from pyscf.data.gyro import get_nuc_g_factor


@attr.s
class Cloppa:
    """
    Class for the calculation of NMR J-coupling contributions using
    localized molecular orbitals of any kind
    e.g Foster-Boys, Pipek-Mezey, etc, using the CLOPPA method.
    This method follows: Molecular Physics 91: 1, 105-112
    """

    mo_coeff_loc = attr.ib(
        default=None, type=np.array, validator=attr.validators.instance_of(np.ndarray)
    )
    mol_loc = attr.ib(
        default=None, validator=attr.validators.instance_of(gto.mole.Mole)
    )
    mo_occ_loc = attr.ib(
        default=None, type=np.array, validator=attr.validators.instance_of(np.ndarray)
    )

    def __attrs_post_init__(self):
        self.occidx = np.where(self.mo_occ_loc > 0)[0]
        self.viridx = np.where(self.mo_occ_loc == 0)[0]

        self.orbv = self.mo_coeff_loc[:, self.viridx]
        self.orbo = self.mo_coeff_loc[:, self.occidx]

        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        self.mo = np.hstack((self.orbo, self.orbv))
        self.nmo = self.nocc + self.nvir

    @property
    def fock_matrix_canonical(self):
        """This property calculates a Hartree-Fock calculation and obtains
        the Fock Matrix in canonical basis.

        Returns:
                numpy.array: Fock matrix in canonical basis
        """
        self.mf = scf.RHF(self.mol_loc).run(verbose=0)
        self.fock_canonical = self.mf.get_fock()
        return self.fock_canonical

    def M(self, triplet=True):
        """Principal Propagator Inverse, defined as M = A + B

        A[i,a,j,b] =  delta_{ij}F_{ab} -delta_{ab}F_{ij} + (ia||bj)
        B[i,a,j,b] = (ia||jb)

        ref: Molecular Physics 91: 1, 105-112


        Args:
                        triplet (bool, optional): defines if the response is triplet (TRUE)
                        or singlet (FALSE), that changes the Matrix M. Defaults is True.

        Returns:
                                numpy.ndarray: M matrix in localized basis
        """
        m = np.zeros((self.nocc, self.nvir, self.nocc, self.nvir))
        fock = self.fock_matrix_canonical
        orbo = self.orbo
        orbv = self.orbv
        nocc = self.nocc
        nvir = self.nvir
        nmo = self.nmo
        mol_loc = self.mol_loc
        mo = self.mo
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        if a == b:
                            m[i, a, j, b] -= orbo[:, i].T @ fock @ orbo[:, j]
                        if i == j:
                            m[i, a, j, b] += orbv[:, a].T @ fock @ orbv[:, b]
        eri_mo = ao2mo.general(mol_loc, [mo, mo, mo, mo], compact=False)
        eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
        m -= np.einsum("ijba->iajb", eri_mo[:nocc, :nocc, nocc:, nocc:])
        if triplet:
            m -= np.einsum("jaib->iajb", eri_mo[:nocc, nocc:, :nocc, nocc:])
        elif not triplet:
            m += np.einsum("jaib->iajb", eri_mo[:nocc, nocc:, :nocc, nocc:])
        m = m.reshape((nocc * nvir, nocc * nvir))
        return m

    def pert_fc(self, atmlst):
        """Perturbator for the response Fermi-Contact using localized molecular
        orbitals

        Args:
                        atmlst (list): atom order in wich the perturbator is centered

        Returns:
                        h1 = list with the perturbator in a localized basis
        """
        mo_coeff = self.mo_coeff_loc
        mo_occ = self.mo_occ_loc
        mol = self.mol_loc
        coords = mol.atom_coords()
        ao = numint.eval_ao(mol, coords)
        mo = ao.dot(mo_coeff)
        orbo = mo[:, mo_occ > 0]
        orbv = mo[:, mo_occ == 0]
        fac = 8 * np.pi / 3 * 0.5  # *.5 due to s = 1/2 * pauli-matrix
        h1 = []
        for ia in atmlst:
            h1.append(fac * np.einsum("p,i->pi", orbv[ia], orbo[ia]))
        return h1

    def pert_fcsd(self, atmlst):
        """Perturbator for the response Fermi-Contact + Spin-Dependent
                contribution in a localized MO basis

        Args:
                atmlst (list): atom order in wich the perturbator is centered

        Returns:
                h1 = list with the perturbator in a Localized MO basis
        """
        orbo = self.orbo
        orbv = self.orbv
        h1 = []
        for ia in atmlst:
            h1ao = self._get_integrals_fcsd(ia)
            for i in range(3):
                for j in range(3):
                    h1.append(orbv.T.conj().dot(h1ao[i, j]).dot(orbo) * 0.5)
        return h1

    def pert_pso(self, atmlst):
        """PSO perturbator in a LMO basis

        Args:
                        atmlst (list): list with the atom in with is centered the perturbator

        Returns:
                        list: pso perturbator in LMO basis
        """
        orbo = self.orbo
        orbv = self.orbv

        h1 = []
        for ia in atmlst:
            self.mol_loc.set_rinv_origin(self.mol_loc.atom_coord(ia))
            h1ao = -self.mol_loc.intor_asymmetric("int1e_prinvxp", 3)
            h1 += [reduce(np.dot, (orbv.T.conj(), x, orbo)) for x in h1ao]
        return h1

    def _get_integrals_fcsd(self, atm_id):
        """
        AO integrals for FC + SD contribution
        Args:
                atm_id (int): int with atom1 order

        Returns:
                h1ao= numpy.ndarray with fc+sd AO integrals
        """
        mol = self.mol_loc
        nao = mol.nao
        with mol.with_rinv_origin(mol.atom_coord(atm_id)):
            a01p = mol.intor("int1e_sa01sp", 12).reshape(3, 4, nao, nao)
            h1ao = -(a01p[:, :3] + a01p[:, :3].transpose(0, 1, 3, 2))
        return h1ao

    def pp_pso_pathways(
        self,
        princ_prop,
        n_atom1,
        occ_atom1,
        vir_atom1,
        n_atom2,
        occ_atom2,
        vir_atom2,
        all_pathways,
        elements,
    ):
        """Function with PSO response using all LMOs. Or use a set of
        ocuppied LMOs and all virtual LMOs, or use a set of occupied LMOs
        and a set of virtuals LMOs.

        Args:
                princ_prop (numpy.array): principal propagator
                n_atom1 (int): atom 1 id
                occ_atom1 (int): occupied atom corresponding to atom 1
                vir_atom1 (int): virtual atom corresponding to atom 1
                n_atom2 (int): atom 2 id
                occ_atom2 (int): set of occupied atom corresponding to atom 2
                vir_atom2 (int): set of virtual atom corresponding to atom 1
                all_pathways (bool): If True, calculate pso response with all
                                                                                occupied and virtual LMOs
                elements (bool): If True, return the perturbators and principal
                                                                                propagator of a specific pathway.

        Returns:
                numpy.array: PSO response
        """
        nvir = self.nvir
        nocc = self.nocc

        if princ_prop.all() == None:
            m = self.M(triplet=False)
            p = np.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
        else:
            p = princ_prop
            p = -p.reshape(nocc, nvir, nocc, nvir)

        h1 = self.pert_pso(n_atom1)
        h1 = np.asarray(h1).reshape(1, 3, nvir, nocc)
        h1_pathway = np.zeros(h1.shape)
        h2 = self.pert_pso(n_atom2)
        h2 = np.asarray(h2).reshape(1, 3, nvir, nocc)
        h2_pathway = np.zeros(h2.shape)

        if all_pathways == True:
            h1_pathway[0, :, :, :] += h1[0, :, :, :]
            h2_pathway[0, :, :, :] += h2[0, :, :, :]

        elif vir_atom1 == None:
            h1_pathway[0, :, :, occ_atom1] += h1[0, :, :, occ_atom1]
            h2_pathway[0, :, :, occ_atom2] += h2[0, :, :, occ_atom2]

        else:
            h1_pathway[0, :, vir_atom1 - nocc, occ_atom1] += h1[
                0, :, vir_atom1 - nocc, occ_atom1
            ]
            h2_pathway[0, :, vir_atom2 - nocc, occ_atom2] += h2[
                0, :, vir_atom2 - nocc, occ_atom2
            ]

        para = []
        e = np.einsum("iax,iajb,jby->xy", h1_pathway[0].T, p, h2_pathway[0].T)
        para.append(e * 4)  # *4 for +c.c. and double occupnacy
        pso = np.asarray(para) * nist.ALPHA**4
        if elements == False:
            return pso
        elif elements == True:
            return h1_pathway[0].T, p, h2_pathway[0].T

    def pp_fcsd_pathways(
        self,
        princ_prop,
        n_atom1,
        occ_atom1,
        vir_atom1,
        n_atom2,
        occ_atom2,
        vir_atom2,
        all_pathways,
        elements,
    ):
        """Function with FCSD response using all LMOs. Or use a set of
        ocuppied LMOs and all virtual LMOs, or use a set of occupied LMOs
        and a set of virtuals LMOs.

        Args:
                        princ_prop (numpy.array): principal propagator
                        n_atom1 (int): atom 1 id
                        occ_atom1 (list): set of occupied atom corresponding to atom 1
                        vir_atom1 (list): set of virtual atom corresponding to atom 1
                        n_atom2 (int): atom 2 id
                        occ_atom2 (list): set of occupied atom corresponding to atom 2
                        vir_atom2 (list): set of virtual atom corresponding to atom 1
                        all_pathways (bool): If True, calculate FCSD response with all
                                                                                        occupied and virtual LMOs
                        elements (bool): If True, return the perturbators and principal
                                                                        propagator of a specific pathway.

        Returns:
                numpy.array: FCSD response
        """
        nvir = self.nvir
        nocc = self.nocc

        if princ_prop.all() == None:
            m = self.M(triplet=True)
            p = np.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
        else:
            p = princ_prop
            p = -p.reshape(nocc, nvir, nocc, nvir)

        h1 = self.pert_fcsd(n_atom1)
        h1 = np.asarray(h1).reshape(-1, 3, 3, nvir, nocc)
        h1_pathway = np.zeros((1, 3, 3, nvir, nocc))
        h2 = self.pert_fcsd(n_atom2)
        h2 = np.asarray(h2).reshape(-1, 3, 3, nvir, nocc)
        h2_pathway = np.zeros((1, 3, 3, nvir, nocc))

        if all_pathways == True:
            h1_pathway[0, :, :, :] += h1[0, :, :, :]
            h2_pathway[0, :, :, :] += h2[0, :, :, :]
        elif vir_atom1 == None:
            h1_pathway[0, :, :, :, occ_atom1] += h1[0, :, :, :, occ_atom1]
            h2_pathway[0, :, :, :, occ_atom2] += h2[0, :, :, :, occ_atom2]
        else:
            h1_pathway[0, :, :, vir_atom1 - nocc, occ_atom1] += h1[
                0, :, :, vir_atom1 - nocc, occ_atom1
            ]
            h2_pathway[0, :, :, vir_atom2 - nocc, occ_atom2] += h2[
                0, :, :, vir_atom2 - nocc, occ_atom2
            ]

        para = []
        e = np.einsum("iawx,iajb,jbwy->xy", h1_pathway[0].T, p, h2_pathway[0].T)
        para.append(e * 4)
        fcsd = np.asarray(para) * nist.ALPHA**4
        if elements == False:
            return fcsd
        elif elements == True:
            return h1_pathway[0].T, p, h2_pathway[0].T

    def pp_fc_pathways(
        self,
        princ_prop,
        n_atom1,
        occ_atom1,
        vir_atom1,
        n_atom2,
        occ_atom2,
        vir_atom2,
        all_pathways,
        elements,
    ):
        """Function with FC response using all LMOs. Or use a set of
        ocuppied LMOs and all virtual LMOs, or use a set of occupied LMOs
        and a set of virtuals LMOs.

        Args:
                princ_prop (numpy.array): principal propagator
                n_atom1 (int): atom 1 id
                occ_atom1 (list): set of occupied atom corresponding to atom 1
                vir_atom1 (list): set of virtual atom corresponding to atom 1
                n_atom2 (int): atom 2 id
                occ_atom2 (list): set of occupied atom corresponding to atom 2
                vir_atom2 (list): set of virtual atom corresponding to atom 1
                all_pathways (bool): If True, calculate FC response with all
                                                                                occupied and virtual LMOs
                elements (bool): If True, return the perturbators and principal
                                                                                propagator of a specific pathway.

        Returns:
                numpy.array: FC response
        """
        nvir = self.nvir
        nocc = self.nocc
        if princ_prop.all() == None:
            m = self.M(triplet=True)
            p = np.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
        else:
            p = princ_prop
            p = -p.reshape(nocc, nvir, nocc, nvir)

        h1 = self.pert_fc(n_atom1)
        h2 = self.pert_fc(n_atom2)
        h1_pathway = np.zeros(h1[0].shape)
        h2_pathway = np.zeros(h2[0].shape)
        if all_pathways == True:
            h1_pathway[:, :] += h1[0][:, :]
            h2_pathway[:, :] += h2[0][:, :]
        elif vir_atom1 == None:
            h1_pathway[:, occ_atom1] += h1[0][:, occ_atom1]
            h2_pathway[:, occ_atom2] += h2[0][:, occ_atom2]
        else:
            h1_pathway[vir_atom1 - nocc, occ_atom1] += h1[0][
                vir_atom1 - nocc, occ_atom1
            ]
            h2_pathway[vir_atom2 - nocc, occ_atom2] += h2[0][
                vir_atom2 - nocc, occ_atom2
            ]

        para = []
        e = np.einsum("ia,iajb,jb", h1_pathway.T, p, h2_pathway.T)
        para.append(e * 4)  # *4 for +c.c. and for double occupancy
        fc = np.einsum(",k,xy->kxy", nist.ALPHA**4, para, np.eye(3))
        if elements == False:
            return fc
        elif elements == True:
            return h1_pathway.T, p, h2_pathway.T

    def obtain_atom_order(self, atom):
        """Function that return the atom order in the molecule input
        given the atom label

        Args:
                atom (str): atom label

        Returns:
                int: atom orden in the mol
        """
        for i in range(self.mol_loc.natm):
            atom_ = self.mol_loc.atom_symbol(i)
            if atom_ == atom:
                return i

    def ssc_pathway(
        self,
        FC=True,
        FCSD=False,
        PSO=False,
        princ_prop=np.full((2, 2), None),
        atom1=None,
        occ_atom1=None,
        vir_atom1=None,
        atom2=None,
        occ_atom2=None,
        vir_atom2=None,
        all_pathways=False,
    ):
        """Spin Spin Coupling mechanisms between two nuclei,

        Args:
                        FC (bool, optional): if True, calculate the FC SSC. Defaults to
                                                                                                        False.
                        FCSD (bool, optional): if True, calculate the FC+SD SSC.
                                                                                                                        Defaults to True.
                        PSO (bool, optional): if True, calculate the FC+SD SSC.
                                                                                                                        Defaults to False.
                        princ_prop (numpy.array, optional): Principal Propagator.
                        Must be obtained with whe M function. Defaults to None.
                        n_atom1 (str): Atom1 name
                        vir_atom1 (list): set of virtual atom corresponding to atom 1
                        n_atom2 (int): atom 2 id
                        occ_atom2 (list): set of occupied atom corresponding to atom 2
                        vir_atom2 (list): set of virtual atom corresponding to atom 1
                        all_pathways (bool): If True, calculate FC response with all
                                                                                                        occupied and virtual LMOs
        Returns:
                        real: isotropic ssc mechanism
        """
        n_atom1 = [self.obtain_atom_order(atom1)]
        n_atom2 = [self.obtain_atom_order(atom2)]
        if n_atom1 == n_atom2:
            raise Exception("n_atom1 must be different to n_atom2")
        if FC == FCSD == PSO:
            raise Exception("you can only calculate one mechanisms at a time")

        if FC == FCSD == True:
            raise Exception("you can only calculate one mechanisms at a time")

        if FC == PSO == True:
            raise Exception("you can only calculate one mechanisms at a time")

        if FCSD == PSO == True:
            raise Exception("you can only calculate one mechanisms at a time")

        if FC:
            prop = self.pp_fc_pathways(
                princ_prop=princ_prop,
                n_atom1=n_atom1,
                occ_atom1=occ_atom1,
                vir_atom1=vir_atom1,
                n_atom2=n_atom2,
                occ_atom2=occ_atom2,
                vir_atom2=vir_atom2,
                all_pathways=all_pathways,
                elements=False,
            )
        if PSO:
            prop = self.pp_pso_pathways(
                princ_prop=princ_prop,
                n_atom1=n_atom1,
                occ_atom1=occ_atom1,
                vir_atom1=vir_atom1,
                n_atom2=n_atom2,
                occ_atom2=occ_atom2,
                vir_atom2=vir_atom2,
                all_pathways=all_pathways,
                elements=False,
            )
        if FCSD:
            prop = self.pp_fcsd_pathways(
                princ_prop=princ_prop,
                n_atom1=n_atom1,
                occ_atom1=occ_atom1,
                vir_atom1=vir_atom1,
                n_atom2=n_atom2,
                occ_atom2=occ_atom2,
                vir_atom2=vir_atom2,
                all_pathways=all_pathways,
                elements=False,
            )

        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        iso_ssc = unit * np.einsum("kii->k", prop) / 3

        gyro1 = [get_nuc_g_factor(self.mol_loc.atom_symbol(n_atom1[0]))]
        gyro2 = [get_nuc_g_factor(self.mol_loc.atom_symbol(n_atom2[0]))]
        jtensor = np.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]

    def pathway_elements(
        self,
        FC=False,
        FCSD=False,
        PSO=False,
        princ_prop=np.full((2, 2), None),
        atom1=None,
        occ_atom1=None,
        vir_atom1=None,
        atom2=None,
        occ_atom2=None,
        vir_atom2=None,
    ):

        """Both perturbator, centered en atom1 and atom2, and the principal
        propagator of a definite coupling pathway

        Args:
                        FC (bool, optional): if True, calculate the FC SSC. Defaults to
                                                                                                        False.
                        FCSD (bool, optional): if True, calculate the FC+SD SSC.
                                                                                                                        Defaults to True.
                        PSO (bool, optional): if True, calculate the FC+SD SSC.
                                                                                                                        Defaults to False.
                        princ_prop (numpy.array, optional): Principal Propagator.
                        Must be obtained with whe M function. Defaults to None.
                        atom1 (str): Atom1 name
                        vir_atom1 (int): virtual atom corresponding to atom 1
                        atom2 (str): atom2 name
                        occ_atom2 (int): occupied atom corresponding to atom 2
                        vir_atom2 (int): virtual atom corresponding to atom 1
        Returns:
                        real: perturbators and principal propagator, in case of PSO or
                        FC+SD returns the perturbator trace
        """

        if occ_atom1 == occ_atom2 == None:
            raise Exception(
                """in order to calculate a definite coupling 
			pathway, you must choose a couple atoms  """
            )
        if vir_atom1 == vir_atom2 == None:
            raise Exception(
                """in order to calculate a definite coupling 
							pathway, you must choose a virtual LMOs"""
            )
        if occ_atom1 == occ_atom2 == None:
            raise Exception(
                """in order to calculate a definite coupling 
							pathway, you must choose a occupied LMOs"""
            )

        n_atom1 = [self.obtain_atom_order(atom1)]
        n_atom2 = [self.obtain_atom_order(atom2)]

        if FC:
            p1, m, p2 = self.pp_fc_pathways(
                princ_prop=princ_prop,
                n_atom1=n_atom1,
                occ_atom1=occ_atom1,
                vir_atom1=vir_atom1,
                n_atom2=n_atom2,
                occ_atom2=occ_atom2,
                vir_atom2=vir_atom2,
                elements=True,
                all_pathways=False,
            )
            p1_pathway = p1[occ_atom1, vir_atom1 - self.nocc]
            m_pathway = m[
                occ_atom1, vir_atom1 - self.nocc, occ_atom2, vir_atom2 - self.nocc
            ]
            p2_pathway = p2[occ_atom2, vir_atom2 - self.nocc]
            return p1_pathway, m_pathway, p2_pathway
        if PSO:
            p1, m, p2 = self.pp_pso_pathways(
                princ_prop=princ_prop,
                n_atom1=n_atom1,
                occ_atom1=occ_atom1,
                vir_atom1=vir_atom1,
                n_atom2=n_atom2,
                occ_atom2=occ_atom2,
                vir_atom2=vir_atom2,
                elements=True,
                all_pathways=False,
            )
            p1_pathway = p1[occ_atom1, vir_atom1 - self.nocc, :]
            m_pathway = m[
                occ_atom1, vir_atom1 - self.nocc, occ_atom2, vir_atom2 - self.nocc
            ]
            p2_pathway = p2[occ_atom2, vir_atom2 - self.nocc, :]

            return np.sum(p1) / 3, m_pathway, np.sum(p2) / 3
        if FCSD:
            p1, m, p2 = self.pp_fcsd_pathways(
                princ_prop=princ_prop,
                n_atom1=n_atom1,
                occ_atom1=occ_atom1,
                vir_atom1=vir_atom1,
                n_atom2=n_atom2,
                occ_atom2=occ_atom2,
                vir_atom2=vir_atom2,
                elements=True,
                all_pathways=False,
            )
            p1_pathway = p1[occ_atom1, vir_atom1 - self.nocc, :, :]
            m_pathway = m[
                occ_atom1, vir_atom1 - self.nocc, occ_atom2, vir_atom2 - self.nocc
            ]
            p2_pathway = p2[occ_atom2, vir_atom2 - self.nocc, :, :]

            return np.trace(p1_pathway) / 3, m_pathway, np.trace(p2_pathway) / 3
