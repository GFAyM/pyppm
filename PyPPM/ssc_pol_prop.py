from pyscf import gto, scf
from pyscf.gto import Mole
import numpy
from pyscf import lib
import attr
from pyscf import ao2mo
from pyscf.dft import numint
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from pyscf import tools
import sys
from functools import reduce


def uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i, j in nuc_pair]))
    atm2lst = sorted(set([j for i, j in nuc_pair]))
    atm1dic = dict([(ia, k) for k, ia in enumerate(atm1lst)])
    atm2dic = dict([(ia, k) for k, ia in enumerate(atm2lst)])
    return atm1dic, atm2dic


@attr.s
class Prop_pol:
    """
    Full-featured class for computing non-relativistic singlet and triplet
    Spin-Spin coupling mechanisms for RPA approach
    """

    mf = attr.ib(
        default=None, type=scf.hf.RHF, validator=attr.validators.instance_of(scf.hf.RHF)
    )

    def __attrs_post_init__(self):
        self.mo_occ = self.mf.mo_occ
        self.mo_energy = self.mf.mo_energy
        self.mo_coeff = self.mf.mo_coeff
        self.mol = self.mf.mol
        self.nuc_pair = [(i, j) for i in range(self.mol.natm) for j in range(i)]
        self.occidx = numpy.where(self.mo_occ > 0)[0]
        self.viridx = numpy.where(self.mo_occ == 0)[0]
        self.orbv = self.mo_coeff[:, self.viridx]
        self.orbo = self.mo_coeff[:, self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]

        self.atm1dic, self.atm2dic = uniq_atoms(nuc_pair=self.nuc_pair)

    # @property
    def M(self, triplet=True):
        """Principal Propagator Inverse, defined as M = A+B

        A[i,a,j,b] = delta_{ab}delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)

        ref: G.A Aucar  https://doi.org/10.1002/cmr.a.20108


        Args:
                triplet (bool, optional): defines if the response is triplet (TRUE)
                or singlet (FALSE), that changes the Matrix M. Defaults is True.

        Returns:
                numpy.ndarray: M matrix
        """
        mo = numpy.hstack((self.orbo, self.orbv))
        nmo = self.nocc + self.nvir

        e_ia = lib.direct_sum(
            "a-i->ia", self.mo_energy[self.viridx], self.mo_energy[self.occidx]
        )
        a = numpy.diag(e_ia.ravel()).reshape(self.nocc, self.nvir, self.nocc, self.nvir)
        b = numpy.zeros_like(a)

        eri_mo = ao2mo.general(self.mol, [self.orbo, mo, mo, mo], compact=False)
        eri_mo = eri_mo.reshape(self.nocc, nmo, nmo, nmo)
        a -= numpy.einsum(
            "ijba->iajb", eri_mo[: self.nocc, : self.nocc, self.nocc :, self.nocc :]
        )
        if triplet:
            b -= numpy.einsum(
                "jaib->iajb", eri_mo[: self.nocc, self.nocc :, : self.nocc, self.nocc :]
            )
        elif not triplet:
            b += numpy.einsum(
                "jaib->iajb", eri_mo[: self.nocc, self.nocc :, : self.nocc, self.nocc :]
            )
        m = a + b
        m = m.reshape(self.nocc * self.nvir, self.nocc * self.nvir, order="C")

        return m

    def pert_fc(self, atmlst):
        """Perturbator for the response Fermi-Contact

        Args:
                atmlst (lsit): atom order in wich the perturbator is centered

        Returns:
                h1 = list with the perturbator
        """
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mol = self.mol
        coords = mol.atom_coords()
        ao = numint.eval_ao(mol, coords)
        mo = ao.dot(mo_coeff)
        orbo = mo[:, mo_occ > 0]
        orbv = mo[:, mo_occ == 0]
        fac = 8 * numpy.pi / 3 * 0.5  # *.5 due to s = 1/2 * pauli-matrix
        h1 = []
        for ia in atmlst:
            h1.append(fac * numpy.einsum("p,i->pi", orbv[ia], orbo[ia]))
        return h1

    def pp_fc(self, atom1, atom2):
        """Fermi Contact Response, calculated as
        ^{FC}J = sum_{ia,jb} ^{FC}P_{ia}(atom1) ^3M_{iajb} ^{FC}P_{jb}(atom2)


        Args:
                atom1 (list): list with atom1 order
                atom2 (list): list with atom2 order

        Returns:
                fc = numpy.ndarray with fc matrix response
        """
        nvir = self.nvir
        nocc = self.nocc

        h1 = self.pert_fc(atom1)
        h2 = self.pert_fc(atom2)
        m = self.M(triplet=True)
        p = numpy.linalg.inv(m)
        p = -p.reshape(nocc, nvir, nocc, nvir)
        para = []
        e = numpy.einsum("ia,iajb,jb", h1[0].T, p, h2[0].T)
        # print(e)
        para.append(e * 4)  # *4 for +c.c. and for double occupancy

        fc = numpy.einsum(",k,xy->kxy", nist.ALPHA**4, para, numpy.eye(3))
        return fc

    def pert_fcsd(self, atmlst):
        """Perturbator for the response Fermi-Contact + Spin-Dependent
            contribution
        Args:
                atmlst (lsit): atom order in wich the perturbator is centered

        Returns:
                h1 = list with the perturbator
        """
        orbo = self.mo_coeff[:, self.mo_occ > 0]
        orbv = self.mo_coeff[:, self.mo_occ == 0]

        h1 = []
        for ia in atmlst:
            h1ao = self.get_integrals_fcsd(ia)
            for i in range(3):
                for j in range(3):
                    h1.append(orbv.T.conj().dot(h1ao[i, j]).dot(orbo) * 0.5)
        return h1

    def get_integrals_fcsd(self, atm_id):
        """
        AO integrals for FC + SD contribution
        Args:
            atm_id (int): int with atom1 order

        Returns:
            h1ao= numpy.ndarray with fc+sd AO integrals
        """

        mol = self.mol
        nao = mol.nao
        with mol.with_rinv_origin(mol.atom_coord(atm_id)):
            # Note the fermi-contact part is different to the fermi-contact
            # operator in HFC, as well as the FC operator in EFG.
            # FC here is associated to the the integrals of
            # (-\nabla \nabla 1/r + I_3x3 \nabla\dot\nabla 1/r), which includes the
            # contribution of Poisson equation twice, i.e. 8\pi rho.
            # Therefore, -1./3 * (8\pi rho) is used as the contact contribution in
            # function _get_integrals_fc to remove the FC part.
            # In HFC or EFG, the factor of FC part is 4\pi/3.
            a01p = mol.intor("int1e_sa01sp", 12).reshape(3, 4, nao, nao)
            h1ao = -(a01p[:, :3] + a01p[:, :3].transpose(0, 1, 3, 2))
        return h1ao

    def pert_pso(self, atmlst):
        """PSO perturbator

        Args:
                atmlst (list): list with the atom in with is centered the perturbator

        Returns:
                list: pso perturbator
        """
        orbo = self.orbo
        orbv = self.orbv

        h1 = []
        for ia in atmlst:
            self.mol.set_rinv_origin(self.mol.atom_coord(ia))
            h1ao = -self.mol.intor_asymmetric("int1e_prinvxp", 3)
            h1 += [reduce(numpy.dot, (orbv.T.conj(), x, orbo)) for x in h1ao]
        return h1

    def obtain_atom_order(self, atom):
        """Function that return the atom order in the molecule input
        given the atom label

        Args:
            atom (str): atom label

        Returns:
            int: atom orden in the mol
        """
        for i in range(self.mol.natm):
            atom_ = self.mol.atom_symbol(i)
            if atom_ == atom:
                return i

    def pp_pso(self, atom1, atom2):
        """
        Paramagnetic spin orbital response, calculated as
        ^{PSO}J = sum_{ia,jb} ^{PSO}P_{ia}(atom1) ^1M_{iajb} ^{PSO}P_{jb}(atom2)

        Args:
            atom1 (list): list with atom1 order
            atom2 (list): list with atom2 order

        Returns:
            numpy.ndarray with PSO matrix response
        """
        para = []
        nvir = self.nvir
        nocc = self.nocc

        m = self.M(triplet=False)
        p = numpy.linalg.inv(m)

        p = -p.reshape(nocc, nvir, nocc, nvir)
        h1 = self.pert_pso(atom1)
        h1 = numpy.asarray(h1).reshape(1, 3, nvir, nocc)
        h2 = self.pert_pso(atom2)
        h2 = numpy.asarray(h2).reshape(1, 3, nvir, nocc)

        e = numpy.einsum("iax,iajb,jby->xy", h1[0].T, p, h2[0].T)
        para.append(e * 4)  # *4 for +c.c. and double occupnacy
        pso = numpy.asarray(para) * nist.ALPHA**4
        return pso

    def pp_fcsd(self, atom1, atom2):
        """Fermi Contact Response, calculated as

        ^{FC+SD}J = sum_{ia,jb} ^{FC+SD}P_{ia}(atom1) ^3M_{iajb} ^{FC+SD}P_{jb}(atom2)


        Args:
                atom1 (list): list with atom1 order
                atom2 (list): list with atom2 order

        Returns:
                fc = numpy.ndarray with FC+SD matrix response
        """
        nvir = self.nvir
        nocc = self.nocc

        h1 = self.pert_fcsd(atom1)
        h1 = numpy.asarray(h1).reshape(-1, 3, 3, nvir, nocc)
        h2 = self.pert_fcsd(atom2)
        h2 = numpy.asarray(h2).reshape(-1, 3, 3, nvir, nocc)
        m = self.M(triplet=True)
        p = numpy.linalg.inv(m)
        p = -p.reshape(nocc, nvir, nocc, nvir)
        para = []
        e = numpy.einsum("iawx,iajb,jbwy->xy", h1[0].T, p, h2[0].T)
        para.append(e * 4)
        fcsd = numpy.asarray(para) * nist.ALPHA**4
        return fcsd

    def ssc(self, FC=True, FCSD=False, PSO=False, atom1=None, atom2=None):
        """
        Function for call the response and multiplicate it by the correspondent
        constants in order to obtain isotropical J-coupling between two nuclei
        (atom1, atom2)


        Args:
                FC (bool, optional): _description_. Defaults to True.
                PSO (bool, optional): _description_. Defaults to False.
                FCSD (bool, optional): Defaults to False
                atom1 (str): atom1 name
                atom2 (str): atom2 name

        Returns:
                jtensor: numpy.ndarray, FC, FC+SD or PSO contribution to J coupling
        """

        atom1_ = [self.obtain_atom_order(atom1)]
        atom2_ = [self.obtain_atom_order(atom2)]

        if FC:
            prop = self.pp_fc(atom1_, atom2_)
        if PSO:
            prop = self.pp_pso(atom1_, atom2_)
        elif FCSD:
            prop = self.pp_fcsd(atom1_, atom2_)

        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        iso_ssc = unit * numpy.einsum("kii->k", prop) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        jtensor = numpy.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]
