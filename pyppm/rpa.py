from functools import reduce

import numpy as np
import scipy as sp
from pyscf import ao2mo, lib, scf
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from pyscf.dft import numint


class RPA:
    """Full-featured class for computing non-relativistic singlet and triplet
    Spin-Spin coupling mechanisms at RPA level of approach
    """

    def __init__(self, mf=None):
        if not isinstance(mf, scf.hf.RHF):
            raise TypeError("mf must be an instance of scf.hf.RHF")

        self.mf = mf
        self.mo_occ = self.mf.mo_occ
        self.mo_energy = self.mf.mo_energy
        self.mo_coeff = self.mf.mo_coeff
        self.mol = self.mf.mol
        self.occidx = np.where(self.mo_occ > 0)[0]
        self.viridx = np.where(self.mo_occ == 0)[0]
        self.orbv = self.mo_coeff[:, self.viridx]
        self.orbo = self.mo_coeff[:, self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        self.mo = np.hstack((self.orbo, self.orbv))
        self.nmo = self.nocc + self.nvir

    def eri_mo(self):
        """Property with all 2-electron Molecular orbital integrals

        Returns:
            np.array: eri_mo, (nmo,nmo,nmo,nmo) shape
        """
        mo = self.mo
        nmo = self.nmo
        eri_mo = ao2mo.general(self.mol, [mo, mo, mo, mo], compact=False)
        eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
        return eri_mo

    def M(self, triplet=True):
        """Principal Propagator Inverse, defined as M = A+B

        A[i,a,j,b] = delta_{ab}delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)

        ref: G.A Aucar  https://doi.org/10.1002/cmr.a.20108


        Args:
                triplet (bool, optional): defines if the response is triplet
                or singlet (FALSE), that changes the Matrix M. Defaults is True

        Returns:
                np.ndarray: M matrix
        """
        eri_mo = self.eri_mo()
        e_ia = lib.direct_sum(
            "a-i->ia", self.mo_energy[self.viridx], self.mo_energy[self.occidx]
        )
        a = np.diag(e_ia.ravel()).reshape(
            self.nocc, self.nvir, self.nocc, self.nvir
        )
        b = np.zeros_like(a)
        a -= eri_mo[
            : self.nocc, : self.nocc, self.nocc :, self.nocc :
        ].transpose(0, 3, 1, 2)
        if triplet:
            b -= eri_mo[
                : self.nocc, self.nocc :, : self.nocc, self.nocc :
            ].transpose(2, 1, 0, 3)
        elif not triplet:
            b += eri_mo[
                : self.nocc, self.nocc :, : self.nocc, self.nocc :
            ].transpose(2, 1, 0, 3)
        m = a + b
        m = m.reshape(self.nocc * self.nvir, self.nocc * self.nvir, order="C")

        return m

    def Communicator(self, triplet=True):
        """Principal Propagator Inverse, defined as M = A+B without A(0) matrix

        A[i,a,j,b] = (ia||bj)
        B[i,a,j,b] = (ia||jb)

        ref: G.A Aucar  https://doi.org/10.1002/cmr.a.20108


        Args:
                triplet (bool, optional): defines if the response is triplet
                or singlet (FALSE), that changes the Matrix M. Defaults is True

        Returns:
                np.ndarray: M matrix
        """
        orbo = self.orbo
        orbv = self.orbv
        mo = np.hstack((orbo, orbv))
        nmo = self.nocc + self.nvir
        nocc = self.nocc
        nvir = self.nvir
        a = np.zeros((nocc, nvir, nocc, nvir))
        b = np.zeros_like(a)
        eri_mo = ao2mo.general(
            self.mol, [self.orbo, mo, mo, mo], compact=False
        )
        eri_mo = eri_mo.reshape(self.nocc, nmo, nmo, nmo)
        a -= eri_mo[
            : self.nocc, : self.nocc, self.nocc :, self.nocc :
        ].transpose(0, 3, 1, 2)
        if triplet:
            b -= eri_mo[
                : self.nocc, self.nocc :, : self.nocc, self.nocc :
            ].transpose(2, 1, 0, 3)

        elif not triplet:
            b += eri_mo[
                : self.nocc, self.nocc :, : self.nocc, self.nocc :
            ].transpose(2, 1, 0, 3)

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
        mol = self.mol
        coords = mol.atom_coords()
        ao = numint.eval_ao(mol, coords)
        mo = ao.dot(mo_coeff)
        orbo = mo[:, :]
        orbv = mo[:, :]
        fac = 8 * np.pi / 3  # *.5 due to s = 1/2 * pauli-matrix
        h1 = []
        for ia in atmlst:
            h1.append(fac * np.einsum("p,i->pi", orbv[ia], orbo[ia]))
        return h1

    def pp_fc(self, atm1lst, atm2lst, elements=False):
        """Fermi Contact Response, calculated as
        ^{FC}J = sum_{ia,jb} ^{FC}P_{ia}(atom1) ^3M_{iajb} ^{FC}P_{jb}(atom2)


        Args:
                atom1 (list): list with atom1 order
                atom2 (list): list with atom2 order

        Returns:
                fc = np.ndarray with fc matrix response
        """
        nvir = self.nvir
        nocc = self.nocc

        h1 = 0.5 * self.pert_fc(atm1lst)[0][:nocc, nocc:]
        h2 = 0.5 * self.pert_fc(atm2lst)[0][:nocc, nocc:]
        m = self.M(triplet=True)
        if elements:
            return h1, m, h2
        else:
            p = sp.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e1 = np.tensordot(h1, p, axes=([0, 1], [0, 1]))
            del p
            e = np.tensordot(e1, h2, axes=([0, 1], [0, 1]))
            para.append(e * 4)  # *4 for +c.c. and for double occupancy
            fc = np.einsum(",k,xy->kxy", nist.ALPHA**4, para, np.eye(3))
            return fc

    def pert_fcsd(self, atmlst):
        """Perturbator for the response Fermi-Contact + Spin-Dependent
        contribution
        Args:
            atmlst (lsit): atom order in wich the perturbator is centered

        Returns:
            h1 = list with the perturbator
        """
        orbo = self.mo_coeff[:, :]
        orbv = self.mo_coeff[:, :]
        h1 = []
        for ia in atmlst:
            h1ao = self.get_integrals_fcsd(ia)
            for i in range(3):
                for j in range(3):
                    h1.append(orbv.T.conj().dot(h1ao[i, j]).dot(orbo) * 0.5)
        return h1

    def get_integrals_fcsd(self, atm_id):
        """AO integrals for FC + SD contribution
        Args:
            atm_id (int): int with atom1 order

        Returns:
            h1ao= np.ndarray with fc+sd AO integrals
        """

        mol = self.mol
        nao = mol.nao
        with mol.with_rinv_origin(mol.atom_coord(atm_id)):
            a01p = mol.intor("int1e_sa01sp", 12).reshape(3, 4, nao, nao)
            h1ao = -(a01p[:, :3] + a01p[:, :3].transpose(0, 1, 3, 2))
        return h1ao

    def pert_pso(self, atmlst):
        """PSO perturbator

        Args:
            atmlst (list): list with the atom in which is centered the
            perturbator

        Returns:
            list: pso perturbator
        """
        mo = self.mo_coeff
        h1 = []
        for ia in atmlst:
            self.mol.set_rinv_origin(self.mol.atom_coord(ia))
            h1ao = -self.mol.intor_asymmetric("int1e_prinvxp", 3)
            h1 += [reduce(np.dot, (mo.T.conj(), x, mo)) for x in h1ao]
        return h1

    def obtain_atom_order(self, atom):
        """Function that return the atom order in the molecule input
        given the atom label

        Args:
            atom (str): atom label

        Returns:
            int: atom orden in the mol
        """
        atoms = []
        for i in range(self.mol.natm):
            atoms.append(self.mol._atom[i][0])
        if atom not in atoms:
            raise Exception(f"{atom} must be one of the labels {atoms}")
        for i in range(self.mol.natm):
            atom_ = self.mol.atom_symbol(i)
            if atom_ == atom:
                return i

    def pp_pso(self, atm1lst, atm2lst, elements=False):
        """
        Paramagnetic spin orbital response, calculated as
        ^{PSO}J = sum_{ia,jb} ^{PSO}P_{ia}(atm1) ^1M_{iajb} ^{PSO}P_{jb}(atm2)

        Args:
            atom1 (list): list with atom1 order
            atom2 (list): list with atom2 order

        Returns:
            np.ndarray with PSO matrix response
        """
        para = []
        nvir = self.nvir
        nocc = self.nocc
        ntot = nocc + nvir
        m = self.M(triplet=False)
        h1 = self.pert_pso(atm1lst)
        h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)
        h1 = h1[0][:, :nocc, nocc:]
        h2 = self.pert_pso(atm2lst)
        h2 = np.asarray(h2).reshape(1, 3, ntot, ntot)
        h2 = h2[0][:, :nocc, nocc:]
        if elements:
            return h1, m, h2
        else:
            p = sp.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
            e1 = np.tensordot(h1, p, axes=([1, 2], [0, 1]))
            del p
            e = np.tensordot(e1, h2, axes=([1, 2], [1, 2]))
            para.append(e * 4)  # *4 for +c.c. and double occupnacy
            pso = np.asarray(para) * nist.ALPHA**4
            return pso

    def pp_fcsd(self, atm1lst, atm2lst, elements=False):
        """Fermi Contact Response, calculated as

        ^{FC+SD}J = sum_{ia,jb} ^{FC+SD}P_{ia}(atom1)
                    ^3M_{iajb} ^{FC+SD}P_{jb}(atom2)

        Args:
            atom1 (list): list with atom1 order
            atom2 (list): list with atom2 order

        Returns:
            fc = np.ndarray with FC+SD matrix response
        """
        nvir = self.nvir
        nocc = self.nocc
        ntot = nocc + nvir
        h1 = self.pert_fcsd(atm1lst)
        h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[
            0, :, :, :nocc, nocc:
        ]
        h2 = self.pert_fcsd(atm2lst)
        h2 = np.asarray(h2).reshape(-1, 3, 3, ntot, ntot)[
            0, :, :, :nocc, nocc:
        ]
        m = self.M(triplet=True)
        if elements:
            return h1, m, h2
        else:
            p = sp.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e1 = np.tensordot(h1, p, axes=([2, 3], [0, 1]))
            del p
            e = np.tensordot(e1, h2, axes=([0, 2, 3], [0, 2, 3]))
            para.append(e * 4)
            fcsd = np.asarray(para) * nist.ALPHA**4
        return fcsd

    def ssc(self, FC=False, FCSD=False, PSO=False, atom1=None, atom2=None):
        """Function for call the response and multiplicate it by the
        correspondent constants in order to obtain isotropical J-coupling
        between two nuclei (atom1, atom2)


        Args:
            FC (bool, optional): _description_. Defaults to True.
            PSO (bool, optional): _description_. Defaults to False.
            FCSD (bool, optional): Defaults to False
            atom1 (str): atom1 name
            atom2 (str): atom2 name

        Returns:
            jtensor: np.ndarray, FC, FC+SD or PSO contribution to J coupling
        """

        atm1lst = [self.obtain_atom_order(atom1)]
        atm2lst = [self.obtain_atom_order(atom2)]

        if FC:
            prop = self.pp_fc(atm1lst, atm2lst)
        if PSO:
            prop = self.pp_pso(atm1lst, atm2lst)
        elif FCSD:
            prop = self.pp_fcsd(atm1lst, atm2lst)

        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        iso_ssc = unit * np.einsum("kii->k", prop) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atm1lst[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atm2lst[0]))]
        jtensor = np.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]

    def elements(self, atm1lst, atm2lst, FC=False, FCSD=False, PSO=False):
        """Function that return principal propagator inverse and perturbators

        Args:
            FC (bool, optional): _description_. Defaults to False.
            FCSD (bool, optional): _description_. Defaults to False.
            PSO (bool, optional): _description_. Defaults to False.
            atom1 (_type_, optional): _description_. Defaults to None.
            atom2 (_type_, optional): _description_. Defaults to None.
        """

        if FC:
            h1, m, h2 = self.pp_fc(atm1lst, atm2lst, elements=True)
            h1 = h1 * 2
            h2 = h2 * 2
        if PSO:
            h1, m, h2 = self.pp_pso(atm1lst, atm2lst, elements=True)
        elif FCSD:
            h1, m, h2 = self.pp_fcsd(atm1lst, atm2lst, elements=True)
        return h1 * 2, m, h2 * 2
