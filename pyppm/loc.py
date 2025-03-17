from itertools import product

import numpy as np
import scipy as sp
from pyscf import lib, scf
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

from pyppm.hrpa import HRPA
from pyppm.rpa import RPA


class Loc:
    """Class to perform calculations of $J^{FC}$ mechanism at at RPA and HRPA
    level of of approach using previously localized molecular orbitals.
    Inspired in Andy Danian Zapata HRPA program

    Attributes:
        mf = RHF object
        mo_coeff_loc = localized molecular orbitals
        elec_corr = str with RPA or HRPA. This defines if the correlation
                level is RPA or HRPA.
    """

    def __init__(self, mf=None, mo_coeff_loc=None, elec_corr="RPA"):
        self.mf = mf if mf is not None else scf.hf.RHF()
        self.mo_coeff_loc = mo_coeff_loc
        self.elec_corr = elec_corr
        self.__post_init__()

    def __post_init__(self):
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
        self.mol = self.mf.mol
        self.mo = np.hstack((self.orbo, self.orbv))
        self.occ = [i for i in range(self.nocc)]
        self.vir = [i for i in range(self.nvir)]
        self.rpa_obj = RPA(mf=self.mf)
        if self.elec_corr == "RPA":
            self.obj = RPA(mf=self.mf)
        elif self.elec_corr == "HRPA":
            self.obj = HRPA(mf=self.mf)
        else:
            raise Exception(
                "SOPPA or other method are not available yet. Only RPA & HRPA"
            )

    @property
    def inv_mat(self):
        """Property than obtain the unitary transformation matrix
        Bouman, T. D., Voigt, B., & Hansen, A. E. (1979) JACS
        eqs 19, 22.
        """
        mf = self.mf
        mo_coeff_loc = self.mo_coeff_loc
        nocc = self.nocc
        nvir = self.nvir
        can_inv = sp.linalg.inv(mf.mo_coeff.T)
        c_occ = (mo_coeff_loc[:, :nocc].T.dot(can_inv[:, :nocc])).T

        c_vir = (mo_coeff_loc[:, nocc:].T.dot(can_inv[:, nocc:])).T
        v_transf = np.einsum("ij,ab->iajb", c_occ, c_vir)
        v_transf = v_transf.reshape(nocc * nvir, nocc * nvir)
        return c_occ, v_transf, c_vir

    def pp(self, atom1, atom2, FC=False, PSO=False, FCSD=False, IPPP=False):
        """Fuction that localize perturbators and principal propagator inverse
        of a chosen mechanism
        Args:
            atom1 (str): atom1 label
            atom2 (str): atom2 label
            FC (bool, optional): If true, returns elements from FC mechanisms
            PSO (bool, optional): If true, returns elements for PSO mechanisms
            FCSD (bool, optional): If true, returns elements for FC+SD
            mechanisms

        Returns:
                h1_loc, p_loc, h2_loc: np.ndarrays with perturbators and
                principal propagator in a localized basis
        """
        atom1_ = [self.rpa_obj.obtain_atom_order(atom1)]
        atom2_ = [self.rpa_obj.obtain_atom_order(atom2)]
        obj = self.obj
        if FC:
            h1, m, h2 = obj.elements(atom1_, atom2_, FC=True)
        if FCSD:
            h1, m, h2 = obj.elements(atom1_, atom2_, FCSD=True)
        if PSO:
            h1, m, h2 = obj.elements(atom1_, atom2_, PSO=True)
        c_occ, v_transf, c_vir = self.inv_mat
        h1_loc = c_occ.T @ h1 @ c_vir
        h2_loc = c_occ.T @ h2 @ c_vir
        if IPPP is False:
            m_loc = v_transf.T @ m @ v_transf
            p_loc = -sp.linalg.inv(m_loc)
        if IPPP is True:
            p = -sp.linalg.inv(m)
            p_loc = v_transf.T @ p @ v_transf
        return h1_loc, p_loc, h2_loc

    def ssc(
        self,
        atom1=None,
        atom2=None,
        FC=False,
        PSO=False,
        FCSD=False,
        IPPP=False,
    ):
        """Function that obtains ssc mechanism for two chosen atoms in the
        localized basis

        Args:
            atom1 (str): atom1 label
            atom2 (str): atom2 label
            FC (bool, optional): If true, returs fc-ssc. Defaults to False.
            PSO (bool, optional): If true, returs pso-ssc. Defaults to False
            FCSD (bool, optional): If true, returs fcsd-ssc. Defaults to False

        Returns:
            real: ssc mechanism
        """
        nocc = self.nocc
        nvir = self.nvir
        if FC:
            h1, p, h2 = self.pp(FC=True, atom1=atom1, atom2=atom2, IPPP=IPPP)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e_ = np.tensordot(h1, p, axes=([0, 1], [0, 1]))
            e = np.tensordot(e_, h2, axes=([0, 1], [0, 1]))
            para.append(e / 4)
            prop = lib.einsum(",k,xy->kxy", nist.ALPHA**4, para, np.eye(3))
        if PSO:
            h1, p, h2 = self.pp(atom1=atom1, atom2=atom2, PSO=True, IPPP=IPPP)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e_ = np.tensordot(h1, p, axes=([1, 2], [0, 1]))
            e = np.tensordot(e_, h2, axes=([1, 2], [1, 2]))
            para.append(e)
            prop = np.asarray(para) * nist.ALPHA**4
        elif FCSD:
            h1, p, h2 = self.pp(FCSD=True, atom1=atom1, atom2=atom2)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e_ = np.tensordot(h1, p, axes=([2, 3], [0, 1]))
            e = np.tensordot(e_, h2, axes=([0, 2, 3], [0, 2, 3]))
            para.append(e)
            prop = np.asarray(para) * nist.ALPHA**4

        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        iso_ssc = unit * lib.einsum("kii->k", prop) / 3
        atom1_ = [self.rpa_obj.obtain_atom_order(atom1)]
        atom2_ = [self.rpa_obj.obtain_atom_order(atom2)]
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        jtensor = lib.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]

    def ssc_pathways(
        self,
        atom1=None,
        atom2=None,
        FC=False,
        FCSD=False,
        PSO=False,
        occ_atom1=None,
        vir_atom1=None,
        occ_atom2=None,
        vir_atom2=None,
        IPPP=False,
    ):
        """Function that obtains coupling pathways between two couple of
        exitations or a set of them.
        The shape of perturbator claims which mechanism is.
        For this function, you must introduce the perturbators and principal
        propagators previously calculated with "pp_loc" function, in order to
        only calculate it once, and then evaluate each coupling pathway.

        Args:
            atom1 (str): atom1 label
            atom2 (str): atom2 label
            h1 (np.array): perturbator centered in atom1
            m (np.array): principal propagator inverse
            h2 (np.array): perturbator centeder in atom2
            occ_atom1 (list): list with occupied LMOs centered on atom1
            vir_atom1 (list): list with virtual LMOs centered on atom1
            occ_atom2 (list): list with occupied LMOs centered on atom2
            vir_atom2 (list): list with virtual LMOs centered on atom2

        Returns:
            real: ssc mechanism for the coupling pathway defined for the LMOs
        """
        nocc = self.nocc
        nvir = self.nvir
        atom1_ = [self.rpa_obj.obtain_atom_order(atom1)]
        atom2_ = [self.rpa_obj.obtain_atom_order(atom2)]
        para = []
        h1, p, h2 = self.pp(atom1, atom2, FC=FC, FCSD=FCSD, PSO=PSO, IPPP=IPPP)
        if FC:
            h1_pathway = np.zeros(h1.shape)
            h2_pathway = np.zeros(h1.shape)
            p = p.reshape(nocc, nvir, nocc, nvir)

            if vir_atom1 is None:
                h1_pathway[occ_atom1, :] += h1[occ_atom1, :]
                h2_pathway[occ_atom2, :] += h2[occ_atom2, :]
            else:
                vir_atom1 = [i - nocc for i in vir_atom1]
                vir_atom2 = [i - nocc for i in vir_atom2]
                for i, a in list(product(occ_atom1, vir_atom1)):
                    h1_pathway[i, a] += h1[i, a]
                for j, b in list(product(occ_atom2, vir_atom2)):
                    h2_pathway[j, b] += h2[j, b]
            e_ = np.tensordot(h1_pathway, p, axes=([0, 1], [0, 1]))
            e = np.tensordot(e_, h2_pathway, axes=([0, 1], [0, 1]))
            para.append(e / 4)
            prop = lib.einsum(",k,xy->kxy", nist.ALPHA**4, para, np.eye(3))
        if PSO:
            h1_pathway = np.zeros(h1.shape)
            h2_pathway = np.zeros(h1.shape)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            if vir_atom1 is None:
                h1_pathway[:, occ_atom1, :] += h1[:, occ_atom1, :]
                h2_pathway[:, occ_atom2, :] += h2[:, occ_atom2, :]
            else:
                vir_atom1 = [i - nocc for i in vir_atom1]
                vir_atom2 = [i - nocc for i in vir_atom2]
                for i, a in list(product(occ_atom1, vir_atom1)):
                    h1_pathway[:, i, a] += h1[:, i, a]
                for j, b in list(product(occ_atom2, vir_atom2)):
                    h2_pathway[:, j, b] += h2[:, j, b]
            e_ = np.tensordot(h1_pathway, p, axes=([1, 2], [0, 1]))
            e = np.tensordot(e_, h2_pathway, axes=([1, 2], [1, 2]))
            para.append(e)
            prop = np.asarray(para) * nist.ALPHA**4
        if FCSD:
            h1_pathway = np.zeros(h1.shape)
            h2_pathway = np.zeros(h1.shape)

            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            if vir_atom1 is None:
                h1_pathway[:, :, occ_atom1, :] += h1[:, :, occ_atom1, :]
                h2_pathway[:, :, occ_atom2, :] += h2[:, :, occ_atom2, :]
            else:
                vir_atom1 = [i - nocc for i in vir_atom1]
                vir_atom2 = [i - nocc for i in vir_atom2]
                for i, a in list(product(occ_atom1, vir_atom1)):
                    h1_pathway[:, :, i, a] += h1[:, :, i, a]
                for j, b in list(product(occ_atom2, vir_atom2)):
                    h2_pathway[:, :, j, b] += h2[:, :, j, b]
            e_ = np.tensordot(h1_pathway, p, axes=([2, 3], [0, 1]))
            e = np.tensordot(e_, h2_pathway, axes=([0, 2, 3], [0, 2, 3]))
            para.append(e)
            prop = np.asarray(para) * nist.ALPHA**4

        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        iso_ssc = unit * lib.einsum("kii->k", prop) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        jtensor = lib.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]
