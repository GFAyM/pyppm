from itertools import product

import numpy as np
import pandas as pd
from pyscf import lib
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

    def __init__(
        self,
        mol=None,
        chkfile=None,
        mo_coeff_loc=None,
        mole_name=None,
        calc_int=False,
        elec_corr="RPA",
    ):
        self.mol = mol
        self.mo_coeff_loc = mo_coeff_loc
        self.elec_corr = elec_corr
        self.chkfile = chkfile
        self.mole_name = mole_name
        self.calc_int = calc_int
        self.__post_init__()

    def __post_init__(self):
        self.mo_coeff = lib.chkfile.load(self.chkfile, "scf/mo_coeff")
        self.mo_occ = lib.chkfile.load(self.chkfile, "scf/mo_occ")
        self.occidx = np.where(self.mo_occ > 0)[0]
        self.viridx = np.where(self.mo_occ == 0)[0]
        self.orbv = self.mo_coeff[:, self.viridx]
        self.orbo = self.mo_coeff[:, self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        self.mo = np.hstack((self.orbo, self.orbv))
        self.occ = [i for i in range(self.nocc)]
        self.vir = [i for i in range(self.nvir)]
        if self.elec_corr == "RPA":
            self.obj = RPA(mol=self.mol, chkfile=self.chkfile)
        elif self.elec_corr == "HRPA":
            self.obj = HRPA(
                mol=self.mol,
                chkfile=self.chkfile,
                mole_name=self.mole_name,
                calc_int=self.calc_int,
            )
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
        mo_coeff_loc = self.mo_coeff_loc
        nocc = self.nocc
        nvir = self.nvir
        can_inv = np.linalg.inv(self.mo_coeff.T)
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
        atom1_ = [self.obj.obtain_atom_order(atom1)]
        atom2_ = [self.obj.obtain_atom_order(atom2)]
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
            p_loc = -np.linalg.inv(m_loc)
        if IPPP is True:
            p = -np.linalg.inv(m)
            p_loc = v_transf.T @ p @ v_transf
        return h1_loc, p_loc, h2_loc, m_loc

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
            h1, p, h2, m = self.pp(
                FC=True, atom1=atom1, atom2=atom2, IPPP=IPPP
            )
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = lib.einsum("ia,iajb,jb", h1, p, h2)
            para.append(e / 4)
            prop = lib.einsum(",k,xy->kxy", nist.ALPHA**4, para, np.eye(3))
        if PSO:
            h1, p, h2, m = self.pp(
                atom1=atom1, atom2=atom2, PSO=True, IPPP=IPPP
            )
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = lib.einsum("xia,iajb,yjb->xy", h1, p, h2)
            para.append(e)
            prop = np.asarray(para) * nist.ALPHA**4
        elif FCSD:
            h1, p, h2, m = self.pp(FCSD=True, atom1=atom1, atom2=atom2)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = np.einsum("wxia,iajb,wyjb->xy", h1, p, h2)
            para.append(e)
            prop = np.asarray(para) * nist.ALPHA**4

        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        iso_ssc = unit * lib.einsum("kii->k", prop) / 3
        atom1_ = [self.obj.obtain_atom_order(atom1)]
        atom2_ = [self.obj.obtain_atom_order(atom2)]
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
        exitations or a set of them. The shape of perturbator claims which 
        mechanism is.
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
        atom1_ = [self.obj.obtain_atom_order(atom1)]
        atom2_ = [self.obj.obtain_atom_order(atom2)]
        para = []
        h1, p, h2, m = self.pp(
            atom1, atom2, FC=FC, FCSD=FCSD, PSO=PSO, IPPP=IPPP
        )
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
            e = lib.einsum("ia,iajb,jb", h1_pathway, p, h2_pathway)
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

            e = lib.einsum("xia,iajb,yjb->xy", h1_pathway, p, h2_pathway)
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
            e = lib.einsum("wxia,iajb,wyjb->xy", h1_pathway, p, h2_pathway)
            para.append(e)
            prop = np.asarray(para) * nist.ALPHA**4

        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS) 
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        iso_ssc = unit * lib.einsum("kii->k", prop) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        jtensor = lib.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]

    def ssc_pathways_occ_xlsx(
        self,
        atom1=None,
        atom2=None,
        FC=False,
        FCSD=False,
        PSO=False,
        occ_atom1=None,
        occ_atom2=None,
        IPPP=False,
        file_name=None,
    ):
        """Function that obtains coupling pathways between two couple of 
        exitations
        or a set of them. The shape of perturbator claims which mechanism is.
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
            occ_atom2 (list): list with occupied LMOs centered on atom2

        Returns:
            real: ssc mechanism for the coupling pathway defined for the LMOs
        """
        nocc = self.nocc
        nvir = self.nvir
        atom1_ = [self.obj.obtain_atom_order(atom1)]
        atom2_ = [self.obj.obtain_atom_order(atom2)]
        h1, p, h2, m = self.pp(
            atom1, atom2, FC=FC, FCSD=FCSD, PSO=PSO, IPPP=IPPP
        )
        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        data = []
        if FC:
            p = p.reshape(nocc, nvir, nocc, nvir)
            e = 0
            for i, j in product(occ_atom1, occ_atom2):
                h1_pathway = h1[i, :]
                h2_pathway = h2[j, :]
                p_pathway = p[i, :, j, :]
                e_ = np.einsum("a, ab, b", h1_pathway, p_pathway, h2_pathway)
                e_ = e_ * gyro1[0] * gyro2[0] * (nist.ALPHA**4) * unit / 4
                e += e_
                if abs(e_) > 0:
                    data.append([e_, i, j])
        if PSO:
            p = p.reshape(nocc, nvir, nocc, nvir)
            e = 0
            for i, j in product(occ_atom1, occ_atom2):
                h1_pathway = h1[:, i, :]
                h2_pathway = h2[:, j, :]
                p_pathway = p[i, :, j, :]
                e_ = np.einsum(
                    "xa, ab, yb->xy", h1_pathway, p_pathway, h2_pathway
                )
                e_ = np.diag(e_).sum() / 3
                e_ = e_ * gyro1[0] * gyro2[0] * (nist.ALPHA**4) * unit
                e += e_
                if abs(e_) > 0:
                    data.append([e_, i, j])
        if FCSD:
            p = p.reshape(nocc, nvir, nocc, nvir)
            e = 0
            for i, j in product(occ_atom1, occ_atom2):
                h1_pathway = h1[:, :, i, :]
                h2_pathway = h2[:, :, j, :]
                p_pathway = p[i, :, j, :]
                e_ = np.einsum(
                    "wxa, ab, wyb->xy", h1_pathway, p_pathway, h2_pathway
                )
                e_ = np.diag(e_).sum() / 3
                e_ = e_ * gyro1[0] * gyro2[0] * (nist.ALPHA**4) * unit
                e += e_
                if abs(e_) > 0:
                    data.append([e_, i, j])
        df = pd.DataFrame(
            data,
            columns=["e", "i", "j"],
        )
        df = df.loc[df["e"].abs().sort_values(ascending=False).index]
        df["suma_acumulativa"] = df["e"].cumsum()
        df.insert(1, "suma_acumulativa", df.pop("suma_acumulativa"))
        df.round(4)
        name = file_name
        df.to_excel(name, index=False)
        return e

    def ssc_pathways_vir_xlsx(
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
        file_name=False,
    ):
        """Function that obtains coupling pathways between two couple of 
        exitations
        or a set of them. The shape of perturbator claims which mechanism is.
        For this function, you must introduce the perturbators and principal
        propagators previously calculated with "pp_loc" function, in order to 
        only calculate it once, and then evaluate each coupling pathway.

        Args:
            atom1 (str): atom1 label
            atom2 (str): atom2 label
            occ_atom1 (list): list with occupied LMOs centered on atom1
            vir_atom1 (list): list with virtual LMOs centered on atom1
            occ_atom2 (list): list with occupied LMOs centered on atom2
            vir_atom2 (list): list with virtual LMOs centered on atom2

        Returns:
            real: ssc mechanism for the coupling pathway defined for the LMOs
        """
        nocc = self.nocc
        nvir = self.nvir
        atom1_ = [self.obj.obtain_atom_order(atom1)]
        atom2_ = [self.obj.obtain_atom_order(atom2)]
        h1, p, h2, m = self.pp(
            atom1, atom2, FC=FC, FCSD=FCSD, PSO=PSO, IPPP=IPPP
        )
        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        data = []
        if FC:
            p = p.reshape(nocc, nvir, nocc, nvir)
            m = m.reshape(nocc, nvir, nocc, nvir)
            e = 0
            for i, a, j, b in product(
                occ_atom1, vir_atom1, occ_atom2, vir_atom2
            ):
                h1_pathway = h1[i, a - nocc]
                h2_pathway = h2[j, b - nocc]
                p_pathway = p[i, a - nocc, j, b - nocc]
                m_pathway = m[i, a - nocc, j, b - nocc]
                e_ = h1_pathway * p_pathway * h2_pathway
                e_ = e_ * gyro1[0] * gyro2[0] * (nist.ALPHA**4) * unit / 4

                e += e_
                if abs(e_) > 0.1:
                    data.append(
                        [
                            e_,
                            i,
                            a,
                            j,
                            b,
                            h1_pathway,
                            p_pathway,
                            h2_pathway,
                            m_pathway,
                            1 / m_pathway,
                        ]
                    )
        if PSO:
            p = p.reshape(nocc, nvir, nocc, nvir)
            m = m.reshape(nocc, nvir, nocc, nvir)
            e = 0
            for i, a, j, b in product(
                occ_atom1, vir_atom1, occ_atom2, vir_atom2
            ):
                h1_pathway = h1[:, i, a - nocc]
                h2_pathway = h2[:, j, b - nocc]
                p_pathway = p[i, a - nocc, j, b - nocc]
                m_pathway = m[i, a - nocc, j, b - nocc]
                e_ = np.einsum("x,,y->xy", h1_pathway, p_pathway, h2_pathway)

                # print(e_)
                e_ = np.diag(e_).sum() / 3
                e_ = e_ * gyro1[0] * gyro2[0] * (nist.ALPHA**4) * unit
                e += e_
                if abs(e_) > 0.001:
                    data.append(
                        [
                            e_,
                            i,
                            a,
                            j,
                            b,
                            h1_pathway,
                            p_pathway,
                            h2_pathway,
                            m_pathway,
                            1 / m_pathway,
                        ]
                    )
        elif FCSD:
            p = p.reshape(nocc, nvir, nocc, nvir)
            m = m.reshape(nocc, nvir, nocc, nvir)
            e = 0
            for i, a, j, b in product(
                occ_atom1, vir_atom1, occ_atom2, vir_atom2
            ):
                h1_pathway = h1[:, :, i, a - nocc]
                h2_pathway = h2[:, :, j, b - nocc]
                p_pathway = p[i, a - nocc, j, b - nocc]
                m_pathway = m[i, a - nocc, j, b - nocc]
                e_ = np.einsum("wx,,wy->xy", h1_pathway, p_pathway, h2_pathway)

                # print(e_)
                e_ = np.diag(e_).sum() / 3
                e_ = e_ * gyro1[0] * gyro2[0] * (nist.ALPHA**4) * unit
                e += e_
                if abs(e_) > 0.001:
                    data.append(
                        [
                            e_,
                            i,
                            a,
                            j,
                            b,
                            h1_pathway,
                            p_pathway,
                            h2_pathway,
                            m_pathway,
                            1 / m_pathway,
                        ]
                    )

        df = pd.DataFrame(
            data,
            columns=[
                "e",
                "i",
                "a",
                "j",
                "b",
                "b_ia",
                "P_iajb",
                "b_jb",
                "M_iajb",
                "1/M_iajb",
            ],
        )
        df = df.loc[df["e"].abs().sort_values(ascending=False).index]
        df["suma_acumulativa"] = df["e"].cumsum()
        df.insert(1, "suma_acumulativa", df.pop("suma_acumulativa"))
        df.round(4)
        df.to_excel(file_name, index=False)
        return e
