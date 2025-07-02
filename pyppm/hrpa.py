import os

import h5py
import numpy as np
import scipy as sp
from pyscf import ao2mo, lib
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

from pyppm.rpa import RPA


class HRPA:
    """Class to perform calculations of Indirect J Coupling at HRPA level of
    of approach. This is the p-h part of SOPPA.
    It follows Oddershede, J.; Jørgensen, P.; Yeager, D. L. Compt. Phys. Rep.
    1984, 2, 33 and is inspired in Andy D. Zapata HRPA program

    Returns:
        obj: hrpa object with methods and properties neccesaries to obtain the
        coupling using HRPA
    """

    def __init__(self, mol=None, chkfile=None, mole_name=None, calc_int=False):
        self.mol = mol
        self.chkfile = chkfile
        self.calc_int = calc_int
        self.mole_name = mole_name
        self.mo_coeff = lib.chkfile.load(self.chkfile, "scf/mo_coeff")
        self.mo_occ = lib.chkfile.load(self.chkfile, "scf/mo_occ")
        self.mo_energy = lib.chkfile.load(self.chkfile, "scf/mo_energy")
        self.occidx = np.where(self.mo_occ > 0)[0]
        self.viridx = np.where(self.mo_occ == 0)[0]

        self.orbv = self.mo_coeff[:, self.viridx]
        self.orbo = self.mo_coeff[:, self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        self.mo = np.hstack((self.orbo, self.orbv))
        self.nmo = self.nocc + self.nvir
        self.occ = [i for i in range(self.nocc)]
        self.vir = [i for i in range(self.nvir)]
        self.rpa_obj = RPA(mol=mol, chkfile=self.chkfile)
        self.scratch_dir = os.getenv("SCRATCH", os.getcwd())
        # erifile = os.path.join(
        #    self.scratch_dir, f"full_eri_{self.mole_name}.h5"
        # )
        # self.erifile = erifile
        # if calc_int:
        #   self.eri_mo()
        self.cte = np.sqrt(3)
        self.k_1 = self.kappa(1)
        self.k_2 = self.kappa(2)

    def eri_mo(self, eri_key=None, orbs=None, compact=False):
        """Method to obtain the ERI in MO basis, and saved it
        in a h5py file, if it doesn't exist.
        Then, loaded in a dask array
        """
        mol = self.mol
        erifile = f"{eri_key}_{self.mole_name}.h5"
        if self.calc_int:
            ao2mo.general(
                mol,
                # (self.mo, self.mo, self.mo, self.mo),
                orbs,
                erifile,
                compact=compact,
            )
            # self.mole_name = erifile

    def kappa(self, I_):
        """Method for obtain kappa_{\alpha \beta}^{m n} in a matrix form
        K_{ij}^{a,b} = [(1-delta_{ij})(1-delta_ab]^{I-1}(2I-1)^.5
        * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]
        Oddershede 1984, eq C.7

        Args:
            I (integral): 1 or 2.

        Returns:
            numpy.ndarray: (nocc,nvir,nocc,nvir) array with kappa
        """
        nocc = self.nocc
        nvir = self.nvir
        occidx = self.occidx
        viridx = self.viridx
        mo_energy = self.mo_energy
        e_iajb = lib.direct_sum(
            "i+j-b-a->iajb",
            mo_energy[occidx],
            mo_energy[occidx],
            mo_energy[viridx],
            mo_energy[viridx],
        )
        c = np.sqrt((2 * I_) - 1)
        eri_k = "ovov"
        orbo = self.orbo
        orbv = self.orbv
        if self.calc_int is True:
            self.eri_mo(eri_key=eri_k, orbs=(orbo, orbv, orbo, orbv))
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            eri_mo = f["eri_mo"][:]
            eri_mo = eri_mo.reshape(nocc, nvir, nocc, nvir)
            int1 = eri_mo  # np.transpose(
            # eri_mo[nocc:, :nocc, nocc:, :nocc], (1, 0, 3, 2)
            # eri_mo.transpose(1,0,3,2), (1, 0, 3, 2)
            # )

            int2 = np.transpose(
                # eri_mo[nocc:, :nocc, nocc:, :nocc], (3, 0, 1, 2)
                # eri_mo.transpose(1,0,3,2), (3, 0, 1, 2)
                eri_mo,
                (0, 3, 2, 1),
            )
            K = c * (int1 - ((-1) ** I_) * int2) / e_iajb
            if I_ == 2:
                delta_1 = 1 - np.eye(nocc)
                delta_2 = 1 - np.eye(nvir)
                deltas = np.einsum("ij,ab->iajb", delta_1, delta_2)
                K = K * deltas
            return K

    @property
    def part_a2(self):
        """Method for obtain A(2) matrix, C.13 in Oddershede 1984
        The A = (A + A _ )/2 term is because of C13 equation

        Returns:
            np.ndarray: (nocc,nvir,nocc,nvir)
        """
        nocc = self.nocc
        nvir = self.nvir
        k_1 = self.k_1
        k_2 = self.k_2
        cte = self.cte
        k = k_1 + cte * k_2
        eri_k = "ovov"
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int_ = f["eri_mo"][:]
            int_ = int_.reshape(nocc, nvir, nocc, nvir)
        A1 = np.tensordot(int_, k, axes=([1, 2, 3], [1, 2, 3])).transpose(1, 0)
        A2 = np.tensordot(int_, k, axes=([0, 1, 2], [2, 3, 0])).transpose(1, 0)
        mask_mn = np.eye(nvir)
        mask_ab = np.eye(nocc)
        A = np.einsum("mn,ij->minj", mask_mn, -0.5 * A1)
        A += np.einsum("ij,mn->minj", mask_ab, -0.5 * A2)
        A_ = np.transpose(A, (2, 3, 0, 1))
        A = (A + A_) / 2
        A = np.transpose(A, (1, 0, 3, 2))
        return A

    def part_b2(self, S):
        """Method for obtain B(2) matrix (eq. 14 in Oddershede Paper)
        but using einsum function
        Args:
            S (int): Multiplicity of the response, triplet (S=1) or single(S=0)

        Returns:
            np.array: (nvir,nocc,nvir,nocc) array with B(2) matrix
        """
        nocc = self.nocc
        nvir = self.nvir
        orbo = self.orbo
        orbv = self.orbv
        k_1 = self.k_1
        k_2 = self.k_2
        cte = self.cte
        cte2 = (-1) ** S
        eri_k = "ovov"
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int1 = f["eri_mo"][:]
            int1 = int1.reshape(nocc, nvir, nocc, nvir)
            int1 = int1.transpose(0, 1, 3, 2)
        eri_k = "oovv"
        if self.calc_int is True:
            self.eri_mo(eri_key=eri_k, orbs=(orbo, orbo, orbv, orbv))
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int2 = f["eri_mo"][:]
            int2 = int2.reshape(nocc, nocc, nvir, nvir)
        eri_k = "oooo"
        if self.calc_int is True:
            self.eri_mo(eri_key=eri_k, orbs=(orbo, orbo, orbo, orbo))
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int3 = f["eri_mo"][:]
            int3 = int3.reshape(nocc, nocc, nocc, nocc)
        eri_k = "vvvv"
        if self.calc_int is True:
            self.eri_mo(
                eri_key=eri_k, orbs=(orbv, orbv, orbv, orbv), compact=True
            )
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int4 = f["eri_mo"][:]
            int4 = ao2mo.restore(1, int4, nvir)
        k_b1 = (k_1 + cte * k_2) * 0.5
        k_b2 = (k_1 + (cte / (1 - 4 * S)) * k_2) * 0.5 * cte2
        B = np.tensordot(int1, k_b1, axes=([2, 3], [3, 2])).transpose(
            0, 3, 2, 1
        )
        B += np.tensordot(int1, k_b1, axes=([2, 3], [3, 2])).transpose(
            2, 1, 0, 3
        )
        B += np.tensordot(int2, k_b2, axes=([1, 2], [0, 3])).transpose(
            0, 2, 3, 1
        )
        B += np.tensordot(int2, k_b2, axes=([1, 2], [0, 3])).transpose(
            3, 1, 0, 2
        )

        B -= np.tensordot(int3, k_b2, axes=([1, 3], [2, 0])).transpose(
            1, 2, 0, 3
        )
        B -= np.tensordot(int4, k_b2, axes=([0, 2], [3, 1])).transpose(
            3, 0, 2, 1
        )
        return B

    @property
    def S2(self):
        """Property with S(2) matrix elements (eq. C.9 in Oddershede 1984)
        This matrix will be multiplied by energy

        Returns:
            np.ndarray: (nocc,nvir,nocc,nvir) array with S(2) matrix
        """
        mo_energy = self.mo_energy
        occidx = self.occidx
        viridx = self.viridx
        nocc = self.nocc
        nvir = self.nvir
        e_iajb = lib.direct_sum(
            "i+j-a-b->iajb",
            mo_energy[occidx],
            mo_energy[occidx],
            mo_energy[viridx],
            mo_energy[viridx],
        )
        k_1 = self.k_1
        k_2 = self.k_2

        eri_k = "ovov"
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int_ = f["eri_mo"][:]
            int_ = int_.reshape(nocc, nvir, nocc, nvir)
        k = k_1 + self.cte * k_2
        S2 = np.tensordot(
            int_ / e_iajb, k, axes=([1, 2, 3], [1, 2, 3])
        ).transpose(1, 0)
        S_ = np.tensordot(
            int_ / e_iajb, k, axes=([0, 1, 2], [2, 3, 0])
        ).transpose(1, 0)
        mask_mn = np.eye(nvir)
        mask_ij = np.eye(nocc)
        S2 = np.einsum("mn,ij->imjn", mask_mn, -0.5 * S2)
        S2 += np.einsum("ij,mn->imjn", mask_ij, -0.5 * S_)
        S2 = -0.5 * S2 * e_iajb
        return S2

    @property
    def kappa_2(self):
        """property with kappa in equation C.24
        with einsum
        Returns:
            np.narray: (nocc,nvir) array
        """
        nocc = self.nocc
        nvir = self.nvir
        orbo = self.orbo
        orbv = self.orbv
        mo_energy = self.mo_energy
        occidx = self.occidx
        viridx = self.viridx
        k_1 = self.k_1
        k_2 = self.k_2
        e_ia = lib.direct_sum("i-a->ia", mo_energy[occidx], mo_energy[viridx])
        eri_k = "ovvv"
        if self.calc_int is True:
            self.eri_mo(eri_key=eri_k, orbs=(orbo, orbv, orbv, orbv))
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int1 = f["eri_mo"][:]
            int1 = int1.reshape(nocc, nvir, nvir, nvir)
        k = k_1 + self.cte * k_2
        eri_k = "ovoo"
        if self.calc_int is True:
            self.eri_mo(eri_key=eri_k, orbs=(orbo, orbv, orbo, orbo))
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int2 = f["eri_mo"][:]
            int2 = int2.reshape(nocc, nvir, nocc, nocc)
        kappa = np.tensordot(int1, k, axes=([0, 1, 3], [0, 1, 3])).transpose(
            1, 0
        )
        kappa -= np.tensordot(int2, k, axes=([0, 1, 2], [0, 1, 2]))
        kappa *= 1 / e_ia
        return kappa

    def correction_pert(self, FC=False, PSO=False, FCSD=False, atmlst=None):
        """Method with eq. C.25, which is the first correction to perturbator

        Args:
            atmlst (list): Nuclei to which will calculate the correction

        Returns:
            np.ndarray: array with first correction to Perturbator (nocc,nvir)
        """

        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        kappa = self.kappa_2
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            p_virt = h1[nocc:, nocc:]
            pert = np.tensordot(kappa, p_virt, axes=([1], [1]))
            p_occ = h1[:nocc, :nocc]
            pert -= np.tensordot(kappa, p_occ, axes=([0], [0])).transpose(1, 0)
        if PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)[0]
            p_virt = h1[:, nocc:, nocc:]
            pert = np.tensordot(kappa, p_virt, axes=([1], [2])).transpose(
                1, 0, 2
            )
            p_occ = h1[:, :nocc, :nocc]
            pert -= np.tensordot(kappa, p_occ, axes=([0], [1])).transpose(
                1, 2, 0
            )
        elif FCSD:
            h1 = self.rpa_obj.pert_fcsd(atmlst)
            h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0, :, :, :, :]
            p_virt = h1[:, :, nocc:, nocc:]
            p_occ = h1[:, :, :nocc, :nocc]
            pert = np.tensordot(kappa, p_virt, axes=([1], [3])).transpose(
                1, 2, 0, 3
            )
            pert -= np.tensordot(kappa, p_occ, axes=([0], [2])).transpose(
                1, 2, 3, 0
            )
        return pert

    def correction_pert_2(self, FC=False, PSO=False, FCSD=False, atmlst=None):
        """Method with C.26 correction, which is a correction to perturbator
        centered in atmslt

        Args:
            atmlst (list): nuclei in which is centered the correction

        Returns:
            np.ndarray: array with second correction to Perturbator
            (nocc,nvir)
        """
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        e_iajb = lib.direct_sum(
            "i+j-a-b->iajb",
            self.mo_energy[self.occidx],
            self.mo_energy[self.occidx],
            self.mo_energy[self.viridx],
            self.mo_energy[self.viridx],
        )
        k = self.k_1 + self.cte * self.k_2
        eri_k = "ovov"
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int_ = f["eri_mo"][:]
            int_ = int_.reshape(nocc, nvir, nocc, nvir)
            int_e = int_ / e_iajb
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            h1 = h1[nocc:, :nocc]
            pert_ = np.tensordot(int_e, k, axes=([1, 2, 3], [1, 2, 3]))
            pert = -np.tensordot(pert_, h1, axes=([0], [1]))
            pert_ = np.tensordot(int_e, k, axes=([0, 1, 2], [2, 3, 0]))
            pert -= np.tensordot(pert_, h1, axes=([0], [0])).transpose(1, 0)
            return pert
        if PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)
            h1 = h1[0][:, nocc:, :nocc]
            pert_ = np.tensordot(int_e, k, axes=([1, 2, 3], [1, 2, 3]))
            pert = -np.tensordot(pert_, h1, axes=([0], [2])).transpose(1, 0, 2)
            pert_ = np.tensordot(int_e, k, axes=([0, 1, 2], [2, 3, 0]))
            pert -= np.tensordot(pert_, h1, axes=([0], [1])).transpose(1, 2, 0)
            return pert

        elif FCSD:
            h1 = self.rpa_obj.pert_fcsd(atmlst)
            h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[
                0, :, :, nocc:, :nocc
            ]
            pert_ = np.tensordot(int_e, k, axes=([1, 2, 3], [1, 2, 3]))
            pert = -np.tensordot(pert_, h1, axes=([0], [3])).transpose(
                1, 2, 0, 3
            )
            pert_ = np.tensordot(int_e, k, axes=([0, 1, 2], [2, 3, 0]))
            pert -= np.tensordot(pert_, h1, axes=([0], [2])).transpose(
                1, 2, 3, 0
            )
            return pert

    def Communicator(self, triplet):
        """Function for obtain Communicator matrix, i.e., the principal
        propagatorinverse without the A(0) matrix

        Args:
            triplet (bool, optional): Triplet or singlet quantum communicator
            matrix.
            Defaults to True.

        Returns:
            np.ndarray: Quantum communicator matrix
        """
        nvir = self.nvir
        nocc = self.nocc
        m = self.M_rpa(triplet=triplet, communicator=True)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m += self.S2
        if triplet:
            m -= self.part_b2(1)
        else:
            m += self.part_b2(0)
        m = m.reshape(nocc * nvir, nocc * nvir)
        return m

    def M_rpa(self, triplet, communicator=False):
        """Principal Propagator Inverse at RPA, defined as M = A+B

        A[i,a,j,b] = delta_{ab}delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)

        ref: G.A Aucar  https://doi.org/10.1002/cmr.a.20108

        Args:
                triplet (bool, optional): defines if the response is triplet
                or singlet (FALSE), that changes the Matrix M. Defaults is True

        Returns:
                numpy.ndarray: M matrix
        """
        nocc = self.nocc
        e_ia = lib.direct_sum(
            "a-i->ia", self.mo_energy[self.viridx], self.mo_energy[self.occidx]
        )
        nocc = self.nocc
        nvir = self.nvir
        if communicator is False:
            a = np.diag(e_ia.ravel()).reshape(
                self.nocc, self.nvir, self.nocc, self.nvir
            )
        else:
            a = np.zeros((nocc, nvir, nocc, nvir))
        b = np.zeros_like(a)
        eri_k = "oovv"
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int1 = f["eri_mo"][:]
            int1 = int1.reshape(nocc, nocc, nvir, nvir)
        a -= np.transpose(int1, (0, 3, 1, 2))
        eri_k = "ovov"
        with h5py.File(f"{eri_k}_{self.mole_name}.h5", "r") as f:
            int2 = f["eri_mo"][:]
            int2 = int2.reshape(nocc, nvir, nocc, nvir)
        if triplet:
            b -= np.transpose(int2, (2, 1, 0, 3))
        elif not triplet:
            b += np.transpose(int2, (2, 1, 0, 3))
        m = a + b
        m = m.reshape(self.nocc * self.nvir, self.nocc * self.nvir, order="C")
        return m

    def pp_ssc_fc(self, atm1lst, atm2lst, elements=False):
        """Method that obtain the linear response between two FC perturbation
        at HRPA level of approach between two nuclei
        Args:
            atm1lst (list): First nuclei
            atm2lst (list): Second nuclei

        Returns:
            real: FC response at HRPA level of approach
        """
        nvir = self.nvir
        nocc = self.nocc
        h1 = self.rpa_obj.pert_fc(atm1lst)[0][:nocc, nocc:]
        h2 = self.rpa_obj.pert_fc(atm2lst)[0][:nocc, nocc:]

        m = self.M_rpa(triplet=True)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m -= self.part_b2(1)
        m += self.S2
        h1_corr1 = self.correction_pert(atmlst=atm1lst, FC=True)
        h1_corr2 = self.correction_pert_2(atmlst=atm1lst, FC=True)
        h2_corr1 = self.correction_pert(atmlst=atm2lst, FC=True)
        h2_corr2 = self.correction_pert_2(atmlst=atm2lst, FC=True)
        h1 = (2 * h1) + h1_corr1 + h1_corr2
        h2 = (2 * h2) + h2_corr1 + h2_corr2
        m = m.reshape(nocc * nvir, nocc * nvir)
        if elements:
            return h1, m, h2
        else:
            p = -sp.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e_ = np.tensordot(h1, p, axes=([0, 1], [0, 1]))
            e = np.tensordot(e_, h2, axes=([0, 1], [0, 1]))
            para.append(e / 4)
            fc = np.einsum(",k,xy->kxy", nist.ALPHA**4, para, np.eye(3))
            return fc

    def pp_ssc_pso(self, atm1lst, atm2lst, elements=False):
        """Method that obtain the linear response between PSO perturbation at
        HRPA level of approach between two nuclei
        Args:
            atm1lst (list): First nuclei
            atm2lst (list): Second nuclei

        Returns:
            real: PSO response at HRPA level of approach
        """
        nvir = self.nvir
        nocc = self.nocc
        ntot = nocc + nvir
        h1 = self.rpa_obj.pert_pso(atm1lst)
        h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)
        h1 = h1[0][:, :nocc, nocc:]
        h2 = self.rpa_obj.pert_pso(atm2lst)
        h2 = np.asarray(h2).reshape(1, 3, ntot, ntot)
        h2 = h2[0][:, :nocc, nocc:]

        h1_corr1 = self.correction_pert(atmlst=atm1lst, PSO=True)
        h1_corr2 = self.correction_pert_2(atmlst=atm1lst, PSO=True)
        h2_corr1 = self.correction_pert(atmlst=atm2lst, PSO=True)
        h2_corr2 = self.correction_pert_2(atmlst=atm2lst, PSO=True)

        h1 = (-2 * h1) + h1_corr1 + h1_corr2
        h2 = (-2 * h2) + h2_corr1 + h2_corr2
        m = self.M_rpa(triplet=False)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m += self.part_b2(0)
        m += self.S2
        m = m.reshape(nocc * nvir, nocc * nvir)
        if elements:
            return h1, m, h2
        else:
            p = sp.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e_ = np.tensordot(h1, p, axes=([1, 2], [0, 1]))
            e = np.tensordot(e_, h2, axes=([1, 2], [1, 2]))
            para.append(e)
            pso = np.asarray(para) * nist.ALPHA**4
            return pso

    def pp_ssc_fcsd(self, atm1lst, atm2lst, elements=False):
        """Method that obtain the linear response between two FC+SD
        perturbation  at HRPA level of approach between two nuclei
        Args:
            atm1lst (list): First nuclei
            atm2lst (list): Second nuclei

        Returns:
            real: FC+SD response at HRPA level of approach
        """
        nvir = self.nvir
        nocc = self.nocc
        ntot = nocc + nvir
        h1 = self.rpa_obj.pert_fcsd(atm1lst)
        h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[
            0, :, :, :nocc, nocc:
        ]
        h2 = self.rpa_obj.pert_fcsd(atm2lst)
        h2 = np.asarray(h2).reshape(-1, 3, 3, ntot, ntot)[
            0, :, :, :nocc, nocc:
        ]
        h1_corr1 = self.correction_pert(atmlst=atm1lst, FCSD=True)
        h1_corr2 = self.correction_pert_2(atmlst=atm1lst, FCSD=True)
        h2_corr1 = self.correction_pert(atmlst=atm2lst, FCSD=True)
        h2_corr2 = self.correction_pert_2(atmlst=atm2lst, FCSD=True)

        h1 = (2 * h1) + h1_corr1 + h1_corr2
        h2 = (2 * h2) + h2_corr1 + h2_corr2
        m = self.M_rpa(triplet=True)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m -= self.part_b2(1)
        m += self.S2
        m = m.reshape(nocc * nvir, nocc * nvir)
        if elements:
            return h1, m, h2
        else:
            p = -sp.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e_ = np.tensordot(h1, p, axes=([2, 3], [0, 1]))
            e = np.tensordot(e_, h2, axes=([0, 2, 3], [0, 2, 3]))
            para.append(e)
            fcsd = np.asarray(para) * nist.ALPHA**4
            return fcsd

    def obtain_atom_order(self, atom):
        atom_ = self.rpa_obj.obtain_atom_order(atom)
        return atom_

    def ssc(self, atom1, atom2, FC=False, FCSD=False, PSO=False):
        """Function for Spin-Spin Coupling calculation at HRPA level of
        approach. It take the value of the responses and multiplicates it
        for the constants.

        Args:
            FC (bool, optional): Fermi Contact. Defaults to False.
            FCSD (bool, optional): FC+SD. Defaults to False.
            PSO (bool, optional): PSO. Defaults to False.
            atom1 (str): Atom1 nuclei
            atom2 (str): Atom2 nuclei.

        Returns:
            ssc: Real. SSC value, in Hertz.
        """

        atom1_ = [self.rpa_obj.obtain_atom_order(atom1)]
        atom2_ = [self.rpa_obj.obtain_atom_order(atom2)]
        if FC:
            prop = self.pp_ssc_fc(atm1lst=atom1_, atm2lst=atom2_)
        if PSO:
            prop = self.pp_ssc_pso(atm1lst=atom1_, atm2lst=atom2_)
        elif FCSD:
            prop = self.pp_ssc_fcsd(atm1lst=atom1_, atm2lst=atom2_)
        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton**2
        iso_ssc = unit * np.einsum("kii->k", prop) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        jtensor = np.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]

    def elements(self, atm1lst, atom2lst, FC=False, FCSD=False, PSO=False):
        """Function that return perturbators and principal propagators
        of a selected mechanism

        Args:
            atm1lst (list): atom1 list in which is centered h1
            atom2lst (list): atom2 list in which is centered h2
            FC (bool, optional): FC mechanims. Defaults to False.
            FCSD (bool, optional): FC+SD mechanisms. Defaults to False.
            PSO (bool, optional): PSO mechanism. Defaults to False.

        Returns:
            np.ndarray, np.ndarray, np.ndarray:
            perturbator h1, principal propagator inverse, perturbator 2
        """

        if FC:
            h1, m, h2 = self.pp_ssc_fc(atm1lst, atom2lst, elements=True)
        if PSO:
            h1, m, h2 = self.pp_ssc_pso(atm1lst, atom2lst, elements=True)
        elif FCSD:
            h1, m, h2 = self.pp_ssc_fcsd(atm1lst, atom2lst, elements=True)
        return h1, m, h2
