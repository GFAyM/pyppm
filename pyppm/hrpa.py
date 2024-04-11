from pyscf import lib, ao2mo, scf
import attr
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from pyppm.rpa import RPA
import numpy as np
import dask.array as da
import h5py


@attr.s
class HRPA:
    """Class to perform calculations of $J^{FC}$ mechanism at HRPA level of
    of approach. This is the p-h part of SOPPA level of approah. The HRPA class
    enherits from RPA of pyppm.rpa because they share several methods
    It follows Oddershede, J.; Jørgensen, P.; Yeager, D. L. Compt. Phys. Rep.
    1984, 2, 33
    and is inspired in Andy Danian Zapata HRPA program

    Returns:
        obj: hrpa object with methods and properties neccesaries to obtain the
        coupling using HRPA
    """

    mf = attr.ib(
        default=None, type=scf.hf.RHF, validator=attr.validators.instance_of(scf.hf.RHF)
    )
    h5_file = attr.ib(default=None)

    def __attrs_post_init__(self):
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
        self.occ = [i for i in range(self.nocc)]
        self.vir = [i for i in range(self.nvir)]
        self.rpa_obj = RPA(mf=self.mf)
        self.eri_mo()
        self.cte = np.sqrt(3)
        self.k_1 = self.kappa(1)
        self.k_2 = self.kappa(2)

    def eri_mo(self):
        """Method to obtain the ERI in MO basis, and saved it
        in a h5py file, if it doesn't exist.
        Then, loaded in a dask array
        """
        mol = self.mol
        if self.h5_file is None:
            ao2mo.general(
                mol,
                (self.mo, self.mo, self.mo, self.mo),
                f"{mol.nao}.h5",
                compact=False,
            )
            self.h5_file = f"{mol.nao}.h5"

    def kappa(self, I_):
        """
        Method for obtain kappa_{\alpha \beta}^{m n} in a matrix form
        K_{ij}^{a,b} = [(1-delta_{ij})(1-delta_ab]^{I-1}(2I-1)^.5
                        * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]

        for i noteq j, a noteq b

        K_{ij}^{a,b}=1^{I-1}(2I-1)^.5 * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]

        Oddershede 1984, eq C.7

        Args:
            I (integral): 1 or 2.

        Returns:
            numpy.ndarray: (nocc,nvir,nocc,nvir) array with kappa
        """
        nocc = self.nocc
        occidx = self.occidx
        viridx = self.viridx
        occ = self.occ
        vir = self.vir
        nmo = self.nmo
        mo_energy = self.mo_energy
        e_iajb = lib.direct_sum(
            "i+j-b-a->iajb",
            mo_energy[occidx],
            mo_energy[occidx],
            mo_energy[viridx],
            mo_energy[viridx],
        )
        eri_mo = self.eri_mo()
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks=(nmo // 4, nmo // 4))
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)

            int1 = da.einsum("aibj->iajb", eri_mo[nocc:, :nocc, nocc:, :nocc]).compute()
            int2 = da.einsum("ajbi->iajb", eri_mo[nocc:, :nocc, nocc:, :nocc]).compute()
        c = np.sqrt((2 * I_) - 1)
        K = (int1 - ((-1) ** I_) * int2) / e_iajb
        K *= c

        if I_ == 2:
            i, j = np.meshgrid(occ, occ, indexing="ij")
            i_ = np.where(i == j)[0]
            j_ = np.where(i == j)[1]
            K[i_, :, j_, :] = 0
            a, b = np.meshgrid(vir, vir, indexing="ij")
            a_ = np.where(a == b)[0]
            b_ = np.where(a == b)[1]
            K[:, a_, :, b_] = 0
        return K

    def da_from_array(self, array):
        """Method to convert np array to dask array
        Args:
            array (np.ndarray): array to convert
        """
        chunk = (
            array.shape[0] // 2,
            array.shape[1],
            array.shape[2] // 2,
            array.shape[3],
        )

        array_da = da.from_array(array, chunks=chunk)

        return array_da

    @property
    def part_a2(self):
        """Method for obtain A(2) matrix using einsum
        equation C.13 in Oddershede 1984
        The A = (A + A_)/2 term is because of c.13a equation

        Returns:
            np.ndarray: (nocc,nvir,nocc,nvir) array with A(2) contribution
                #A = np.einsum('mn,ij->minj', mask_mn, A)
        #A = -.5*np.einsum('jadb,iadb->ij',int_,k)

        """
        nocc = self.nocc
        nvir = self.nvir
        self.k_1 = self.kappa(1)
        self.k_2 = self.kappa(2)
        k_1 = self.k_1
        k_2 = self.k_2
        cte = self.cte
        k = k_1 + cte * k_2
        nmo = self.nmo
        k_da = self.da_from_array(k)
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks=(nmo // 4, nmo // 4))
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int_ = eri_mo[:nocc, nocc:, :nocc, nocc:]
            A1 = da.einsum("jadb,iadb->ij", int_, k_da).compute()
            A2 = da.einsum("dbpn,pmdb->mn", int_, k_da).compute()

        mask_mn = np.eye(nvir)
        mask_ab = np.eye(nocc)
        A = np.einsum("mn,ij->minj", mask_mn, -0.5 * A1)
        A += np.einsum("ij,mn->minj", mask_ab, -0.5 * A2)
        A_ = np.einsum("aibj->bjai", A)
        A = (A + A_) / 2
        A = np.einsum("aibj->iajb", A)
        return A

    def part_b2(self, S):
        """Method for obtain B(2) matrix (eq. 14 in Oddershede Paper)
        but using einsum function
        Args:
            S (int): Multiplicity of the response, triplet (S=1) or singlet(S=0)

        Returns:
            np.array: (nvir,nocc,nvir,nocc) array with B(2) matrix
        """
        nocc = self.nocc
        nmo = self.nmo
        k_1 = self.k_1
        k_2 = self.k_2
        cte = self.cte
        cte2 = (-1) ** S
        k_b1 = self.da_from_array(k_1 + cte * k_2)
        k_b2 = self.da_from_array(k_1 + (cte / (1 - 4 * S)) * k_2)
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")

            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1 = eri_mo[:nocc, nocc:, nocc:, :nocc]
            int2 = eri_mo[:nocc, :nocc, nocc:, nocc:]
            int3 = eri_mo[:nocc, :nocc, :nocc, :nocc]
            int4 = eri_mo[nocc:, nocc:, nocc:, nocc:]
            B = da.einsum("anrp,bmpr->manb", int1, k_b1).compute()
            B += da.einsum("bmrp,anpr->manb", int1, k_b1).compute()
            B += cte2 * da.einsum("aprn,pmbr->manb", int2, k_b2).compute()
            B += cte2 * da.einsum("bprm,pnar->manb", int2, k_b2).compute()
            B -= cte2 * da.einsum("bpad,dmpn->manb", int3, k_b2).compute()
            B -= cte2 * da.einsum("qmpn,bpaq->manb", int4, k_b2).compute()
        B = np.einsum("aibj->iajb", B)
        return 0.5 * B

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
        k = self.da_from_array(k_1 + self.cte * k_2)
        nmo = self.nmo
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int_ = eri_mo[:nocc, nocc:, :nocc, nocc:]
            S2 = da.einsum("japb,iapb->ij", int_ / e_iajb, k).compute()
            S2_ = da.einsum("dapn,pmda->mn", int_ / e_iajb, k).compute()
        mask_mn = np.eye(nvir)
        mask_ij = np.eye(nocc)
        S2 = np.einsum("mn,ij->minj", mask_mn, -0.5 * S2)
        S2 += np.einsum("ij,mn->minj", mask_ij, -0.5 * S2_)
        E = lib.direct_sum(
            "a+b-i-j->aibj",
            mo_energy[viridx],
            mo_energy[viridx],
            mo_energy[occidx],
            mo_energy[occidx],
        )
        S2 = np.einsum("aibj->iajb", 0.5 * S2 * E)
        return S2

    @property
    def kappa_2(self):
        """property with kappa in equation C.24
        with einsum
        Returns:
            np.narray: (nocc,nvir) array
        """
        nocc = self.nocc
        mo_energy = self.mo_energy
        occidx = self.occidx
        viridx = self.viridx
        nmo = self.nmo
        k_1 = self.k_1
        k_2 = self.k_2
        e_ia = lib.direct_sum("i-a->ia", mo_energy[occidx], mo_energy[viridx])
        k = k_1 + self.cte * k_2
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1 = eri_mo[:nocc, nocc:, nocc:, nocc:]
            int2 = eri_mo[:nocc, nocc:, :nocc, :nocc]
            kappa = da.einsum("pamb,paib->im", int1, k).compute()
            kappa -= da.einsum("padi,padm->im", int2, k).compute()
        return kappa / e_ia

    def correction_pert(self, FC=False, PSO=False, FCSD=False, atmlst=None):
        """Method with eq. C.25, which is the first correction to perturbator

        Args:
            atmlst (list): Nuclei to which will calculate the correction

        Returns:
            np.ndarray: array with first correction to Perturbator (nocc,nvir)
        """

        kappa = self.kappa_2
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            p_virt = h1[nocc:, nocc:]
            pert = np.einsum("an,mn->am", kappa, p_virt)
            p_occ = h1[:nocc, :nocc]
            pert -= np.einsum("bm,ba->am", kappa, p_occ)
        if PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)[0]
            p_virt = h1[:, nocc:, nocc:]
            pert = np.einsum("an,xmn->xam", kappa, p_virt)
            p_occ = h1[:, :nocc, :nocc]
            pert -= np.einsum("bm,xba->xam", kappa, p_occ)
        elif FCSD:
            h1 = self.rpa_obj.pert_fcsd(atmlst)
            h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0, :, :, :, :]
            p_virt = h1[:, :, nocc:, nocc:]
            p_occ = h1[:, :, :nocc, :nocc]
            pert = np.einsum("an,wxmn->wxam", kappa, p_virt)
            pert -= np.einsum("bm, wxba->wxam", kappa, p_occ)
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
        nmo = self.nocc + self.nvir
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        k_1 = self.k_1
        k_2 = self.k_2
        nmo = self.nmo
        e_iajb = lib.direct_sum(
            "i+j-a-b->iajb",
            self.mo_energy[self.occidx],
            self.mo_energy[self.occidx],
            self.mo_energy[self.viridx],
            self.mo_energy[self.viridx],
        )
        k = self.da_from_array(k_1 + self.cte * k_2)
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            h1 = h1[nocc:, :nocc]
            with h5py.File(str(self.h5_file), "r") as f:
                eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
                eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
                int_e = eri_mo[:nocc, nocc:, :nocc, nocc:] / e_iajb
                pert = -da.einsum("dapb,md,iapb->im", int_e, h1, k).compute()
                pert -= da.einsum("dapb,bi,pmda->im", int_e, h1, k).compute()
        if PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)
            h1 = h1[0][:, nocc:, :nocc]
            h1 = da.from_array(
                h1, chunks=(h1[0].shape[0], h1[1].shape[1] // 2, h1[2].shape[0] // 2)
            )
            with h5py.File(str(self.h5_file), "r") as f:
                eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
                eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
                int_e = eri_mo[:nocc, nocc:, :nocc, nocc:] / e_iajb
                pert = -da.einsum("dapb,xmd,iapb->xim", int_e, h1, k).compute()
                pert -= da.einsum("dapb,xbi,pmda->xim", int_e, h1, k).compute()
        elif FCSD:
            h1 = self.rpa_obj.pert_fcsd(atmlst)
            h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0, :, :, nocc:, :nocc]
            h1 = self.da_from_array(h1)
            with h5py.File(str(self.h5_file), "r") as f:
                eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
                eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
                int_e = eri_mo[:nocc, nocc:, :nocc, nocc:] / e_iajb
                pert = -da.einsum("dapb,wxmd,iapb->wxim", int_e, h1, k).compute()
                pert -= da.einsum("dapb,wxbi,pmda->wxim", int_e, h1, k).compute()
        return pert

    def Communicator(self, triplet):
        """Function for obtain Communicator matrix, i.e., the principal propagator
        inverse without the A(0) matrix

        Args:
            triplet (bool, optional): Triplet or singlet quantum communicator matrix.
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

    def pp_ssc_fc(self, atm1lst, atm2lst, elements=False):
        """Method that obtain the linear response between two FC perturbation at
        HRPA level of approach between two nuclei
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
            p = -np.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = np.einsum("ia,iajb,jb", h1, p, h2)
            para.append(e / 4)
            fc = np.einsum(",k,xy->kxy", nist.ALPHA**4, para, np.eye(3))
            return fc

    def M_rpa(self, triplet, communicator=False):
        """Principal Propagator Inverse at RPA, defined as M = A+B

        A[i,a,j,b] = delta_{ab}delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)

        ref: G.A Aucar  https://doi.org/10.1002/cmr.a.20108

        Args:
                triplet (bool, optional): defines if the response is triplet (TRUE)
                or singlet (FALSE), that changes the Matrix M. Defaults is True.

        Returns:
                numpy.ndarray: M matrix
        """
        nmo = self.nmo
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
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            a -= da.einsum(
                "ijba->iajb", eri_mo[: self.nocc, : self.nocc, self.nocc :, self.nocc :]
            ).compute()
            if triplet:
                b -= da.einsum(
                    "jaib->iajb",
                    eri_mo[: self.nocc, self.nocc :, : self.nocc, self.nocc :],
                ).compute()
            elif not triplet:
                b += da.einsum(
                    "jaib->iajb",
                    eri_mo[: self.nocc, self.nocc :, : self.nocc, self.nocc :],
                ).compute()
        m = a + b
        m = m.reshape(self.nocc * self.nvir, self.nocc * self.nvir, order="C")
        return m

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
            p = np.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = np.einsum("xia,iajb,yjb->xy", h1, p, h2)
            para.append(e)
            pso = np.asarray(para) * nist.ALPHA**4
            return pso

    def pp_ssc_fcsd(self, atm1lst, atm2lst, elements=False):
        """Method that obtain the linear response between two FC+SD perturbation at
        HRPA level of approach between two nuclei
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
        h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0, :, :, :nocc, nocc:]
        h2 = self.rpa_obj.pert_fcsd(atm2lst)
        h2 = np.asarray(h2).reshape(-1, 3, 3, ntot, ntot)[0, :, :, :nocc, nocc:]
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
            p = -np.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = np.einsum("wxia,iajb,wyjb->xy", h1, p, h2)

            para.append(e)
            fcsd = np.asarray(para) * nist.ALPHA**4
            return fcsd

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
