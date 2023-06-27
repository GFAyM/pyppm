from pyscf import gto, scf
from pyscf.gto import Mole
import numpy
from pyscf import lib
import attr
from pyscf import ao2mo
from pyscf.dft import numint
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from functools import reduce
from pyppm.ssc_pol_prop import Prop_pol


@attr.s
class HRPA(Prop_pol):
    """Class to perform calculations of $J^{FC}$ mechanism at HRPA level of
    of approach. This is the p-h part of SOPPA level of approah. The HRPA class
    enherits from Prop_pol of pyppm.ssc_pol_prop because they share several methods
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

    def __attrs_post_init__(self):
        self.mo_occ = self.mf.mo_occ
        self.mo_energy = self.mf.mo_energy
        self.mo_coeff = self.mf.mo_coeff
        self.mol = self.mf.mol
        self.occidx = numpy.where(self.mo_occ > 0)[0]
        self.viridx = numpy.where(self.mo_occ == 0)[0]

        self.orbv = self.mo_coeff[:, self.viridx]
        self.orbo = self.mo_coeff[:, self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        self.mo = numpy.hstack((self.orbo, self.orbv))
        self.nmo = self.nocc + self.nvir
        mol = self.mol
        mo = self.mo
        nmo = self.nmo
        eri_mo = ao2mo.general(mol, [mo, mo, mo, mo], compact=False)
        self.eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)

    def kappa(self, I):
        """
        Method for obtain kappa_{\alpha \beta}^{m n} in a matrix form
        K_{ij}^{a,b} = [(1-\delta_{ij})(1-\delta_ab]^{I-1}(2I-1)^.5
                        * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]

        for i \noteq j, a \noteq b

        K_{ij}^{a,b}=1^{I-1}(2I-1)^.5 * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]

        Oddershede 1984, eq C.7

        But, this follows Andy's program, the terms i==j and a==b are not seted
        to zero in order to get to the right values

        Args:
            I (integral): 1 or 2.

        Returns:
            numpy.ndarray: (nocc,nvir,nocc,nvir) array with \kappa
        """
        nocc = self.nocc
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
        int1 = numpy.einsum("aibj->iajb", self.eri_mo[nocc:, :nocc, nocc:, :nocc])
        int2 = numpy.einsum("ajbi->iajb", self.eri_mo[nocc:, :nocc, nocc:, :nocc])
        c = numpy.sqrt((2 * I) - 1)
        K = (int1 - (((-1) ** I) * int2)) / e_iajb
        K = K * c
        # for i in range(nocc):
        #    for j in range(nocc):
        #        if i==j:
        #            K[i,:,j,:]=0
        # for a in range(nvir):
        #    for b in range(nvir):
        #        if a==b:
        #            K[:,a,:,b]=0
        return K

    @property
    def part_a2(self):
        """Method for obtain A(2) matrix
        equation C.13 in Oddershede 1984
        The A = (A + A_)/2 term is because of c.13a equation

        Returns:
            numpy.ndarray: (nocc,nvir,nocc,nvir) array with A(2) contribution
        """
        nocc = self.nocc
        nvir = self.nvir
        int = self.eri_mo[:nocc, nocc:, :nocc, nocc:]
        k_1 = self.kappa(1)
        k_2 = self.kappa(2)
        A = numpy.zeros((nvir, nocc, nvir, nocc))

        for alfa in range(nocc):
            for beta in range(nocc):
                for m in range(nvir):
                    for n in range(nvir):
                        if n == m:
                            k = k_1[alfa, :, :, :] + (
                                numpy.sqrt(3) * k_2[alfa, :, :, :]
                            )
                            A[m, alfa, n, beta] -= (0.5) * numpy.einsum(
                                "adb,adb->", int[beta, :, :, :], k
                            )
                        if alfa == beta:
                            k = k_1[:, m, :, :] + (numpy.sqrt(3) * k_2[:, m, :, :])
                            A[m, alfa, n, beta] -= (0.5) * numpy.einsum(
                                "dbp,pdb->", int[:, :, :, n], k
                            )

        A_ = numpy.einsum("aibj->bjai", A)
        A = (A + A_) / 2
        A = numpy.einsum("aibj->iajb", A)
        return A

    def part_b2(self, S):
        """Method for obtain B(2) matrix (eq. 14 in Oddershede Paper)

        Args:
            S (int): Multiplicity of the response, triplet (S=1) or singlet(S=0)

        Returns:
            numpy.array: (nvir,nocc,nvir,nocc) array with B(2) matrix
        """
        nocc = self.nocc
        nvir = self.nvir
        eri_mo = self.eri_mo
        int1 = eri_mo[:nocc, nocc:, nocc:, :nocc]
        int2 = eri_mo[:nocc, :nocc, nocc:, nocc:]
        int3 = eri_mo[:nocc, :nocc, :nocc, :nocc]
        int4 = eri_mo[nocc:, nocc:, nocc:, self.nocc :]
        k_1 = self.kappa(1)
        k_2 = self.kappa(2)
        B = numpy.zeros((nvir, nocc, nvir, nocc))
        for alfa in range(nocc):
            for beta in range(nocc):
                for m in range(nvir):
                    for n in range(nvir):
                        k_b1 = k_1[beta, m, :, :] + numpy.sqrt(3) * k_2[beta, m, :, :]
                        k_b2 = k_1[alfa, n, :, :] + numpy.sqrt(3) * k_2[alfa, n, :, :]
                        k_b3 = (
                            k_1[:, m, beta, :]
                            + (numpy.sqrt(3) / (1 - (4 * S))) * k_2[:, m, beta, :]
                        )
                        k_b4 = (
                            k_1[:, n, alfa, :]
                            + (numpy.sqrt(3) / (1 - (4 * S))) * k_2[:, n, alfa, :]
                        )
                        k_b5 = (
                            k_1[:, m, :, n]
                            + (numpy.sqrt(3) / (1 - (4 * S))) * k_2[:, m, :, n]
                        )
                        k_b6 = (
                            k_1[beta, :, alfa, :]
                            + (numpy.sqrt(3) / (1 - 4 * S)) * k_2[beta, :, alfa, :]
                        )
                        # B[alfa,m,beta,n]
                        B[m, alfa, n, beta] += 0.5 * (
                            numpy.einsum("rp,pr->", int1[alfa, n, :, :], k_b1)
                            + numpy.einsum("rp,pr->", int1[beta, m, :, :], k_b2)
                            + ((-1) ** S)
                            * (
                                (
                                    numpy.einsum("pr,pr->", int2[alfa, :, :, n], k_b3)
                                    + numpy.einsum("pr,pr->", int2[beta, :, :, m], k_b4)
                                )
                                - numpy.einsum("pd,dp->", int3[beta, :, alfa, :], k_b5)
                                - numpy.einsum("qp,pq->", int4[:, m, :, n], k_b6)
                            )
                        )
        B = numpy.einsum("aibj->iajb", B)
        return B

    @property
    def S2(self):
        """Property with S(2) matrix elements (eq. C.9 in Oddershede 1984)
        This matrix will be multiplied by energy

        Returns:
            numpy.ndarray: (nocc,nvir,nocc,nvir) array with S(2) matrix
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
        k_1 = self.kappa(1)
        k_2 = self.kappa(2)
        S2 = numpy.zeros((nvir, nocc, nvir, nocc))
        int = self.eri_mo[:nocc, nocc:, :nocc, nocc:]
        for alfa in range(nocc):
            for beta in range(nocc):
                for m in range(nvir):
                    for n in range(nvir):
                        if m == n:
                            k_s2_1 = (
                                k_1[alfa, :, :, :] + numpy.sqrt(3) * k_2[alfa, :, :, :]
                            )
                            S2[m, alfa, n, beta] -= 0.5 * numpy.einsum(
                                "apb,apb->",
                                int[beta, :, :, :] / e_iajb[beta, :, :, :],
                                k_s2_1,
                            )
                        if alfa == beta:
                            k_s2_2 = k_1[:, m, :, :] + (numpy.sqrt(3) * k_2[:, m, :, :])

                            S2[m, alfa, n, beta] -= 0.5 * numpy.einsum(
                                "dap,pda->",
                                int[:, :, :, n] / e_iajb[:, :, :, n],
                                k_s2_2,
                            )
        E = lib.direct_sum(
            "a+b-i-j->aibj",
            mo_energy[viridx],
            mo_energy[viridx],
            mo_energy[occidx],
            mo_energy[occidx],
        )
        S2 = numpy.einsum("aibj->iajb", 0.5 * S2 * E)
        return S2

    @property
    def kappa_2(self):
        """property with \kappa in equation C.24

        Returns:
            numpy.narray: (nocc,nvir) array
        """
        nocc = self.nocc
        nvir = self.nvir
        mo_energy = self.mo_energy
        occidx = self.occidx
        viridx = self.viridx
        k_1 = self.kappa(1)
        k_2 = self.kappa(2)
        e_ia = lib.direct_sum("i-a->ia", mo_energy[occidx], mo_energy[viridx])
        int1 = self.eri_mo[:nocc, nocc:, nocc:, nocc:]
        int2 = self.eri_mo[:nocc, nocc:, :nocc, :nocc]
        kappa = numpy.zeros((nocc, nvir))
        for alfa in range(nocc):
            for m in range(nvir):
                kappa[alfa, m] = numpy.einsum(
                    "pab,pab->",
                    int1[:, :, m, :],
                    (k_1[:, :, alfa, :] + numpy.sqrt(3) * k_2[:, :, alfa, :]),
                )
                kappa[alfa, m] -= numpy.einsum(
                    "pad,pad->",
                    int2[:, :, :, alfa],
                    (k_1[:, :, :, m] + numpy.sqrt(3) * k_2[:, :, :, m]),
                )
                kappa[alfa, m] = kappa[alfa, m] / e_ia[alfa, m]
        return kappa

    def correction_pert(self, atmlst):
        """Method with eq. C.25, which is the first correction to perturbator

        Args:
            atmlst (list): Nuclei to which will calculate the correction

        Returns:
            numpy.ndarray: array with first correction to Perturbator (nocc,nvir)
        """
        h1 = self.pert_fc(atmlst)[0]
        kappa = self.kappa_2
        nocc = self.nocc
        nvir = self.nvir
        pert = numpy.zeros((nvir, nocc))
        for alfa in range(self.nocc):
            for m in range(self.nvir):
                p_virt = h1[nocc:, nocc:]
                pert[m, alfa] += numpy.einsum("n,n->", kappa[alfa, :], p_virt[m, :])
                p_occ = h1[:nocc, :nocc]
                pert[m, alfa] -= numpy.einsum("b,b->", kappa[:, m], p_occ[:, alfa])
        pert = numpy.einsum("ai->ia", pert)
        return pert

    def correction_pert_2(self, atmlst):
        """Method with C.26 correction, which is a correction to perturbator
        centered in atmslt

        Args:
            atmlst (list): nuclei in which is centered the correction

        Returns:
            numpy.ndarray: array with second correction to Perturbator
            (nocc,nvir)
        """
        mo = numpy.hstack((self.orbo, self.orbv))
        nmo = self.nocc + self.nvir
        nocc = self.nocc
        nvir = self.nvir
        eri_mo = ao2mo.general(self.mol, [mo, mo, mo, mo], compact=False)
        eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
        int1 = eri_mo[:nocc, nocc:, :nocc, nocc:]
        c1 = numpy.sqrt(3)
        k_1 = self.kappa(1)
        k_2 = self.kappa(2)
        e_iajb = lib.direct_sum(
            "i+j-a-b->iajb",
            self.mo_energy[self.occidx],
            self.mo_energy[self.occidx],
            self.mo_energy[self.viridx],
            self.mo_energy[self.viridx],
        )
        h1 = self.pert_fc(atmlst)[0]
        pert = numpy.zeros((nvir, nocc))

        h1 = h1[nocc:, :nocc]
        for alfa in range(nocc):
            for m in range(nvir):
                t = numpy.einsum(
                    "dapb,d,apb->",
                    int1[:, :, :, :] / e_iajb[:, :, :, :],
                    h1[m, :],
                    (k_1[alfa, :, :, :] + c1 * k_2[alfa, :, :, :]),
                )
                t += numpy.einsum(
                    "dapb,b,pda",
                    int1[:, :, :, :] / e_iajb[:, :, :, :],
                    h1[:, alfa],
                    (k_1[:, m, :, :] + c1 * k_2[:, m, :, :]),
                )
                t = -t
                pert[m, alfa] = t
        pert = numpy.einsum("ai->ia", pert)
        return pert

    def pp_ssc_fc_select(self, atom1, atom2):
        """Method that obtain the linear response between two FC perturbation at
        HRPA level of approach between two nuclei
        Args:
            atom1 (list): First nuclei
            atom2 (list): Second nuclei

        Returns:
            real: FC response at HRPA level of approach
        """
        nvir = self.nvir
        nocc = self.nocc
        h1 = self.pert_fc(atom1)[0][:nocc, nocc:]
        h2 = self.pert_fc(atom2)[0][:nocc, nocc:]
        h1_corr1 = self.correction_pert(atom1)
        h1_corr2 = self.correction_pert_2(atom1)
        h2_corr1 = self.correction_pert(atom2)
        h2_corr2 = self.correction_pert_2(atom2)

        h1 = (2 * h1) + h1_corr1 + h1_corr2
        h2 = (2 * h2) + h2_corr1 + h2_corr2
        m = self.M(triplet=True)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m -= self.part_b2(1)
        m += self.S2
        m = m.reshape(nocc * nvir, nocc * nvir)
        p = numpy.linalg.inv(m)
        p = p.reshape(nocc, nvir, nocc, nvir)
        para = []
        e = numpy.einsum("ia,iajb,jb", h1, p, h2)
        #para.append(e)
        # fc = numpy.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))
        return e