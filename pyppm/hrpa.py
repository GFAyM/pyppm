from pyscf import gto, scf
from pyscf.gto import Mole
import numpy
from pyscf import lib
import attr
from pyscf import ao2mo
from pyscf import lib
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from functools import reduce
from pyppm.rpa import RPA
from itertools import product


@attr.s
class HRPA(RPA):
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
        self.occ = [i for i in range(self.nocc)]
        self.vir = [i for i in range(self.nvir)]    

    def kappa(self, I):
        """
        Method for obtain kappa_{\alpha \beta}^{m n} in a matrix form
        K_{ij}^{a,b} = [(1-\delta_{ij})(1-\delta_ab]^{I-1}(2I-1)^.5
                        * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]

        for i \noteq j, a \noteq b

        K_{ij}^{a,b}=1^{I-1}(2I-1)^.5 * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]

        Oddershede 1984, eq C.7

        Args:
            I (integral): 1 or 2.

        Returns:
            numpy.ndarray: (nocc,nvir,nocc,nvir) array with \kappa
        """
        nocc = self.nocc
        nvir = self.nvir
        occidx = self.occidx
        viridx = self.viridx
        occ = self.occ
        vir = self.vir
        mo_energy = self.mo_energy
        e_iajb = lib.direct_sum(
            "i+j-b-a->iajb",
            mo_energy[occidx],
            mo_energy[occidx],
            mo_energy[viridx],
            mo_energy[viridx],
        )
        int1 = lib.einsum("aibj->iajb", self.eri_mo[nocc:, :nocc, nocc:, :nocc])
        int2 = lib.einsum("ajbi->iajb", self.eri_mo[nocc:, :nocc, nocc:, :nocc])
        c = numpy.sqrt((2 * I) - 1)
        K = (int1 - (((-1) ** I) * int2)) / e_iajb
        K = K * c
        if I==2:
            for i,j in list(product(occ,occ)):
                if i==j:
                    K[i,:,j,:]=0
            for a,b in list(product(vir,vir)):
                if a==b:
                    K[:,a,:,b]=0
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
        occ = self.occ
        vir = self.vir
        for alfa, beta, m, n in list(product(occ,occ,vir,vir)):
            if n == m:
                k = k_1[alfa, :, :, :] + (
                    numpy.sqrt(3) * k_2[alfa, :, :, :]
                )
                A[m, alfa, n, beta] -= (0.5) * lib.einsum(
                    "adb,adb->", int[beta, :, :, :], k
                )
            if alfa == beta:
                k = k_1[:, m, :, :] + (numpy.sqrt(3) * k_2[:, m, :, :])
                A[m, alfa, n, beta] -= (0.5) * lib.einsum(
                    "dbp,pdb->", int[:, :, :, n], k
                )
        A_ = lib.einsum("aibj->bjai", A)
        A = (A + A_) / 2
        A = lib.einsum("aibj->iajb", A)
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
        occ = self.occ
        vir = self.vir
        eri_mo = self.eri_mo
        int1 = eri_mo[:nocc, nocc:, nocc:, :nocc]
        int2 = eri_mo[:nocc, :nocc, nocc:, nocc:]
        int3 = eri_mo[:nocc, :nocc, :nocc, :nocc]
        int4 = eri_mo[nocc:, nocc:, nocc:, self.nocc :]
        k_1 = self.kappa(1)
        k_2 = self.kappa(2)
        B = numpy.zeros((nvir, nocc, nvir, nocc))
        for alfa, beta, m, n in list(product(occ,occ,vir,vir)):
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
            B[m, alfa, n, beta] += 0.5 * (
                lib.einsum("rp,pr->", int1[alfa, n, :, :], k_b1)
                + lib.einsum("rp,pr->", int1[beta, m, :, :], k_b2)
                + ((-1) ** S)
                * (
                    (
                        lib.einsum("pr,pr->", int2[alfa, :, :, n], k_b3)
                        + lib.einsum("pr,pr->", int2[beta, :, :, m], k_b4)
                    )
                    - lib.einsum("pd,dp->", int3[beta, :, alfa, :], k_b5)
                    - lib.einsum("qp,pq->", int4[:, m, :, n], k_b6)
                )
            )
        B = lib.einsum("aibj->iajb", B)
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
        occ = self.occ
        vir = self.vir
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
        for alfa, beta, m, n in list(product(occ,occ,vir,vir)):
            if m == n:
                k_s2_1 = (
                    k_1[alfa, :, :, :] + numpy.sqrt(3) * k_2[alfa, :, :, :]
                )
                S2[m, alfa, n, beta] -= 0.5 * lib.einsum(
                    "apb,apb->",
                    int[beta, :, :, :] / e_iajb[beta, :, :, :],
                    k_s2_1,
                )
            if alfa == beta:
                k_s2_2 = k_1[:, m, :, :] + (numpy.sqrt(3) * k_2[:, m, :, :])

                S2[m, alfa, n, beta] -= 0.5 * lib.einsum(
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
        S2 = lib.einsum("aibj->iajb", 0.5 * S2 * E)
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
        occ = self.occ
        vir = self.vir
        k_1 = self.kappa(1)
        k_2 = self.kappa(2)
        e_ia = lib.direct_sum("i-a->ia", mo_energy[occidx], mo_energy[viridx])
        int1 = self.eri_mo[:nocc, nocc:, nocc:, nocc:]
        int2 = self.eri_mo[:nocc, nocc:, :nocc, :nocc]
        kappa = numpy.zeros((nocc, nvir))
        for alfa, m in list(product(occ,vir)):
            kappa[alfa, m] = lib.einsum(
                "pab,pab->",
                int1[:, :, m, :],
                (k_1[:, :, alfa, :] + numpy.sqrt(3) * k_2[:, :, alfa, :]),
            )
            kappa[alfa, m] -= lib.einsum(
                "pad,pad->",
                int2[:, :, :, alfa],
                (k_1[:, :, :, m] + numpy.sqrt(3) * k_2[:, :, :, m]),
            )
            kappa[alfa, m] = kappa[alfa, m] / e_ia[alfa, m]
        return kappa

    def correction_pert(self, FC=False,PSO=False,FCSD=False, atmlst=None):
        """Method with eq. C.25, which is the first correction to perturbator

        Args:
            atmlst (list): Nuclei to which will calculate the correction

        Returns:
            numpy.ndarray: array with first correction to Perturbator (nocc,nvir)
        """
        kappa = self.kappa_2
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        occ = self.occ
        vir = self.vir
        if FC:
            h1 = self.pert_fc(atmlst)[0]
            pert = numpy.zeros((nvir, nocc))
            for alfa, m in list(product(occ,vir)):
                p_virt = h1[nocc:, nocc:]
                pert[m, alfa] += lib.einsum("n,n->", kappa[alfa, :], p_virt[m, :])
                p_occ = h1[:nocc, :nocc]
                pert[m, alfa] -= lib.einsum("b,b->", kappa[:, m], p_occ[:, alfa])
            pert = lib.einsum("ai->ia", pert)
        if PSO:
            h1 = self.pert_pso(atmlst)
            h1 = numpy.asarray(h1).reshape(1, 3, ntot, ntot)[0]
            pert = numpy.zeros((3,nvir,nocc))
            for alfa, m in list(product(occ,vir)):
                p_virt = h1[:,nocc:,nocc:]
                pert[:,m,alfa] += lib.einsum('n,xn->x', kappa[alfa,:],p_virt[:,m,:])
                p_occ = h1[:,:nocc,:nocc]
                pert[:,m,alfa] -= lib.einsum('b,xb->x', kappa[:,m],p_occ[:,:,alfa])
            pert = lib.einsum('xai->xia', pert)
        elif FCSD:
            h1 = self.pert_fcsd(atmlst)
            h1 = numpy.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0,:,:,:,:]
            pert = numpy.zeros((3,3,nvir,nocc))
            for alfa, m in list(product(occ,vir)):
                p_virt = h1[:,:,nocc:,nocc:]
                pert[:,:,m,alfa] += lib.einsum('n,wxn->wx', kappa[alfa,:],p_virt[:,:,m,:])
                p_occ = h1[:,:,:nocc,:nocc]
                pert[:,:,m,alfa] -= lib.einsum('b,wxb->wx', kappa[:,m],p_occ[:,:,:,alfa])
            pert = lib.einsum('wxai->wxia', pert)
        return pert

    def correction_pert_2(self, FC=False,PSO=False,FCSD=False, atmlst=None):
        """Method with C.26 correction, which is a correction to perturbator
        centered in atmslt

        Args:
            atmlst (list): nuclei in which is centered the correction

        Returns:
            numpy.ndarray: array with second correction to Perturbator
            (nocc,nvir)
        """
        nmo = self.nocc + self.nvir
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        occ = self.occ
        vir = self.vir
        eri_mo = self.eri_mo.reshape(nmo, nmo, nmo, nmo)
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
        if FC:
            h1 = self.pert_fc(atmlst)[0]
            pert = numpy.zeros((nvir, nocc))

            h1 = h1[nocc:, :nocc]
            for alfa, m in list(product(occ,vir)):
                t = lib.einsum(
                    "dapb,d,apb->",
                    int1[:, :, :, :] / e_iajb[:, :, :, :],
                    h1[m, :],
                    (k_1[alfa, :, :, :] + c1 * k_2[alfa, :, :, :]),
                )
                t += lib.einsum(
                    "dapb,b,pda",
                    int1[:, :, :, :] / e_iajb[:, :, :, :],
                    h1[:, alfa],
                    (k_1[:, m, :, :] + c1 * k_2[:, m, :, :]),
                )
                t = -t
                pert[m, alfa] = t
            pert = lib.einsum("ai->ia", pert)
        if PSO:
            h1 = self.pert_pso(atmlst)
            h1 = numpy.asarray(h1).reshape(1, 3, ntot, ntot)
            h1 = h1[0]
            pert = numpy.zeros((3,nvir,nocc))
            
            h1 = h1[:,nocc:,:nocc]
            for alfa, m in list(product(occ,vir)):    
                t = lib.einsum('dapb,xd,apb->x',int1[:,:,:,:]/e_iajb[:,:,:,:], h1[:,m,:],
                                    (k_1[alfa,:,:,:]+c1*k_2[alfa,:,:,:]))
                t += lib.einsum('dapb,xb,pda->x',int1[:,:,:,:]/e_iajb[:,:,:,:],h1[:,:,alfa],
                                    (k_1[:,m,:,:]+c1*k_2[:,m,:,:]))
                t = -t#*numpy.sqrt(2)/2  
                pert[:,m,alfa] = t
            pert = lib.einsum('xai->xia', pert)
        elif FCSD:
            h1 = self.pert_fcsd(atmlst)
            h1 = numpy.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0,:,:,nocc:,:nocc]
            pert = numpy.zeros((3,3,nvir,nocc))
            for alfa, m in list(product(occ,vir)):
                t = lib.einsum('dapb,wxd,apb->wx',int1[:,:,:,:]/e_iajb[:,:,:,:], h1[:,:,m,:],
                                    (k_1[alfa,:,:,:]+c1*k_2[alfa,:,:,:]))
                t += lib.einsum('dapb,wxb,pda->wx',int1[:,:,:,:]/e_iajb[:,:,:,:],h1[:,:,:,alfa],
                                    (k_1[:,m,:,:]+c1*k_2[:,m,:,:]))
                t = -t#*numpy.sqrt(2)/2  
                pert[:,:,m,alfa] = t
            pert = lib.einsum('wxai->wxia', pert)
        return pert
    
    def communicator_matrix_hrpa(self, triplet):
        """Function for obtain Communicator matrix, i.e., the principal propagator 
        inverse without the A(0) matrix

        Args:
            triplet (bool, optional): Triplet or singlet quantum communicator matrix. 
            Defaults to True.

        Returns:
            numpy.ndarray: Quantum communicator matrix
        """
        nvir = self.nvir
        nocc = self.nocc
        m = self.M_sin_a0(triplet=triplet)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m += self.S2
        if triplet:
            m -= self.part_b2(1)
        else:            
            m += self.part_b2(0)
        m = m.reshape(nocc * nvir, nocc * nvir)
        return m

    def pp_ssc_fc_select(self, atom1=None, atom2=None, elements=False):
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
        h1_corr1 = self.correction_pert(atmlst=atom1, FC=True)
        h1_corr2 = self.correction_pert_2(atmlst=atom1, FC=True)
        h2_corr1 = self.correction_pert(atmlst=atom2, FC=True)
        h2_corr2 = self.correction_pert_2(atmlst=atom2, FC=True)

        h1 = (2 * h1) + h1_corr1 + h1_corr2
        h2 = (2 * h2) + h2_corr1 + h2_corr2
        m = self.M(triplet=True)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m -= self.part_b2(1)
        m += self.S2
        m = m.reshape(nocc * nvir, nocc * nvir)
        if elements:
            return h1, m, h2
        else:
            p = -numpy.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = lib.einsum("ia,iajb,jb", h1, p, h2)
            para.append(e/4)
            fc = lib.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))
            return fc

    def pp_ssc_pso_select(self, atom1, atom2, elements=False):
        """Method that obtain the linear response between PSO perturbation at
        HRPA level of approach between two nuclei
        Args:
            atom1 (list): First nuclei
            atom2 (list): Second nuclei

        Returns:
            real: PSO response at HRPA level of approach
        """
        nvir = self.nvir
        nocc = self.nocc
        ntot = nocc + nvir        
        h1 = self.pert_pso(atom1)
        h1 = numpy.asarray(h1).reshape(1, 3, ntot, ntot)
        h1 = h1[0][:,:nocc,nocc:]
        h2 = self.pert_pso(atom2)
        h2 = numpy.asarray(h2).reshape(1, 3, ntot, ntot)
        h2 = h2[0][:,:nocc,nocc:]

        h1_corr1 = self.correction_pert(atmlst=atom1, PSO=True)
        h1_corr2 = self.correction_pert_2(atmlst=atom1, PSO=True)
        h2_corr1 = self.correction_pert(atmlst=atom2, PSO=True)
        h2_corr2 = self.correction_pert_2(atmlst=atom2, PSO=True)

        h1 = (-2*h1) + h1_corr1 + h1_corr2
        h2 = (-2*h2) + h2_corr1 + h2_corr2
        m = self.M(triplet=False)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m += self.part_b2(0)
        m += self.S2
        m = m.reshape(nocc*nvir,nocc*nvir)
        if elements:
            return h1, m, h2
        else:
            p = numpy.linalg.inv(m)
            p = -p.reshape(nocc,nvir,nocc,nvir)
            para = []
            e = lib.einsum('xia,iajb,yjb->xy', h1, p , h2)
            para.append(e)
            pso = numpy.asarray(para) * nist.ALPHA ** 4
            return pso

    def pp_ssc_fcsd_select(self, atom1=None, atom2=None, elements=False):
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
        ntot = nocc + nvir 
        h1 = self.pert_fcsd(atom1)
        h1 = numpy.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0,:,:,:nocc, nocc:]
        h2 = self.pert_fcsd(atom2)
        h2 = numpy.asarray(h2).reshape(-1, 3, 3, ntot, ntot)[0,:,:,:nocc, nocc:]
        h1_corr1 = self.correction_pert(atmlst=atom1, FCSD=True)
        h1_corr2 = self.correction_pert_2(atmlst=atom1, FCSD=True)
        h2_corr1 = self.correction_pert(atmlst=atom2, FCSD=True)
        h2_corr2 = self.correction_pert_2(atmlst=atom2, FCSD=True)

        h1 = (2 * h1) + h1_corr1 + h1_corr2
        h2 = (2 * h2) + h2_corr1 + h2_corr2
        m = self.M(triplet=True)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m -= self.part_b2(1)
        m += self.S2
        m = m.reshape(nocc * nvir, nocc * nvir)
        if elements:
            return h1, m, h2
        else:
            p = -numpy.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = numpy.einsum("wxia,iajb,wyjb->xy", h1, p, h2)

            para.append(e)
            fcsd = numpy.asarray(para) * nist.ALPHA ** 4
            return fcsd    

    def ssc_hrpa(self, FC=True, FCSD=False, PSO=False, atom1=None, atom2=None):

        atom1_ = [self.obtain_atom_order(atom1)]
        atom2_ = [self.obtain_atom_order(atom2)]
        if FC:
            prop = self.pp_ssc_fc_select(atom1=atom1_, atom2=atom2_)
        if PSO:
            prop = self.pp_ssc_pso_select(atom1=atom1_, atom2=atom2_)
        elif FCSD:
            prop = self.pp_ssc_fcsd_select(atom1=atom1_, atom2=atom2_)
        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton ** 2
        iso_ssc = unit * lib.einsum("kii->k", prop) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        jtensor = lib.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]

    def elements(self, atom1, atom2, FC=False, FCSD=False, PSO=False):
        """_summary_

        Args:
            FC (bool, optional): _description_. Defaults to False.
            FCSD (bool, optional): _description_. Defaults to False.
            PSO (bool, optional): _description_. Defaults to False.
            atom1 (_type_, optional): _description_. Defaults to None.
            atom2 (_type_, optional): _description_. Defaults to None.
        """

        
        if FC:
            h1,m,h2 = self.pp_ssc_fc_select(atom1=atom1, atom2=atom2, elements=True)
        if PSO:
            h1,m,h2 = self.pp_ssc_pso_select(atom1, atom2, elements=True)
        elif FCSD:
            h1,m,h2 = self.pp_ssc_fcsd_select(atom1=atom1, atom2=atom2, elements=True)
        return h1,m,h2