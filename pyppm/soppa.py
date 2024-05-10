from pyscf import lib, ao2mo, scf
import attr
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from pyppm.rpa import RPA
import numpy as np
import dask.array as da
from dask import delayed
import operator
import h5py
from memory_profiler import profile
#from line_profiler import profile

@attr.s
class SOPPA:
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
        nvir = self.nvir
        occidx = self.occidx
        viridx = self.viridx
        nmo = self.nmo
        mo_energy = self.mo_energy
        e_iajb = lib.direct_sum(
            "i+j-b-a->iajb",
            mo_energy[occidx],
            mo_energy[occidx],
            mo_energy[viridx],
            mo_energy[viridx],
        )
        self.eri_mo()
        c = np.sqrt((2 * I_) - 1)
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1 = da.einsum("aibj->iajb", eri_mo[nocc:, :nocc, nocc:, :nocc])
            int2 = da.einsum("ajbi->iajb", eri_mo[nocc:, :nocc, nocc:, :nocc])
            K = c*(int1 - ((-1) ** I_) * int2) / e_iajb

            if I_ == 2:
                delta_1 = 1 - np.eye(nocc)
                delta_2 = 1 - np.eye(nvir)
                deltas = np.einsum('ij,ab->iajb', delta_1, delta_2)
                K = K*deltas
            return K.compute()

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
        k_1 = self.k_1
        k_2 = self.k_2
        cte = self.cte
        k = k_1 + cte * k_2
        nmo = self.nmo
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            k_da = da.from_array(k, chunks='auto')
            int_ = eri_mo[:nocc, nocc:, :nocc, nocc:]
            A1 = da.einsum("jadb,iadb->ij", int_, k_da)
            A2 = da.einsum("dbpn,pmdb->mn", int_, k_da)
            mask_mn = da.eye(nvir)
            mask_ab = da.eye(nocc)
            A = da.einsum("mn,ij->minj", mask_mn, -0.5 * A1)
            A += da.einsum("ij,mn->minj", mask_ab, -0.5 * A2)
            A_ = da.einsum("aibj->bjai", A)
            A = (A + A_) / 2
            A = da.einsum("aibj->iajb", A)
            return A.compute()

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
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")

            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            k_b1 = da.from_array(k_1 + cte * k_2,
                                 chunks= 'auto')
            k_b2 = da.from_array(k_1 + (cte / (1 - 4 * S)) * k_2,
                                 chunks='auto')

            int1 = eri_mo[:nocc, nocc:, nocc:, :nocc]
            int2 = eri_mo[:nocc, :nocc, nocc:, nocc:]
            int3 = eri_mo[:nocc, :nocc, :nocc, :nocc]
            int4 = eri_mo[nocc:, nocc:, nocc:, nocc:]
            B = da.einsum("anrp,bmpr->ambn", int1, .5*k_b1)
            B += da.einsum("bmrp,anpr->ambn", int1, .5*k_b1)
            B += cte2 * da.einsum("aprn,pmbr->ambn", int2, .5*k_b2)
            B += cte2 * da.einsum("bprm,pnar->ambn", int2, .5*k_b2)
            B -= cte2 * da.einsum("bpad,dmpn->ambn", int3, .5*k_b2)
            B -= cte2 * da.einsum("qmpn,bpaq->ambn", int4, .5*k_b2)
            return  B.compute()
        
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
        nmo = self.nmo
        with h5py.File(str(self.h5_file), "r") as f:
            k = da.from_array(k_1 + self.cte * k_2, chunks='auto')
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int_ = eri_mo[:nocc, nocc:, :nocc, nocc:]
            S2 = da.einsum("japb,iapb->ij", int_ / e_iajb, k)
            S2_ = da.einsum("dapn,pmda->mn", int_ / e_iajb, k)
            mask_mn = da.eye(nvir)
            mask_ij = da.eye(nocc)
            S2 = da.einsum("mn,ij->imjn", mask_mn, -0.5 * S2)
            S2 += da.einsum("ij,mn->imjn", mask_ij, -0.5 * S2_)
            S2 = -.5*S2*e_iajb
            return S2.compute()
    
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
        with h5py.File(str(self.h5_file), "r") as f:
            k = da.from_array(k_1 + self.cte * k_2, chunks='auto')
            eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1 = eri_mo[:nocc, nocc:, nocc:, nocc:]
            int2 = eri_mo[:nocc, nocc:, :nocc, :nocc]
            kappa = da.einsum("pamb,paib->im", int1, k)
            kappa -= da.einsum("padi,padm->im", int2, k)
            kappa *= 1/e_ia
            return kappa.compute()

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
        kappa = da.from_array(self.kappa_2, chunks='auto')
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            p_virt = h1[nocc:, nocc:]
            p_virt = da.from_array(p_virt, chunks='auto')
            pert = da.einsum("an,mn->am", kappa, p_virt)
            p_occ = h1[:nocc, :nocc]
            p_occ = da.from_array(p_occ, chunks='auto')
            pert -= da.einsum("bm,ba->am", kappa, p_occ)
        if PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)[0]
            p_virt = h1[:, nocc:, nocc:]
            p_virt = da.from_array(p_virt, chunks='auto')
            pert = da.einsum("an,xmn->xam", kappa, p_virt)
            p_occ = h1[:, :nocc, :nocc]
            p_occ = da.from_array(p_occ, chunks='auto')
            pert -= da.einsum("bm,xba->xam", kappa, p_occ)
        elif FCSD:
            h1 = self.rpa_obj.pert_fcsd(atmlst)
            h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0, :, :, :, :]
            p_virt = h1[:, :, nocc:, nocc:]
            p_virt = da.from_array(p_virt, chunks='auto')
            p_occ = h1[:, :, :nocc, :nocc]
            p_occ = da.from_array(p_occ, chunks='auto')
            pert = da.einsum("an,wxmn->wxam", kappa, p_virt)
            pert -= da.einsum("bm, wxba->wxam", kappa, p_occ)
        return pert.compute()

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
        nmo = self.nmo
        e_iajb = lib.direct_sum(
            "i+j-a-b->iajb",
            self.mo_energy[self.occidx],
            self.mo_energy[self.occidx],
            self.mo_energy[self.viridx],
            self.mo_energy[self.viridx],
        )
        k = da.from_array(self.k_1 + self.cte * self.k_2, 
                          chunks='auto')
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            h1 = h1[nocc:, :nocc]
            with h5py.File(str(self.h5_file), "r") as f:
                eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
                eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
                int_e = eri_mo[:nocc, nocc:, :nocc, nocc:] / e_iajb
                pert = -da.einsum("dapb,md,iapb->im", int_e, h1, k)
                pert -= da.einsum("dapb,bi,pmda->im", int_e, h1, k)
                return pert.compute()
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
                pert = -da.einsum("dapb,xmd,iapb->xim", int_e, h1, k)
                pert -= da.einsum("dapb,xbi,pmda->xim", int_e, h1, k)
                return pert.compute()

        elif FCSD:
            h1 = self.rpa_obj.pert_fcsd(atmlst)
            h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0, :, :, nocc:, :nocc]
            h1 = self.da_from_array(h1)
            with h5py.File(str(self.h5_file), "r") as f:
                eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
                eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
                int_e = eri_mo[:nocc, nocc:, :nocc, nocc:] / e_iajb
                pert = -da.einsum("dapb,wxmd,iapb->wxim", int_e, h1, k)
                pert -= da.einsum("dapb,wxbi,pmda->wxim", int_e, h1, k)
                return pert.compute()
            
    
    def D(self):
        """D(0) matrix, eq C.18 in Oddershede 1984

        Returns:
            numpy.array: D(0) matrix
        """
        mo_energy = self.mo_energy
        occidx = self.occidx
        viridx = self.viridx
        #e_aibj = lib.direct_sum(
        #    "a+b-m-n->nbma",
        #    mo_energy[occidx],
        #    mo_energy[occidx],
        #    mo_energy[viridx],
        #    mo_energy[viridx])
        e_aibj = lib.direct_sum(
            "n+m-a-b->nbma",
            mo_energy[viridx],
            mo_energy[viridx],
            mo_energy[occidx],
            mo_energy[occidx]
        )
        #d = np.diag(e_aibj.ravel())
        #nocc = self.nocc
        #nvir = self.nvir
        d = e_aibj.ravel()

        return d

    @profile
    def c_1_singlet(self):
        """C.16 equation, for obtain 2p-2h C(i=1) for singlet 
        properties

        Returns:
            _type_: _description_
        """
        nmo = self.nmo
        nocc = self.nocc
        nvir = self.nvir
        with h5py.File(str(self.h5_file), "r") as f:
            #eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = f["eri_mo"][:]
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int1 = np.einsum('mbnp->nbmp', int1_)
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int2 = np.einsum('mpnb->nbmp', int2_)
            c1 = int1 + int2
            mask_ag = -np.eye(nocc)
            c_1 = np.einsum('ag,nbmp->nbmapg', mask_ag, c1)
            #second term eq. C.16
            int1 = np.einsum('manp->nmap', int1_)
            int2 = np.einsum('mpna->nmap', int2_)
            c2 = int1 + int2
            mask_bg = -np.eye(nocc)
            c_1 += np.einsum('bg,nmap->nbmapg', mask_bg, c2)
            #third term eq C.16
            mask_np = np.eye(nvir)
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int1 = np.einsum('magb->bmag', int3_)
            int2 = np.einsum('mbga->bmag', int3_)
            c3 = int1 + int2
            c_1 += np.einsum('np,bmag->nbmapg', mask_np, c3)
            # fourth term eq C.16
            mask_mp = np.eye(nvir)
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            int1 = np.einsum('gbna->nbag', int4_)
            int2 = np.einsum('ganb->nbag', int4_)
            c4 = int1 + int2
            c_1 += np.einsum('mp,nbag->nbmapg', mask_mp, c4)
            c_1 = c_1/np.sqrt(2)
            
        return -c_1

    @profile
    def c_1_singlet_best(self):
        """C.16 equation, for obtain 2p-2h C(i=1) for singlet 
        properties

        Returns:
            _type_: _description_
        """
        nmo = self.nmo
        nocc = self.nocc
        nvir = self.nvir
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks='auto')
            #eri_mo = f["eri_mo"][:]
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int1 = da.einsum('mbnp->nbmp', int1_)
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int2 = da.einsum('mpnb->nbmp', int2_)
            c1 = da.add(int1,int2)
            mask_ag = -np.eye(nocc)
            c_1 = da.einsum('ag,nbmp->nbmapg', mask_ag, c1)
            #second term eq. C.16
            int1 = da.einsum('manp->nmap', int1_)
            int2 = da.einsum('mpna->nmap', int2_)
            c2 = da.add(int1,int2)
            mask_bg = -da.eye(nocc)
            c_1 = da.add(c_1,da.einsum('bg,nmap->nbmapg', mask_bg, c2))
            #third term eq C.16
            mask_np = da.eye(nvir)
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int1 = da.einsum('magb->bmag', int3_)
            int2 = da.einsum('mbga->bmag', int3_)
            c3 = da.add(int1,int2)
            c_1 = da.add(c_1,da.einsum('np,bmag->nbmapg', mask_np, c3))
            # fourth term eq C.16
            mask_mp = da.eye(nvir)
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            int1 = da.einsum('gbna->nbag', int4_)
            int2 = da.einsum('ganb->nbag', int4_)
            c4 = da.add(int1,int2)
            c_1 = da.add(c_1, da.einsum('mp,nbag->nbmapg', mask_mp, c4))
            c_1 = -c_1/np.sqrt(2)
            #da.to_hdf5('c_1.h5', '/data', c_1, compute=True)
            return c_1.compute()
    @profile
    def c_1_singlet_other(self):
        """C.16 equation, for obtain 2p-2h C(i=1) for singlet 
        properties

        Returns:
            _type_: _description_
        """
        nmo = self.nmo
        nocc = self.nocc
        nvir = self.nvir
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array(f["eri_mo"][:], chunks="auto")
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int1 = delayed(np.einsum)('mbnp->nbmp', int1_)
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int2 = delayed(np.einsum)('mpnb->nbmp', int2_)
            c1 = delayed(operator.add)(int1, int2)
            mask_ag = -np.eye(nocc)
            c_1 = delayed(np.einsum)('ag,nbmp->nbmapg', mask_ag, c1)
            #second term eq. C.16
            int1 = delayed(np.einsum)('manp->nmap', int1_)
            int2 = delayed(np.einsum)('mpna->nmap', int2_)
            c2 = delayed(operator.add)(int1, int2)
            mask_bg = -np.eye(nocc)
            c_1 = delayed(operator.add)(c_1, delayed(np.einsum)('bg,nmap->nbmapg', mask_bg, c2))
            #third term eq C.16
            mask_np = np.eye(nvir)
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int1 = delayed(np.einsum)('magb->bmag', int3_)
            int2 = delayed(np.einsum)('mbga->bmag', int3_)
            c3 = delayed(operator.add)(int1, int2)
            c_1 = delayed(operator.add)(c_1, delayed(np.einsum)('np,bmag->nbmapg', mask_np, c3))
            # fourth term eq C.16
            mask_mp = np.eye(nvir)
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            int1 = delayed(np.einsum)('gbna->nbag', int4_)
            int2 = delayed(np.einsum)('ganb->nbag', int4_)
            c4 = delayed(operator.add)(int1, int2)
            c_1 = delayed(operator.add)(c_1, delayed(np.einsum)('mp,nbag->nbmapg', mask_mp, c4))
            c_1 = delayed(operator.truediv)(c_1, np.sqrt(2))

            return -c_1.compute()

    def c_2_singlet(self):
        """C.16 equation, for obtain 2p-2h C(i=2) for singlet 
        properties

        Returns:
            _type_: _description_
        """
        nmo = self.nmo
        nocc = self.nocc
        nvir = self.nvir
        with h5py.File(str(self.h5_file), "r") as f:
            #eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = f["eri_mo"][:]
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int1 = np.einsum('mbnp->nbmp', int1_)
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int2 = np.einsum('mpnb->nbmp', int2_)
            c1 = int1 - int2
            mask_ag = np.eye(nocc)
            c_1 = np.einsum('ag,nbmp->nbmapg', mask_ag, c1)
            #second term eq. C.16
            int1 = np.einsum('mpna->nmap', int2_)
            int2 = np.einsum('manp->nmap', int1_)
            c2 =  int1 - int2
            mask_bg = np.eye(nocc)
            c_1 += np.einsum('bg,nmap->nbmapg', mask_bg, c2)
            #third term eq C.16
            mask_np = np.eye(nvir)
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int1 = np.einsum('magb->bmag', int3_)
            int2 = np.einsum('mbga->bmag', int3_)
            c3 = int1 - int2
            c_1 += np.einsum('np,bmag->nbmapg', mask_np, c3)
            # fourth term eq C.16
            mask_mp = np.eye(nvir)
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            int1 = np.einsum('ganb->nbag', int4_)
            int2 = np.einsum('gbna->nbag', int4_)
            c4 = int1 - int2
            c_1 += np.einsum('mp,nbag->nbmapg', mask_mp, c4)
            # firt deltas
            delta1 = 1 - np.eye(nvir)
            delta2 = 1 - np.eye(nocc)
            deltas = np.einsum('nm,ab->nbma', delta1, delta2)
            cte = -np.sqrt(3)/np.sqrt(2)
            c_1 = np.einsum('nbma,nbmapg->nbmapg', cte*deltas, c_1)
            
        return c_1
    
    def c_1_triplet(self):
        """C.21 equation, for obtain 2p-2h C(i=2) for triplet 
        properties

        Returns:
            _type_: _description_
        """
        nmo = self.nmo
        nocc = self.nocc
        nvir = self.nvir
        with h5py.File(str(self.h5_file), "r") as f:
            #eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = f["eri_mo"][:]
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int1 = np.einsum('mbnp->nbmp', int1_)
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int2 = np.einsum('mpnb->nbmp', int2_)
            c1 = int1 + int2
            mask_ag = np.eye(nocc)
            c_1 = np.einsum('ag,nbmp->nbmapg', mask_ag, c1)
            #second term 
            int1 = np.einsum('mpna->nmap', int2_)
            int2 = np.einsum('manp->nmap', int1_)
            c2 =  int1 + int2
            mask_bg = -np.eye(nocc)
            c_1 += np.einsum('bg,nmap->nbmapg', mask_bg, c2)
            #third term 
            mask_np = np.eye(nvir)
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int1 = np.einsum('magb->bmag', int3_)
            int2 = np.einsum('mbga->bmag', int3_)
            c3 = int1 - int2
            c_1 += np.einsum('np,bmag->nbmapg', mask_np, c3)
            # fourth term eq C.16
            mask_mp = np.eye(nvir)
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            int1 = np.einsum('ganb->nbag', int4_)
            int2 = np.einsum('gbna->nbag', int4_)
            c4 = -int1 + int2
            c_1 += np.einsum('mp,nbag->nbmapg', mask_mp, c4)
            # firt deltas
            delta = 1 - np.eye(nocc)
            cte = 1/np.sqrt(2)
            c_1 = np.einsum('ab,nbmapg->nbmapg', delta, c_1)
        return cte*c_1

    def c_2_triplet(self):
        """C.22 equation, for obtain 2p-2h C(i=3) for triplet 
        properties

        Returns:
            _type_: _description_
        """
        nmo = self.nmo
        nocc = self.nocc
        nvir = self.nvir
        with h5py.File(str(self.h5_file), "r") as f:
            #eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = f["eri_mo"][:]
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int1 = np.einsum('mbnp->nbmp', int1_)
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int2 = np.einsum('mpnb->nbmp', int2_)
            c1 = -int1 + int2
            mask_ag = np.eye(nocc)
            c_1 = np.einsum('ag,nbmp->nbmapg', mask_ag, c1)
            #second term 
            int1 = np.einsum('mpna->nmap', int2_)
            int2 = np.einsum('manp->nmap', int1_)
            c2 =  int1 - int2
            mask_bg = np.eye(nocc)
            c_1 += np.einsum('bg,nmap->nbmapg', mask_bg, c2)
            #third term 
            mask_np = np.eye(nvir)
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int1 = np.einsum('magb->bmag', int3_)
            int2 = np.einsum('mbga->bmag', int3_)
            c3 = int1 + int2
            c_1 += np.einsum('np,bmag->nbmapg', mask_np, c3)
            # fourth term
            mask_mp = -np.eye(nvir)
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            int1 = np.einsum('ganb->nbag', int4_)
            int2 = np.einsum('gbna->nbag', int4_)
            c4 = int1 + int2
            c_1 += np.einsum('mp,nbag->nbmapg', mask_mp, c4)
            # firt deltas
            delta = 1 - np.eye(nvir)
            #deltas = np.einsum('nm,ab->nbma', delta1, delta2)
            cte = -1/np.sqrt(2)
            c_1 = np.einsum('nm,nbmapg->nbmapg', delta, c_1)
        return cte*c_1

    def da0(self, triplet):
        d = self.D()
        nocc = self.nocc
        nvir = self.nvir
        d = np.linalg.inv(d).reshape(nvir,nocc,nvir,nocc,nvir,nocc,nvir,nocc)
        c_1 = self.c_1_singlet()
        c_1_t = np.einsum('nbmapg->pgnbma',c_1).conj()
        if triplet:
            c2 = self.c_1_triplet()
            c2_t = np.einsum('nbmapg->pgnbma',c2).conj()
            c3 = self.c_2_triplet()
            c3_t = np.einsum('nbmapg->pgnbma',c3).conj()
            c_1 = (-np.sqrt(2)/np.sqrt(3))*c_1
            c_1_t = (-np.sqrt(2)/np.sqrt(3))*c_1_t
            da0 = np.einsum('pgnbma,nbmanbma,nbmaqd->pgqd', c_1_t, d, c_1)
            da0 += np.einsum('pgnbma,nbmanbma,nbmaqd->pgqd', c2_t, d, c2)
            da0 += np.einsum('pgnbma,nbmanbma,nbmaqd->pgqd', c3_t, d, c3)
        else:
            da0 = np.einsum('pgnbma,nbmacejk,cejkqd->pgqd',c_1_t, d, c_1)
            c_2 = self.c_2_singlet()
            c_2_t = np.einsum('nbmapg->pgnbma',c_2)#.conj()
            da0 += np.einsum('pgnbma,nbmacejk,cejkqd->pgqd',c_2_t, d, c_2)
        
        da0 = np.einsum('pgqd->gpdq',da0/4)
        return da0
    
    @profile
    def da0_best(self):
        mo_energy = self.mo_energy
        occidx = self.occidx
        viridx = self.viridx

        e_aibj = lib.direct_sum(
            "n+m-a-b->nbma",
            mo_energy[viridx],
            mo_energy[viridx],
            mo_energy[occidx],
            mo_energy[occidx]
        )
        e_aibj = 1/e_aibj
        nocc = self.nocc
        nvir = self.nvir
        nmo = nocc + nvir
        cte = -np.sqrt(3)/np.sqrt(2)
        with h5py.File(str(self.h5_file), "r") as f:
            eri_mo = da.from_array((f["eri_mo"]), chunks=
                                   (nvir,nvir))
            d = da.diag(
                e_aibj.ravel())
            
            d = d.reshape(
                    nvir,nocc,nvir,nocc,nvir,nocc,nvir,nocc).rechunk((5,1,5,1,5,1,5,1))
            #print(type(d))                  
            
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo).rechunk((nmo//2,nmo//2,nmo//2,nmo//2))
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int1 = da.einsum('mbnp->nbmp', int1_)
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int2 = da.einsum('mpnb->nbmp', int2_)
            c1 = int1 + int2
            mask_ag = da.eye(nocc)
            c_1 = da.einsum('ag,nbmp->nbmapg', -mask_ag, c1)
            c1_ = int1-int2
            c_2 = da.einsum('ag,nbmp->nbmapg', mask_ag, c1_)
            #second term eq. C.16
            int1 = da.einsum('manp->nmap', int1_)
            int2 = da.einsum('mpna->nmap', int2_)
            c2 = int1 + int2
            mask_bg = da.eye(nocc)
            c_1 += da.einsum('bg,nmap->nbmapg', -mask_bg, c2)
            c2_ =  int2-int1
            c_2 +=  da.einsum('bg,nmap->nbmapg', mask_bg, c2_)
            #third term eq C.16
            mask_np = da.eye(nvir)
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int1 = da.einsum('magb->bmag', int3_)
            int2 = da.einsum('mbga->bmag', int3_)
            c3 = int1+int2
            c_1 += da.einsum('np,bmag->nbmapg', mask_np, c3)
            c3_ = int1-int2
            c_2 += da.einsum('np,bmag->nbmapg', mask_np, c3_)
            # fourth term eq C.16
            mask_mp = da.eye(nvir)
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            int1 = da.einsum('gbna->nbag', int4_)
            int2 = da.einsum('ganb->nbag', int4_)
            c4 = int1+int2
            c_1 +=  da.einsum('mp,nbag->nbmapg', mask_mp, c4)
            c_1 = -c_1/np.sqrt(2)
            c4_ = int2-int1
            c_2 += da.einsum('mp,nbag->nbmapg', mask_mp, c4_)            



            # firt deltas
            delta1 = 1 - da.eye(nvir)
            delta2 = 1 - da.eye(nocc)
            deltas = da.einsum('nm,ab->nbma', delta1, delta2)
            
            #c_2 = da.einsum('nbma,nbmapg->nbmapg', cte*deltas, c_2).compute()
            c_1_t = da.einsum('nbmapg->pgnbma',c_1).conj()
            #c_2_t = da.einsum('nbmapg->pgnbma',c_2)
            da0 = da.einsum('pgnbma,nbmacejk,cejkqd->pgqd',c_1_t, d, c_1)
            
            #da0 += da.einsum('pgnbma,nbmacejk,cejkqd->pgqd',c_2_t, d, c_2)
            #da0 = da.einsum('pgqd->gpdq',da0/4)
            return da0.compute()# c_1.sum().compute(), c_2.sum().compute() # 


    def trans_mat_1(self, atmlst, FC=False, PSO=False):
        """C.29 oddershede eq
        """
        k_1 = self.k_1
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            p_virt = h1[nocc:, nocc:]
            pert = np.einsum("nc,ambc->manb", p_virt, k_1)
            pert += np.einsum("mc,anbc->manb", p_virt, k_1)
            p_occ = h1[:nocc, :nocc]
            pert -= np.einsum('pb,ampn->manb', p_occ, k_1)
            pert -= np.einsum('pa,bmpn->manb', p_occ, k_1)
        elif PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)[0]
            p_virt = h1[:, nocc:, nocc:]
            pert = np.einsum("xnc,ambc->xmanb", p_virt, k_1)
            pert += np.einsum("xmc,anbc->xmanb", p_virt, k_1)
            p_occ = h1[:, :nocc, :nocc]
            pert -= np.einsum('xpb,ampn->xmanb', p_occ, k_1)
            pert -= np.einsum('xpa,bmpn->xmanb', p_occ, k_1)
        return pert          

    def trans_mat_2(self, atmlst, FC=False, PSO=False):
        """C.30 oddershede eq
        """
        k_2 = self.k_2
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            p_virt = h1[nocc:, nocc:]
            pert = np.einsum("nc,ambc->manb", p_virt, k_2)
            pert -= np.einsum("mc,anbc->manb", p_virt, k_2)
            p_occ = h1[:nocc, :nocc]
            pert -= np.einsum('pb,ampn->manb', p_occ, k_2)
            pert += np.einsum('pa,bmpn->manb', p_occ, k_2)
        elif PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)[0]
            p_virt = h1[:, nocc:, nocc:]
            pert = np.einsum("xnc,ambc->xmanb", p_virt, k_2)
            pert -= np.einsum("xmc,anbc->xmanb", p_virt, k_2)
            p_occ = h1[:, :nocc, :nocc]
            pert -= np.einsum('xpb,ampn->xmanb', p_occ, k_2)
            pert += np.einsum('xpa,bmpn->xmanb', p_occ, k_2)
        return pert

    def correction_pert_3(self, atmlst, FC=False, PSO=False):
        "C.28 oddershede 1984 paper"
        d = self.D()
        nocc = self.nocc
        nvir = self.nvir
        d = np.linalg.inv(d).reshape(nvir,nocc,nvir,nocc,nvir,nocc,nvir,nocc)
        c_1 = self.c_1_singlet()
        c_2 = self.c_2_singlet()
        if FC:
            trans_mat_1 = self.trans_mat_1(atmlst, FC=True)
            trans_mat_2 = self.trans_mat_2(atmlst, FC=True)
            t = np.einsum('manb,manbmanb,manbpg->pg', trans_mat_1, d, c_1)
            t += np.einsum('manb,manbmanb,manbpg->pg', trans_mat_2, d, c_2)
            t = np.einsum('pg->gp', t)
        elif PSO:
            trans_mat_1 = self.trans_mat_1(atmlst, PSO=True)
            trans_mat_2 = self.trans_mat_2(atmlst, PSO=True)
            t = np.einsum('xmanb,manbcejk,cejkpg->xpg', trans_mat_1, d, c_1)
            t += np.einsum('xmanb,manbcejk,cejkpg->xpg', trans_mat_2, d, c_2)
            t = np.einsum('xpg->xgp', t)

        return t/4

    def w4(self, atm1lst, atm2lst, FC=False, PSO=False):
        'eq 5.34 Oddershede'
        d = self.D()
        nocc = self.nocc
        nvir = self.nvir
        d = np.linalg.inv(d).reshape(nvir,nocc,nvir,nocc,nvir,nocc,nvir,nocc)
        if FC:
            h1_trans_mat_1 = self.trans_mat_1(atm1lst, FC=True)
            h1_trans_mat_2 = self.trans_mat_2(atm1lst, FC=True)
            h2_trans_mat_1 = self.trans_mat_1(atm2lst, FC=True)
            h2_trans_mat_2 = self.trans_mat_2(atm2lst, FC=True)
            w4 = np.einsum('manb,manbcejk,cejk', h2_trans_mat_1, d, h1_trans_mat_1)
            w4 += np.einsum('manb,manbcejk,cejk', h2_trans_mat_2, d, h1_trans_mat_2)
            
        elif PSO:
            h1_trans_mat_1 = self.trans_mat_1(atm1lst, PSO=True)
            h1_trans_mat_2 = self.trans_mat_2(atm1lst, PSO=True)
            h2_trans_mat_1 = self.trans_mat_1(atm2lst, PSO=True)
            h2_trans_mat_2 = self.trans_mat_2(atm2lst, PSO=True)
            w4 = np.einsum('xmanb,manbcejk,ycejk->xy', h2_trans_mat_1, d, h1_trans_mat_1)
            w4 += np.einsum('xmanb,manbcejk,ycejk->xy', h2_trans_mat_2, d, h1_trans_mat_2)
        return 2*w4/4

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
        h1_corr3 = self.correction_pert_3(atmlst=atm1lst, PSO=True)
        h2_corr1 = self.correction_pert(atmlst=atm2lst, PSO=True)
        h2_corr2 = self.correction_pert_2(atmlst=atm2lst, PSO=True)
        h2_corr3 = self.correction_pert_3(atmlst=atm2lst, PSO=True)

        h1 = (-2 * h1) + h1_corr1 + h1_corr2 - h1_corr3
        h2 = (-2 * h2) + h2_corr1 + h2_corr2 - h2_corr3
        m = self.M_rpa(triplet=False)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m += self.part_b2(0)
        m += self.S2
        m -= self.da0()
        m = m.reshape(nocc * nvir, nocc * nvir)
        w4 = self.w4(atm1lst,atm2lst, PSO=True)        
        if elements:
            return h1, m, h2
        else:
            p = np.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = np.einsum("xia,iajb,yjb->xy", h1, p, h2)
            e += w4
            para.append(e)
            pso = np.asarray(para) * nist.ALPHA**4
            return pso

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
        m -= self.da0(triplet=True)
        h1_corr1 = self.correction_pert(atmlst=atm1lst, FC=True)
        h1_corr2 = self.correction_pert_2(atmlst=atm1lst, FC=True)
        h1_corr3 = self.correction_pert_3(atmlst=atm1lst, FC=True)

        h2_corr1 = self.correction_pert(atmlst=atm2lst, FC=True)
        h2_corr2 = self.correction_pert_2(atmlst=atm2lst, FC=True)
        h2_corr3 = self.correction_pert_3(atmlst=atm2lst, FC=True)
        h1 = (2 * h1) + h1_corr1 + h1_corr2 - h1_corr3
        h2 = (2 * h2) + h2_corr1 + h2_corr2 - h2_corr3
        m = m.reshape(nocc * nvir, nocc * nvir)
        w4 = self.w4(atm1lst,atm2lst, FC=True)
        if elements:
            return h1, m, h2
        else:
            p = -np.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = np.einsum("ia,iajb,jb", h1, p, h2)
            e += w4*4
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
        #print(prop*unit*gyro1*gyro2, )
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


    def c_2_singlet_for(self):
        nmo = self.nmo
        nocc = self.nocc
        nvir = self.nvir
        with h5py.File(str(self.h5_file), "r") as f:
            #eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = f["eri_mo"][:]
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            c_1 = np.zeros((nvir,nocc,nvir,nocc,nvir,nocc))
            for a in range(nocc):
                for b in range(nocc):
                    for g in range(nocc):
                        for m in range(nvir):
                            for n in range(nvir):
                                for p in range(nvir):
                                    if a == g:
                                        if a!=b and n!=m:
                                            c_1[n,b,m,a,p,g] -= np.sqrt(3)/np.sqrt(2)*(int1_[m,b,n,p] - int2_[m,p,n,b])
                                    if b == g:
                                        if a!=b and n!=m:
                                            c_1[n,b,m,a,p,g] -= np.sqrt(3)/np.sqrt(2)*(int2_[m,p,n,a] - int1_[m,a,n,p])
                                    if n == p:
                                        if a!=b and n!=m:
                                            c_1[n,b,m,a,p,g] -= np.sqrt(3)/np.sqrt(2)*(int3_[m,a,g,b] - int3_[m,b,g,a])
                                    if m == p:
                                        if a!=b and n!=m:
                                            c_1[n,b,m,a,p,g] -= np.sqrt(3)/np.sqrt(2)*(int4_[g,a,n,b] - int4_[g,b,n,a])
        return c_1

    def c_1_singlet_for(self):
        nmo = self.nmo
        nocc = self.nocc
        nvir = self.nvir
        with h5py.File(str(self.h5_file), "r") as f:
            #eri_mo = da.from_array((f["eri_mo"]), chunks="auto")
            eri_mo = f["eri_mo"][:]
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
            int1_ = eri_mo[nocc:, :nocc, nocc:, nocc:]
            int2_ = eri_mo[nocc:, nocc:, nocc:, :nocc]
            int3_ = eri_mo[nocc:, :nocc, :nocc, :nocc]
            int4_ = eri_mo[:nocc,:nocc,nocc:,:nocc]
            c_1 = np.zeros((nvir,nocc,nvir,nocc,nvir,nocc))
            for a in range(nocc):
                for b in range(nocc):
                    for g in range(nocc):
                        for m in range(nvir):
                            for n in range(nvir):
                                for p in range(nvir):
                                    if a == g:
                                        if a==b and n==m:
                                            c_1[n,b,m,a,p,g] -= 1/np.sqrt(2)*(1/np.sqrt(4))*(int1_[m,b,n,p] + int2_[m,p,n,b])
                                        elif a==b and n!=m:
                                            c_1[n,b,m,a,p,g] -= 1/np.sqrt(2)*(1/np.sqrt(2))*(int1_[m,b,n,p] + int2_[m,p,n,b])
                                        elif a!=b and n==m:
                                            c_1[n,b,m,a,p,g] -= 1/np.sqrt(2)*(1/np.sqrt(2))*(int1_[m,b,n,p] + int2_[m,p,n,b])
                                        else:
                                            c_1[n,b,m,a,p,g] -= 1/np.sqrt(2)*(int1_[m,b,n,p] + int2_[m,p,n,b])
                                    if b == g:
                                        if a==b and n==m:
                                            c_1[n,b,m,a,p,g] -= 1/np.sqrt(2)*(1/np.sqrt(4))*(int1_[m,a,n,p] + int2_[m,p,n,a])
                                        elif a==b and n!=m:
                                            c_1[n,b,m,a,p,g] -= 1/np.sqrt(2)*(1/np.sqrt(2))*(int1_[m,a,n,p] + int2_[m,p,n,a])
                                        elif a!=b and n==m:
                                            c_1[n,b,m,a,p,g] -= 1/np.sqrt(2)*(1/np.sqrt(2))*(int1_[m,a,n,p] + int2_[m,p,n,a])
                                        else:
                                            c_1[n,b,m,a,p,g] -= 1/np.sqrt(2)*(int1_[m,a,n,p] + int2_[m,p,n,a])
                                    if n == p:
                                        if a==b and n==m:
                                            c_1[n,b,m,a,p,g] += 1/np.sqrt(2)*(1/np.sqrt(4))*(int3_[m,a,g,b] + int3_[m,b,g,a])
                                        elif a==b and n!=m:
                                            c_1[n,b,m,a,p,g] += 1/np.sqrt(2)*(1/np.sqrt(2))*(int3_[m,a,g,b] + int3_[m,b,g,a])
                                        elif a!=b and n==m:
                                            c_1[n,b,m,a,p,g] += 1/np.sqrt(2)*(1/np.sqrt(2))*(int3_[m,a,g,b] + int3_[m,b,g,a])
                                        else:
                                            c_1[n,b,m,a,p,g] += 1/np.sqrt(2)*(int3_[m,a,g,b] + int3_[m,b,g,a])
                                    if m == p:
                                        if a==b and n==m:
                                            c_1[n,b,m,a,p,g] += 1/np.sqrt(2)*(1/np.sqrt(4))*(int4_[g,b,n,a] + int4_[g,a,n,b])
                                        elif a==b and n!=m:
                                            c_1[n,b,m,a,p,g] += 1/np.sqrt(2)*(1/np.sqrt(2))*(int4_[g,b,n,a] + int4_[g,a,n,b])
                                        elif a!=b and n==m:
                                            c_1[n,b,m,a,p,g] += 1/np.sqrt(2)*(1/np.sqrt(2))*(int4_[g,b,n,a] + int4_[g,a,n,b])
                                        else:
                                            c_1[n,b,m,a,p,g] += 1/np.sqrt(2)*(int4_[g,b,n,a] + int4_[g,a,n,b])
        return c_1

    def D_for(self):
        'same as D but with for loops'
        mo_energy = self.mo_energy
        occidx = self.occidx
        viridx = self.viridx
        nocc = self.nocc
        nvir = self.nvir
        d = np.zeros((nvir,nocc,nvir,nocc,nvir,nocc,nvir,nocc))
        for a in range(nocc):
                for b in range(nocc):
                    for m in range(nvir):
                        for n in range(nvir):
                            d[n,b,m,a,n,b,m,a] += mo_energy[viridx][n] + mo_energy[viridx][m] - mo_energy[occidx][a] - mo_energy[occidx][b]

        d = d.reshape((nvir*nocc*nvir*nocc,nvir*nocc*nvir*nocc))
        return d
