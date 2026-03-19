import os

import h5py
import numpy as np
import opt_einsum as oe
import scipy as sp
from pyscf import lib
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

from pyppm.hrpa import HRPA


class SOPPA(HRPA):
    """Class to perform calculations of $J^{FC}$ mechanism at SOPPA level of
    approach. It inherits from HRPA class, so it has all the methods and
    properties of HRPA class, but it has some additional methods to perform
    calculations at SOPPA level of approach.
    """

    def __init__(self, mol=None, chkfile=None, mole_name=None, calc_int=False):
        super().__init__(
            mol=mol, chkfile=chkfile, mole_name=mole_name, calc_int=calc_int
        )

    @property
    def da0_triplet_t23(self):
        """CD^-1C triplet matrices, based in Oddershede 1984 rev eq. C.19
        and C.21
        """
        nocc = self.nocc
        nvir = self.nvir
        int1_, int2_, int3_, int4_, d = self.integrals_soppa
        int_mbnp = int1_.transpose(2, 1, 0, 3)
        int_mpnb = int2_.transpose(2, 3, 0, 1)
        int_magb = int3_.transpose(3, 0, 1, 2)
        int_mbga = int3_.transpose(1, 0, 3, 2)
        int_manp = int1_.transpose(2, 0, 1, 3)
        int_mpna = int2_.transpose(2, 0, 3, 1)
        int_gbna = int4_.transpose(2, 1, 3, 0)
        int_ganb = int4_.transpose(2, 3, 1, 0)
        mask_bg = np.eye(nocc)
        mask_ag = mask_bg
        mask_np = np.eye(nvir)
        mask_mp = mask_np
        delta_vir = 1 - mask_np
        delta_occ = 1 - mask_bg
        d_occ = oe.contract("ab,nbma->nbma", delta_occ, d)
        d_vir = oe.contract("nm,nbma->nbma", delta_vir, d)
        c2_t_1 = int_mbnp + int_mpnb
        c3_t_1 = -int_mbnp + int_mpnb
        o = oe.contract("nbmp, nbma, nbmq->paq", c2_t_1, d_occ, c2_t_1)
        da0 = oe.contract("paq, ag, ad->pgqd", o, mask_ag, mask_ag)
        o = oe.contract("nbmp, nbma, nbmq->paq", c3_t_1, d_vir, c3_t_1)
        da0 += oe.contract("paq, ag, ad->pgqd", o, mask_ag, mask_ag)
        c2_t_2 = int_mpna + int_manp
        c3_t_2 = int_mpna - int_manp
        o = oe.contract("nmap,nbma,nmaq->pbq", c2_t_2, d_occ, c2_t_2)
        da0 += oe.contract("pbq, bg, bd->pgqd", o, mask_bg, mask_bg)
        o = oe.contract("nmap,nbma,nmaq->pbq", c3_t_2, d_vir, c3_t_2)
        da0 += oe.contract("pbq, bg, bd->pgqd", o, mask_bg, mask_bg)
        c2_t_3 = int_magb - int_mbga
        c3_t_3 = int_magb + int_mbga
        o = oe.contract("bmag,nbma,bmad->gnd", c2_t_3, d_occ, c2_t_3)
        da0 += oe.contract("gnd, np, nq->pgqd", o, mask_np, mask_np)
        o = oe.contract("bmag,nbma,bmad->gnd", c3_t_3, d_vir, c3_t_3)
        da0 += oe.contract("gnd, np, nq->pgqd", o, mask_np, mask_np)
        c2_t_4 = -int_ganb + int_gbna
        c3_t_4 = int_ganb + int_gbna
        o = oe.contract("nbag, nbma, nbad->gmd", c2_t_4, d_occ, c2_t_4)
        da0 += oe.contract("gmd, mp, mq->pgqd", o, mask_mp, mask_mp)
        o = oe.contract("nbag, nbma, nbad->gmd", c3_t_4, d_vir, c3_t_4)
        da0 += oe.contract("gmd, mp, mq->pgqd", o, mask_mp, mask_mp)
        o = oe.contract("nbmp, nbma, nmaq->bpaq", c2_t_1, d_occ, c2_t_2)
        da0_ = oe.contract("bpaq, ag, bd->pgqd", o, mask_ag, -mask_bg)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp, nbma, nmaq->bpaq", c3_t_1, d_vir, c3_t_2)
        da0_ = oe.contract("bpaq, ag, bd->pgqd", o, mask_ag, mask_bg)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp, nbma, bmad->npad", c2_t_1, d_occ, c2_t_3)
        da0_ = oe.contract("npad, ag, nq->pgqd", o, mask_ag, mask_np)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp, nbma, bmad->npad", c3_t_1, d_vir, c3_t_3)
        da0_ = oe.contract("npad, ag, nq->pgqd", o, mask_ag, mask_np)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp, nbma, nbad->mapd", c2_t_1, d_occ, c2_t_4)
        da0_ = oe.contract("mapd, ag, mq->pgqd", o, mask_ag, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp, nbma, nbad->mapd", c3_t_1, d_vir, c3_t_4)
        da0_ = oe.contract("mapd, ag, mq->pgqd", o, mask_ag, -mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nmap, nbma, bmad->bnpd", c2_t_2, d_occ, c2_t_3)
        da0_ = oe.contract("bnpd, bg, nq->pgqd", o, -mask_bg, mask_np)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nmap, nbma, bmad->bnpd", c3_t_2, d_vir, c3_t_3)
        da0_ = oe.contract("bnpd, bg, nq->pgqd", o, mask_bg, mask_np)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nmap, nbma, nbad->mbpd", c2_t_2, d_occ, c2_t_4)
        da0_ = oe.contract("mbpd, bg, mq->pgqd", o, -mask_bg, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nmap, nbma, nbad->mbpd", c3_t_2, d_vir, c3_t_4)
        da0_ = oe.contract("mbpd, bg, mq->pgqd", o, mask_bg, -mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("bmag, nbma, nbad->mgnd", c2_t_3, d_occ, c2_t_4)
        da0_ = oe.contract("mgnd, np, mq->pgqd", o, mask_np, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("bmag, nbma, nbad->mgnd", c3_t_3, d_vir, c3_t_4)
        da0_ = oe.contract("mgnd, np, mq->pgqd", o, mask_np, -mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        da0 = oe.contract("pgqd->gpdq", da0 / 2)
        return da0

    @property
    def da0_singlet_s1(self):
        """CD^-1C first product for SOPPA calculations for singlet, based in
        Oddershede 1984 rev eq. C.15 and C.17

        Args:
            PSO (bool, optional): _description_. Defaults to False.
            FC (bool, optional): _description_. Defaults to False.
            FCSD (bool, optional): _description_. Defaults to False.
            atm1lst (_type_, optional): _description_. Defaults to None.
            atm2lst (_type_, optional): _description_. Defaults to None.
        """
        nocc = self.nocc
        nvir = self.nvir
        int1_, int2_, int3_, int4_, d = self.integrals_soppa
        int_mbnp = int1_.transpose(2, 1, 0, 3)
        int_mpnb = int2_.transpose(2, 3, 0, 1)
        int_magb = int3_.transpose(3, 0, 1, 2)
        int_mbga = int3_.transpose(1, 0, 3, 2)
        int_manp = int1_.transpose(2, 0, 1, 3)
        int_mpna = int2_.transpose(2, 0, 3, 1)
        int_gbna = int4_.transpose(2, 1, 3, 0)
        int_ganb = int4_.transpose(2, 3, 1, 0)
        mask_bg = np.eye(nocc)
        mask_ag = mask_bg
        mask_np = np.eye(nvir)
        mask_mp = mask_np
        c1_1 = int_mbnp + int_mpnb
        o = oe.contract("nbmp, nbma, nbmq->paq", c1_1, d, c1_1)
        da0 = oe.contract("paq, ad, ag->pgqd", o, mask_ag, mask_ag)
        c1_2 = int_manp + int_mpna
        o = oe.contract("nmap,nbma,nmaq->pbq", c1_2, d, c1_2)
        da0 += oe.contract("pbq, bg,bd->pgqd", o, mask_bg, mask_bg)
        c1_3 = int_magb + int_mbga
        o = oe.contract("bmag,nbma,bmad->gnd", c1_3, d, c1_3)
        da0 += oe.contract("gnd, np, nq->pgqd", o, mask_np, mask_np)
        c1_4 = int_gbna + int_ganb
        o = oe.contract("nbag,nbma,nbad->gmd", c1_4, d, c1_4)
        da0 += oe.contract("gmd,mp,mq->pgqd", o, mask_mp, mask_mp)
        o = oe.contract("nbmp,nbma,nmaq->bpaq", c1_1, d, c1_2)
        da0_ = oe.contract("bpaq,ag,bd->pgqd", o, mask_ag, mask_bg)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp, nbma, bmad ->npad", c1_1, d, c1_3)
        da0_ = oe.contract("npad, ag, nq->pgqd", o, -mask_ag, mask_np)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp,nbma,nbad->mapd", c1_1, d, c1_4)
        da0_ = oe.contract("mapd,ag,mq->pgqd", o, -mask_ag, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nmap,nbma,bmad->nbpd", c1_2, d, c1_3)
        da0_ = oe.contract("nbpd,bg,nq->pgqd", o, -mask_bg, mask_np)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nmap,nbma,nbad->mbpd", c1_2, d, c1_4)
        da0_ = oe.contract("mbpd,bg,mq->pgqd", o, -mask_bg, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("bmag,nbma,nbad->mgnd", c1_3, d, c1_4)
        da0_ = oe.contract("mgnd,np,mq->pgqd", o, mask_np, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        da0 = da0 / 2
        da0 = oe.contract("pgqd->gpdq", da0)
        return da0

    def t1_opt_PSO(self, atmlst=None):
        """Optimized T1 for SOPPA calculations for singlet, based in Oddershede
        1984 rev eq. C.28. Having in count the sign correction in eq. C.16

        Args:
            atm1lst (_type_, optional): _description_. Defaults to None.
        """
        nocc = self.nocc
        nvir = self.nvir
        int1_, int2_, int3_, int4_, d = self.integrals_soppa
        tm_h1 = self.trans_mat_1(atmlst, PSO=True)
        tm2_h1 = self.trans_mat_2(atmlst, PSO=True)
        int_mbnp = int1_.transpose(2, 1, 0, 3)
        int_mpnb = int2_.transpose(2, 3, 0, 1)
        int_magb = int3_.transpose(3, 0, 1, 2)
        int_mbga = int3_.transpose(1, 0, 3, 2)
        int_manp = int1_.transpose(2, 0, 1, 3)
        int_mpna = int2_.transpose(2, 0, 3, 1)
        int_gbna = int4_.transpose(2, 1, 3, 0)
        int_ganb = int4_.transpose(2, 3, 1, 0)
        mask_bg = np.eye(nocc)
        mask_ag = mask_bg
        mask_np = np.eye(nvir)
        mask_mp = mask_np
        delta_vir = 1 - mask_np
        delta_occ = 1 - mask_bg
        deltas = oe.contract("nm,ab->nbma", delta_vir, delta_occ)
        cte = np.sqrt(3 / 2)
        d_ = deltas * d * deltas * cte
        c2_1 = int_mbnp - int_mpnb
        c1_1 = int_mbnp + int_mpnb
        o = oe.contract("xnbma, nbma, nbmp->xap", tm_h1, d, c1_1)
        t1 = oe.contract("xap, ag->xpg", o / np.sqrt(2), mask_ag)
        o = oe.contract("xnbma, nbma, nbmp->xap", tm2_h1, d_, c2_1)
        t1 -= oe.contract("xap, ag->xpg", o, mask_ag)
        c1_2 = int_manp + int_mpna
        c2_2 = int_mpna - int_manp
        o = oe.contract("xnbma, nbma, nmap->xbp", tm_h1, d, c1_2)
        t1 -= oe.contract("xbp, bg->xpg", o / np.sqrt(2), -mask_bg)
        o = oe.contract("xnbma, nbma, nmap->xbp", tm2_h1, d_, c2_2)
        t1 -= oe.contract("xbp, bg->xpg", o, mask_bg)
        c1_3 = int_magb + int_mbga
        c2_3 = int_magb - int_mbga
        o = oe.contract("xnbma, nbma, bmag->xng", tm_h1, d, c1_3)
        t1 -= oe.contract("xng, np->xpg", o / np.sqrt(2), mask_np)
        o = oe.contract("xnbma, nbma, bmag->xng", tm2_h1, d_, c2_3)
        t1 -= oe.contract("xng, np->xpg", o, mask_np)
        c1_4 = int_gbna + int_ganb
        c2_4 = int_ganb - int_gbna
        o = oe.contract("xnbma,nbma,nbag->xmg", tm_h1, d, c1_4)
        t1 -= oe.contract("xmg, mp->xpg", o / np.sqrt(2), mask_mp)
        o = oe.contract("xnbma,nbma,nbag->xmg", tm2_h1, d_, c2_4)
        t1 -= oe.contract("xmg, mp->xpg", o, mask_mp)
        t1 = oe.contract("xpg->xgp", t1)
        return t1

    def t1_opt_triplet(self, FC=False, FCSD=False, atmlst=None):
        """Optimized T1 for SOPPA calculations for triplet, based in Oddershede
        1984 rev eq. C.28 and C.29

        Args:
            PSO (bool, optional): _description_. Defaults to True.
            atm1lst (_type_, optional): _description_. Defaults to None.
        """
        nocc = self.nocc
        nvir = self.nvir
        int1_, int2_, int3_, int4_, d = self.integrals_soppa
        if FC:
            tm_h1 = self.trans_mat_1(atmlst, FC=FC)
            tm2_h1 = self.trans_mat_2(atmlst, FC=FC)
        elif FCSD:
            tm_h1 = self.trans_mat_1(atmlst, FCSD=FCSD)
            tm2_h1 = self.trans_mat_2(atmlst, FCSD=FCSD)
        int_mbnp = int1_.transpose(2, 1, 0, 3)
        int_mpnb = int2_.transpose(2, 3, 0, 1)
        int_magb = int3_.transpose(3, 0, 1, 2)
        int_mbga = int3_.transpose(1, 0, 3, 2)
        int_manp = int1_.transpose(2, 0, 1, 3)
        int_mpna = int2_.transpose(2, 0, 3, 1)
        int_gbna = int4_.transpose(2, 1, 3, 0)
        int_ganb = int4_.transpose(2, 3, 1, 0)
        mask_bg = np.eye(nocc)
        mask_ag = mask_bg
        mask_np = np.eye(nvir)
        mask_mp = mask_np
        delta_vir = 1 - mask_np
        delta_occ = 1 - mask_bg
        deltas = oe.contract("nm,ab->nbma", delta_vir, delta_occ)
        cte = np.sqrt(3 / 2)
        d_ = deltas * d * deltas * cte
        c2_1 = int_mbnp - int_mpnb
        c1_1 = int_mbnp + int_mpnb
        c1_2 = int_manp + int_mpna
        c2_2 = int_mpna - int_manp
        c1_3 = int_magb + int_mbga
        c2_3 = int_magb - int_mbga
        c1_4 = int_gbna + int_ganb
        c2_4 = int_ganb - int_gbna
        if FC:
            o = oe.contract("nbma, nbma, nbmp->ap", tm_h1, d, c1_1)
            t1 = oe.contract("ap, ag->pg", o / np.sqrt(2), mask_ag)
            o = oe.contract("nbma, nbma, nbmp->ap", tm2_h1, d_, c2_1)
            t1 -= oe.contract("ap, ag->pg", o, mask_ag)
            o = oe.contract("nbma, nbma, nmap->bp", tm_h1, d, c1_2)
            t1 -= oe.contract("bp, bg->pg", o / np.sqrt(2), -mask_bg)
            o = oe.contract("nbma, nbma, nmap->bp", tm2_h1, d_, c2_2)
            t1 -= oe.contract("bp, bg->pg", o, mask_bg)
            o = oe.contract("nbma, nbma, bmag->ng", tm_h1, d, c1_3)
            t1 -= oe.contract("ng, np->pg", o / np.sqrt(2), mask_np)
            o = oe.contract("nbma, nbma, bmag->ng", tm2_h1, d_, c2_3)
            t1 -= oe.contract("ng, np->pg", o, mask_np)
            o = oe.contract("nbma,nbma,nbag->mg", tm_h1, d, c1_4)
            t1 -= oe.contract("mg, mp->pg", o / np.sqrt(2), mask_mp)
            o = oe.contract("nbma,nbma,nbag->mg", tm2_h1, d_, c2_4)
            t1 -= oe.contract("mg, mp->pg", o, mask_mp)
            t1 = oe.contract("pg->gp", t1)
        elif FCSD:
            o = oe.contract("wxnbma, nbma, nbmp->wxap", tm_h1, d, c1_1)
            t1 = oe.contract("wxap, ag->wxpg", o / np.sqrt(2), mask_ag)
            o = oe.contract("wxnbma, nbma, nbmp->wxap", tm2_h1, d_, c2_1)
            t1 -= oe.contract("wxap, ag->wxpg", o, mask_ag)
            o = oe.contract("wxnbma, nbma, nmap->wxbp", tm_h1, d, c1_2)
            t1 -= oe.contract("wxbp, bg->wxpg", o / np.sqrt(2), -mask_bg)
            o = oe.contract("wxnbma, nbma, nmap->wxbp", tm2_h1, d_, c2_2)
            t1 -= oe.contract("wxbp, bg->wxpg", o, mask_bg)
            o = oe.contract("wxnbma, nbma, bmag->wxng", tm_h1, d, c1_3)
            t1 -= oe.contract("wxng, np->wxpg", o / np.sqrt(2), mask_np)
            o = oe.contract("wxnbma, nbma, bmag->wxng", tm2_h1, d_, c2_3)
            t1 -= oe.contract("wxng, np->wxpg", o, mask_np)
            o = oe.contract("wxnbma,nbma,nbag->wxmg", tm_h1, d, c1_4)
            t1 -= oe.contract("wxmg, mp->wxpg", o / np.sqrt(2), mask_mp)
            o = oe.contract("wxnbma,nbma,nbag->wxmg", tm2_h1, d_, c2_4)
            t1 -= oe.contract("wxmg, mp->wxpg", o, mask_mp)
            t1 = oe.contract("wxpg->wxgp", t1)
        return t1

    def w4(self, PSO=False, FC=False, FCSD=False, atm1lst=None, atm2lst=None):
        """Optimized W4 for SOPPA calculations for singlet, based in Oddershede
        1984 rev eq. C.30 and C.31

        Args:
            PSO (bool, optional): _description_. Defaults to True.
            atm1lst (_type_, optional): _description_. Defaults to None.
            atm2lst (_type_, optional): _description_. Defaults to None.
        """
        int1_, int2_, int3_, int4_, d = self.integrals_soppa
        if PSO:
            tm_h1 = self.trans_mat_1(atm1lst, PSO=True)
            tm2_h1 = self.trans_mat_2(atm1lst, PSO=True)
            tm_h2 = self.trans_mat_1(atm2lst, PSO=True)
            tm2_h2 = self.trans_mat_2(atm2lst, PSO=True)
            w4 = oe.contract("xmanb,manb,ymanb->xy", tm_h2, d, tm_h1)
            w4 += oe.contract("xmanb,manb,ymanb->xy", tm2_h1, d, tm2_h2)
        elif FC:
            tm_h1 = self.trans_mat_1(atm1lst, FC=True)
            tm2_h1 = self.trans_mat_2(atm1lst, FC=True)
            tm_h2 = self.trans_mat_1(atm2lst, FC=True)
            tm2_h2 = self.trans_mat_2(atm2lst, FC=True)
            w4 = oe.contract("manb,manb,manb->", tm_h1, d, tm_h2)
            w4 += oe.contract("manb,manb,manb->", tm2_h1, d, tm2_h2)
        elif FCSD:
            tm_h1 = self.trans_mat_1(atm1lst, FCSD=True)
            tm2_h1 = self.trans_mat_2(atm1lst, FCSD=True)
            tm_h2 = self.trans_mat_1(atm2lst, FCSD=True)
            tm2_h2 = self.trans_mat_2(atm2lst, FCSD=True)
            w4 = oe.contract("wxmanb,manb,wymanb->xy", tm_h1, d, tm_h2)
            w4 += oe.contract("wxmanb,manb,wymanb->xy", tm2_h1, d, tm2_h2)
        return w4

    @property
    def da0_singlet_s2(self):
        """CD^-1C second product for SOPPA calculations for singlet, based in
        Oddershede 1984 rev eq. C.15 and C.17

        Args:
            PSO (bool, optional): _description_. Defaults to False.
            FC (bool, optional): _description_. Defaults to False.
            FCSD (bool, optional): _description_. Defaults to False.
            atm1lst (_type_, optional): _description_. Defaults to None.
            atm2lst (_type_, optional): _description_. Defaults to None.
        """
        nocc = self.nocc
        nvir = self.nvir
        int1_, int2_, int3_, int4_, d = self.integrals_soppa
        int_mbnp = int1_.transpose(2, 1, 0, 3)
        int_mpnb = int2_.transpose(2, 3, 0, 1)
        int_magb = int3_.transpose(3, 0, 1, 2)
        int_mbga = int3_.transpose(1, 0, 3, 2)
        int_manp = int1_.transpose(2, 0, 1, 3)
        int_mpna = int2_.transpose(2, 0, 3, 1)
        int_gbna = int4_.transpose(2, 1, 3, 0)
        int_ganb = int4_.transpose(2, 3, 1, 0)
        mask_bg = np.eye(nocc)
        mask_ag = mask_bg
        mask_np = np.eye(nvir)
        mask_mp = mask_np
        delta_vir = 1 - mask_np
        delta_occ = 1 - mask_bg
        deltas = oe.contract("nm,ab->nbma", delta_vir, delta_occ)
        cte = 3 / 2
        d_ = deltas * d * deltas * cte
        c2_1 = int_mbnp - int_mpnb
        o = oe.contract("nbmp,nbma,nbmq->paq", c2_1, d_, c2_1)
        da0 = oe.contract("paq,ag,ad->pgqd", o, mask_ag, mask_ag)
        c2_2 = int_mpna - int_manp
        o = oe.contract("nmap,nbma,nmaq->pbq", c2_2, d_, c2_2)
        da0 += oe.contract("pbq,bg,bd->pgqd", o, mask_bg, mask_bg)
        c2_3 = int_magb - int_mbga
        o = oe.contract("bmag,nbma,bmad->gnd", c2_3, d_, c2_3)
        da0 += oe.contract("gnd,np,nq->pgqd", o, mask_np, mask_np)
        c2_4 = int_ganb - int_gbna
        o = oe.contract("nbag,nbma,nbad->gmd", c2_4, d_, c2_4)
        da0 += oe.contract("gmd,mp,mq->pgqd", o, mask_mp, mask_mp)
        o = oe.contract("nbmp,nbma,nmaq->bpaq", c2_1, d_, c2_2)
        da0_ = oe.contract("bpaq,ag,bd->pgqd", o, mask_ag, mask_bg)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp,nbma,bmad->npad", c2_1, d_, c2_3)
        da0_ = oe.contract("npad,ag,nq->pgqd", o, mask_ag, mask_np)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nbmp,nbma,nbad->mapd", c2_1, d_, c2_4)
        da0_ = oe.contract("mapd,ag,mq->pgqd", o, mask_ag, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nmap,nbma,bmad->bnpd", c2_2, d_, c2_3)
        da0_ = oe.contract("bnpd,bg,nq->pgqd", o, mask_bg, mask_np)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("nmap,nbma,nbad->mbpd", c2_2, d_, c2_4)
        da0_ = oe.contract("mbpd,bg,mq->pgqd", o, mask_bg, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        o = oe.contract("bmag,nbma,nbad->mgnd", c2_3, d_, c2_4)
        da0_ = oe.contract("mgnd,np,mq->pgqd", o, mask_np, mask_mp)
        da0 += da0_
        da0 += da0_.transpose(2, 3, 0, 1)
        da0 = oe.contract("pgqd->gpdq", da0)
        return da0

    @property
    def integrals_soppa(self):
        """Method that returns all integrals needed for the SOPPA method

        Returns:
            2e-integrals: numpy.array
        """
        mo_energy = self.mo_energy
        occidx = self.occidx
        viridx = self.viridx
        e_aibj = lib.direct_sum(
            "n+m-a-b->nbma",
            mo_energy[viridx],
            mo_energy[viridx],
            mo_energy[occidx],
            mo_energy[occidx],
        )
        d = 1 / e_aibj
        nocc = self.nocc
        nvir = self.nvir
        eri_k = "ovvv"
        erifile = f"{eri_k}_{self.mole_name}.h5"
        erifile = os.path.join(self.scratch_dir, erifile)
        with h5py.File(erifile, "r") as f:
            int1_ = f["eri_mo"][:].reshape(nocc, nvir, nvir, nvir)
        int1_ = int1_.transpose(1, 0, 2, 3)
        int2_ = int1_.transpose(2, 3, 0, 1)
        eri_k = "ovoo"
        erifile = f"{eri_k}_{self.mole_name}.h5"
        erifile = os.path.join(self.scratch_dir, erifile)
        with h5py.File(erifile, "r") as f:
            int3_ = f["eri_mo"][:].reshape(nocc, nvir, nocc, nocc)
        int3_ = int3_.transpose(1, 0, 2, 3)
        int4_ = int3_.transpose(2, 3, 0, 1)

        return int1_, int2_, int3_, int4_, d

    def trans_mat_1(self, atmlst, FC=False, PSO=False, FCSD=False):
        """C.29 oddershede eq"""
        k_1 = self.k_1
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            p_virt = h1[nocc:, nocc:]
            pert = oe.contract("nc,ambc->manb", p_virt, k_1)
            pert += oe.contract("mc,anbc->manb", p_virt, k_1)
            p_occ = h1[:nocc, :nocc]
            pert -= oe.contract("pb,ampn->manb", p_occ, k_1)
            pert -= oe.contract("pa,bmpn->manb", p_occ, k_1)
        if PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)[0]
            p_virt = h1[:, nocc:, nocc:]
            pert = oe.contract("xnc,ambc->xmanb", p_virt, k_1)
            pert += oe.contract("xmc,anbc->xmanb", p_virt, k_1)
            p_occ = h1[:, :nocc, :nocc]
            pert -= oe.contract("xpb,ampn->xmanb", p_occ, k_1)
            pert -= oe.contract("xpa,bmpn->xmanb", p_occ, k_1)
        elif FCSD:
            h1 = self.rpa_obj.pert_fcsd(atmlst)
            h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0, :, :, :, :]
            p_virt = h1[:, :, nocc:, nocc:]
            pert = oe.contract("wxnc,ambc->wxmanb", p_virt, k_1)
            pert += oe.contract("wxmc,anbc->wxmanb", p_virt, k_1)
            p_occ = h1[:, :, :nocc, :nocc]
            pert -= oe.contract("wxpb,ampn->wxmanb", p_occ, k_1)
            pert -= oe.contract("wxpa,bmpn->wxmanb", p_occ, k_1)
        return pert

    def trans_mat_2(self, atmlst, FC=False, PSO=False, FCSD=False):
        """C.30 oddershede eq"""
        k_2 = self.k_2
        nocc = self.nocc
        nvir = self.nvir
        ntot = nocc + nvir
        if FC:
            h1 = self.rpa_obj.pert_fc(atmlst)[0]
            p_virt = h1[nocc:, nocc:]
            pert = oe.contract("nc,ambc->manb", p_virt, k_2)
            pert -= oe.contract("mc,anbc->manb", p_virt, k_2)
            p_occ = h1[:nocc, :nocc]
            pert -= oe.contract("pb,ampn->manb", p_occ, k_2)
            pert += oe.contract("pa,bmpn->manb", p_occ, k_2)
        if PSO:
            h1 = self.rpa_obj.pert_pso(atmlst)
            h1 = np.asarray(h1).reshape(1, 3, ntot, ntot)[0]
            p_virt = h1[:, nocc:, nocc:]
            pert = oe.contract("xnc,ambc->xmanb", p_virt, k_2)
            pert -= oe.contract("xmc,anbc->xmanb", p_virt, k_2)
            p_occ = h1[:, :nocc, :nocc]
            pert -= oe.contract("xpb,ampn->xmanb", p_occ, k_2)
            pert += oe.contract("xpa,bmpn->xmanb", p_occ, k_2)
        if FCSD:
            h1 = self.rpa_obj.pert_fcsd(atmlst)
            h1 = np.asarray(h1).reshape(-1, 3, 3, ntot, ntot)[0, :, :, :, :]
            p_virt = h1[:, :, nocc:, nocc:]
            pert = oe.contract("wxnc,ambc->wxmanb", p_virt, k_2)
            pert -= oe.contract("wxmc,anbc->wxmanb", p_virt, k_2)
            p_occ = h1[:, :, :nocc, :nocc]
            pert -= oe.contract("wxpb,ampn->wxmanb", p_occ, k_2)
            pert += oe.contract("wxpa,bmpn->wxmanb", p_occ, k_2)
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
            m -= self.da0_triplet_t23 / 4
            m -= self.da0_singlet_s2 * (2 / 3) / 4
        else:
            m += self.part_b2(0)
            m -= self.da0_singlet_s1 / 4
            m -= self.da0_singlet_s2 / 4
        m = m.reshape(nocc * nvir, nocc * nvir)
        return m

    def pp_ssc_pso(self, atm1lst, atm2lst, elements=False):
        """Method that obtain the linear response between PSO perturbation at
        SOPPA level of approach between two nuclei
        Args:
            atm1lst (list): First nuclei
            atm2lst (list): Second nuclei

        Returns:
            real: PSO response at SOPPA level of approach
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
        da0 = self.da0_singlet_s1
        da0 += self.da0_singlet_s2
        h1_corr3 = self.t1_opt_PSO(atmlst=atm1lst)
        h2_corr3 = self.t1_opt_PSO(atmlst=atm2lst)
        w4 = self.w4(PSO=True, atm1lst=atm1lst, atm2lst=atm2lst)
        h1 = (-2 * h1) + h1_corr1 + h1_corr2 - h1_corr3 / 4
        h2 = (-2 * h2) + h2_corr1 + h2_corr2 - h2_corr3 / 4
        m = self.M_rpa(triplet=False)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m += self.part_b2(0)
        m += self.S2
        m -= da0 / 4
        m = m.reshape(nocc * nvir, nocc * nvir)
        if elements:
            return h1, m, h2
        else:
            p = sp.linalg.inv(m)
            p = -p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = oe.contract("xia,iajb,yjb->xy", h1, p, h2)
            e -= w4 / 2
            para.append(e)
            pso = np.asarray(para) * nist.ALPHA**4
            return pso

    def pp_ssc_fc(self, atm1lst, atm2lst, elements=False):
        """Method that obtain the linear response between two FC perturbation 
        at SOPPA level of approach between two nuclei
        Args:
            atm1lst (list): First nuclei
            atm2lst (list): Second nuclei

        Returns:
            real: FC response at SOPPA level of approach
        """
        nvir = self.nvir
        nocc = self.nocc
        h1 = self.rpa_obj.pert_fc(atm1lst)[0][:nocc, nocc:]
        h2 = self.rpa_obj.pert_fc(atm2lst)[0][:nocc, nocc:]
        da0 = self.da0_triplet_t23
        da0 += self.da0_singlet_s2 * (2 / 3)
        h1_corr3 = self.t1_opt_triplet(FC=True, atmlst=atm1lst)
        h2_corr3 = self.t1_opt_triplet(FC=True, atmlst=atm2lst)
        w4 = self.w4(FC=True, atm1lst=atm1lst, atm2lst=atm2lst)
        m = self.M_rpa(triplet=True)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m -= self.part_b2(1)
        m += self.S2
        m -= da0 / 4
        h1_corr1 = self.correction_pert(atmlst=atm1lst, FC=True)
        h1_corr2 = self.correction_pert_2(atmlst=atm1lst, FC=True)
        h2_corr1 = self.correction_pert(atmlst=atm2lst, FC=True)
        h2_corr2 = self.correction_pert_2(atmlst=atm2lst, FC=True)
        h1 = (2 * h1) + h1_corr1 + h1_corr2 + h1_corr3 / 4
        h2 = (2 * h2) + h2_corr1 + h2_corr2 + h2_corr3 / 4
        m = m.reshape(nocc * nvir, nocc * nvir)
        if elements:
            return h1, m, h2
        else:
            p = -sp.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = oe.contract("ia,iajb,jb", h1, p, h2)
            e -= w4 / 2
            para.append(e / 4)
            fc = oe.contract(",k,xy->kxy", nist.ALPHA**4, para, np.eye(3))
            return fc

    def pp_ssc_fcsd(self, atm1lst, atm2lst, elements=False):
        """Method that obtain the linear response between two FC+SD 
        perturbation at SOPPA level of approach between two nuclei
        Args:
            atm1lst (list): First nuclei
            atm2lst (list): Second nuclei

        Returns:
            real: FC+SD response at SOPPA level of approach
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
        da0 = self.da0_triplet_t23
        da0 += self.da0_singlet_s2 * (2 / 3)
        h1_corr3 = self.t1_opt_triplet(FCSD=True, atmlst=atm1lst)
        h2_corr3 = self.t1_opt_triplet(FCSD=True, atmlst=atm2lst)
        w4 = self.w4(FCSD=True, atm1lst=atm1lst, atm2lst=atm2lst)

        h1 = (2 * h1) + h1_corr1 + h1_corr2 - h1_corr3 / 4
        h2 = (2 * h2) + h2_corr1 + h2_corr2 - h2_corr3 / 4
        m = self.M_rpa(triplet=True)
        m = m.reshape(nocc, nvir, nocc, nvir)
        m += self.part_a2
        m -= self.part_b2(1)
        m += self.S2
        m -= da0 / 4
        m = m.reshape(nocc * nvir, nocc * nvir)
        if elements:
            return h1, m, h2
        else:
            p = -sp.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = oe.contract("wxia,iajb,wyjb->xy", h1, p, h2)
            e -= w4 / 2
            para.append(e)
            fcsd = np.asarray(para) * nist.ALPHA**4
            return fcsd

    def ssc(self, atom1, atom2, FC=False, FCSD=False, PSO=False):
        """Function for Spin-Spin Coupling calculation at SOPPA level of
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

        iso_ssc = unit * oe.contract("kii->k", prop) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        # print(prop*unit*gyro1*gyro2, )
        jtensor = oe.contract("i,i,j->i", iso_ssc, gyro1, gyro2)
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
