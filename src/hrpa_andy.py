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
from pyscf.lib import logger
import scipy

def uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

@attr.s
class Prop_pol:
    """_summary_
    """
    mf = attr.ib(default=None, type=scf.RHF, validator=attr.validators.instance_of(scf.hf.RHF))

    def __attrs_post_init__(self):
        self.mol = self.mf.mol
        self.mo_coeff = self.mf.mo_coeff
        self.mo_occ = self.mf.mo_occ
        self.mo_energy = self.mf.mo_energy
        self.nuc_pair = [(i,j) for i in range(self.mol.natm) for j in range(i)]
        self.occidx = numpy.where(self.mo_occ > 0)[0]
        self.viridx = numpy.where(self.mo_occ == 0)[0]
        
        self.orbv = (self.mo_coeff[:,self.viridx])
        self.orbo = (self.mo_coeff[:,self.occidx])
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        self.numpyrim = self.nocc + self.nvir

        eri_mo = ao2mo.general(self.mol, [self.mo_coeff,self.mo_coeff,self.mo_coeff,self.mo_coeff], compact=False)
        eri_mo = eri_mo.reshape(self.numpyrim,self.numpyrim,self.numpyrim,self.numpyrim)
        self.ao2moj =  eri_mo[self.nocc:,self.nocc:,:self.nocc,:self.nocc]
        self.ao2mok = eri_mo[self.nocc:,:self.nocc,self.nocc:,:self.nocc]
        self.ao2mooc = eri_mo[:self.nocc,:self.nocc,:self.nocc,:self.nocc]
        self.ao2movi = eri_mo[self.nocc:,self.nocc:,self.nocc:,self.nocc:]

        self.ao2mo1o3v = eri_mo[self.nocc:,self.nocc:,self.nocc:,:self.nocc]
        self.ao2mo3o1v = eri_mo[self.nocc:,:self.nocc,:self.nocc,:self.nocc]


    def kappa(self,i,alpha,beta,m,n,eom):
        nocc = self.nocc
        ao2mok = self.ao2mok
        c1 = numpy.sqrt(3.0)
        if i == 2 :
            if alpha == beta :
                return 0.0
            if m == n :
                return 0.0

        energies = eom[alpha] + eom[beta] - eom[m+nocc] - eom[n+nocc]
        Na = ao2mok[m,alpha,n,beta]
        if i == 1 :
            Nb = -ao2mok[m,beta,n,alpha]
        else:
            Nb = ao2mok[m,beta,n,alpha]

        kapa = Na - Nb
        kapa = kapa/energies
        if i == 2:
            kapa = c1*kapa

        return kapa


    def A21(self,beta,alpha,eom):
        """[summary]
            Construye el primer término de A(2)
        """
        ao2mok = self.ao2mok
        A2a = 0.0
        c1 = numpy.sqrt(3.0)
        nvir = self.nvir
        nocc = self.nocc
        for a in range(nvir):
            for b in range(nvir):
                for delta in range(nocc):
                    kapa = 0.0
    #     kappas
                    kappa1 = self.kappa(1,alpha,delta,a,b,eom)
                    kappa2 = self.kappa(2,alpha,delta,a,b,eom)
                    kapa = kappa1 + c1*kappa2
                    A2a  += ao2mok[a,beta,b,delta]*kapa

        return A2a
##########################################################
    def A22(self,m,n,eom):
        """[summary]
            Construye el segundo término de A(2)
        """
        A2b = 0.0
        c1 = numpy.sqrt(3.0)
        ao2mok = self.ao2mok
        nocc = self.nocc
        nvir = self.nvir
        for pi in range(nocc):
            for delta in range(nocc):
                for b in range(nvir):
    #     kappas
                    kappa1 = self.kappa(1,pi,delta,m,b,eom)
                    kappa2 = self.kappa(2,pi,delta,m,b,eom)
                    kapa = kappa1 + c1*kappa2
                    A2b  += ao2mok[b,delta,n,pi]*kapa
        return A2b
##########################################################
    def B21(self,alpha,beta,m,n,eom):
        """[summary]
            Construye los 2 primeros término de B(2)
        """
        B2a  = 0.0
        B2a1 = 0.0
        B2a2 = 0.0
        ao2mok = self.ao2mok
        nocc = self.nocc
        nvir = self.nvir
        c1 = numpy.sqrt(3.0)
        for r in range(nvir):
            for pi in range(nocc):
    #Primer término
    #     kappas
                kappa1 = self.kappa(1,beta,pi,m,r,eom)
                kappa2 = self.kappa(2,beta,pi,m,r,eom)
                kapa = kappa1 + c1*kappa2
                #B2a1  += ao2mok[alpha,n,r,pi]*kapa
                B2a1  += ao2mok[n,alpha,r,pi]*kapa
    #Segundo término
    #     kappas
                kappa1 = self.kappa(1,alpha,pi,n,r,eom)
                kappa2 = self.kappa(2,alpha,pi,n,r,eom)
                #print(kappa1,kappa2)
                kapa = kappa1 + c1*kappa2
                #B2a2  += ao2mok(beta,m,r,pi)*kapa
                B2a2  += ao2mok[m,beta,r,pi]*kapa
        B2a = B2a1 + B2a2
        return B2a
##########################################################
    def B22(self,alpha,beta,m,n,eom,MS):
        """[summary]
            Construye el tercero y cuarto término de B(2)
        """
        B2b  = 0.0
        B2b1 = 0.0
        B2b2 = 0.0
        ao2mok = self.ao2mok
        ao2moj = self.ao2moj
        nvir = self.nvir
        nocc = self.nocc
        if MS == 1:
            c1 = -numpy.sqrt(3.0)/3.0
        else:
            c1 = numpy.sqrt(3.0)

        for r in range(nvir):
            for pi in range(nocc):
    #Primer término
    #     kappas
                kappa1 = self.kappa(1,pi,beta,m,r,eom)
                kappa2 = self.kappa(2,pi,beta,m,r,eom)
                kapa = kappa1 + c1*kappa2
                #B2a1  += ao2mok[alpha,pi,r,n]*kapa
                B2b1  += ao2moj[r,n,alpha,pi]*kapa
#Segundo término
#     kappas
                kappa1 = self.kappa(1,pi,alpha,n,r,eom)
                kappa2 = self.kappa(2,pi,alpha,n,r,eom)
                kapa = kappa1 + c1*kappa2
                #B2a2  += ao2mok[beta,pi,r,m]*kapa

                B2b2  += ao2moj[r,m,beta,pi]*kapa
        B2b = B2b1 + B2b2
        return B2b
##########################################################
    def B23(self,alpha,beta,m,n,eom,MS):
        """[summary]
            Construye el quinto término de B(2)
        """
        ao2mok = self.ao2mok
        ao2mooc = self.ao2mooc
        B2c  = 0.0
        nocc = self.nocc
        if MS == 1:
            c1 = -numpy.sqrt(3.0)/3.0
        else:
            c1 = numpy.sqrt(3.0)

        for pi in range(nocc):
            for delta in range(nocc):
    #     kappas
                kappa1 = self.kappa(1,delta,pi,m,n,eom)
                kappa2 = self.kappa(2,delta,pi,m,n,eom)
                #print(kappa1,kappa2)
                kapa = kappa1 + c1*kappa2
                B2c  += ao2mooc[beta,pi,alpha,delta]*kapa
        return B2c
##########################################################
    def B24(self,alpha,beta,m,n,eom,MS):
        """[summary]
            Construye el sexto término de B(2)
        """
        B2d  = 0.0
        ao2mok = self.ao2mok
        ao2movi = self.ao2movi
        nocc = self.nocc
        nvir = self.nvir
        if MS == 1:
            c1 = -numpy.sqrt(3.0)/3.0
        else:
            c1 = numpy.sqrt(3.0)

        for p in range(nvir):
            for q in range(nvir):
    #     kappas
                kappa1 = self.kappa(1,beta,alpha,p,q,eom)
                kappa2 = self.kappa(2,beta,alpha,p,q,eom)
                kapa = kappa1 + c1*kappa2
                B2d  += ao2movi[q,m,p,n]*kapa
        return B2d
########################################################
########################################################
########################################################
    def SO(self,alpha,beta,m,n,eom,MS):
        """[summary]
        Ecuación 60 de nielsen 1980, la cual se relaciona con
        la Ecuación 40, S(0,2) = (q+|q+)
        J. Chern. Phys., Vol. 73, No. 12,15 December 1980
        """
        if alpha == beta and m == n :
            c1 = 1.0
        else:
            c1 = 0.0
        c2 = -0.5
        c3 = numpy.sqrt(3.0)
        nvir = self.nvir
        nocc = self.nocc
        ao2mok = self.ao2mok
        pq2bph  = 0.0
        pq2bph1 = 0.0
        pq2bph2 = 0.0
        for a in range(nvir):
            for pi in range(nocc):
                if m == n :  #rho_{i,j}^(2)
                    for b in range(nvir):
                        kappa1 = self.kappa(1,beta,pi,a,b,eom)
                        kappa2 = self.kappa(2,beta,pi,a,b,eom)
                        kappab = kappa1 + c3*kappa2
                        e4 = eom[alpha] + eom[pi] - eom[a+nocc] - eom[b+nocc]
                        pq2bph1 += c2*ao2mok[a,alpha,b,pi]*kappab/e4
                if alpha == beta:
                    for delta in range(nocc):
                        kappa1 = self.kappa(1,pi,delta,m,a,eom)
                        kappa2 = self.kappa(2,pi,delta,m,a,eom)
                        kappac = kappa1 + c3*kappa2
                        e4 = eom[pi] + eom[delta] - eom[a+nocc] - eom[n+nocc]
                        pq2bph2 += c2*ao2mok[a,delta,n,pi]*kappac/e4
        pq2bph = c1 + pq2bph1+pq2bph2
        S2 = 0.0
        S2 = pq2bph1+pq2bph2
        return pq2bph, S2
########################################################
########################################################
    def a2b2(self,m,n,beta,alpha,eom,MS):

        ao2mok = self.ao2mok


        nocc = self.nocc
        nvir = self.nvir

        tpc = 0
        if tpc == 0:
        #!!!Empiezo contrucción de A(2)_{m\alpha,n\beta}
        #md: 0.5 por la delta dirac
            md = -0.5
            if   alpha == beta and m == n :
        #Construcción del primer término de A2
                A2a1 = self.A21(beta,alpha,eom)
                A2a1 = md*A2a1
        #Construcción del segundo término de A2
                A2b1 = self.A22(m,n,eom)
                A2b1 = md*A2b1
        #Sumo ambos términos
                A2l = A2a1 + A2b1
            elif alpha == beta and m != n :
                A2l = self.A22(m,n,eom)
                A2l = md*A2l
            elif m == n and alpha != beta :
                A2l = self.A21(beta,alpha,eom)
                A2l = md*A2l
            else:
                A2l = 0.0

        #!!!Empiezo contrucción de B(2)_{m\alpha,n\beta}
        #Construcción de los primeros dos términos
            #print("B2a1")
            B2a1 = self.B21(alpha,beta,m,n,eom)
        #Construcción del tercero y cuarto
            B2b1 = self.B22(alpha,beta,m,n,eom,MS)
        #Construcción del quinto término
            B2c1 = self.B23(alpha,beta,m,n,eom,MS)
        #Construcción del sexto término
            B2d1 = self.B24(alpha,beta,m,n,eom,MS)
            if MS == 0 :
                B2l = B2a1 + B2b1 - B2c1 - B2d1  #signos iguales a j phys b at mol opt phys 1997, 30, 3773
                    #el menos en B2a1 lo tomo del articulo phys rev a 1970, 2, 2208 Eq 64
            else:
                B2l = B2a1 - B2b1 + B2c1 + B2d1
            B2l = 0.5*B2l

        #!!!Empiezo contrucción de A(2)_{n\beta,m\alpha}
        #cambiando de signo alguno de los 2 términos no mejora pso, aunque produce que converja
        #en C2H2
            if   alpha == beta and m == n :
        #Construcción del primer término de A2
                A2a2 = self.A21(alpha,beta,eom)
                A2a2 = md*A2a2
        #Construcción del segundo término de A2
                A2b2 = self.A22(n,m,eom)
                A2b2 = md*A2b2
        #Sumo ambos términos
                A2r = A2a2 + A2b2
            elif alpha == beta and m != n :
                A2r = self.A22(n,m,eom)
                A2r = md*A2r
            elif m == n and alpha != beta :
                A2r = self.A21(alpha,beta,eom)
                A2r = md*A2r
            else:
                A2r = 0.0
            S0S2, S2 = self.SO(alpha,beta,m,n,eom,MS)
            
            A2 =  0.5*(A2l + A2r)
            A0S2a = (eom[m+nocc]-eom[alpha])*S2
            A0S2b = (eom[n+nocc]-eom[beta])*S2
            A0S2  = 0.5*(A0S2a+A0S2b)
            B2 = B2l

            if MS == 1:
        #Se resta, debido a que es una excitación triplete
                A2B2 = A2-B2+A0S2
            else:
                A2B2 = -A2-B2-A0S2#*1.5
            return A2B2, S0S2, S2
        else:
            S2 = 0.0
            S0S2 = 0.0
            S0S2, S2 = self.SO(alpha,beta,m,n,eom,MS)
            A0S2a = (eom[m+nocc]-eom[alpha])*S2
            A0S2b = (eom[n+nocc]-eom[beta])*S2
            A0S2  = 0.5*(A0S2a+A0S2b)
            A2a = self.A2_T(alpha,beta,m,n,ao2mok,eom,nocc,nvir)
            A2b = self.A2_T(beta,alpha,n,m,ao2mok,eom,nocc,nvir)
            #if abs(A2a - A2b) > 0.0001 :
            #    print("A diferencia ",A2a, A2b)
            A2  = 0.5*(A2a + A2b)
            #B2 es simetrica al itercambio de índices
            B2 = self.B2_T(alpha,beta,m,n,eom)
            A2B2 = -A2-B2#-A0S2
            return A2B2, S0S2, S2
#########################################################
##########################################################
    def A2_T(self,alpha,beta,m,n,eom):
        nvir = self.nvir
        nocc = self.nocc
        ao2mok = self.ao2mok
        A2a = 0.0
        A2b = 0.0
        A2  = 0.0
        if alpha == beta:
            for q in range(nvir):
                for pi in range(nocc):
                    for gamma in range(nocc):
                        kappa1 = ao2mok[m,pi,q,gamma] - ao2mok[m,gamma,q,pi]
                        kappa2 = ao2mok[q,pi,n,gamma] - ao2mok[n,pi,q,gamma]
                        e4     = eom[pi] + eom[gamma] - eom[m+nocc] - eom[q+nocc]
                        A2a   += kappa1 * kappa2/e4
        if m == n:
            for q in range(nvir):
                for p in range(nvir):
                    for pi in range(nocc):
                        kappa1 = ao2mok[q,beta,p,pi] - ao2mok[p,beta,q,pi]
                        kappa2 = ao2mok[q,alpha,p,pi] - ao2mok[q,pi,p,alpha]
                        e4     = eom[alpha] + eom[pi] - eom[q+nocc] - eom[p+nocc]
                        A2b   += kappa1 * kappa2/e4
        A2 = 0.5*(A2b-A2a)
        return A2

    def B2_T(self,alpha,beta,m,n,eom):
        nvir = self.nvir
        nocc = self.nocc
        ao2moj = self.ao2moj
        ao2mok = self.ao2mok
        ao2movi = self.ao2movi
        ao2mooc = self.ao2mooc
        B21 = 0.0
        B22 = 0.0
        B23 = 0.0
        B24 = 0.0
        B2  = 0.0

        for q in range(nvir):
            for pi in range(nocc):
                kappa1 = ao2moj[q,m,beta,pi] - ao2mok[q,pi,m,beta]
                kappa2 = ao2mok[q,pi,n,alpha] - ao2mok[q,alpha,n,pi]
                e4     = eom[pi] + eom[alpha] - eom[n+nocc] - eom[q+nocc]
                B21    += kappa1 * kappa2/e4

                kappa1 = ao2moj[q,n,alpha,pi] - ao2mok[n,alpha,q,pi]
                kappa2 = ao2mok[m,beta,q,pi] - ao2mok[m,pi,q,beta]
                e4     = eom[beta] + eom[pi] - eom[m+nocc] - eom[q+nocc]
                B22    += kappa1 * kappa2/e4

        for p in range(nvir) :
            for q in range(nvir) :
                kappa1 = ao2movi[p,n,q,m] - ao2movi[p,m,q,n]
                kappa2 = ao2mok[p,beta,q,alpha] - ao2mok[p,alpha,q,beta]
                e4     = eom[beta] + eom[alpha] - eom[p+nocc] - eom[q+nocc]
                B23    += kappa1 * kappa2/e4

        for pi in range(nocc) :
            for delta in range(nocc) :
                kappa1 = ao2mooc[alpha,pi,beta,delta] - ao2mooc[alpha,delta,beta,pi]
                kappa2 = ao2mok[m,pi,n,delta] - ao2mok[m,delta,n,pi]
                e4     = eom[pi] + eom[delta] - eom[m+nocc] - eom[n+nocc]
                B24    += kappa1 * kappa2/e4

        B2 = B21 + B22 + 0.5*(B23 + B24)
        B2 = -B2
        return B2

    def M(self, multiplicidad):
        nvir = self.nvir
        nocc = self.nocc
        EOM = self.mo_energy
        k = 0
        irow = 0
        icol = 0
        Einv = numpy.zeros((nvir * nocc, nvir * nocc), dtype=float)
        MN = numpy.zeros((nvir * nocc, nvir * nocc), dtype=float)
        indexrow = numpy.zeros((nvir, nvir, nocc, nocc), dtype=int)
        indexcol = numpy.zeros((nvir, nvir, nocc, nocc), dtype=int)
        ao2moj = self.ao2moj
        ao2mok = self.ao2mok
        tpcinv = 1
        for i in range(nocc):
            for a in range(nvir):
                s = a + nocc
                Einv[k, k] = 1.0e0 / (EOM[s] - EOM[i])
                k += 1
                for j in range(nocc):
                    for b in range(nvir):
                        t = b + nocc
                        if multiplicidad == 1:  # <ab|ji> + <aj|bi>
                            EMO = 0.0
                            if i == j and a == b and tpcinv == 1:
                                EMO = EOM[s] - EOM[i]
                            MN[irow, icol] = -EMO + ao2moj[a, b, j, i] + ao2mok[a, j, b, i]
                            temp = 0.0
                            temp, s0s2, vS2 = self.a2b2(a,b,j,i,EOM,multiplicidad)
                            MN[irow, icol] = MN[irow, icol] - temp
                            
                        indexrow[a, b, j, i] = irow
                        indexcol[a, b, j, i] = icol
                        icol = icol + 1
                        if icol > nocc * nvir - 1:
                            irow += 1
                        if icol > nocc * nvir - 1:
                            icol = 0
        return MN

    def PQ2a(self,alpha,m,P,eom,MS):
        """[summary]
        Construye segundo término Ec C.27, necesario para corregir
        los momentos de transición ph en SOPPA o HRPA
        P son los elementos del valor medio en MO sin multiplicar por
        2
        """
        nvir = self.nvir
        nocc = self.nocc
        ao2mo1o3v = self.ao2mo1o3v
        ao2mo3o1v = self.ao2mo3o1v
        c1 = numpy.sqrt(3.0)
        pq2bpha = 0.0
        for n in range(nvir):
            pq2bph1 = 0.0
            pq2bph2 = 0.0
            for a in range(nvir):
                for pi in range(nocc):
                    for b in range(nvir):
                        kappa1 = self.kappa(1,pi,alpha,a,b,eom)
                        kappa2 = self.kappa(2,pi,alpha,a,b,eom)
                        kapa = kappa1 + c1*kappa2
                        pq2bph1 += ao2mo1o3v[b,n,a,pi]*kapa
                    for delta in range(nocc):
                        kappa1 = self.kappa(1,pi,delta,a,n,eom)
                        kappa2 = self.kappa(2,pi,delta,a,n,eom)
                        kapa = kappa1 + c1*kappa2
                        pq2bph2 += ao2mo3o1v[a,pi,delta,alpha]*kapa

            e2 = eom[alpha]-eom[n+nocc]
            #print("kappa_alpha^n ",alpha,n,c2*(pq2bph1-pq2bph2)/e2)
            pq2bpha += P[m+nocc,n+nocc]*(pq2bph1-pq2bph2)/e2

        pq2bphb = 0.0
        for beta in range(nocc):
            pq2bph1 = 0.0
            pq2bph2 = 0.0
            for a in range(nvir):
                for pi in range(nocc):
                    for b in range(nvir):
                        kappa1 = self.kappa(1,pi,beta,a,b,eom)
                        kappa2 = self.kappa(2,pi,beta,a,b,eom)
                        kapa = kappa1 + c1*kappa2
                        pq2bph1 += ao2mo1o3v[b,m,a,pi]*kapa
                    for delta in range(nocc):
                        kappa1 = self.kappa(1,pi,delta,a,m,eom)
                        kappa2 = self.kappa(2,pi,delta,a,m,eom)
                        kapa = kappa1 + c1*kappa2
                        pq2bph2 += ao2mo3o1v[a,pi,delta,beta]*kapa

            e2 = eom[beta]-eom[m+nocc]
            #print("kappa_beta^m ",c2*(pq2bph1-pq2bph2)/e2)
            pq2bphb += P[beta,alpha]*(pq2bph1-pq2bph2)/e2
        #print("ph : ",pq2bphb)

        if MS == 1 :
            pq2bph = (pq2bpha-pq2bphb)
        else:
            pq2bph = (pq2bpha+pq2bphb)

        return pq2bph

    def PQ2b(self,alpha,m,P,eom):
        """[summary]
        Construye tercer término Ec C.27, necesario para corregir
        los momentos de transición ph en SOPPA o HRPA
        P son los elementos del valor medio en MO sin multiplicar por
        2
        """
        pq2bph = 0.0
        pq2bph1 = 0.0
        pq2bph2 = 0.0
        nvir = self.nvir
        nocc = self.nocc
        ao2mok = self.ao2mok

        c1 = numpy.sqrt(3.0)
        c2 = -numpy.sqrt(2.0)/2.0
        for a in range(nvir):
            for b in range(nvir):
                for pi in range(nocc):
                    kappa1 = self.kappa(1,alpha,pi,a,b,eom)
                    kappa2 = self.kappa(2,alpha,pi,a,b,eom)
                    kappab = kappa1 + c1*kappa2
                    for delta in range(nocc):
                        kappa1 = self.kappa(1,pi,delta,m,a,eom)
                        kappa2 = self.kappa(2,pi,delta,m,a,eom)
                        kappac = kappa1 + c1*kappa2

                        pq2bph1 = P[m+nocc,delta]*kappab
                        pq2bph2 = P[b+nocc,alpha]*kappac

                        e4 = eom[pi] + eom[delta] - eom[a+nocc] - eom[b+nocc]

                        pq2bph += (ao2mok[a,delta,b,pi]*(pq2bph1+pq2bph2))/e4
        return pq2bph



    def pert_fc(self,atmlst):
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mol = self.mol
        coords = mol.atom_coords()
        ao = numint.eval_ao(mol, coords)
        mo = ao.dot(mo_coeff)
        orbo = mo[:,mo_occ> 0]
        #orbo = mo[:,:]
        orbv = mo[:,mo_occ==0]
        #orbv = mo[:,:]
        fac = 8*numpy.pi/3 *.5  # *.5 due to s = 1/2 * pauli-matrix
        h1 = []
        for ia in atmlst:
                h1.append(fac * numpy.einsum('p,i->pi', orbv[ia], orbo[ia]))
        return h1

    def pert_corr(self, pert, multiplicidad):
        nocc = self.nocc
        nvir = self.nvir
        k = 0
        EOM = self.mo_energy
        gpvph = numpy.zeros(nocc*nvir)
        for i in range(nocc):
            for a in range(nvir):
                s = a + nocc
                k += 1
                # Armando vector gradient property
            
                PQA2 = self.PQ2a(i, a, pert, EOM, multiplicidad)
                PQB2 = self.PQ2b(i, a, pert, EOM)
                if multiplicidad == 1:
                    gpvph[a + i * nvir] = 2.0 * pert[ i, s] + (PQA2 - PQB2)
                else:
                    gpvph[a + i * nvir] = -2.0 * pert[ i, s] - (PQA2 + PQB2)
        return gpvph.reshape(nocc,nvir, order='F')




    def pp_ssc_fc(self, atom1, atom2):
        nvir = self.nvir
        nocc = self.nocc
        h1 = self.pert_fc(atom1)
        h2 = self.pert_fc(atom2)    
        #h1 = self.pert_corr(h1[0],1)
        #h2 = self.pert_corr(h2[0],1)
        m = self.M(multiplicidad=1)
        
        p = numpy.linalg.inv(m)
        p = -p.reshape(nocc,nvir,nocc,nvir)
        para = []
        e = numpy.einsum('ia,iajb,jb', h1[0].T, p , h2[0].T)
        para.append(e*4)  # *4 for +c.c. and for double occupancy
        fc = numpy.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))    
        return fc        



    def kernel(self, atom1=None, atom2=None):
        mol = self.mol

        e11 = self.pp_ssc_fc(atom1=atom1, atom2=atom2)
        #e11 = self.ssc_4c_pyscf()
        nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS) 
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag ** 2 * numpy.einsum('kii->k', e11) / 3
        natm = mol.natm


        gyro1 = [get_nuc_g_factor(mol.atom_symbol(atom1[0]))]
        gyro2 = [get_nuc_g_factor(mol.atom_symbol(atom2[0]))]
        jtensor = numpy.einsum('i,i,j->i', iso_ssc, gyro1, gyro2)
                    
        return jtensor
