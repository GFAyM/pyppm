#Andy Zapata   IMIT-2021
#
# Obtiene cada uno de los términos de la matriz del propagador
# principal que están conformado por cantidades de SOPPA d
# diferentes de las de RPA, como son A(2) y B(2) que hacen parte
# de p - h
#
# En espectativa la parte de 2p-2h
#
# Este programa es basado en el trabajo
#   Computer Physics Reports 2 (1984) 33-92
#
# Greek : ocupados, holes
# Roma  : desocupados, particles
#
#!!Nota: tener en cuenta que la Ecuación C.16 tiene el signo cambiado

#importo modulos
import numpy as np
from numpy.lib.type_check import real

def kappa(i,alpha,beta,m,n,eom,ao2mok,nocc):

    c1 = np.sqrt(3.0)
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
############################################################
def A21(beta,alpha,ao2mok,eom,nocc,nvir):
    """[summary]
        Construye el primer término de A(2)
    """
    A2a = 0.0
    c1 = np.sqrt(3.0)
    for a in range(nvir):
        for b in range(nvir):
            for delta in range(nocc):
                kapa = 0.0
#     kappas
                kappa1 = kappa(1,alpha,delta,a,b,eom,ao2mok,nocc)
                kappa2 = kappa(2,alpha,delta,a,b,eom,ao2mok,nocc)
                kapa = kappa1 + c1*kappa2
                A2a  += ao2mok[a,beta,b,delta]*kapa

    return A2a
##########################################################
def A22(m,n,ao2mok,eom,nocc,nvir):
    """[summary]
        Construye el segundo término de A(2)
    """
    A2b = 0.0
    c1 = np.sqrt(3.0)
    for pi in range(nocc):
        for delta in range(nocc):
            for b in range(nvir):
#     kappas
                kappa1 = kappa(1,pi,delta,m,b,eom,ao2mok,nocc)
                kappa2 = kappa(2,pi,delta,m,b,eom,ao2mok,nocc)
                kapa = kappa1 + c1*kappa2
                A2b  += ao2mok[b,delta,n,pi]*kapa
    return A2b
##########################################################
def B21(alpha,beta,m,n,ao2mok,eom,nocc,nvir):
    """[summary]
        Construye los 2 primeros término de B(2)
    """
    B2a  = 0.0
    B2a1 = 0.0
    B2a2 = 0.0
    c1 = real(np.sqrt(3.0))
    for r in range(nvir):
        for pi in range(nocc):
#Primer término
#     kappas
            kappa1 = kappa(1,beta,pi,m,r,eom,ao2mok,nocc)
            kappa2 = kappa(2,beta,pi,m,r,eom,ao2mok,nocc)
            #print(kappa1,kappa2)
            kapa = kappa1 + c1*kappa2
            #B2a1  += ao2mok[alpha,n,r,pi]*kapa
            B2a1  += ao2mok[n,alpha,r,pi]*kapa
#Segundo término
#     kappas
            kappa1 = kappa(1,alpha,pi,n,r,eom,ao2mok,nocc)
            kappa2 = kappa(2,alpha,pi,n,r,eom,ao2mok,nocc)
            #print(kappa1,kappa2)
            kapa = kappa1 + c1*kappa2
            #B2a2  += ao2mok(beta,m,r,pi)*kapa
            B2a2  += ao2mok[m,beta,r,pi]*kapa
    B2a = B2a1 + B2a2
    return B2a
##########################################################
def B22(alpha,beta,m,n,ao2mok,ao2moj,eom,nocc,nvir,MS):
    """[summary]
        Construye el tercero y cuarto término de B(2)
    """
    B2b  = 0.0
    B2b1 = 0.0
    B2b2 = 0.0

    if MS == 1:
        c1 = -real(np.sqrt(3.0)/3.0)
    else:
        c1 = real(np.sqrt(3.0))

    for r in range(nvir):
        for pi in range(nocc):
#Primer término
#     kappas
            kappa1 = kappa(1,pi,beta,m,r,eom,ao2mok,nocc)
            kappa2 = kappa(2,pi,beta,m,r,eom,ao2mok,nocc)
            kapa = kappa1 + c1*kappa2
            #B2a1  += ao2mok[alpha,pi,r,n]*kapa
            B2b1  += ao2moj[r,n,alpha,pi]*kapa
#Segundo término
#     kappas
            kappa1 = kappa(1,pi,alpha,n,r,eom,ao2mok,nocc)
            kappa2 = kappa(2,pi,alpha,n,r,eom,ao2mok,nocc)
            kapa = kappa1 + c1*kappa2
            #B2a2  += ao2mok[beta,pi,r,m]*kapa

            B2b2  += ao2moj[r,m,beta,pi]*kapa
    B2b = B2b1 + B2b2
    return B2b
##########################################################
def B23(alpha,beta,m,n,ao2mok,ao2mooc,eom,nocc,MS):
    """[summary]
        Construye el quinto término de B(2)
    """
    B2c  = 0.0

    if MS == 1:
        c1 = -real(np.sqrt(3.0)/3.0)
    else:
        c1 = real(np.sqrt(3.0))

    for pi in range(nocc):
        for delta in range(nocc):
#     kappas
            kappa1 = kappa(1,delta,pi,m,n,eom,ao2mok,nocc)
            kappa2 = kappa(2,delta,pi,m,n,eom,ao2mok,nocc)
            #print(kappa1,kappa2)
            kapa = kappa1 + c1*kappa2
            B2c  += ao2mooc[beta,pi,alpha,delta]*kapa
    return B2c
##########################################################
def B24(alpha,beta,m,n,ao2mok,ao2movi,eom,nocc,nvir,MS):
    """[summary]
        Construye el sexto término de B(2)
    """
    B2d  = 0.0

    if MS == 1:
        c1 = -real(np.sqrt(3.0)/3.0)
    else:
        c1 = real(np.sqrt(3.0))

    for p in range(nvir):
        for q in range(nvir):
#     kappas
            kappa1 = kappa(1,beta,alpha,p,q,eom,ao2mok,nocc)
            kappa2 = kappa(2,beta,alpha,p,q,eom,ao2mok,nocc)
            kapa = kappa1 + c1*kappa2
            B2d  += ao2movi[q,m,p,n]*kapa
    return B2d
########################################################
def PQ2a(alpha,m,nvir,P,nocc,eom,ao2mok,ao2mo3o1v,ao2mo1o3v,MS):
    """[summary]
    Construye segundo término Ec C.27, necesario para corregir
    los momentos de transición ph en SOPPA o HRPA
    P son los elementos del valor medio en MO sin multiplicar por
    2
    """
    c1 = np.sqrt(3.0)
    c2 = np.sqrt(2.0)/2.0
    pq2bpha = 0.0
    for n in range(nvir):
        pq2bph1 = 0.0
        pq2bph2 = 0.0
        for a in range(nvir):
            for pi in range(nocc):
                for b in range(nvir):
                    kappa1 = kappa(1,pi,alpha,a,b,eom,ao2mok,nocc)
                    kappa2 = kappa(2,pi,alpha,a,b,eom,ao2mok,nocc)
                    kapa = kappa1 + c1*kappa2
                    pq2bph1 += ao2mo1o3v[b,n,a,pi]*kapa
                for delta in range(nocc):
                    kappa1 = kappa(1,pi,delta,a,n,eom,ao2mok,nocc)
                    kappa2 = kappa(2,pi,delta,a,n,eom,ao2mok,nocc)
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
                    kappa1 = kappa(1,pi,beta,a,b,eom,ao2mok,nocc)
                    kappa2 = kappa(2,pi,beta,a,b,eom,ao2mok,nocc)
                    kapa = kappa1 + c1*kappa2
                    pq2bph1 += ao2mo1o3v[b,m,a,pi]*kapa
                for delta in range(nocc):
                    kappa1 = kappa(1,pi,delta,a,m,eom,ao2mok,nocc)
                    kappa2 = kappa(2,pi,delta,a,m,eom,ao2mok,nocc)
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
########################################################
def PQ2b(alpha,m,nvir,P,nocc,eom,ao2mok):
    """[summary]
    Construye tercer término Ec C.27, necesario para corregir
    los momentos de transición ph en SOPPA o HRPA
    P son los elementos del valor medio en MO sin multiplicar por
    2
    """
    pq2bph = 0.0
    pq2bph1 = 0.0
    pq2bph2 = 0.0
    c1 = np.sqrt(3.0)
    c2 = -np.sqrt(2.0)/2.0
    for a in range(nvir):
        for b in range(nvir):
            for pi in range(nocc):
                kappa1 = kappa(1,alpha,pi,a,b,eom,ao2mok,nocc)
                kappa2 = kappa(2,alpha,pi,a,b,eom,ao2mok,nocc)
                kappab = kappa1 + c1*kappa2
                for delta in range(nocc):
                    kappa1 = kappa(1,pi,delta,m,a,eom,ao2mok,nocc)
                    kappa2 = kappa(2,pi,delta,m,a,eom,ao2mok,nocc)
                    kappac = kappa1 + c1*kappa2

                    pq2bph1 = P[m+nocc,delta]*kappab
                    pq2bph2 = P[b+nocc,alpha]*kappac

                    e4 = eom[pi] + eom[delta] - eom[a+nocc] - eom[b+nocc]

                    pq2bph += (ao2mok[a,delta,b,pi]*(pq2bph1+pq2bph2))/e4
    #pq2bph *= c2
    return pq2bph
########################################################
def SO(alpha,beta,m,n,nocc,nvir,eom,ao2mok,MS):
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
    c3 = real(np.sqrt(3.0))
    pq2bph  = 0.0
    pq2bph1 = 0.0
    pq2bph2 = 0.0
    for a in range(nvir):
        for pi in range(nocc):
            if m == n :  #rho_{i,j}^(2)
                for b in range(nvir):
                    kappa1 = kappa(1,beta,pi,a,b,eom,ao2mok,nocc)
                    kappa2 = kappa(2,beta,pi,a,b,eom,ao2mok,nocc)
                    kappab = kappa1 + c3*kappa2
                    e4 = eom[alpha] + eom[pi] - eom[a+nocc] - eom[b+nocc]
                    pq2bph1 += c2*ao2mok[a,alpha,b,pi]*kappab/e4
            if alpha == beta:
                for delta in range(nocc):
                    kappa1 = kappa(1,pi,delta,m,a,eom,ao2mok,nocc)
                    kappa2 = kappa(2,pi,delta,m,a,eom,ao2mok,nocc)
                    kappac = kappa1 + c3*kappa2
                    e4 = eom[pi] + eom[delta] - eom[a+nocc] - eom[n+nocc]
                    pq2bph2 += c2*ao2mok[a,delta,n,pi]*kappac/e4
    pq2bph = c1 + pq2bph1+pq2bph2
    S2 = 0.0
    #if MS == 0:
    #    S2  = 0.5*(2*eom[n+nocc]-eom[beta]-eom[alpha])*pq2bph1
    #    S2 += 0.5*(eom[n+nocc]+eom[m+nocc]-2*eom[alpha])*pq2bph2
    #else:
    S2 = pq2bph1+pq2bph2
    return pq2bph, S2
    #return pq2bph, (pq2bph1+pq2bph2)
########################################################
def GA0S2(i,alpha,beta,m,n,ao2mok,eom,nocc,nvir):
    """[summary]
    Calcula A'(2) = 0.5*(A(0)*S(2)+S(2)*A(0))
    J. Chern. Phys., Vol. 73, No. 12,15 December 1980
    """
    c1 = -0.5
    c2 = np.sqrt(3.0)
    A0S2 = 0.0
    A0S2a = 0.0
    A0S2b = 0.0
    e2 = 0.0
    if alpha == beta and m == n:
        e2 = c1*(eom[m+nocc] - eom[alpha])

    for a in range(nvir):
        for b in range(nvir):
            if m == n :
                for pi in range(nocc):
                    kappa1 = kappa(1,alpha,pi,a,b,eom,ao2mok,nocc)
                    kappa2 = kappa(2,alpha,pi,a,b,eom,ao2mok,nocc)
                    kapa = kappa1 + c2*kappa2
                    e4 = eom[beta] + eom[pi] - eom[a+nocc] - eom[b+nocc]
                    A0S2a += ao2mok[a,beta,b,pi]*kapa/e4
    for a in range(nvir):
        for pi in range(nocc):
            if alpha == beta :
                for delta in range(nocc):
                    kappa1 = kappa(1,pi,delta,m,a,eom,ao2mok,nocc)
                    kappa2 = kappa(2,pi,delta,m,a,eom,ao2mok,nocc)
                    kapa = kappa1 + c2*kappa2
                    e4 = eom[delta] + eom[pi] - eom[a+nocc] - eom[n+nocc]
                    A0S2b += ao2mok[a,delta,n,pi]*kapa/e4
    #else:
    #    return A0S2
    if i == 1 :
        A0S2 = e2*(A0S2a + A0S2b)
    else:
        A0S2 = c1*(A0S2a + A0S2b)
    return A0S2
########################################################
def a2b2(m,n,beta,alpha,ao2moj,ao2mok,ao2mooc,ao2movi,\
    eom,nocc,nvir,MS):

    tpc = 0
    if tpc == 0:
    #!!!Empiezo contrucción de A(2)_{m\alpha,n\beta}
    #md: 0.5 por la delta dirac
        md = -0.5
        if   alpha == beta and m == n :
    #Construcción del primer término de A2
            A2a1 = A21(beta,alpha,ao2mok,eom,nocc,nvir)
            A2a1 = md*A2a1
    #Construcción del segundo término de A2
            A2b1 = A22(m,n,ao2mok,eom,nocc,nvir)
            A2b1 = md*A2b1
    #Sumo ambos términos
            A2l = A2a1 + A2b1
        elif alpha == beta and m != n :
            A2l = A22(m,n,ao2mok,eom,nocc,nvir)
            A2l = md*A2l
        elif m == n and alpha != beta :
            A2l = A21(beta,alpha,ao2mok,eom,nocc,nvir)
            A2l = md*A2l
        else:
            A2l = 0.0

    #!!!Empiezo contrucción de B(2)_{m\alpha,n\beta}
    #Construcción de los primeros dos términos
        #print("B2a1")
        B2a1 = B21(alpha,beta,m,n,ao2mok,eom,nocc,nvir)
    #Construcción del tercero y cuarto
        B2b1 = B22(alpha,beta,m,n,ao2mok,ao2moj,eom,nocc,nvir,MS)
    #Construcción del quinto término
        B2c1 = B23(alpha,beta,m,n,ao2mok,ao2mooc,eom,nocc,MS)
    #Construcción del sexto término
        B2d1 = B24(alpha,beta,m,n,ao2mok,ao2movi,eom,nocc,nvir,MS)
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
            A2a2 = A21(alpha,beta,ao2mok,eom,nocc,nvir)
            A2a2 = md*A2a2
    #Construcción del segundo término de A2
            A2b2 = A22(n,m,ao2mok,eom,nocc,nvir)
            A2b2 = md*A2b2
    #Sumo ambos términos
            A2r = A2a2 + A2b2
        elif alpha == beta and m != n :
            A2r = A22(n,m,ao2mok,eom,nocc,nvir)
            A2r = md*A2r
        elif m == n and alpha != beta :
            A2r = A21(alpha,beta,ao2mok,eom,nocc,nvir)
            A2r = md*A2r
        else:
            A2r = 0.0
        S0S2, S2 = SO(alpha,beta,m,n,nocc,nvir,eom,ao2mok,MS)
        #A0S2a = GA0S2(1,alpha,beta,m,n,ao2mok,eom,nocc,nvir)

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
        S0S2, S2 = SO(alpha,beta,m,n,nocc,nvir,eom,ao2mok,MS)
        A0S2a = (eom[m+nocc]-eom[alpha])*S2
        A0S2b = (eom[n+nocc]-eom[beta])*S2
        A0S2  = 0.5*(A0S2a+A0S2b)
        A2a = A2_T(alpha,beta,m,n,ao2mok,eom,nocc,nvir)
        A2b = A2_T(beta,alpha,n,m,ao2mok,eom,nocc,nvir)
        #if abs(A2a - A2b) > 0.0001 :
        #    print("A diferencia ",A2a, A2b)
        A2  = 0.5*(A2a + A2b)
        #B2 es simetrica al itercambio de índices
        B2 = B2_T(alpha,beta,m,n,ao2moj,ao2mok,ao2mooc,ao2movi,eom,nocc,nvir)
        A2B2 = -A2-B2#-A0S2
        return A2B2, S0S2, S2
#########################################################
#!!!Empiezo contrucción de B(2)_{n\beta,m\alpha}
#Nota: se tiene encuenta el menos porque se toma la negativa del propagador
#Construcción de los primeros dos términos
    #print("B2a2")
#    B2a2 = B21(beta,alpha,n,m,ao2mok,eom,nocc,nvir)
    #print("B2a ",B2a1,B2a2)
#Construcción del tercero y cuarto
#    B2b2 = B22(beta,alpha,n,m,ao2mok,ao2moj,eom,nocc,nvir,MS)
    #print("B22 ",B2b1,B2b2)
#Construcción del quinto término
#    B2c2 = B23(beta,alpha,n,m,ao2mok,ao2mooc,eom,nocc,MS)
   #print("B23 ",B2c1,B2c2)
#Construcción del sexto término
#    B2d2 = B24(beta,alpha,n,m,ao2mok,ao2movi,eom,nocc,nvir,MS)
   #print("B24 ",B2d1,B2d2)
#    if MS == 0 :
#        B2r = B2a2 + B2b2 - B2c2 - B2d2
#    else:
#        B2r = B2a2 - B2b2 + B2c2 + B2d2
#    B2r = 0.5*B2r
##########################################################
def A2_T(alpha,beta,m,n,ao2mok,eom,nocc,nvir):
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
def B2_T(alpha,beta,m,n,ao2moj,ao2mok,ao2mooc,ao2movi,eom,nocc,nvir):
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

########################################################
def triplete_c1d0(p,q,gamma,delta,m,n,beta,alpha,ao2mo1o3v,ao2mo3o1v,eom,nocc,nvir):

#!Empiezo contrucción de C^+(1)(-D(0))^-1C(1)
#Construcción de C(1)
    c1 = 1.0/np.sqrt(2.0)
    C  = 0.0
    CT = 0.0
#parte derecha
    dda1 = 0.0
    dda2 = 0.0
    dbd1 = 0.0
    dbd2 = 0.0
    dnq1 = 0.0
    dnq2 = 0.0
    dmq1 = 0.0
    dmq2 = 0.0
#parte izquierda
    dag1 = 0.0
    dag2 = 0.0
    dbg1 = 0.0
    dbg2 = 0.0
    dnp1 = 0.0
    dnp2 = 0.0
    dmp1 = 0.0
    dmp2 = 0.0
    #if alpha != beta  or m != n :
    if alpha != delta : #con el Li el int[m beta n p] = int[n p m beta]
        dda1 = ao2mo1o3v[n,q,m,beta]
        dda2 = ao2mo1o3v[m,q,n,beta]
    if alpha != gamma :
        dag1 = ao2mo1o3v[n,p,m,beta]
        dag2 = ao2mo1o3v[m,p,n,beta]

    if beta != delta :
        dbd1 = ao2mo1o3v[m,q,n,alpha]
        dbd2 = ao2mo1o3v[n,q,m,alpha]
    if beta != gamma :
        dbg1 = ao2mo1o3v[m,p,n,alpha]
        dbg2 = ao2mo1o3v[n,p,m,alpha]

    if n != q :
        dnq1 = ao2mo3o1v[m,alpha,delta,beta]
        dnq2 = ao2mo3o1v[m,beta,delta,alpha]
    if n != p :
        dnp1 = ao2mo3o1v[m,alpha,gamma,beta]
        dnp2 = ao2mo3o1v[m,beta,gamma,alpha]

    if m != q :
        dmq1 = ao2mo3o1v[n,beta,delta,alpha]
        dmq2 = ao2mo3o1v[n,alpha,delta,beta]
    if m != p :
        dmp1 = ao2mo3o1v[n,beta,gamma,alpha]
        dmp2 = ao2mo3o1v[n,alpha,gamma,beta]
##? 1)   (T^+(1)_n beta,m alpha|V|Q^+(1)_p gamma)
#parte derecha (T^+(1)|V|Q(1))
    T1Q1r = 0.0
    if alpha != beta  and n != m :
        T1Q1r = dda1 - dda2
        T1Q1r = T1Q1r + (dbd1 - dbd2)
        T1Q1r = T1Q1r + (dnq1 - dnq2)
        T1Q1r = T1Q1r + (dmq1 - dmq2)
#parte izquierda (T^+(1)|V|Q(1))
    T1Q1l = 0.0
    if alpha != beta  and m != n :
        T1Q1l = dag1 - dag2
        T1Q1l = T1Q1l + (dbg1 - dbg2)
        T1Q1l = T1Q1l + (dnp1 - dnp2)
        T1Q1l = T1Q1l + (dmp1 - dmp2)
##? 1)   (T^+(2)_n beta,m alpha|V|Q^+(1)_p gamma)
#parte derecha (T^+(2)|V|Q(1))
    T2Q1r = 0.0
    if alpha != beta :
        c2 = 1.0
#        if n == m :
#            c2 = 1.0/np.sqrt(2.0)
        T2Q1r = dda1 + dda2
        T2Q1r = T2Q1r - (dbd1 + dbd2)
        T2Q1r = T2Q1r + (dnq1 - dnq2)
        T2Q1r = T2Q1r + (-dmq1 + dmq2)
        T2Q1r = c1*c2*T2Q1r
#parte izquierda (T^+(2)|V|Q(1))
    T2Q1l = 0.0
    if alpha != beta :
        c2 = 1.0
    #    if n == m :
    #        c2 = 1.0/np.sqrt(2.0)
        T2Q1l = dag1 + dag2
        T2Q1l = T2Q1l - (dbg1 + dbg2)
        T2Q1l = T2Q1l + (dnp1 - dnp2)
        T2Q1l = T2Q1l + (-dmp1 + dmp2)
        T2Q1l = c1*c2*T2Q1l
##? 1)   (T^+(3)_n beta,m alpha|V|Q^+(1)_p gamma)
#parte derecha (T^+(3)|V|Q(1))
    T3Q1r = 0.0
    if m != n :
        c2 = 1.0
    #    if alpha == beta :
    #        c2 = 1.0/np.sqrt(2.0)
        T3Q1r = -dda1 + dda2
        T3Q1r = T3Q1r + (dbd1 - dbd2)
        T3Q1r = T3Q1r + (dnq1 + dnq2)
        T3Q1r = T3Q1r - (dmq1 + dmq2)
        T3Q1r = -c1*c2*T3Q1r
#parte izquierda (T^+(3)|V|Q(1))
    T3Q1l = 0.0
    if n != m :
        c2 = 1.0
    #    if alpha == beta :
    #        c2 = 1.0/np.sqrt(2.0)
        T3Q1l = -dag1 + dag2
        T3Q1l = T3Q1l + (dbg1 - dbg2)
        T3Q1l = T3Q1l + (dnp1 + dnp2)
        T3Q1l = T3Q1l - (dmp1 + dmp2)
        T3Q1l = -c1*c2*T3Q1l

    C  = (T1Q1r + T2Q1r + T3Q1r)/4.0
    CT  = (T1Q1l + T2Q1l + T3Q1l)/4.0

    #if alpha == 0 and beta == 0 and m == 0 and n == 1:
    #    print(T1Q1r, T2Q1r, T3Q1r, C, CT)
    #print("C^+(D)^-1C  ",A2B2,CDC)
    return T1Q1l/4.0, T1Q1r/4.0, T2Q1l/4.0, T2Q1r/4.0, T3Q1l/4.0, T3Q1r/4.0