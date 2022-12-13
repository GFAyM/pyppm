#!/usr/bin/python3

import numpy as np
from pyscf.dft import numint
# from scipy.linalg import sqrtm #root to matriz
import scipy.linalg as la
import os
import sys  # para control de memoría
import time
from resource import getrusage, RUSAGE_SELF

############

############
# import soppa
import soppa_triplete as soppa
import pyscf 
from pyscf import gto, scf, ao2mo




mol = gto.M(atom='''
F     0.0000000000    0.0000000000     0.1319629808
H     0.0000000000    0.0000000000    -1.6902522555
''', basis='6-311g', unit='angstrom')

#mol, ctr_coeff = mol.decontract_basis()
# Hartree-Fock
#mf = scf.RHF(mol)
#mf.kernel()
# MP2
mf = mol.RHF().run()
mf.MP2().run()


iprint = 1
mo_coeff = mf.mo_coeff
mo_energy = mf.mo_energy
mo_occ = mf.mo_occ
##Lectura de Coeficientes y Energías de los OMs desde el SIRIFC
occidx = np.where(mo_occ>0)[0]
viridx = np.where(mo_occ==0)[0]
orbv = mo_coeff[:,viridx]
orbo = mo_coeff[:,occidx]

nvir = orbv.shape[1]
nocc = orbo.shape[1]

print("nocc, nvir ",nocc,nvir)

mo = np.hstack((orbo,orbv))
nprim = nocc+nvir

C = mo_coeff
EOM = mo_energy

###################################################################
# Lectura de las integrales en base atómica de los pertubadores
#          <0|A|n>       <n|B|0>

# Tipo de propagador principal el de RPA o SOPPA
tpp = 'SOPPA'
#!#################################################################

nrot = nocc * nvir  
multiplicidad = 1

# Número de propiedades
nproperties = 2

#################################################################
# I2C AO -> MO
eri_mo = ao2mo.general(mol, [mo,mo,mo,mo], compact=False)
eri_mo = eri_mo.reshape(nprim,nprim,nprim,nprim)
ao2moj =  eri_mo[nocc:,nocc:,:nocc,:nocc]
ao2mok = eri_mo[nocc:,:nocc,nocc:,:nocc]

#print("J ",ao2moj.shape)
#print("\nK",ao2mok.shape)

ao2mooc = eri_mo[:nocc,:nocc,:nocc,:nocc]
ao2movi = eri_mo[nocc:,nocc:,nocc:,nocc:]

ao2mo1o3v = eri_mo[nocc:,nocc:,nocc:,:nocc]
ao2mo3o1v = eri_mo[nocc:,:nocc,:nocc,:nocc]

mb_oc = 0.0
mb_vi = 0.0

def header(nc):
    for i in range(nc):
        print("=", end=" ")
    print()


#####################################################
propertyi = np.zeros((nproperties, nprim, nprim), dtype=float)

def pert_fc(atmlst):
    coords = mol.atom_coords()
    ao = numint.eval_ao(mol, coords)
    mo = ao.dot(mo_coeff)
    orbo = mo[:,:]
    orbv = mo[:,:]
    fac = 8*np.pi/3 
    h1 = []
    for ia in atmlst:
        h1.append(fac * np.einsum('p,i->pi',orbo[ia], orbv[ia] ))
    return h1

propertyimo = np.asarray(pert_fc([0,1]))

###################################################################
#  Calculos RPA haciendo uso de la relación de Aucar y Cavasotto
# 					implementado por Andy Zapata
#            PP = E_{ai} \sum_i^p ^mN_{abji} E_{bj}
####################################################################
header(27)
if tpp == "RPA":
    print("                   Calculo    RPA  ")
else:
    print("                   Calculo    SOPPA  ")
print("               Relación de Aucar y Cavasotto")
print("Implementador: Andy Zapata 2020")
print("Nota: La matriz de Fock es diagonal en canónicos")
header(27)

timei = time.time()  # segundos

k = 0
irow = 0
icol = 0
Einv = np.zeros((nvir * nocc, nvir * nocc), dtype=float)
MN = np.zeros((nvir * nocc, nvir * nocc), dtype=float)
gpvph = np.zeros((nproperties, nocc * nvir))

if tpp != "RPA":
    total4 = nocc * nocc * nvir * nvir
    # MS0S2  = np.zeros((nvir*nocc,nvir*nocc),dtype=float)
    S2 = np.zeros((nvir * nocc, nvir * nocc), dtype=float)
    # MN2   = np.zeros((nvir*nocc,nvir*nocc),dtype=float)
indexrow = np.zeros((nvir, nvir, nocc, nocc), dtype=int)
indexcol = np.zeros((nvir, nvir, nocc, nocc), dtype=int)

typec = 0
id = 0
icol1 = 0
tpcinv = 1
for i in range(nocc):
    for a in range(nvir):
        s = a + nocc
        Einv[k, k] = 1.0e0 / (EOM[s] - EOM[i])
        k += 1
        # Armando vector gradient property
        if tpp == "SOPPA":
            for j in range(nproperties):

                PQA2 = soppa.PQ2a(
                    i,
                    a,
                    nvir,
                    propertyimo[j, :, :],
                    nocc,
                    EOM,
                    ao2mok,
                    ao2mo3o1v,
                    ao2mo1o3v,
                    multiplicidad,
                )
                PQB2 = soppa.PQ2b(i, a, nvir, propertyimo[j, :, :], nocc, EOM, ao2mok)
                if multiplicidad == 1:
                    gpvph[j, a + i * nvir] = 2.0 * propertyimo[j, i, s] + (PQA2 - PQB2)
                else:
                    gpvph[j, a + i * nvir] = -2.0 * propertyimo[j, i, s] - (PQA2 + PQB2)
                    if abs(gpvph[j, a + i * nvir]) > 1.0e-16:
                        gpvph[j, a + i * nvir] = -gpvph[j, a + i * nvir]
                    else:
                        gpvph[j, a + i * nvir] = 0.0
                # if j == 0 :
                # 	print(gpvph[j,a+i*nvir])
        else:
            for j in range(nproperties):
                gpvph[j, a + i * nvir] = 2.0 * propertyimo[j, i, s]
                
        for j in range(nocc):
            for b in range(nvir):
                t = b + nocc
                #!MN
                if multiplicidad == 1:  # <ab|ji> + <aj|bi>
                    EMO = 0.0
                    if i == j and a == b and tpcinv == 1:
                        EMO = EOM[s] - EOM[i]
                    MN[irow, icol] = -EMO + ao2moj[a, b, j, i] + ao2mok[a, j, b, i]
                    # print(EMO-ao2moj[a,b,j,i]-ao2mok[a,j,b,i]) #,EMO-ao2moj[a,b,j,i],-ao2mok[a,j,b,i])
                    temp = 0.0
                    if tpp == "SOPPA":
                        temp, s0s2, vS2 = soppa.a2b2(
                            a,
                            b,
                            j,
                            i,
                            ao2moj,
                            ao2mok,
                            ao2mooc,
                            ao2movi,
                            EOM,
                            nocc,
                            nvir,
                            multiplicidad,
                        )
                        MN[irow, icol] = MN[irow, icol] - temp
                        # MS0S2[irow,icol] = s0s2
                        S2[irow, icol] = vS2
                        # tempo = soppa.SO(i,j,a,b,nocc,nvir,EOM,ao2mok)
                        # MN2[irow,icol] = soppa_triplete.triplete_a2b2\
                        # 		(a,b,j,i,ao2moj,ao2mok,ao2mooc,ao2movi,\
                        # 		EOM,nocc,nvir,multiplicidad)
                elif multiplicidad == 0:
                    EMO = 0.0
                    if i == j and a == b and tpcinv == 1:
                        EMO = EOM[s] - EOM[i]
                    # MN[irow,icol] =\
                    # -EMO + ao2mok[a,j,b,i] - ao2moj[a,b,j,i]\
                    # 	- 2.0E+0*ao2mok[a,j,b,i] + 2.0E+0*ao2moj[a,b,i,j]
                    MN[irow, icol] = -EMO - ao2mok[a, j, b, i] + ao2moj[a, b, j, i]  # \
                    # -2.0*ao2mok[a,i,b,j] + 2.0*ao2mok[a,i,b,j]
                    temp = 0.0
                    if tpp == "SOPPA":
                        temp, s0s2, vS2 = soppa.a2b2(
                            a,
                            b,
                            j,
                            i,
                            ao2moj,
                            ao2mok,
                            ao2mooc,
                            ao2movi,
                            EOM,
                            nocc,
                            nvir,
                            multiplicidad,
                        )
                        MN[irow, icol] = MN[irow, icol] + temp
                        # S2[irow,icol] = vS2
                else:
                    print("La multiplicidad debe ser 0 o 1")
                    exit()

                indexrow[a, b, j, i] = irow
                indexcol[a, b, j, i] = icol
                icol = icol + 1
                if icol > nocc * nvir - 1:
                    irow += 1
                if icol > nocc * nvir - 1:
                    icol = 0
#for x in gpvph[0,:]:
#    print(x)                
timepp = time.time() - timei  # segundos

del ao2moj, ao2mok

if tpp == "SOPPA":
    A2 = np.zeros((nvir * nocc, nvir * nocc), dtype=float)
    # Multiplicación de (A(1)+B(1)) con A(O)
    A2 = np.dot(np.linalg.inv(Einv), S2)
    A2 += np.dot(S2, np.linalg.inv(Einv))
    A2 = 0.5 * A2
    # print(A2)
    if multiplicidad == 1:
        MN = MN  # - A2
    else:
        MN = MN  # + A2
    del ao2mooc, ao2movi, ao2mo3o1v, ao2mo1o3v

EinvSMNEinvi = np.zeros((nvir * nocc, nvir * nocc), dtype=float)
EinvSMNEinvi = np.linalg.inv(MN)
del Einv, MN
# rpaac1 con 0.5*(A(0)S(2) + S(2)A(0)) : 2.558849
# rpaac1 sin 0.5*(A(0)S(2) + S(2)A(0)) : 2.761606
# rpaac1 sin B(2)      : 2.365725
# rpaac1 sin A(2)      : 2.308266
# rpaac1 sin A(2)+B(2) : 2.149489
# DALTON PSO 1; PSO 4 = 2.124190553158
# rpaac1 sin A(2)+B(2) : 5.334945
# DALTON PSO 1; PSO 4 = 5.348586454838


# print(EinvSMNEinvi[:,:],"\n")
# print(0.5*np.dot(EinvSMNEinvi,gpvph[5,:]))
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!    Empieza la construcción de los caminos
#!!
#!!    virtual -> ocupado -> virtual -> ocupado
#!!
#!!    Path_ia,jb = A_ia PP_ia,jb B_bj
#!!
#!!    PP_ia,jb = (E_ia^-1 Sum_{i=1}^p [^mspinN_ia,jb E_jb^-1]^i )_ia,jb
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# print(propertyimo[0,:,:]) #,"\n\n",propertyimo[1,:,:],"\n")

# Los labels de las propiedades en una misma dirección deben estar continuas
# en input
# propertya x
# propertyb x

for k in range(int(nproperties / 2)):
    itp = k * 2
    iset = 0
    ipath = 0
    vpathT = 0.0e0
    tpzoa = 0.0e0
    appb = np.zeros((nocc * nvir * nocc * nvir), dtype=float)
    pzoa = np.zeros((nocc * nvir * nocc * nvir), dtype=float)
    vv = 0.0

    for i in range(nocc):
        for j in range(nocc):
            #!
            iset = iset + 1
            count = 1
            spath = 0.0e0
            tipzoa = 0.0e0
            #!
            for a in range(nvir):
                s = a + nocc
                for b in range(nvir):
                    t = b + nocc
                    #!<i|FC(C)|s><t|FC(H)|j>
                    #!2 D2 por ias o tbj, es necesario para reproducir valores del DALTON
                    appb[ipath] = 0.0e0
                    appb[ipath] = (
                        -1.0 * gpvph[itp, a + i * nvir] * gpvph[itp + 1, b + j * nvir]
                    )
                    #!
                    irow = indexrow[a, b, j, i]
                    icol = indexcol[a, b, j, i]
                    #!Pure zero order approximation
                    if i == j and s == t:
                        pzoa[ipath] = appb[ipath] / (EOM[s] - EOM[i])
                        tpzoa = tpzoa + pzoa[ipath]
                        tipzoa = tipzoa + pzoa[ipath]
                    #!<i|FC(C)|s><t|FC(H)|j> ESTNEi
                    pp = 0.0e0
                    pp = EinvSMNEinvi[irow, icol]
                    appb[ipath] = appb[ipath] * pp
                    #!Sum path
                    vpath = 0.0e0
                    vpath = vpath + appb[ipath]
                    vpathT = vpathT + vpath
                    # print("vpathT ",vpathT)
                    #!path > 0.1
                    if iprint > 4:
                        print("   #    t    i    s    j  ")
                    if iprint > 4 and abs(vpath) > 0.1:
                        print(t, i, s, j, vpath)
                    #!
                    spath = spath + vpath
                    count = count + 1
                    #!
                    if iprint > 2 and ipath / (nvir * nvir) == iset:
                        print("Total path", spath)
                        print("Total path PZOA", tipzoa)
                        print()
                    #!
                    ipath = ipath + 1

    print()
    print("************************************")
    label1 = (
        "-<<"
        + 'FC(0)'
        + ";"
        + 'FC(1)'
        + ">>"
    )
    print(label1)
    if tpp == "RPA":
        print(f"RPA         {vpathT:.6f}")
    else:
        print(f"SOPPA       {vpathT:.6f}")
    print(f"PZOA        {tpzoa:.6f}")
    dif = tpzoa - vpathT
    print(f"Cont. I2C   {dif:.6f}")
    print("***********************************")

    
# Tamaño en bites de los objetos en bites
# 1 bytes es 9.5367431640625×10-7 MB

# print("\n Memoría de los obtejos en MB: RPAAC \n")
# objects = [
# MOITC,EOM,propertyimo,thrd,iprint,nocc,nvir,nip,nprim,nproperties
# ]

# namev=[
# 	"MOITC","EOM","properties","thrd","iprint","nocc","nvir","nip","nprim","nproperties"
# ]

# for obj in objects:
#    print('{:>10} : {}'.format(type(obj).__name__,

#######################################

# rpaene.rpaac(MOITCJ,MOITCK,EOM,propertyimo,thrd,iprint,nip,nocc,nvir,nprim,nproperties)
# rpaene.rpaac(MOI2C,EOM,propertyimo,thrd,iprint,nip,nocc,nvir,nprim,nproperties)
