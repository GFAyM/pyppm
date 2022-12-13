#!/usr/bin/env python

'''
Integral transformation to compute 2-electron integrals for no-pair
Dirac-Coulomb Hamiltonian. The molecular orbitals are based on RKB basis.
'''

import h5py
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf import ao2mo

mol = gto.M(
    atom = '''
    O   0.   0.       0.
    H   0.   -0.757   0.587
    H   0.   0.757    0.587
    ''',
    basis = 'ccpvdz',
)

mf = scf.DHF(mol).run()

def no_pair_ovov(mol, mo_coeff):
    '''
    2-electron integrals ( o v | o v ) for no-pair Hamiltonian under RKB basis
    '''
    c = lib.param.LIGHT_SPEED
    n4c, nmo = mo_coeff.shape
    n2c = n4c // 2
    nocc = mol.nelectron
    nvir = nmo - nocc
    mo_pos_l = mo_coeff[:n2c,:]
    mo_pos_s = mo_coeff[n2c:,:] * (.5/c)
    Lo = mo_pos_l[:,:nocc]
    So = mo_pos_s[:,:nocc]
    Lv = mo_pos_l[:,nocc:]
    Sv = mo_pos_s[:,nocc:]


    eri_a1 = ao2mo.general(mol,  (Lv, Lo, Lo, Lv), intor='int2e_spinor')
    eri_a1 += ao2mo.general(mol, (Sv, So, So, Sv), intor='int2e_spsp1spsp2_spinor')
    eri_a1 += ao2mo.general(mol, (Sv, So, Lo, Lv), intor='int2e_spsp1_spinor'     )
    eri_a1 += ao2mo.general(mol, (Lv, Lo, So, Sv), intor='int2e_spsp2_spinor'     )
    eri_a1 = eri_a1.reshape(nocc,nvir,nocc,nvir)

    eri_a2 = ao2mo.general(mol,  (Lv, Lv, Lo, Lo), intor='int2e_spinor')
    eri_a2 += ao2mo.general(mol, (Sv, Sv, So, So), intor='int2e_spsp1spsp2_spinor')
    eri_a2 += ao2mo.general(mol, (Sv, Sv, Lo, Lo), intor='int2e_spsp1_spinor'     )
    eri_a2 += ao2mo.general(mol, (Lv, Lv, So, So), intor='int2e_spsp2_spinor'     )
    eri_a2 = eri_a2.reshape(nocc,nvir,nocc,nvir)
    
    return eri_a1, eri_a2

a1,a2 = no_pair_ovov(mol, mf.mo_coeff)

print(a1.shape,a2.shape)

