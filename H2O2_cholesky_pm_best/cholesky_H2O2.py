from pyscf import gto, scf
from pyscf.gto import Mole
from pyscf.lo.edmiston import ER
from pyscf.lo.ibo import PM
from pyscf.scf import RHF
from pyscf.lo import Boys
from pyscf.lo import PipekMezey
from pyscf.lo import EdmistonRuedenberg
from pyscf.lo.cholesky import cholesky_mos
from pyscf.tools import molden, mo_mapping
import numpy as np

def cholesky(ang):
    mol = gto.M(atom= str('''
        O1   1
        O2   1 1.45643942
        H3   2 0.97055295  1 99.79601616
        H4   1 0.97055295  2 99.79601616  3 {}
        ''').format(ang) , basis = '631g**' )
    mf = scf.RHF(mol).run()

    # determine the number of occupied orbitals
    nocc = np.count_nonzero(mf.mo_occ > 0)
    # localize the occupied orbitals separately
    #lmo_occ = cholesky_mos(mf.mo_coeff[:, :nocc])
    # localize the virtual orbitals separately
    lmo_virt = cholesky_mos(mf.mo_coeff[:, nocc:])
    # merge the MO coefficients in one matrix
    #lmo_merged = np.hstack((lmo_occ, lmo_virt))

    ##lmo_merged as mo_init

    #Boys.conv_tol = 1e-8
    #Boys.max_stepsize=0.01
    #lmo_occ = Boys(mol).kernel(mf.mo_coeff[:, :nocc])
    #lmo_virt = Boys(mol).kernel(mf.mo_coeff[:, nocc:])
    PipekMezey.conv_tol = 1e-6
    PipekMezey.max_stepsize = 0.015
    PipekMezey.max_iters = 70
    lmo_occ = PipekMezey(mol).kernel(mf.mo_coeff[:, :nocc])
    #lmo_occ = PipekMezey(mol).kernel(lmo_occ)
    PipekMezey.conv_tol = 1e-7
    PipekMezey.max_stepsize = 0.0035
    PipekMezey.max_iters = 50    
    lmo_virt = PipekMezey(mol).kernel(lmo_virt)


    lmo_merged = np.hstack((lmo_occ, lmo_virt))
    return mol, mf.mo_occ, lmo_merged


def orbitals(mol,mo_coeff):
    comp_H1 = mo_mapping.mo_comps('H3', mol, mo_coeff)
    vector_H1 = np.array([])
    for i,c in enumerate(comp_H1):
        if 0.3 < c < 1:
            vector_H1=np.append(vector_H1,(c,i))
    comp_H2 = mo_mapping.mo_comps('H4', mol, mo_coeff)
    vector_H2 = np.array([])
    for i,c in enumerate(comp_H2):
        if 0.3 < c < 1:
            vector_H2=np.append(vector_H2,(c,i))
    return vector_H1, vector_H2



for ang in range(130,140,10):
    mol, mo_occ, lmo_merged = cholesky(ang)
    filename = str('H2O2_mezcla_{}.molden').format(ang)
    print('Dumping the orbitals in file:', filename)
    molden.from_mo(mol, filename, lmo_merged, occ=mo_occ)
    H1, H2 = orbitals(mol, lmo_merged)
    print( H1, H2)
    print(len(H1), len(H2))
