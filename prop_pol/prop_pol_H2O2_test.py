import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf, gto, tdscf
from pyscf.dft import numint

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.polaritization_propagator import Prop_pol as pp


M_list = [[],[]]

ang=100
mol = gto.M(atom='''
        O1   1
        O2   1 1.45643942
        H3   2 0.97055295  1 99.79601616
        H4   1 0.97055295  2 99.79601616  3 {}
        '''.format(ang*10), basis='ccpvdz', verbose=0)

mf = scf.RHF(mol).run()

def h1_fc_pyscf(atmlst):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mol = mf.mol
    coords = mol.atom_coords()
    ao = numint.eval_ao(mol, coords)
    mo = ao.dot(mo_coeff)
    orbo = mo[:,mo_occ> 0]
    orbv = mo[:,mo_occ==0]
    fac = 8*np.pi/3 *.5  # *.5 due to s = 1/2 * pauli-matrix
    h1 = []
    for ia in atmlst:
        h1.append(fac * np.einsum('p,i->pi', orbv[ia], orbo[ia]))
    return h1


def _uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic


#ppobj = pp(mf)
nuc_pair_ = [(i,j) for i in range(mol.natm) for j in range(i)]

atm1, atm2 = _uniq_atoms(nuc_pair_)
print(atm1,atm2)

amtlst2 = sorted(atm1.keys())

#h1 = h1_fc_pyscf(amtlst2)

#print(len(h1))
#print(h1[3].shape)

    #print(mol.natm)
    
    #pol_prop = ppobj.polarization_propagator(2,3)

#    pol_prop = np.sum(np.diag(np.linalg.inv(ppobj.m_matrix_triplet)))
 #   M_list[0].append(ang*10)#, np.sum(m)])#,  "Propagador Pol"])
 #   M_list[1].append(pol_prop)


#fig = plt.figure(figsize=(8, 8))
#x = M_list[0]
#y = M_list[1]
#plt.plot(x,y)
#plt.title("J(H-H) coupling without any constant, in the canonical MO basis")

#plt.savefig('J_H2O2.png')
#plt.show()

