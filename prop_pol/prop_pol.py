from tabnanny import verbose
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




mol = gto.M(atom='''
            H 0  0.34252627 0.34252627
            F 0  1.54719985 1.54719985 
            ''',
            basis='ccpvdz', unit='Bohr')

mf = scf.RHF(mol).run()

mo_coeff = mf.mo_coeff
mo_occ = mf.mo_occ
mol = mf.mol

coords = mol.atom_coords()
ao = numint.eval_ao(mol, coords)
ao = ao[0]
mo = ao.dot(mo_coeff)

orbo = mo[mo_occ> 0]
orbv = mo[mo_occ==0]


h1 = np.einsum('p,i->pi', orbv, orbo).ravel()
#print(h1)
print(h1.shape)
#fac = 8*numpy.pi/3 *.5  # *.5 due to s = 1/2 * pauli-matrix
#h1 = []
#for ia in atmlst:
#    h1.append(fac * numpy.einsum('p,i->pi', orbv[ia], orbo[ia]))


m = pp(mf).m_matrix_triplet

mh1 = m@h1
print(mh1.shape)

ao = numint.eval_ao(mol, coords)
ao = ao[1]
mo = ao.dot(mo_coeff)

orbo = mo[mo_occ> 0]
orbv = mo[mo_occ==0]
h2 = np.einsum('p,i->pi', orbv, orbo).ravel()


h2mh1 = h2@mh1

print(h2mh1)