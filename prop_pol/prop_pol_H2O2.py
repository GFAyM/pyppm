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

for ang in range(1,18):
    mol = gto.M(atom='''
            O1   1
            O2   1 1.45643942
            H3   2 0.97055295  1 99.79601616
            H4   1 0.97055295  2 99.79601616  3 {}
            '''.format(ang*10), basis='ccpvdz')

    mf = scf.RHF(mol).run()

    ppobj = pp(mf)
#    pol_prop = ppobj.polarization_propagator(0,3)
    pol_prop = np.sum(np.diag(np.linalg.inv(ppobj.m_matrix_triplet)))
    M_list[0].append(ang*10)#, np.sum(m)])#,  "Propagador Pol"])
    M_list[1].append(pol_prop)
    fig = plt.figure(figsize=(8, 8))


x = M_list[0]
y = M_list[1]
plt.plot(x,y)
plt.title("J(H-H) coupling without any constant, in the canonical MO basis")

plt.savefig('J_H2O2.png')

