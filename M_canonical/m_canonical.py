import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf, gto, tdscf

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.polaritization_propagator import Prop_pol as pp


M_list = []
M_diag_list = []
inv_M_diag_list = []

for ang in range(0,19,1):
    mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(10*ang)
    
    mol = gto.M(atom=mol_H2O2, basis='6-31G**')   
    mf = scf.RHF(mol).run()
    m = pp(mf).m_matrix_triplet
    #print(m, m.shape)


    M_list.append([ang*10, np.sum(m)])#,  "Propagador Pol"])
    M_diag_list.append([ang*10, np.sum(np.diag(m))])#,  "Propagador Pol"])
    inv_M_diag_list.append([ang*10, np.sum(np.diag(np.linalg.inv(m)))])#,  "Propagador Pol"])


df = pd.DataFrame(M_list, columns=['angulo', 'Propagator'])#,   'Polarization Propagator'])

df.plot(x='angulo', y='Propagator')

plt.savefig('sum_canonical_M.png')

df = pd.DataFrame(M_diag_list, columns=['angulo', 'Propagator'])#,   'Polarization Propagator'])

df.plot(x='angulo', y='Propagator')

plt.savefig('sum_diag_canonical_M.png')

df = pd.DataFrame(inv_M_diag_list, columns=['angulo', 'Propagator'])#,   'Polarization Propagator'])

df.plot(x='angulo', y='Propagator')

plt.savefig('sum_inv_diag_canonical_M.png')
