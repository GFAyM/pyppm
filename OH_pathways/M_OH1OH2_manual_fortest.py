import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa_full
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import ao2mo
import itertools


M_list = []

for ang in range(10,11,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
    '''.format(10*ang)


    occidx_OH2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H4', .3, .5)

    occidx_OH1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
            'H3', .3, .5)

    viridx_OH2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'H4', .7, 1)

    viridx_OH1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'H3', .7, 1)


    viridx_OH2_2s = viridx_OH2[0]
    viridx_OH2_2pz = viridx_OH2[1]
    viridx_OH2_2px = viridx_OH2[3]


    viridx_OH1_2s = viridx_OH1[0]
    viridx_OH1_2pz = viridx_OH1[1]
    viridx_OH1_2px = viridx_OH1[3]

    

    i = mo_coeff_loc[:, [occidx_OH1]]
    a = mo_coeff_loc[:, [viridx_OH1_2s,viridx_OH1_2pz,viridx_OH1_2px]]
    j = mo_coeff_loc[:, [occidx_OH2]]
    b = mo_coeff_loc[:, [viridx_OH2_2s,viridx_OH2_2pz,viridx_OH2_2px]]


    m_12 = -ao2mo.general(mol_loc, [a,b,j,i], compact=False)
    m_12 = m_12.reshape(len(a[0]),len(a[0]))
    m_12 -= ao2mo.general(mol_loc, [i,b,j,a], compact=False)

    #print(m_12)
    m_21 = -ao2mo.general(mol_loc, [b,a,i,j], compact=False)
    m_21 = m_21.reshape(len(a[0]),len(a[0]))
    m_21 -= ao2mo.general(mol_loc, [j,a,i,b], compact=False)
    #print(m_21)
    m_11 = np.zeros((len(a[0]),len(a[0])))
    m_22 = np.zeros((len(a[0]),len(a[0])))
    
    block1 = np.concatenate((m_11, m_12), axis=1)
    block2 = np.concatenate((m_21, m_22), axis=1)
    m = np.concatenate((block1, block2), axis=0)
    print(m)
    eig = np.linalg.eigvals(m)
    print(eig)
    print(eig.sum())
    
    
