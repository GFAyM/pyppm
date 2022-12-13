import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.ppe_3 import M_matrix

from src.help_functions import extra_functions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



occ1 = [22, 23, 23, 22, 22, 22, 22, 22, 22, 22, 23, 22, 23, 22, 23, 22, 22, 22]
occ2 = [23, 22, 22, 23, 23, 23, 23, 23, 23, 23, 22, 23, 22, 23, 22, 23, 23, 23]

v1_1 = [63, 93, 93, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63]
v1_2 = [93, 64, 64, 64, 53, 92, 64, 64, 64, 64, 64, 64, 64, 64, 53, 64, 53, 64]

v2_1 = [34, 35, 35, 35, 36, 36, 36, 36, 36, 35, 36, 36, 36, 36, 36, 35, 35, 36]
v2_2 = [35, 34, 34, 34, 35, 35, 35, 35, 35, 36, 35, 35, 35, 35, 35, 34, 34, 35]

ang = 10

mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H2F4_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

inv_prop = M_matrix(mol=mol, mo_coeff=mo_coeff, mo_occ=mo_occ,
            occ = [  occ1[ang], occ2[ang]],
            vir = [ v1_1[ang],
                    v1_2[ang]])   
#eig = inv_prop.rho

print(inv_prop.entropy_iajb_2)

