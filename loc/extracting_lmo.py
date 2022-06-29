import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


occ1 = []
occ2 = []
for ang in range(0,19,1):
    a = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").mo_hibridization_for_list_several(
        'H3 1s', .3, 1)
    #a = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('O1 3s', .4, 1)
    
    b = extra_functions(molden_file=f"C2H6_{ang*10}_ccpvdz_Cholesky_PM.molden").mo_hibridization_for_list_several(
        'H7 1s',.3, 1)
    #b = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_fixed('O1 3dx', 26, .1, 1)
    #print(ang*10,a,b)
    occ1.append(a[0])
    occ2.append(b[0])

print(occ1)
print(occ2)
#    print(ang*10,a, b)
