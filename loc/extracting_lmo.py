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
for ang in range(100,110,10):
    a = extra_functions(molden_file=f"C2H2F2_{ang}_ccpvdz_Cholesky_PM.molden").mo_hibridization(
        'C1', .1, 1)
    b = extra_functions(molden_file=f"C2H2F2_{ang}_ccpvdz_Cholesky_PM.molden").mo_hibridization(
        'C2', .1, 1)
    #occ1.append(a)
    #occ2.append(b)
    print(a)
    print(b)

#print(occ1)
#print(occ2)