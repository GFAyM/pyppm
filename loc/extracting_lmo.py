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
for ang in range(180,190,10):
    a = extra_functions(molden_file=f"C2H2F4_{ang}_ccpvdz_Cholesky_PM.molden").mo_hibridization(
        'H3', .1, .21)
    b = extra_functions(molden_file=f"C2H2F4_{ang}_ccpvdz_Cholesky_PM.molden").mo_hibridization(
        'H7', .1, .21)
    #occ1.append(a)
    #occ2.append(b)
    print(a)
    print(b)

#print(f"v6_1 = {occ1}")
#print(f"v6_2 = {occ2}")