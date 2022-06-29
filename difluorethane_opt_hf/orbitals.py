import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import pandas as pd
from pyscf import scf, gto 
from src.help_functions import extra_functions

lmo1 = []
lmo2 = []
for ang in range(0,190,10):
    extra_obj = extra_functions(molden_file=f"difluorethane_cc-pvdz_{ang}_Cholesky_PM.molden")
    #idx1_cont = extra_obj.mo_hibridization_for_list('C2 2px',0.2,1)
    #idx2_cont = extra_obj.mo_hibridization_for_list('C1 2px',0.2,1)   
    #idx2_cont = extra_obj.mo_hibridization('F7 3dz',0.2,1)
    
    #print(ang, idx1_cont, idx2_cont)
    

    idx1 = extra_obj.mo_hibridization_for_list('C1 2px',0.2,1)
    #idx2 = extra_obj.mo_hibridization_for_list('F7 3dz',0.2,1)
    #print(idx1)
    lmo1.append(idx1)
    #lmo2.append(idx2)


print(lmo1)
#print(lmo2)

