import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_50.molden").extraer_coeff
full_M_obj = Cloppa(
    mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
    mo_occ_loc=mo_occ_loc)

#full_M_obj.kernel(FC=False, FCSD=False, PSO=True)
obj = full_M_obj
n_atom1=[2]
n_atom2=[3]

#j = full_M_obj.kernel_pathway_occ(n_atom1=n_atom1, occ_atom1=5, n_atom2=n_atom2, occ_atom2=3)
ssc_tot = 0
m = obj.M(triplet=True)
p = np.linalg.inv(m)
#ssc = full_M_obj.kernel_pathway_occ(princ_prop=p,n_atom1=n_atom1,occ_atom1=5,vir_atom1=36, n_atom2=n_atom2,occ_atom2=4, vir_atom2=14)     
#print(ssc)
print(' a  b    J_SD+FC      Total' '\n')
for a in range(9,38,1):
    for b in range(9,38,1):
        ssc = full_M_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
                                        princ_prop=p,
                                        n_atom1=n_atom1,occ_atom1=5,vir_atom1=a, 
                                        n_atom2=n_atom2,occ_atom2=4,vir_atom2=b)
        ssc_tot += ssc
        if abs(ssc) > 0.5:
          print(a, b, ' ',ssc[0],' ', ssc_tot[0])

print('la contribución de todos los virtuales a la contribución J_ia,jb, siendo i,j los ocupados O-H, es ',ssc_tot)      
fc = obj.kernel_pathway(FC=True, FCSD=False, PSO=False, n_atom1=n_atom1, n_atom2=n_atom2,
                                    princ_prop=p,
									occ_atom1=5, occ_atom2=4)
print(fc)



