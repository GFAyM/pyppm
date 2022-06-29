import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.polaritization_propagator import Prop_pol as pp
from src.help_functions import extra_functions
from src.cloppa import Cloppa
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf

mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"difluorethane_cc-pvdz_150_Cholesky_PM.molden").extraer_coeff
cloppa_obj = Cloppa(
    mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
    mo_occ_loc=mo_occ_loc)

n_atom1=[2]
n_atom2=[6]

ssc_tot = 0
m = cloppa_obj.M(triplet=True)
p = np.linalg.inv(m)
print('i  j   ssc_pso ssc_pso_total')

for i in range(cloppa_obj.nocc):
	for j in range(cloppa_obj.nocc):
#		for a in range(9,38,1):
#			for b in range(9,38,1):
		ssc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
										princ_prop=p,
										n_atom1=n_atom1, occ_atom1=i, 
										n_atom2=n_atom2, occ_atom2=j)
		ssc_tot += ssc
		if abs(ssc) > 10: 
			print(i, j, ssc, ssc_tot)    
print(ssc_tot)
					
mf = scf.RHF(mol_loc).run()
ppobj = pp(mf)
pso = ppobj.kernel_select(FC=True, FCSD=False, PSO=False,atom1=[2], atom2=[6])
print(pso)


