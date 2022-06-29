import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
import numpy as np
from src.polaritization_propagator import Prop_pol as pp
from src.help_functions import extra_functions
from src.cloppa import Cloppa

if os.path.exists('ssc_mechanism_C2H6_ccpvdz.txt'):
    os.remove('ssc_mechanism_C2H6_ccpvdz.txt')

for ang in range(0,190,10):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"C2H6_{ang}_ccpvdz_Cholesky_PM.molden").extraer_coeff


    cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
		                mo_occ_loc=mo_occ_loc)
    m = cloppa_obj.M(triplet=True)
    p = np.linalg.inv(m)

    fcsd = cloppa_obj.kernel_pathway(FC=False, FCSD=True, PSO=False,n_atom1=[2], n_atom2=[6], princ_prop=p,all_pathways=True)
    fc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,n_atom1=[2], n_atom2=[6], princ_prop=p,all_pathways=True)
    #print(fcsd, fc)
    m = cloppa_obj.M(triplet=False)
    p = np.linalg.inv(m)
    pso = cloppa_obj.kernel_pathway(FC=False, FCSD=True, PSO=True,n_atom1=[2], n_atom2=[6], princ_prop=p,all_pathways=True)
    with open('ssc_mechanism_C2H6_ccpvdz.txt', 'a') as f:
        f.write(f'{ang} {fcsd[0]} {fc[0]} {pso[0]} \n')