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

file = 'cloppa_fc_ab_H2O2.txt'

#if os.path.exists(file):#
#	os.remove(file)

H3_1s_occ = [5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5]
H4_1s_occ = [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 3, 7, 4, 4, 4]



H3_1s =  [36, 13, 36, 13, 36, 36, 13, 36, 36, 36, 36, 36, 36, 36, 36, 36, 14, 36]
H3_2s =  [12, 11, 12, 11, 12, 12, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 12]
H3_2px =  [29, 22, 29, 23, 29, 28, 29, 28, 27, 27, 27, 28, 30, 29, 29, 29, 23, 29]
H3_2py =  [20, 20, 23, 20, 23, 23, 24, 24, 24, 23, 24, 24, 23, 23, 23, 23, 20, 23]
H3_2pz =  [17, 19, 17, 19, 17, 17, 19, 17, 17, 17, 17, 17, 17, 17, 17, 17, 19, 17]

H4_1s =  [13, 36, 13, 36, 13, 13, 36, 13, 14, 14, 14, 14, 14, 14, 14, 14, 36, 14]
H4_2s =  [11, 12, 11, 12, 11, 11, 12, 11, 13, 13, 13, 13, 13, 13, 13, 13, 12, 13]
H4_2px =  [23, 29, 22, 29, 22, 22, 23, 29, 30, 30, 29, 29, 22, 22, 22, 22, 28, 22]
H4_2py =  [22, 23, 20, 21, 20, 20, 21, 23, 23, 24, 23, 23, 20, 20, 20, 20, 22, 20]
H4_2pz =  [19, 17, 19, 18, 19, 19, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 19]


occ_lmo = [(H3_1s_occ,'H1_lig'), (H4_1s_occ,'H2_lig')]

occ_lmo1 = [(H3_1s_occ,'H1_lig')]
occ_lmo2 = [(H4_1s_occ,'H2_lig')]


lmo_vir = vir_lmo = [(H3_1s, 'H1_1s'), (H4_1s, 'H2_1s'), (H3_2s, 'H1_2s'), (H4_2s, 'H2_2s'), 
			  (H3_2px, 'H1_2px'), (H4_2px, 'H2_2px'), (H3_2py, 'H1_2py'), (H4_2py, 'H2_2py'), 
				(H3_2pz, 'H1_2pz'),(H4_2pz, 'H2_2pz')]

#lmo_vir1 = [(H3_1s, 'H1_1s'), (H3_2s, 'H1_2s'), 
#			  (H3_2px, 'H1_2px'), (H3_2py, 'H1_2py'), (H3_2pz, 'H1_2pz')]

#lmo_vir2 = [(H4_1s, 'H2_1s'), (H4_2s, 'H2_2s'), 
#			   (H4_2px, 'H2_2px'), (H4_2py, 'H2_2py'), (H4_2pz, 'H2_2pz')]


for ang in range(12,16,1):
	ssc_tot = 0
	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(
		molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff    
	cloppa_obj = Cloppa(
				mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
				mo_occ_loc=mo_occ_loc)
	m = cloppa_obj.M(triplet=True)
	p = np.linalg.inv(m)
	ssc_ij = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
                                            princ_prop=p,
                                            n_atom1=[2], occ_atom1=H3_1s_occ[ang], 
                                            n_atom2=[3], occ_atom2=H4_1s_occ[ang])
	print(ssc_ij, H3_1s_occ[ang], H4_1s_occ[ang], ang)
#    for a, aa in lmo_vir:
#		for b, bb in lmo_vir:
#			ssc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
 #                                           princ_prop=p,
  #                                          n_atom1=[2], occ_atom1=H3_1s_occ[ang], vir_atom1=a[ang], 
   #                                         n_atom2=[3], occ_atom2=H4_1s_occ[ang], vir_atom2=b[ang])
			
            #with open(file, 'a') as f:
		#		f.write(f'{ang*10} {ssc_ij[0]} {ssc[0]} {aa} {bb} \n')     
            