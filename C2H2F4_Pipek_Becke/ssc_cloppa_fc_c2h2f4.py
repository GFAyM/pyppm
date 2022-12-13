import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.cloppa import Cloppa
from src.help_functions import extra_functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

text = 'ssc_fc_c2h2f4.txt'
#print('number of threads:',lib.num_threads())
if os.path.exists(text):
	os.remove(text)

for ang in range(0,18,1): 
	mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H2F4_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

	cloppa_obj = Cloppa(
				mo_coeff_loc=mo_coeff, mol_loc=mol, 
				mo_occ_loc=mo_occ)
	
	m = cloppa_obj.M(triplet=True)
	p = np.linalg.inv(m)

	ssc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False,
									princ_prop=p,
									n_atom1=[2],   
									n_atom2=[6], all_pathways=True)
	with open(text, 'a') as f:
		f.write(f'{ang*10} {ssc[0]} \n')     

data_J = pd.read_csv(text, sep='\s+', header=None)
data_J.columns = ['ang', 'fcsd']


plt.figure(figsize=(10,8))
plt.plot(data_J.ang, data_J.fcsd, 'b>-')#f'a={orb1} b={orb2}')
#plt.plot(ang, fc, 'g<-', label='$^{FC}J(H-H)$')
plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.suptitle('FC+SD contribution to $^3J(H-H)_{i,j}$ en C$_2$F$_2$H$_4$, cc-pVDZ')
#plt.title('i=C-F$_1$(2s,2p$_z$, 2p$_x$), j=C-F$_2$(2s,2p$_z$,2p$_x$), a = b = all')# f'a={orb1}, b={orb2}')
#plt.savefig(f'FC_occ_C2F2H4_sum_2.png', dpi=200)
plt.show() 




