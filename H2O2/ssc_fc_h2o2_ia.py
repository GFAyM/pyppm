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
from pyscf import lib


text = str('fc_H2O2.txt')
if os.path.exists(text):
	os.remove(text)


H4_1s_occ = [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 3, 7, 4, 4, 4]
H3_1s_occ = [5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5]

O_O = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 8, 8, 8, 8]


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

#vir = [H3_1s,H3_2s,H3_2px,H3_2py,H3_2pz,H4_1s,H4_2s,H4_2px,H4_2py,H4_2pz]
vir1 = [H3_1s,H3_2s,H3_2px,H3_2py,H3_2pz]
vir2 = [H4_1s,H4_2s,H4_2px,H4_2py,H4_2pz]


for ang in range(0,18,1):
	mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
	ssc_tot = 0
	cloppa_obj = Cloppa(
		mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
		mo_occ_loc=mo_occ_loc)
	
	m = cloppa_obj.M(triplet=True, energy_m=True, pzoa=True)
	p = np.linalg.inv(m)

	fc = cloppa_obj.kernel_pathway(FC=True, FCSD=False, PSO=False, n_atom1=[2], n_atom2=[3], princ_prop=p, all_pathways=True)

	ssc = 0
	for v1 in vir1:
		for v2 in vir2:
			ssc += cloppa_obj.kernel_pathway(FC=False, FCSD=True, PSO=False,
										princ_prop=p,
										n_atom1=[2], occ_atom1=H3_1s_occ[ang],vir_atom1=v1[ang], 
										n_atom2=[3], occ_atom2=H4_1s_occ[ang],vir_atom2=v2[ang],
										all_pathways=False)


	with open(text, 'a') as f:
		f.write(f'{ang*10} {ssc[0]} {fc[0]} \n')        		
	

df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang','fc_iajb', 'fc']

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,8))
#plt.figure(figsize=(10,8))
ax1.plot(df.ang, df.fc_iajb, 'b>-', label='H1') #f'a={orb1} b={orb2}')
ax1.set_title(r'$FC$')
ax1.legend()
ax2.plot(df.ang, df.fc, 'b>-', label='H1') #f'a={orb1} b={orb2}')
ax2.set_title(r'$FC_i$')
ax2.legend()

plt.suptitle(r'''Elements of Polarization Propagator $J^{FC}(H-H)$ in H$_2$O$_2$''')
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
#plt.savefig('FC_elements_H2O2_OH1_OH2.png', dpi=200)
plt.show()  




