import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)

from src.help_functions import extra_functions
from src.ppe_2 import M_matrix

from src.help_functions import extra_functions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

text = str('entropy_triplet.txt')
if os.path.exists(text):
	os.remove(text)


occ1 = [22, 23, 23, 22, 22, 22, 22, 22, 22, 22, 23, 22, 23, 22, 23, 22, 22, 22]
occ2 = [23, 22, 22, 23, 23, 23, 23, 23, 23, 23, 22, 23, 22, 23, 22, 23, 23, 23]

v1_1 = [63, 93, 93, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63]
v1_2 = [93, 64, 64, 64, 53, 92, 64, 64, 64, 64, 64, 64, 64, 64, 53, 64, 53, 64]

v2_1 = [34, 35, 35, 35, 36, 36, 36, 36, 36, 35, 36, 36, 36, 36, 36, 35, 35, 36]
v2_2 = [35, 34, 34, 34, 35, 35, 35, 35, 35, 36, 35, 35, 35, 35, 35, 34, 34, 35]


for ang in range(0,18,1):
	mol, mo_coeff, mo_occ = extra_functions(molden_file=f"C2H2F4_{ang*10}_ccpvdz_Cholesky_PM.molden").extraer_coeff

	inv_prop = M_matrix(mol=mol, mo_coeff=mo_coeff, mo_occ=mo_occ,
				occ = [  occ1[ang], occ2[ang]],
				vir = [ v1_1[ang],
						v1_2[ang]])   
	ent_iajb = inv_prop.entropy_iajb_1

	with open(text, 'a') as f:
		f.write(f'{ang*10} {ent_iajb}  \n')


df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang','ent_iajb']

fig, (ax1) = plt.subplots(1, 1, figsize=(8,8))
#plt.figure(figsize=(10,8))
ax1.plot(df.ang, df.ent_iajb, 'b>-', label='M') #f'a={orb1} b={orb2}')
ax1.set_title(r'$S_{iajb}$')
ax1.legend()
#plt.ylabel('Hz')
ax1.set_xlabel('Ángulo diedro')
plt.suptitle(r'''Von Newmann entropy in C$_2$H$_2$F$_4$''')
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
#plt.savefig('M_C2H2F4_CH1_CH1*_CH2_CH2**_.png', dpi=200)
plt.show()  
