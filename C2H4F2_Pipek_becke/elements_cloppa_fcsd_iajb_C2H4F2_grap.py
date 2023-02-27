from cProfile import label
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

#print('number of threads:',lib.num_threads())
text = str('elementos_cloppa_C2H4F2_fcsd_iajb.txt')


df = pd.read_csv(text, sep='\s+', header=None)

df.columns = ['ang','p1','p2','m']

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,8))
ax1.plot(df.ang, df.p1, 'b>-', label='H1') #f'a={orb1} b={orb2}')
ax1.set_title(r'${b}^{FC+SD}_{i=F-C_1,a=F_13pz}$')
ax1.legend()
ax3.plot(df.ang, -abs(df.p2), 'b>-', label='H2' )#f'a={orb1} b={orb2}')
ax3.set_title(r'${b}^{FC+SD}_{j=LP_2,b=F_23pz}$')
ax3.legend()
ax2.plot(df.ang, abs(df.m), 'b>-', label='$^{FC}J_{ij}(H-H)$' )#f'a={orb1} b={orb2}')
ax2.set_title(r'$^3{P}_{ia,jb}$')
#plt.ylabel('Hz')
ax1.set_xlabel('Ángulo diedro')
ax2.set_xlabel('Ángulo diedro')
ax3.set_xlabel('Ángulo diedro')

ax4.plot(df.ang, df.p1*df.p2*df.m, 'r>-')
ax4.set_title(r'${b}^{FC+SD}_{i=FC_1,a=F_13pz}* ^3{P}_{ia,jb} *{b}^{FC+SD}_{j=LP_2,b=F_23p_z}$')
ax4.yaxis.tick_right()
ax4.set_xlabel('Angulo diedro')
plt.suptitle(r'''Elements of Polarization Propagator $J^{FC+SD}(H-H)$ in C$_2$H$_2$F$_4$''')
#plt.title(f'i={i}, a={a}, j={j}, b = {b}')# f'a={orb1}, b={orb2}')
plt.savefig('FCSD_elements_C2H4F2_iajb_lig_LP.png', dpi=200)
plt.show()  