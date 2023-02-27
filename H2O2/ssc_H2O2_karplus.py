import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.polaritization_propagator import Prop_pol as pp
from src.help_functions import extra_functions
import matplotlib.pyplot as plt
import pandas as pd

text = 'test_w_T_B.txt'


data_J = pd.read_csv(text, sep='\s+', header=None)
data_J = pd.DataFrame(data_J)


ang = data_J[0]
FCSD = data_J[1]
FC = data_J[2]
PSO = data_J[3]

#plt.plot(ang, DSO, 'ro', label='DSO')
fig = plt.gcf()
plt.plot(ang, PSO, 'b^-', label='PSO')
#plt.plot(ang, DSO, 'm--', label='DSO')
plt.plot(ang, FC, 'go-', label='FC')
plt.plot(ang, FCSD+PSO, 'm--', label='Total')
#plt.plot(ang, FCSD, 'r+-', label='FC+SD')
plt.plot(ang, FCSD - FC, 'r*-', label='SD')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.title('Acoplamiento entre espines nucleares A(1)+B(1) $^3J(H-H)$ en H$_2$O$_2$, 6-31G**')
fig.set_size_inches(8, 8)
plt.savefig('mecanismos_H2O2_631G.png', dpi=200)
plt.show()


    