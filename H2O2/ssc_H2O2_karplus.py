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



for ang in range(10,180,10):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang}.molden").extraer_coeff


    mf = scf.RHF(mol_loc).run()
    ppobj = pp(mf)
    #print('SSC in Hz with canonical orbitals')
    fcsd = ppobj.kernel_select(FC=False, FCSD=True, PSO=False,atom1=[2], atom2=[3])
    fc = ppobj.kernel_select(FC=True, FCSD=False, PSO=False,atom1=[2], atom2=[3])
    pso = ppobj.kernel_select(FC=False, FCSD=False, PSO=True,atom1=[2], atom2=[3])
    with open('mechanism_H2O2_631G_a1+b1.txt', 'a') as f:
        f.write(f'{ang} {fcsd[0]} {fc[0]} {pso[0]} \n')

data_J = pd.read_csv('mechanism_H2O2_631G_a1+b1.txt', sep='\s+', header=None)
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
plt.plot(ang, FCSD, 'r+-', label='FC+SD')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.title('Acoplamiento entre espines nucleares A(1)+B(1) $^3J(H-H)$ en H$_2$O$_2$, 6-31G**')
fig.set_size_inches(6.5, 6.5)
plt.show()
#plt.savefig('mecanismos_H2O2_631G.png', dpi=200)


    