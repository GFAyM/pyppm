import os
from tabnanny import verbose
import matplotlib.pyplot as plt
import pandas as pd
from pyscf import scf, gto


#print('number of threads:',lib.num_threads())
file = str('energy_dihedral_angle_C2H2F4.txt')

if os.path.exists(file):
	os.remove(file)



data = []

for ang in range(0,360,10): 
    mol = gto.M(atom=f'''
        C1   1
        C2   1 1.509596
        H1   2 1.086084    1  111.973
        F1   2 1.334998    1  108.649  3  121.2700 
        F2   2 1.334998    1  108.649  3 -121.2700
        F3   1 1.334998    2  108.649  3 {ang+121.27}
        H2   1 1.086084    2  111.973  3 {ang}
        F4   1 1.334998    2  108.649  3 {ang-121.27}
        ''', basis='ccpvdz', verbose=0)
    mf = scf.RHF(mol).run()
    #print(mf.e_tot)
    with open(file, 'a') as f:
        f.write(f'{ang} {mf.e_tot} \n')

data = pd.read_csv(file, sep='\s+', header=None)
data = pd.DataFrame(data)


ang = data[0]
energy = data[1]

fig = plt.gcf()
plt.plot(ang, energy, 'b^-', label='Total Energy')

plt.legend()
plt.ylabel('Hartree')
plt.xlabel('Ángulo diedro')
plt.title(r'''Total energy of C$_2$H$_2$F$_4$ molecule with cc-pVDZ basis 
varying the dihedral angle between Hydrogen atoms''')
fig.set_size_inches(10,8)
plt.savefig('energy_C2H4F2.png', dpi=200)
plt.show()