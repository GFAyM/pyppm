import os
from tabnanny import verbose
import matplotlib.pyplot as plt
import pandas as pd
from pyscf import scf, gto


#print('number of threads:',lib.num_threads())
file = str('energy_dihedral_angle_C2H4F2.txt')

if os.path.exists(file):
	os.remove(file)



data = []

for ang in range(0,360,10): 
    mol = gto.M(atom=f'''
        C1   1
        C2   1 1.509253
        F1   2 1.369108    1  108.060
        H1   2 1.088919    1  110.884  3  119.16 
        H2   2 1.088919    1  110.884  3 -119.16
        H3   1 1.088919    2  110.884  3  {ang+119.16}
        F2   1 1.369108    2  108.060  3  {ang}
        H4   1 1.088919    2  110.884  3  {ang-119.16}
        ''', basis='ccpvdz', verbose=0)
    mf = scf.RHF(mol).run()
    #print(mf.e_tot)
    with open(file, 'a') as f:
        f.write(f'{ang} {mf.e_tot} \n')


data = pd.read_csv(file, sep='\s+', header=None)

data.columns = ['ang', 'energy']

fig, (ax1) = plt.subplots(1, 1, figsize=(10,6))

ax1.plot(data.ang, data.energy, 'b>-', label='Energy' )#f'a={orb1} b={orb2}')
ax1.set_xlabel('Hartree')
plt.suptitle(r'''Total energy of C$_2$H$_4$F$_2$ molecule 
varying the dihedral angle between the Fluorine atoms''')
plt.savefig('energy_c2h4f2.png')

plt.show()

#ax4.set_xlabel('Dihedral angle')
#ax4.plot(data_J.ang, data_J.mutual, 'b>-', label='$^{FC}J_{ij}(F-F)$' )#f'a={orb1} b={orb2}')
#ax4.set_title('Mutual Information ')# f'a={orb1}, b={orb2}')
#plt.show()
