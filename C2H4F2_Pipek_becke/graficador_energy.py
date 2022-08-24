import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('energy_dihedral_angle_C2H4F2.txt', sep='\s+', header=None)
data = pd.DataFrame(data)


ang = data[0]
energy = data[1]

fig = plt.gcf()
plt.plot(ang, energy, 'b^-', label='Total Energy')

plt.legend()
plt.ylabel('Hartree')
plt.xlabel('Ángulo diedro')
plt.title(r'''Total energy of C$_2$H$_4$F$_2$ molecule 
varying the dihedral angle between the Fluorine atoms''')
fig.set_size_inches(10,8)
plt.savefig('energy_C2H4F2.png', dpi=200)
plt.show()