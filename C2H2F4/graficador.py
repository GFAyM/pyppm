import matplotlib.pyplot as plt
import pandas as pd


data_J = pd.read_csv('mechanism_C2H2F4_ccpvdz.txt', sep='\s+', header=None)
data_J = pd.DataFrame(data_J)


ang = data_J[0]
FCSD = data_J[1]
FC = data_J[2]
PSO = data_J[3]

#plt.plot(ang, DSO, 'ro', label='DSO')
fig = plt.gcf()
plt.plot(ang, PSO, 'b^-', label='PSO')
plt.plot(ang, FC, 'go-', label='FC')
plt.plot(ang, FCSD+PSO, 'm--', label='FC+SD+PSO')
plt.plot(ang, -FC + FCSD, 'r+-', label='SD')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.title('Acoplamiento indirecto entre espines nucleares $^3J(H-H)$ en C$_2$H$_2$F$_4$, cc-pVDZ')
fig.set_size_inches(8,8)
plt.savefig('mecanismos_C2H4F2_ccpvdz.png', dpi=200)
plt.show()

