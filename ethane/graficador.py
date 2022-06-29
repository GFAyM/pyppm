import matplotlib.pyplot as plt
import pandas as pd

data_J = pd.read_csv('ssc_mechanism_C2H6_ccpvdz.txt', sep='\s+', header=None)
data_J = pd.DataFrame(data_J)


ang = data_J[0]
FCSD = data_J[1]
FC = data_J[2]
PSO = data_J[3]

#plt.plot(ang, DSO, 'ro', label='DSO')
fig = plt.gcf()
plt.plot(ang, PSO, 'bo-', label='PSO')
#plt.plot(ang, DSO, 'm--', label='DSO')
plt.plot(ang, FC, 'g>-', label='FC')
plt.plot(ang, FCSD+PSO, 'r<--', label='Total')
plt.plot(ang, FCSD - FC, 'm:', label='SC')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Ángulo diedro')
plt.title('J coupling $^3J(H-H)$ with Pipek-Mezey LMOs in C$_2$H$_6$, cc-pVDZ')
fig.set_size_inches(8.5, 8.5)
#
plt.savefig('ssc_mechanims_C2H6_ccpvdz.png', dpi=200)
plt.show()
