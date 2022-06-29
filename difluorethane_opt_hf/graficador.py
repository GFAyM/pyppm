import matplotlib.pyplot as plt
import pandas as pd

data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
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
plt.title('Acoplamiento entre espines nucleares $^3J(F-F)$ en C$_2$H$_4$F$_2$, cc-pVDZ')
fig.set_size_inches(6.5, 6.5)
#plt.show()
plt.savefig('mecanismos_C2H4F2_ccpvdz.png', dpi=200)

