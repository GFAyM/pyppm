import matplotlib.pyplot as plt
import pandas as pd


data_J = pd.read_csv('mechanism_C2H4F2_ccpvdz.txt', sep='\s+', header=None)
data_J = pd.DataFrame(data_J)


ang = data_J[0]
FCSD = data_J[1]
FC = data_J[2]
PSO = data_J[3]

#plt.plot(ang, DSO, 'ro', label='DSO')
plt.figure(figsize=(10,8))
plt.plot(ang, PSO, 'b^-', label='PSO')
plt.plot(ang, FC, 'go-', label='FC')
plt.plot(ang, FCSD+PSO, 'm--', label='FC+SD+PSO')
plt.plot(ang, -FC + FCSD, 'r+-', label='SD')

plt.legend()
plt.ylabel('Hz')
plt.xlabel('Dihedral Angle')
plt.title('J-coupling $^3J$(F-F) in C$_2$H$_4$F$_2$ with cc-pVDZ basis')
plt.savefig('mecanismos_C2H4F2_ccpvdz.png', dpi=400)
plt.show()

