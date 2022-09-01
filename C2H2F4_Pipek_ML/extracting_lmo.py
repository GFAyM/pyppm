import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions


occ1 = []
occ2 = []

for ang in range(0,180,10):
    a = extra_functions(molden_file=f"C2H2F4_{ang}_ccpvdz_Cholesky_PM.molden").mo_hibridization(
        'H3 2p', .1, 1)
    b = extra_functions(molden_file=f"C2H2F4_{ang}_ccpvdz_Cholesky_PM.molden").mo_hibridization(
        'H7 2p', .1, 1)
    #occ1.append(a[1])
    #occ2.append(b[1])
    print(ang,a)
    print(ang,b)

print('v1_1 =', occ1)
print('v1_2 =', occ2)