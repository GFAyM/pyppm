import sys
sys.path.append("/home/fer/pyPPE/src")
from help_functions import extra_functions
import itertools

v1 = []
v2 = []
for ang in range(0,18,1):
    mol, mo_coeff = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff

    orbital_1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization('H3 2', .7, 1)
    
    orbital_2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization('H4 2', .7, 1)
    
    print(ang*10, orbital_1, orbital_2)