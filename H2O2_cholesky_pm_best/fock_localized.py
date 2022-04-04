import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.help_functions import extra_functions
from src.cloppa import full_M_two_elec
from src.cloppa import Cloppa_test
import plotly.express as px
import pandas as pd
import numpy as np
from pyscf import ao2mo, gto, scf 


ang=10



mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(
                                    molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff



occ = mo_coeff_loc[:,[8,9,1]]

vir = mo_coeff_loc[:,[14,15]]


ang=100
mol_H2O2 = '''
    O1   1
    O2   1 1.45643942
    H3   2 0.97055295  1 99.79601616
    H4   1 0.97055295  2 99.79601616  3 {}
'''.format(ang)

mol = gto.M(atom=str(mol_H2O2), basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()

fock_can = mf.get_fock()


M = np.zeros((occ,vir,occ,vir))


































































































 




print(M)


