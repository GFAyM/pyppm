import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.localization import localization


mol = '''
C1       0.0000000000            0.0000003929            0.7625314764
C1       0.0000000000           -0.0000003929           -0.7625314764
H3       0.8820493404           -0.5092509547            1.1587310600
H3      -0.8820493404           -0.5092509547            1.1587310600
H3       0.8820493404            0.5092509547           -1.1587310600
H3      -0.8820493404            0.5092509547           -1.1587310600
H4       0.0000000000            1.0185034327            1.1587304620
H4       0.0000000000           -1.0185034327           -1.1587304620
'''

#mf = scf.RHF(mol).run()

loc_obj = localization(mol_input=mol, basis='ccpvdz', no_second_loc=False, molecule_name='C2H6', atom_for_searching1='H3',
    atom_for_searching2='H7', dihedral_angle='opt_benk')
loc_obj.kernel
