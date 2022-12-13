import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.localization import localization


mol = '''
C1       0.0000000000            0.4663393949            0.5932864698
C2       0.0000000000           -0.4663393949           -0.5932864698
H1       0.8884363804            1.0959623133            0.5921146257
H2      -0.8884363804            1.0959623133            0.5921146257
H3       0.8884363804           -1.0959623133           -0.5921146257
H4      -0.8884363804           -1.0959623133           -0.5921146257
F1       0.0000000000           -0.2947283454            1.7313694084
F2       0.0000000000            0.2947283454           -1.7313694084
'''

loc_obj = localization(mol_input=mol, basis='ccpvdz', no_second_loc=True, molecule_name='C2H4F2', atom_for_searching1='F1',
    atom_for_searching2='F2', dihedral_angle='opt')
loc_obj.kernel
