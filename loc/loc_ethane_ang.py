import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.localization import localization

ang = 0

for ang in range(0,190,10):
    mol = str(f'''
        C1   1
        C2   1 1.529534
        H3   2 1.103184    1  111.481
        H4   2 1.103184    1  111.481  3  120.0000 
        H5   2 1.103184    1  111.481  3 -120.0000
        H6   1 1.103184    2  111.481  3 {ang -120}
        H7   1 1.103184    2  111.481  3 {ang}
        H8   1 1.103184    2  111.481  3 {ang+120}
        ''')

    loc_obj = localization(mol_input=mol, basis='ccpvdz', no_second_loc=False, molecule_name='C2H6', atom_for_searching1='H3',
    atom_for_searching2='H7', dihedral_angle=ang)
    loc_obj.kernel

