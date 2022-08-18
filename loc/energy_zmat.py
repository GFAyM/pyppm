import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.localization import localization

ang = 180

mol = f'''
        C1   1
        C2   1 1.509253
        F1   2 1.369108    1  108.060
        H1   2 1.088919    1  110.884  3  119.16 
        H2   2 1.088919    1  110.884  3 -119.16
        H3   1 1.088919    2  110.884  3  {ang+119.16}
        F2   1 1.369108    2  108.060  3  {ang}
        H4   1 1.088919    2  110.884  3  {ang-119.16}
        '''

loc_obj = localization(mol_input=mol, basis='ccpvdz', no_second_loc=True, molecule_name='C2H4F2', atom_for_searching1='F3',
    atom_for_searching2='F7', dihedral_angle='opt_zmat')
loc_obj.kernel