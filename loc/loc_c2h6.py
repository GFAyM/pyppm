import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.localization import localization


for ang in range(10,190,10):
    mol = str(f'''
        C1   1
        C2   1 1.525063
        H1   2 1.092850    1  111.256
        H2   2 1.092850    1  111.256  3  120 
        H3   2 1.092850    1  111.256  3 -120
        H4   1 1.092850    2  111.256  3  {ang+120}
        H5   1 1.092850    2  111.256  3  {ang}
        H6   1 1.092850    2  111.256  3  {ang-120}
        ''')

    loc_obj = localization(mol_input=mol, basis='ccpvdz', no_second_loc=False, molecule_name='C2H6', atom_for_searching1='H3',
                           atom_for_searching2='H7',pm_pop_method_second='becke', dihedral_angle=ang)
    loc_obj.kernel