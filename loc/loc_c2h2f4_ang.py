import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.localization import localization

ang = 0

for ang in range(0,100,10):
    mol = str(f'''
        C1   1
        C2   1 1.509596
        H1   2 1.086084    1  111.973
        F1   2 1.334998    1  108.649  3  121.2700 
        F2   2 1.334998    1  108.649  3 -121.2700
        F3   1 1.334998    2  108.649  3 {ang+121.27}
        H2   1 1.086084    2  111.973  3 {ang}
        F4   1 1.334998    2  108.649  3 {ang-121.27}
        ''')

    loc_obj = localization(mol_input=mol, basis='ccpvdz', no_second_loc=False, molecule_name='C2H2F4', atom_for_searching1='H3',
                           atom_for_searching2='H7',pm_pop_method_second='becke', dihedral_angle=ang)
    loc_obj.kernel