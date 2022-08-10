import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.localization import localization


mol = '''
C	-0.2424010	0.7147920	0.0000000
C	0.2424010	-0.7147920	0.0000000
H	-1.3267140	0.7762050	0.0000000
H	1.3267140	-0.7762050	0.0000000
F	0.2424010	1.3300300	1.0810690
F	0.2424010	1.3300300	-1.0810690
F	-0.2424010	-1.3300300	1.0810690
F	-0.2424010	-1.3300300	-1.0810690
'''

#mf = scf.RHF(mol).run()

loc_obj = localization(mol_input=mol, basis='ccpvdz', no_second_loc=False, molecule_name='C2H2F4', atom_for_searching1='H7',
    atom_for_searching2='H8', dihedral_angle='opt_benk')
loc_obj.kernel
