import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from pyscf.tools import molden


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

mol = gto.M(atom=str(mol), basis='cc-pvdz')

mf = scf.RHF(mol).run()
molden.from_mo(mol, 'c2h4f2_canonical.molden', mf.mo_coeff)