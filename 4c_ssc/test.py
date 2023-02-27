import os
import sys
from pyscf import gto, scf, lib
from pyscf import gto
import time

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.pp_4c_jj import Prop_pol
from pyscf import gto, scf

mol_h2o = gto.M(
            atom = ''' O 2 1 0; H1 1 0 0; H2 1 0 1''',
            basis = 'sto-3g')

rhf = scf.DHF(mol_h2o).run()

pp = Prop_pol(rhf)

j = pp.kernel()