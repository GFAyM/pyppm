import os
import sys
from pyscf import gto, scf, lib
from pyscf import gto
import time

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.pp_4c import Prop_pol
from pyscf import gto, scf

mol_h2o = gto.M(
            atom = ''' O 1 0.5 0; H1 0 0 0; H2 0 0 1''',
            basis = 'sto-3g')

rhf = scf.DHF(mol_h2o).run()

pp = Prop_pol(rhf)

j = pp.kernel()

#mol_2 = gto.M(
 #           atom = ''' 
  #          S        0.0000000000            0.0000000000           -0.2841815015
   #         H1       0.0000000000            1.0563775694            0.7152065435
    #        H1       0.0000000000           -1.0563775694            0.7152065435
     #       ''',
      #      basis = 'cc-pvdz', unit='Angstrom')


