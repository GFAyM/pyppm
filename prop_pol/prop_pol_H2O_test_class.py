import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.polaritization_propagator import Prop_pol as pp


mol = gto.M(atom = '''O 0 0 0; H  0 1 0; H 0 0 1''', basis='ccpvdz',unit='amstrong', verbose=0)
mf = scf.RHF(mol).run()
ppobj = pp(mf)
ssc = ppobj.kernel
