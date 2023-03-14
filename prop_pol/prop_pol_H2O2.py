from pyscf import scf, gto, tdscf, lib

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.polaritization_propagator import Prop_pol as pp
from pyscf import gto




mol = gto.M(atom='''
        O1   1
        O2   1 1.45643942
        H3   2 0.97055295  1 99.79601616
        H4   1 0.97055295  2 99.79601616  3 100
        ''', basis='6-31G**', verbose=0)

mf = scf.RHF(mol).run()


ppobj = pp(mf)


#ssc = ppobj.kernel(FC=False, FCSD=True, PSO=True)

