import pandas as pd
import numpy 
import matplotlib.pyplot as plt
from pyscf import scf, gto, tdscf, lib
from pyscf.dft import numint
from pyscf.data import nist

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.polaritization_propagator import Prop_pol as pp
from functools import reduce
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import tools
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.scf import _response_functions  # noqa
from pyscf.ao2mo import _ao2mo
from pyscf.dft import numint
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor



mol = gto.M(atom='''
        O1   1
        O2   1 1.45643942
        H3   2 0.97055295  1 99.79601616
        H4   1 0.97055295  2 99.79601616  3 100
        ''', basis='6-31G**', verbose=0)

mf = scf.RHF(mol).run()


ppobj = pp(mf)

ssc = ppobj.kernel(FC=False, FCSD=True, PSO=False)

#pso = ppobj.pp_ssc_pso
#print(pso)
