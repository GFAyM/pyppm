from tabnanny import verbose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf, gto, tdscf
from pyscf.dft import numint

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.polaritization_propagator import Prop_pol as pp




mol = gto.M(atom='''
            H 0  0.34252627 0.34252627
            F 0  1.54719985 1.54719985 
            ''',
            basis='ccpvdz', unit='Bohr')

mf = scf.RHF(mol).run()

ppobj = pp(mf)
pol_prop = ppobj.polarization_propagator

print(pol_prop)