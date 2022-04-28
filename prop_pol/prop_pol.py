from tabnanny import verbose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf, gto, tdscf

import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.polaritization_propagator import Prop_pol as pp




mol = gto.M(atom='''
            O 0 0      0
            H 0 -0.757 0.587
            H 0  0.757 0.587''',
            basis='ccpvdz')

mf = scf.RHF(mol).run()

prop = pp(mf).kernel

print(prop[0].shape)