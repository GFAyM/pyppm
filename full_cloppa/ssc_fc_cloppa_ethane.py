import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"C2H6_ccpvdz_Cholesky_PM.molden").extraer_coeff
mol_H2O2 = '''
C1       0.0000027840           -0.0000020156            0.7647696389;
C2      -0.0000023761            0.0000017425           -0.7647638280;
H3       0.0000044467           -1.0265602958            1.1687399473;
H4       0.8890282899            0.5132777256            1.1687402156;
H5      -0.0000018126            1.0265601512           -1.1687337840;
H6      -0.8890296464           -0.5132749495           -1.1687343145;
H7      -0.8890218879            0.5132759420            1.1687443491;
H8       0.8890202024           -0.5132783004           -1.1687402745;
'''

full_M_obj = Cloppa(
    mol_input=mol_H2O2,basis='ccpvdz',
    mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
    mo_occ_loc=mo_occ_loc)

full_M_obj.kernel





