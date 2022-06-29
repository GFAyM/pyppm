import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.polaritization_propagator import Prop_pol as pp
from src.help_functions import extra_functions


mol = gto.M(atom = '''
C1       0.0000027840           -0.0000020156            0.7647696389
C2      -0.0000023761            0.0000017425           -0.7647638280
H1       0.0000044467           -1.0265602958            1.1687399473
H2       0.8890282899            0.5132777256            1.1687402156
H3      -0.0000018126            1.0265601512           -1.1687337840
H4      -0.8890296464           -0.5132749495           -1.1687343145
F1      -0.8890218879            0.5132759420            1.1687443491
F2       0.8890202024           -0.5132783004           -1.1687402745
''', 
basis='cc-pvdz',unit='amstrong', verbose=0)

#mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"difluorethane_cc-pvdz_100_Cholesky_PM.molden").extraer_coeff


mf = scf.RHF(mol).run()
ppobj = pp(mf)
print('SSC in Hz with canonical orbitals')
ssc = ppobj.kernel(FC=False, FCSD=False, PSO=True)
#print(ppobj.pp_ssc_fcsd)