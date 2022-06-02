import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.polaritization_propagator import Prop_pol as pp


mol = gto.M(atom = '''
C1       0.0000027840           -0.0000020156            0.7647696389;
C2      -0.0000023761            0.0000017425           -0.7647638280;
H3       0.0000044467           -1.0265602958            1.1687399473;
H4       0.8890282899            0.5132777256            1.1687402156;
H5      -0.0000018126            1.0265601512           -1.1687337840;
H6      -0.8890296464           -0.5132749495           -1.1687343145;
H7      -0.8890218879            0.5132759420            1.1687443491;
H8       0.8890202024           -0.5132783004           -1.1687402745;
''', 

basis='ccpvdz',unit='amstrong')
mf = scf.RHF(mol).run()
ppobj = pp(mf)
ssc = ppobj.kernel
