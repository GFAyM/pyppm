import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.polaritization_propagator import Prop_pol as pp

#if os.path.exists('mechanism_C2H2F4_ccpvdz.txt'):
#	os.remove('mechanism_C2H2F4_ccpvdz.txt')

mol = gto.M(atom = f'''
C1       0.7537643361           -0.0394844785            0.0000000000;
C2      -0.7537643361            0.0394844785            0.0000000000;
F1       1.2144144730            0.5939430486            1.0811058803;
F2       1.2144144730            0.5939430486           -1.0811058803;
F3      -1.2144144730           -0.5939430486            1.0811058803;
F4      -1.2144144730           -0.5939430486           -1.0811058803;
H1       1.1068995862           -1.0665551558            0.0000000000;
H2      -1.1068995862            1.0665551558            0.0000000000;
    ''', basis='cc-pvdz')


#mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"difluorethane_cc-pvdz_100_Cholesky_PM.molden").extraer_coeff


mf = scf.RHF(mol).run()
ppobj = pp(mf)

fcsd = ppobj.kernel_select(FC=False, FCSD=True, PSO=False,atom1=[7], atom2=[6])
fc = ppobj.kernel_select(FC=True, FCSD=False, PSO=False,atom1=[7], atom2=[6])
pso = ppobj.kernel_select(FC=False, FCSD=False, PSO=True,atom1=[7], atom2=[6])
print(f'{fcsd[0]} {fc[0]} {pso[0]}')

