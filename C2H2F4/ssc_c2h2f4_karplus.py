import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.polaritization_propagator import Prop_pol as pp

if os.path.exists('mechanism_C2H2F4_ccpvdz.txt'):
	os.remove('mechanism_C2H2F4_ccpvdz.txt')

for ang in range(90,280,10):
    mol = gto.M(atom = f'''
    C1   1
    C2   1 1.509596
    H1   2 1.086084    1  111.973
    F1   2 1.334998    1  108.649  3  121.2700 
    F2   2 1.334998    1  108.649  3 -121.2700
    F3   1 1.334998    2  108.649  3 {ang-121.27}
    H2   1 1.086084    2  111.973  3 {ang}
    F4   1 1.334998    2  108.649  3 {ang+121.27}
      ''', basis='cc-pvdz')


#mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"difluorethane_cc-pvdz_100_Cholesky_PM.molden").extraer_coeff


    mf = scf.RHF(mol).run()
    ppobj = pp(mf)

    fcsd = ppobj.kernel_select(FC=False, FCSD=True, PSO=False,atom1=[2], atom2=[6])
    fc = ppobj.kernel_select(FC=True, FCSD=False, PSO=False,atom1=[2], atom2=[6])
    pso = ppobj.kernel_select(FC=False, FCSD=False, PSO=True,atom1=[2], atom2=[6])
    with open('mechanism_C2H2F4_ccpvdz.txt', 'a') as f:
        f.write(f'{ang} {fcsd[0]} {fc[0]} {pso[0]} \n')

    