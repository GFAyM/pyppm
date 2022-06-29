import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
import pandas as pd
from pyscf import scf, gto 
from src.polaritization_propagator import Prop_pol as pp
from src.help_functions import extra_functions



for ang in range(0,190,10):
    mol = gto.M(atom = '''
      C1   1
      C2   1 1.509253
      F3   2 1.369108    1  108.060
      H4   2 1.088919    1  110.884  3  120.0000 
      H5   2 1.088919    1  110.884  3 -120.0000
      H6   1 1.088919    2  110.884  3 {}
      F7   1 1.369108    2  108.060  3 {}
      H8   1 1.088919    2  110.884  3 {}
      '''.format(-120+ang, ang, 120+ang), basis='cc-pvdz')


#mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"difluorethane_cc-pvdz_100_Cholesky_PM.molden").extraer_coeff


    mf = scf.RHF(mol).run()
    ppobj = pp(mf)
    #print('SSC in Hz with canonical orbitals')
    fcsd = ppobj.kernel_select(FC=False, FCSD=True, PSO=False,atom1=[2], atom2=[6])
    fc = ppobj.kernel_select(FC=True, FCSD=False, PSO=False,atom1=[2], atom2=[6])
    pso = ppobj.kernel_select(FC=False, FCSD=False, PSO=True,atom1=[2], atom2=[6])
    with open('mechanism_C2H4F2_ccpvdz.txt', 'a') as f:
        f.write(f'{ang} {fcsd[0]} {fc[0]} {pso[0]} \n')

    