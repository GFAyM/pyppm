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
C1       0.0000000000            0.4663393949            0.5932864698
C2       0.0000000000           -0.4663393949           -0.5932864698
H1       0.8884363804            1.0959623133            0.5921146257
H2      -0.8884363804            1.0959623133            0.5921146257
H3       0.8884363804           -1.0959623133           -0.5921146257
H4      -0.8884363804           -1.0959623133           -0.5921146257
F1       0.0000000000           -0.2947283454            1.7313694084
F2       0.0000000000            0.2947283454           -1.7313694084
#''', 
basis='cc-pvdz',unit='amstrong', verbose=0)
ang = 100
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
#from /home/fer/Ubuntu_19_files/Dalton/Jcoupling/difluorethane/6-311G :
mol_2 = gto.M(atom = '''
    C1  1                                              
    C2   1   1.519167                                 
    F3   2   1.391086  1  108.456                  
    H4   2   1.103435  1  110.614  3  120.0009  0  
    H5   2   1.103435  1  110.614  3 -120.0090  0  
    H6   1   1.103435  2  110.614  3  -20  0  
    F7   1   1.391086  2  108.456  3  100        0  
    H8   1   1.103435  2  110.614  3  220  0 
''',basis='6-311g')


#mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"difluorethane_cc-pvdz_100_Cholesky_PM.molden").extraer_coeff


mf = scf.RHF(mol_2).run()
ppobj = pp(mf)
print('SSC in Hz with canonical orbitals')
ssc = ppobj.kernel_select(FC=False, FCSD=True, PSO=False,atom1=[2], atom2=[6])
print(ssc)
ssc = ppobj.kernel_select(FC=True, FCSD=False, PSO=False,atom1=[2], atom2=[6])
print(ssc)

#ssc = ppobj.kernel(FC=True, FCSD=False, PSO=False)

#print(ppobj.pp_ssc_fcsd)