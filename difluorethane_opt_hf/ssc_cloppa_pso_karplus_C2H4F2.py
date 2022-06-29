import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from src.polaritization_propagator import Prop_pol as pp
from src.help_functions import extra_functions
from src.cloppa import Cloppa
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf


F3_1s = [3, 3, 2, 2, 2, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 3]
F7_1s = [2, 2, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 3, 2]

F3_2s = [11, 10, 10, 11, 10, 10, 11, 10, 11, 10, 11, 10, 11, 10, 10, 11, 10, 11, 11]
F7_2s = [10, 11, 11, 10, 11, 11, 10, 11, 10, 11, 10, 11, 10, 11, 11, 10, 11, 10, 10]


F3_2pz = [8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
F7_2pz = [9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]


F3_2p1 =[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
F7_2p1 =[5, 5, 6, 6, 6, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5, 5]

F3_2p2 = [7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7]
F7_2p2 = [6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6]

C1_1s = [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]
C2_1s = [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0]

C_C = [14, 14, 15, 15, 15, 15, 15, 15, 14, 14, 14, 15, 15, 15, 15, 15, 15, 14, 14]

occ_lmo = [(F3_1s,'F3_1s'), (F7_1s,'F7_1s'), (F3_2s,'F3_2s'), (F7_2s,'F7_2s'), (F3_2pz,'F3_2pz'), 
(F7_2pz,'F7_2pz'), (F3_2p1,'F3_2p1'), (F7_2p1,'F7_2p1'), (F3_2p2,'F3_2p2'), (F7_2p2,'F7_2p2'), 
(C1_1s,'C1_1s'), (C2_1s,'C2_1s'), (C_C,'C_C')]

#occ_lmo = [(F3_1s,'F3_1s'), (F7_1s,'F7_1s')]

#ssc_tot = 0

for ang in range(4,6,1):
    ssc_tot = 0

    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"difluorethane_cc-pvdz_{ang*10}_Cholesky_PM.molden").extraer_coeff
    cloppa_obj = Cloppa(
                mo_coeff_loc=mo_coeff_loc, mol_loc=mol_loc, #vir=viridx, occ=occidx,
                mo_occ_loc=mo_occ_loc)
    m = cloppa_obj.M(triplet=False)
    p = np.linalg.inv(m)        
    for i, ii in occ_lmo:
        for j, jj in occ_lmo:
            ssc = cloppa_obj.kernel_pathway(FC=False, FCSD=False, PSO=True,
                                            princ_prop=p,
                                            n_atom1=[2], occ_atom1=i[ang], 
                                            n_atom2=[6], occ_atom2=j[ang])
            ssc_tot += ssc
            print(f"{ii} {jj} {ssc} {ssc_tot}")
            #if abs(ssc) > 1: 
#            with open('cloppa_pso_C2H4F2_ccpvdz.txt', 'a') as f:
#                f.write(f'{ang*10} {ssc[0]} {ii}_{jj} \n')        		
    print('the sum of pathways is')
    print(f'{ang*10} {ssc_tot[0]}')
    mf = scf.RHF(mol_loc).run()
    ppobj = pp(mf)
    #print('SSC in Hz with canonical orbitals')
    pso = ppobj.kernel_select(FC=False, FCSD=False, PSO=True,atom1= [2], atom2=[6])
    print('the final ssc value is', ang*10, pso[0])


