import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa_full


M_diag_list = []
inv_M_diag_list = []

for ang in range(3,4,1):
    mol_loc, mo_coeff_loc, mo_occ_loc = extra_functions(molden_file=f"H2O2_mezcla_{ang*10}.molden").extraer_coeff
    occidx_OH = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'H', .3, .5)

    occidx_1 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('O1 1s', 0.5, 1)
    occidx_2 = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('O2 1s', 0.5, 1)

    occidx_O_1s = occidx_1 + occidx_2

    occidx_O1_2s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O1 2s', 0.3, 1) #.4 de 2s, 0.6 de 2p
    occidx_O2_2s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O2 2s', 0.3, 1) #.4 de 2s, 0.6 de 2p
    
    occidx_O_2s = occidx_O1_2s + occidx_O2_2s

    occidx_O1_2p = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O1 2p', 0.7, 1) #.2% de 2s
    occidx_O2_2p = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O2 2p', 0.7, 1) #.2% de 2s

    occidx_O_2p = occidx_O1_2p + occidx_O2_2p

    occidx_O_O = [extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O1 2px', 0.4, 0.5)[0]]
            


    print(f"Occs O1s se encuentra, en el ángulo {ang*10}", occidx_O_1s)
    print(f"Occs O2s se encuentra, en el ángulo {ang*10}", occidx_O_2s)
    print(f"Occs O2p se encuentra, en el ángulo {ang*10}", occidx_O_2p)
    print(f"Occs O-O se encuentra, en el ángulo {ang*10}", occidx_O_O)
    print(f"Occs O-H se encuentra, en el ángulo {ang*10}", occidx_OH)
