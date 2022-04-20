import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa_full
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



for ang in range(1,18,1):
    
    occidx_O1_2p = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O1 2p', 0.7, 1)
    occidx_O2_2p = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O2 2p', 0.7, 1)


    print(ang*10, occidx_O1_2p, occidx_O2_2p)

    occidx_O1_2s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O1 2s', 0.3, 1)
    occidx_O2_2s = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
            'O2 2s', 0.3, 1)
    print(ang*10,occidx_O1_2s,occidx_O2_2s)