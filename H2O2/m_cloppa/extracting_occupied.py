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



for ang in range(10,11,1):
    a = extra_functions(
        molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('O1 1s', 0.1, 1)
    print(ang*10,a)
    b = extra_functions(
    molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several('O2 1s', 0.1, 1)
    print(ang*10,b)
    print(a+b)