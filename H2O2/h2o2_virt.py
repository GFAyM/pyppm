import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
	sys.path.append(module_path)


from src.help_functions import extra_functions
from src.cloppa import Cloppa
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


H4_1s = []
H4_2px = []
H4_2py = []
H4_2pz = []
H4_2s = []

H3_1s = []
H3_2px = []
H3_2py = []
H3_2pz = []
H3_2s = []

for ang in range(0,18,1):
	viridx_OH2_1s = extra_functions(
		molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
			'H4', .5, .7)

	occidx_OH2 = extra_functions(
		molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
			'H4', .3, .5)

	viridx_OH1_1s = extra_functions(
		molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
			'H3', .5, .7)

	occidx_OH1 = extra_functions(
		molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list(
			'H3', .3, .5)

	viridx_OH2 = extra_functions(
		molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
			'H4', .7, 1)


	viridx_OH2_2s = viridx_OH2[0]

	viridx_OH2_2pz = viridx_OH2[1]
	viridx_OH2_2py = viridx_OH2[2]
	viridx_OH2_2px = viridx_OH2[3]

	viridx_OH1 = extra_functions(
		molden_file=f"H2O2_mezcla_{ang*10}.molden").mo_hibridization_for_list_several(
			'H3', .7, 1)

	viridx_OH1_2s = viridx_OH1[0]
	viridx_OH1_2pz = viridx_OH1[1]
	viridx_OH1_2py = viridx_OH1[2]
	viridx_OH1_2px = viridx_OH1[3]
	H3_1s.append(viridx_OH1_1s)
	H3_2px.append(viridx_OH1_2px) 
	H3_2py.append(viridx_OH1_2py)
	H3_2pz.append(viridx_OH1_2pz)
	H3_2s.append(viridx_OH1_2s)

	H4_1s.append(viridx_OH2_1s)
	H4_2px.append(viridx_OH2_2px) 
	H4_2py.append(viridx_OH2_2py)
	H4_2pz.append(viridx_OH2_2pz)
	H4_2s.append(viridx_OH2_2s)

print('H3_1s = ', H3_1s)
print('H3_2s = ', H3_2s)
print('H3_2px = ', H3_2px)
print('H3_2py = ', H3_2py)
print('H3_2pz = ', H3_2pz)

print('H4_1s = ', H4_1s)
print('H4_2s = ', H4_2s)
print('H4_2px = ', H4_2px)
print('H4_2py = ', H4_2py)
print('H4_2pz = ', H4_2pz)




