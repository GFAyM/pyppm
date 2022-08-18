import chemcoord as cc
import time
import pandas as pd

#water = cc.Cartesian.read_xyz('water_dimer.xyz', start_index=1)
water_results = cc.Cartesian.read_xyz('C2H4F2.xyz', start_index=1)
zwater = water_results.get_zmat()


print(zwater)