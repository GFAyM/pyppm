# creating the path (PYTHONPATH) to our module.
# assuming that our 'pyECM' directory is out ('..') of our current directory 
import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from pyppm.ssc_pol_prop import Prop_pol
from pyppm.help_functions import extra_functions
from pyppm.ssc_cloppa import Cloppa
from pyppm.hrpa import HRPA
from pyscf import scf, gto
import numpy as np

mol, mo_coeff_loc, mo_occ = extra_functions(molden_file='C2H6_ccpvdz_Pipek_Mezey.molden').extraer_coeff
mf = scf.RHF(mol)
mf.kernel()
pp = HRPA(mf)
nocc = pp.nocc
nvir = pp.nvir

h1, m, h2 = pp.pp_ssc_pso_select_elements([2], [4])

can_inv = np.linalg.inv(mf.mo_coeff.T)
c_occ = (mo_coeff_loc[:,:nocc].T.dot(can_inv[:,:nocc])).T

c_vir = (mo_coeff_loc[:,nocc:].T.dot(can_inv[:,nocc:])).T
total = np.einsum('ij,ab->iajb',c_occ,c_vir)
total = total.reshape(nocc*nvir,nocc*nvir)
p = np.linalg.inv(m)
p = p.reshape(nocc, nvir, nocc, nvir)
m_loc = total @ m @ total.T
p_loc = np.linalg.inv(m_loc)
p_loc = p_loc.reshape(nocc, nvir, nocc, nvir)
h1_loc = c_occ@h1@c_vir.T
h2_loc = c_occ@h2@c_vir.T

p = np.einsum('xia,iajb,yjb->xy', h1, p , h2)
p_loc = np.einsum('xia,iajb,yjb->xy', h1_loc, p_loc , h2_loc)

print(p_loc)
print(p)