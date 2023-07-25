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
hrpa = HRPA(mf)
nocc = hrpa.nocc
nvir = hrpa.nvir

can_inv = np.linalg.inv(mf.mo_coeff.T)
c_occ = (mo_coeff_loc[:,:nocc].T.dot(can_inv[:,:nocc])).T

c_vir = (mo_coeff_loc[:,nocc:].T.dot(can_inv[:,nocc:])).T
total = np.einsum('ij,ab->iajb',c_occ,c_vir)
total = total.reshape(nocc*nvir,nocc*nvir)

m = hrpa.communicator_matrix_hrpa(triplet=True)

#print(m)

m_loc = total.T @ m @ total

print(m_loc)
m_loc = m_loc.reshape(nocc,nvir,nocc,nvir)
print(m_loc[0,1,3,6])
print(m_loc[4,1,3,6])