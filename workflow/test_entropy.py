# creating the path (PYTHONPATH) to our module.
# assuming that our 'pyECM' directory is out ('..') of our current directory 
import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from pyppm.ssc_cloppa import Cloppa
from pyppm.ssc_pol_prop import Prop_pol
from pyppm.help_functions import extra_functions
from pyscf import scf, gto
from pyscf import lib
from pyppm.hrpa_loc_2 import HRPA_loc
#from pyppm.hrpa_loc_cloppa import HRPA_loc
from pyppm.hrpa import HRPA
import numpy as np


mol, mo_coeff, mo_occ = extra_functions(molden_file='ethane.molden').extraer_coeff
mol, mo_coeff, mo_occ = extra_functions(molden_file='cholesky_ethane_sto-3g_loc.molden').extraer_coeff


#mol, mo_coeff, mo_occ = extra_functions(molden_file='C2H6_ccpvdz_Pipek_Mezey.molden').extraer_coeff

mf = scf.RHF(mol)
mf.kernel()
hrpa_obj = HRPA(mf)
hrpa_loc_obj = HRPA_loc(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
#cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)

#print(cloppa_obj.M(triplet=True))

#f_iajb = np.zeros((nocc,nvir,nocc,nvir))
#orbo = cloppa_obj.orbo
#orbv = cloppa_obj.orbv
#mo = cloppa_obj.mo
#fock_at = mf.get_fock()
#fock_can = mf.mo_coeff.T @ fock_at @ mf.mo_coeff
#fock_loc = orbo.T @ fock_at @ orbo


#print(np.diag(fock_loc).shape)
#print(np.diag(fock_can))
#print(fock_loc[8,:].sum())
#fock=np.einsum('xy->x', fock_loc)
#print(fock.shape)
#f = 0
#for i in range(nocc):
#    for a in range(nvir):
#                f += orbo[:,i].T @ fock_can @ orbo[:,i]
#                f -= orbv[:,a].T @ fock_can @ orbv[:,a]

#fock_loc = mo_coeff.T @ fock_can @ mo_coeff
#fock_mol = mf.mo_coeff.T @ fock_can @ mf.mo_coeff

#print(f.sum())

#print(mf.mo_energy.sum())
#print(np.diag(fock_mol[nocc:]))
#print(np.diag(fock_loc[nocc:]))
#print(HRPA(mf).S2.sum())
#print(hrpa_loc_obj.S2.sum())
print(HRPA(mf).pp_ssc_fc_select([2],[6]))
print(hrpa_loc_obj.pp_ssc_fc_select([2],[6]))