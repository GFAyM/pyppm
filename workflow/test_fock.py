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
from pyppm.hrpa_loc import HRPA_loc

mol, mo_coeff, mo_occ = extra_functions(molden_file='C2H6_ccpvdz_Pipek_Mezey.molden').extraer_coeff
mf = scf.RHF(mol)
mf.kernel()
hrpa_loc_obj = HRPA_loc(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)
cloppa_obj = Cloppa(mo_coeff_loc=mo_coeff, mol_loc=mol, mo_occ_loc=mo_occ)


#mo_energy_occ, mo_energy_vir = hrpa_loc_obj.fock_energy 

#e_iajb = lib.direct_sum(
#    "i+j-b-a->iajb",
#    mo_energy_occ,
#    mo_energy_occ,
#    mo_energy_vir,
#    mo_energy_vir,
#)

#print(s2)
#print(mf.mo_coeff[:,i].T @ cloppa_obj.fock_matrix_canonical @ mf.mo_coeff[:,i])
#print(mo_coeff[:,i].T @ cloppa_obj.fock_matrix_canonical @ mo_coeff[:,i])

print(hrpa_loc_obj.pp_ssc_fc_select([0],[1]))