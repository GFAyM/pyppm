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
from pyscf import scf, gto
import numpy as np

mol, mo_coeff_loc, mo_occ = extra_functions(molden_file='C2H6_ccpvdz_Pipek_Mezey.molden').extraer_coeff
mol, mo_coeff_loc, mo_occ = extra_functions(molden_file='ethane.molden').extraer_coeff

mf = scf.RHF(mol)
mf.kernel()
pp = Prop_pol(mf)
m = pp.M(triplet=False)
nocc = pp.nocc
nvir = pp.nvir

can_inv = np.linalg.inv(mf.mo_coeff.T)
c_occ = (mo_coeff_loc[:,:nocc].T.dot(can_inv[:,:nocc])).T

c_vir = (mo_coeff_loc[:,nocc:].T.dot(can_inv[:,nocc:])).T
total = np.einsum('ij,ab->iajb',c_occ,c_vir)
total = total.reshape(nocc*nvir,nocc*nvir)
p = np.linalg.inv(m)
p = p.reshape(nocc, nvir, nocc, nvir)
m_loc = total.T @ m @ total
#m = m.reshape(nocc, nvir, nocc, nvir)
#m_loc = np.einsum('ij,ab,iajb,ji,ba->iajb',c_occ,c_vir,m,c_occ.T,c_vir.T)
#m_loc = np.einsum('iajb,iajb,aibj->iajb',total,m,total.T)
#m_loc = m
m_loc = m_loc.reshape(nocc* nvir, nocc* nvir)
p_loc = np.linalg.inv(m_loc)
p_loc = p_loc.reshape(nocc, nvir, nocc, nvir)
#print(np.diag(m_loc).sum())
#print(np.diag(m).sum())
h1 = pp.pert_pso([2])[2][:nocc,nocc:]
h2 = pp.pert_pso([4])[2][:nocc,nocc:]

#print((h1**2).sum())
h1_loc = c_occ.T@h1@c_vir
h2_loc = c_occ.T@h2@c_vir

p = np.einsum('ia,iajb,jb', h1, p, h2)
p_loc = np.einsum('ia,iajb,jb',h1_loc,p_loc,h2_loc)
print(p_loc)
print(p)