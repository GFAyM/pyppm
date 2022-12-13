import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.pp_4c import Prop_pol 
from src.polaritization_propagator import Prop_pol as Prop_pol_NR
from pyscf import gto, scf

ang = 100



mol = gto.M(atom='''
O       0.0000000000            0.0000000000           -0.0409868122
H1       0.0000000000            0.7567917171            0.5640254210
H2       0.0000000000           -0.7567917171            0.5640254210
''', basis='631g', unit='angstrom')

#mol, ctr_coeff = mol.decontract_basis()
mf = scf.DHF(mol).newton()
mf.conv_tol_grad = 1e-6
mf.conv_tol = 1e-10
mf.kernel()

pp = Prop_pol(mf)

#print(pp.pp_ssc_4c_select([1],[2]))
j = pp.kernel_select([1],[2])
print(j)

mf = scf.DHF(mol).newton()
mf.conv_tol_grad = 1e-6
mf.conv_tol = 1e-10
mf.kernel()

pp = Prop_pol(mf)

#print(pp.pp_ssc_4c_select([1],[2]))
j = pp.kernel_select([1],[2])
print(j)

#mf = scf.RHF(mol).run()
#pp = Prop_pol_NR(mf)
#j = pp.kernel_select(FC=False,PSO=True, atom1=[1], atom2=[2])
#print(j)
