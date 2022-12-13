import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.hrpa_andy import Prop_pol 
from pyscf import gto, scf
import numpy as np



mol = gto.M(atom='''
F     0.0000000000    0.0000000000     0.1319629808
H     0.0000000000    0.0000000000    -1.6902522555
''', basis='321g', unit='angstrom')

#mol, ctr_coeff = mol.decontract_basis()

#mf = mol.RHF().run()
#mf.MP2().run()

mf = scf.RHF(mol)
mf.kernel()

#mf = scf.RHF(mol).run()
pp = Prop_pol(mf)
#print(pp.nocc, pp.nvir)
#pert = pp.pert_fc([0])[0]
#print(pert.shape)
#pert_corr = pp.pert_corr(pert, 1)
#print(pert_corr.shape)
#print(pert_corr.reshape(pp.nocc,pp.nvir))

inte = pp.kernel(atom1=[0], atom2=[1])
print(inte)


