import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.hrpa import Prop_pol 
from src.hrpa_andy import Prop_pol as Prop_pol_andy 

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
#print(mf.mo_energy[0]+mf.mo_energy[1]-mf.mo_energy[0])
pp = Prop_pol(mf)
#kappa, E = pp.kappa_full(1)
#print(kappa[0,:,:,:]==kappa[:,:,0,:].reshape(pp.nvir,pp.nocc,pp.nvir))
#inte = pp.kernel(atom1=[0], atom2=[1])
#print(inte)
#pp_andy = Prop_pol_andy(mf)
#print(pp_andy.kappa(1,0,1,5,4,mf.mo_energy))

### acá se testea la función part_A2, que es la ecuación C.13 de Oddershede 1984
#B2 = pp.part_b2(1)
#print(B2[:,:,:,:].sum())
#S = pp.S2
#print(S[0,0,:,:])
#print(A2[1,1,:,:])

#print(pp.kappa_2(4,5))
#print(pp.kappa_2_full())
print(pp.correction_pert_2([1]))