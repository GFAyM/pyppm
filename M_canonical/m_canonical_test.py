from tabnanny import verbose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyscf import scf, gto, tdscf, ao2mo

ang = 10

mol_H2O2 = '''
O1   1
O2   1 1.45643942
H3   2 0.97055295  1 99.79601616
H4   1 0.97055295  2 99.79601616  3 {}
'''.format(10*ang)

mol = gto.M(atom=mol_H2O2, basis='6-31G**', verbose=0)   
mf = scf.RHF(mol).run()
#m = pp(mf).m_matrix_triplet_otherway

mo_coeff = mf.mo_coeff


o1 = mo_coeff[:,[4]]
o2 = mo_coeff[:,[5]]

v1 = mo_coeff[:,[10]]
v2 = mo_coeff[:,[11]]
v3 = mo_coeff[:,[13]]


#int2e = ao2mo.general(mol, [occ,occ,vir,vir], compact=False)

orbv = mo_coeff[:,[10,11,13]]
orbo = mo_coeff[:,[4,5]]
nvir = orbv.shape[1]
nocc = orbo.shape[1]
mo = np.hstack((orbo,orbv))
nmo = nocc + nvir

eri_mo = ao2mo.general(mol, [orbo,mo,mo,mo], compact=False)
eri_mo = eri_mo.reshape(nocc,nmo,nmo,nmo)
a = np.einsum('ijba->iajb', eri_mo[:nocc,:nocc,nocc:,nocc:])
print(a.reshape((nocc*nvir,nocc*nvir), order='C'))

#the matrix element "a" has the form: [a,b,j,i] 

print(ao2mo.general(mol, [o1,o1,v1,v1], compact=False))
print(ao2mo.general(mol, [o1,o1,v1,v2], compact=False))
print(ao2mo.general(mol, [o1,o1,v1,v3], compact=False))

print(ao2mo.general(mol, [o2,o1,v1,v1], compact=False))
print(ao2mo.general(mol, [o2,o1,v1,v2], compact=False))
print(ao2mo.general(mol, [o2,o1,v1,v3], compact=False))


