from pyscf import gto, scf, lib
from pyscf.tools import molden


mol = gto.Mole()
mol.atom = '''
H  0  0  0; Br  0  0  2 
'''
mol.unit = 'Angstrom'
mol.basis = {'H':'ccpvdz', 'Br':'dyall_dz'}
#mol.basis = 'ccpvdz'
#mol.verbose = 6
mol.build()
#mf = scf.DHF(mol)
#mf.scf()
#molden.from_mo(mol, 'prueba.molden', mf.mo_coeff)

lib.param.LIGHT_SPEED = 400
print('light speed = 400c')
mf = scf.DHF(mol)
mf.scf()

print('NR')
mf = scf.RHF(mol)
mf.scf()
