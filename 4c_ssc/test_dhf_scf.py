from pyscf import gto, scf
from pyscf.tools import molden


mol = gto.Mole()
mol.atom = '''
H  0  0  0; I  0  0  2 
'''
mol.unit = 'Angstrom'
mol.basis = {'H':'ccpvdz', 'I':'dyall_dz'}
#mol.basis = 'ccpvdz'
#mol.verbose = 6
mol.build()
mf = scf.DHF(mol)
mf.scf()
#molden.from_mo(mol, 'prueba.molden', mf.mo_coeff)

