from pyscf import gto, scf
from pyscf.prop import ssc


mol = gto.M(atom = '''O 0 0 0; H  0 1 0; H 0 0 1''', basis='ccpvtz',unit='amstrong', verbose=0)
mf = scf.RHF(mol).run()

ssc = mf.SSC()
ssc.verbose = 4

ssc.with_fc = False
ssc.with_fcsd = True
ssc.cphf = True

ssc.kernel()
