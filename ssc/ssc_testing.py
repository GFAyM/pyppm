from pyscf import gto, scf
from pyscf.prop import ssc

mol = gto.M(atom='''
            O 0 0      0
            H 0 -0.757 0.587
            H 0  0.757 0.587''',
            basis='ccpvdz')

mf = scf.RHF(mol).run()

ssc = mf.SSC()
ssc.verbose = 4

ssc.with_fc = True
ssc.with_fcsd = True
    
ssc.kernel()
