from pyscf import gto, scf
from pyscf.prop import ssc

ang=100
mol = gto.M(atom='''
        O1   1
        O2   1 1.45643942
        H3   2 0.97055295  1 99.79601616
        H4   1 0.97055295  2 99.79601616  3 {}
        '''.format(ang*10), basis='ccpvdz', verbose=0)

mf = scf.RHF(mol).run()

ssc = mf.SSC()
ssc.verbose = 4

ssc.with_fc = True
#ssc.with_fcsd = True
ssc.cphf = False

ssc.kernel()
