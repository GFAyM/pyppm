from pyscf import gto, scf
from pyscf.prop import ssc

ang=100
mol = gto.M(atom='''
        O1   1
        O2   1 1.45198749
        H3   2 0.96592992  1 100.40878555
        H4   1 0.96592992  2 100.40878555  3 {}
        '''.format(ang*10), basis='ccpvtz', verbose=0)

mf = scf.RHF(mol).run()

ssc = mf.SSC()
ssc.verbose = 4

ssc.with_fc = True
#ssc.with_fcsd = True
ssc.cphf = True

ssc.kernel()
