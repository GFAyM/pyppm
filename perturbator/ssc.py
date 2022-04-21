from pyscf import gto, scf, dft
from pyscf.prop import ssc
mol = gto.M(atom='''
            O 0 0      0
            H 0 -0.757 0.587
            H 0  0.757 0.587''',
            basis='ccpvdz')

mf = scf.RHF(mol).run()
sc = ssc.RHF(mf)


print(sc)

fc = ssc.rhf.make_fc(sc)

print(fc)