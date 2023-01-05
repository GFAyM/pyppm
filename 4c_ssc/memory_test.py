from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.cc import RCCSD
from pyscf.lib import current_memory

import tracemalloc
tracemalloc.start()

mol = Mole()
mol.atom = '''
He  0.0 0.0 0.0
'''
mol.basis = 'def2-QZVPP'
mol.build()

mf = RHF(mol)
mf.verbose = 0
mf.kernel()

cc = RCCSD(mf)
cc.direct = True
cc.verbose = 4
cc.kernel()

t = cc.ccsd_t()

# total memory usage
print("total memory: %.1f MiB" % current_memory()[0])

# in-Python memory usage
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
total = sum(stat.size for stat in top_stats)
print("in-Python memory: %.1f MiB" % (total / 1048576))