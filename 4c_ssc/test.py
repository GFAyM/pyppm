import os
import sys
from pyscf import gto, scf, lib
from pyscf import gto
import time

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)
from src.pp_4c_2 import Prop_pol 
from pyscf import gto, scf
from pyscf.lib import current_memory


mol_h2s = gto.M(atom='''
S      .0000000000        0.0000000000        -.2249058930
H1   -1.4523499293         .0000000000         .8996235720
H2    1.4523499293         .0000000000         .8996235720
''', basis='cc-pvtz', unit='bhor', verbose=3)

#mol = gto.Mole()
#mol.atom='''
#S     0.000000000000   0.000000000000  -1.224905893000
#H1   -1.452349929300   0.000000000000   0.899623572000
#H2    1.452349929300   0.000000000000   0.899623572000
#'''
#mol.basis = 'cc-pvdz'
#mol.unit = 'B'
#mol.verbose=4
#mol.build()
#lib.param.LIGHT_SPEED = 300

#ang = 60
#mol_ethane= gto.Mole()
#mol_ethane.atom = f'''
#        C1   1
#        C2   1 1.525063
#        H1   2 1.092850    1  111.256
#        H2   2 1.092850    1  111.256  3  120 
#        H3   2 1.092850    1  111.256  3 -120
#        H4   1 1.092850    2  111.256  3  {ang+120}
#        H5   1 1.092850    2  111.256  3  {ang}
#        H6   1 1.092850    2  111.256  3  {ang-120}
#        '''
#mol_ethane.verbose=4

#mol_ethane.basis = '6-31G'
#mol_ethane.max_memory = 4000
#mol_ethane.build()

rhf = scf.DHF(mol_h2s).run()
pp = Prop_pol(rhf)

j = pp.kernel()