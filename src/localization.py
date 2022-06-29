from pyscf import gto, scf
from pyscf.gto import Mole
from pyscf.lo.edmiston import ER
from pyscf.lo.ibo import PM, Pipek
from pyscf.scf import RHF
from pyscf.lo import Boys
from pyscf.lo import PipekMezey
from pyscf.lo import EdmistonRuedenberg
from pyscf.lo.cholesky import cholesky_mos
from pyscf.tools import molden, mo_mapping
import numpy as np
import attr
from src.help_functions import extra_functions

@attr.s
class localization:
    """
    Clase para localizar los LMOs de algún sistema molecular definido utilizando la matriz Z,
    y variando su ángulo diedro utilizando dos métodos de localización consecutivamente
    """
    first_loc = attr.ib(default="Cholesky", type=str, validator=attr.validators.in_(["Cholesky", "PM", "Boys"]))
    no_second_loc = attr.ib(default=True, type=bool)
    first_max_stepsize = attr.ib(default=0.05, type=float)
    first_conv_tol = attr.ib(default=1e-6, type=float)
    first_iters = attr.ib(default=50, type=int)
    pm_pop_method_first = attr.ib(default="meta-lowdin", type=str, validator=attr.validators.in_(["lowdin", "becke", "mulliken","meta-lowdin"]))
    second_loc = attr.ib(default="PM", type=str, validator=attr.validators.in_(["PM", "Boys"]))
    second_max_stepsize = attr.ib(default=0.01, type=float)
    second_conv_tol = attr.ib(default=1e-6, type=float)
    second_iters = attr.ib(default=50, type=int)
    pm_pop_method_second = attr.ib(default="meta-lowdin", type=str, validator=attr.validators.in_(["lowdin", "becke", "mulliken","meta-lowdin"]))
    mol_input = attr.ib(default=None, type=str)
    dihedral_angle = attr.ib(default=None, type=int)
    basis = attr.ib(default='cc-pvdz', type=str)
    molecule_name = attr.ib(default=None, type=str)
    atom_for_searching1 = attr.ib(default=None, type=str)
    atom_for_searching2 = attr.ib(default=None, type=str)


    def __attrs_post_init__(self):
        self.mol = gto.M(atom=str(self.mol_input), basis=self.basis)

    @property
    def localiza_first(self):
        mf = scf.RHF(self.mol).run()
        nocc = np.count_nonzero(mf.mo_occ > 0)

        if self.first_loc=="Cholesky":
            lmo_occ_first = cholesky_mos(mf.mo_coeff[:, :nocc])
            lmo_virt_first = cholesky_mos(mf.mo_coeff[:, nocc:])

        elif self.first_loc=="PM":
            #PipekMezey.pop_method = self.pm_pop_method_first
            #PipekMezey.conv_tol = self.first_conv_tol
            #PipekMezey.max_stepsize = self.first_max_stepsize
            #PipekMezey.max_iters = self.first_iters
            #PipekMezey.init_guess = None
            lmo_occ_first = PipekMezey(self.mol).kernel(mf.mo_coeff[:, :nocc])
            lmo_virt_first = PipekMezey(self.mol).kernel(mf.mo_coeff[:, nocc:])
            
        elif self.first_loc=="Boys":
            #Boys.conv_tol = self.first_conv_tol
            #Boys.max_iters = self.first_iters
            #Boys.max_stepsize = self.first_max_stepsize
            lmo_occ_first = Boys(self.mol).kernel(mf.mo_coeff[:, :nocc])
            lmo_virt_first = Boys(self.mol).kernel(mf.mo_coeff[:, nocc:])

        if self.no_second_loc == True:
            return np.hstack((lmo_occ_first, lmo_virt_first))            
        elif self.no_second_loc==False:
            return lmo_occ_first, lmo_virt_first
    
    @property
    def localiza_second(self):
        lmo_occ_first, lmo_virt_first = self.localiza_first
        
        if self.second_loc=="PM":
            PipekMezey.pop_method = self.pm_pop_method_second
            PipekMezey.conv_tol = self.second_conv_tol
            PipekMezey.max_stepsize = self.second_max_stepsize
            PipekMezey.max_iters = self.second_iters
            #PipekMezey.init_guess = None
            lmo_occ_second = PipekMezey(self.mol).kernel(lmo_occ_first)
            #PipekMezey.init_guess = None
            lmo_virt_second = PipekMezey(self.mol).kernel(lmo_virt_first)
            lmo_merged = np.hstack((lmo_occ_second, lmo_virt_second))

        elif self.second_loc=="Boys":
            Boys.conv_tol = self.second_conv_tol
            Boys.max_iters = self.second_iters
            Boys.max_stepsize = self.second_max_stepsize
            lmo_occ_second = Boys(self.mol).kernel(lmo_occ_first)
            lmo_virt_second = Boys(self.mol).kernel(lmo_virt_first)
            lmo_merged = np.hstack((lmo_occ_second, lmo_virt_second))

        return lmo_merged
    
    @property
    def kernel(self):
        #for ang in range(ang_init, ang_second, 10):
        if self.no_second_loc == True:
            lmo_merged = self.localiza_first
            #self.filename = str('{}_{}_{}_{}.molden').format(self.molecule_name, self.basis, self.dihedral_angle, self.first_loc)
            self.filename = f'{self.molecule_name}_{self.dihedral_angle}_{self.basis}_{self.first_loc}.molden'
            
            print('Dumping the orbitals in file:', self.filename)
            molden.from_mo(self.mol, self.filename, lmo_merged)
            orbitals1 = extra_functions(molden_file=self.filename).mo_hibridization(self.atom_for_searching1, 0.1,1)
            orbitals2 = extra_functions(molden_file=self.filename).mo_hibridization(self.atom_for_searching2, 0.1,1)
            print(orbitals1)
            print(orbitals2)

        elif self.no_second_loc == False:
            lmo_merged = self.localiza_second
            #self.filename = str('{}_{}_{}_{}_{}.molden').format(self.molecule_name, self.basis, self.dihedral_angle, self.first_loc, self.second_loc)
            self.filename = f'{self.molecule_name}_{self.dihedral_angle}_{self.basis}_{self.first_loc}_{self.second_loc}.molden'
            
            print('Dumping the orbitals in file:', self.filename)
            molden.from_mo(self.mol, self.filename, lmo_merged)            
            orbitals1 = extra_functions(molden_file=self.filename).mo_hibridization(self.atom_for_searching1, 0.1,1)
            orbitals2 = extra_functions(molden_file=self.filename).mo_hibridization(self.atom_for_searching2, 0.1,1)           
            print(orbitals1)
            print(orbitals2)