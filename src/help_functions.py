import numpy as np
from pyscf import gto, scf, lo, tools, tddft, tdscf, ao2mo
from pyscf.tools import molden, mo_mapping
import attr

@attr.s
class extra_functions:
    molden_file=attr.ib(default=None, type=str, validator=attr.validators.instance_of(str))

    @property
    def extraer_coeff(self):
        ''' Aquí extraigo los mol y mo_coeff de algún archivo molden '''
        self.mol, mo_energy, self.mo_coeff, self.mo_occ, irrep_labels, spins =  tools.molden.load(self.molden_file)
        return self.mol, self.mo_coeff, self.mo_occ

    def mo_hibridization(self, mo_label, lim1, lim2):
        self.mol, self.mo_coeff,self.mo_occ = self.extraer_coeff
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff)
        orbital = np.array([])
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital=np.append(orbital, (i,c))
        return orbital

    def mo_hibridization_fixed(self, mo_label,fixed_orbital, lim1, lim2):
        self.mol, self.mo_coeff, self.mo_occ = self.extraer_coeff
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff[:,[fixed_orbital]])
        orbital = np.array([])
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital=np.append(orbital, c)
        return orbital

    def mo_hibridization_for_list(self,mo_label, lim1, lim2):
        self.mol, self.mo_coeff, self.mo_occ = self.extraer_coeff
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff)
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital=i
        return orbital

    def mo_hibridization_for_list_several(self,mo_label, lim1, lim2):
        self.mol, self.mo_coeff, self.mo_occ = self.extraer_coeff
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff)
        orbital = []
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital.append(i)
        return orbital


#los orbitales que usaremos son aquellos con contribución de F 2pz mayor a 0.1

