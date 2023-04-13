import numpy as np
from pyscf import tools
from pyscf.tools import mo_mapping
import attr

@attr.s
class extra_functions:
    molden_file=attr.ib(default=None, type=str, validator=attr.validators.instance_of(str))

    def __attrs_post_init__(self):
        
        self.mol, mo_energy, self.mo_coeff, self.mo_occ, irrep_labels, spins =  tools.molden.load(self.molden_file)
        self.occidx = np.where(self.mo_occ>0)[0]
        self.viridx = np.where(self.mo_occ==0)[0]

        self.orbv = self.mo_coeff[:,self.viridx]
        self.orbo = self.mo_coeff[:,self.occidx]

        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]

    @property
    def extraer_coeff(self):
        """Function that extractes mol, mo_coeff and mo_occ of a molden file

        Returns:
            gto.mole, np.array, np.array: mole object, mo_coeff and mo_occ
        """
        self.mol, mo_energy, self.mo_coeff, self.mo_occ, irrep_labels, spins =  tools.molden.load(self.molden_file)
        return self.mol, self.mo_coeff, self.mo_occ

    def mo_hibridization(self, mo_label, lim1, lim2):
        """This function gives the orbital index of the MOs that are in the 
        hibridization range

        Args:
            mo_label (str): Which AO i watn to know the composition
            lim1 (int): inferior limit of the hibridization range
            lim2 (int): superior limit of the hibridization range
            cart (bool, optional): If the MOs are in cartesian coordinates,
            use True in cart. Otherwise, let unchange this value. Defaults to False.
            orth_method (str): The localization method to generated orthogonal AO upon which the AO contribution
            are computed. It can be one of ‘meta_lowdin’, ‘lowdin’ or ‘nao’.
        Returns:
            list : atm-id, hibridization coeff  
        """
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff, orth_method='meta_lowdin')
        orbital = np.array([])
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital=np.append(orbital, (i,c))
        return orbital

    def mo_hibridization_fixed(self, mo_label,fixed_orbital, lim1, lim2):
        """Evaluate the 'mo_label' composition of a fixed orbital of the molden
          file between two limits

        Args:
            mo_label (str): In terms of wich AO want to know the composition
            fixed_orbital (int): in which MO wan to know the composition
            lim1 (real): inferior limit of the hibridization range
            lim2 (real): superior limit of the hibridization range

        Returns:
            real: composition of 'mo_label' in a definite fixed_orbital
        """
        self.mol, self.mo_coeff, self.mo_occ = self.extraer_coeff
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff[:,[fixed_orbital]])
        orbital = np.array([])
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital=np.append(orbital, c)
        return orbital[0]



