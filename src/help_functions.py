import numpy as np
from pyscf import gto, scf, lo, tools, tddft, tdscf, ao2mo
from pyscf.tools import molden, mo_mapping
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
        ''' Aquí extraigo los mol y mo_coeff de algún archivo molden '''
        self.mol, mo_energy, self.mo_coeff, self.mo_occ, irrep_labels, spins =  tools.molden.load(self.molden_file)
        return self.mol, self.mo_coeff, self.mo_occ

    def mo_hibridization(self, mo_label, lim1, lim2, occ=False,cart=False, orth_method='meta_lowdin'):
        """This function gives the orbital index of the MOs that are in the hibridization range

        Args:
            mo_label (str): Which AO i watn to know the composition
            lim1 (int): inferior limit of the hibridization range
            lim2 (int): superior limit of the hibridization range
            cart (bool, optional): If the MOs are in cartesian coordinates,
            use True in cart. Otherwise, let unchange this value. Defaults to False.
            orth_method (str): The localization method to generated orthogonal AO upon which the AO contribution
            are computed. It can be one of ‘meta_lowdin’, ‘lowdin’ or ‘nao’.
        Returns:
            _type_: _description_
        """
        #if occ==True:
        #    comp = mo_mapping.mo_comps(mo_label, self.mol, self.orbo, cart=cart,orth_method='meta_lowdin')
        #else:
        #    comp = mo_mapping.mo_comps(mo_label, self.mol, self.orbv, cart=cart,orth_method='meta_lowdin')
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff, cart=cart,orth_method='meta_lowdin')
        orbital = np.array([])
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital=np.append(orbital, (i,c))
        return orbital

    def mo_hibridization_fixed(self, mo_label,fixed_orbital, lim1, lim2, cart=False,orth_method='meta_lowdin'):
        self.mol, self.mo_coeff, self.mo_occ = self.extraer_coeff
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff[:,[fixed_orbital]], cart=cart)
        orbital = np.array([])
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital=np.append(orbital, c)
        return orbital

    def mo_hibridization_for_list(self,mo_label, lim1, lim2, occ=False,cart=False,orth_method='meta_lowdin'):
        #if occ == True:
        #    comp = mo_mapping.mo_comps(mo_label, self.mol, self.orbo, cart=cart)
        #else:
        #    comp = mo_mapping.mo_comps(mo_label, self.mol, self.orbv, cart=cart)
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff, cart=cart)
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital=i
        return orbital 

    def mo_hibridization_for_list_several(self,mo_label, lim1, lim2, cart=False,orth_method='meta_lowdin'):
        self.mol, self.mo_coeff, self.mo_occ = self.extraer_coeff
        comp = mo_mapping.mo_comps(mo_label, self.mol, self.mo_coeff, cart=cart)
        orbital = []
        for i,c in enumerate(comp):
            if lim1 < c < lim2:
                orbital.append(i)
        return orbital

    def mo_hibridization_two_aos(self, mo_label1,mo_label2, lim1_1, lim1_2, lim2_1, lim2_2,
                                lim1_tot, lim2_tot,
                                occ=False,cart=False, orth_method='meta_lowdin'):
        comp1 = mo_mapping.mo_comps(mo_label1, self.mol, self.mo_coeff, cart=cart,orth_method='meta_lowdin')
        orbital = np.array([])
        for i1,c1 in enumerate(comp1):
            if lim1_1 < c1 < lim1_2:
                comp2 = mo_mapping.mo_comps(mo_label2, self.mol, self.mo_coeff[:,[i1]], cart=cart,orth_method='meta_lowdin')
                for i2,c2 in enumerate(comp2):
                    if lim2_1 < c2 < lim2_2:
                        if lim1_tot < c1+c2 < lim2_tot:
                            orbital=np.append(orbital, (i1,c2+c1))
                
        return orbital