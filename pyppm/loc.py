from pyscf import scf
import numpy
from pyscf import lib
import attr
from pyscf.data.gyro import get_nuc_g_factor
from pyppm.hrpa import HRPA
from pyppm.rpa import RPA
from pyscf.data import nist
import numpy
from itertools import product


@attr.s
class Loc:
    """Class to perform calculations of $J^{FC}$ mechanism at at RPA and HRPA level of
    of approach using previously localized molecular orbitals. 
    Inspired in Andy Danian Zapata HRPA program
    attributes:
    mf = RHF object
    mo_coeff_loc = localized molecular orbitals
    elec_corr = str with RPA or HRPA. This defines if the correlation level is RPA or HRPA.
    """
    mf = attr.ib(
        default=None, type=scf.hf.RHF, validator=attr.validators.instance_of(scf.hf.RHF)
    )
    mo_coeff_loc = attr.ib(default=None, type=numpy.ndarray,
                validator=attr.validators.instance_of(numpy.ndarray)
    )
    elec_corr = attr.ib(default='RPA', type=str, 
                        validator=attr.validators.instance_of(str))

    def __attrs_post_init__(self):
        self.mo_occ = self.mf.mo_occ
        self.mo_energy = self.mf.mo_energy
        self.mo_coeff = self.mf.mo_coeff
        self.mol = self.mf.mol
        self.occidx = numpy.where(self.mo_occ > 0)[0]
        self.viridx = numpy.where(self.mo_occ == 0)[0]
        self.orbv = self.mo_coeff[:, self.viridx]
        self.orbo = self.mo_coeff[:, self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        self.mol = self.mf.mol
        self.mo = numpy.hstack((self.orbo, self.orbv))
        self.occ = [i for i in range(self.nocc)]
        self.vir = [i for i in range(self.nvir)]
        self.rpa_obj = RPA(mf=self.mf)
        if self.elec_corr == 'RPA':
            self.obj = RPA(mf=self.mf)
        elif self.elec_corr == 'HRPA':
            self.obj = HRPA(mf=self.mf)
        else:
            raise Exception('SOPPA or other method are not avaible yet. Only RPA and HRPA')


    @property
    def inv_mat(self):
        """Property than obtain the unitary transformation matrix
        Bouman, T. D., Voigt, B., & Hansen, A. E. (1979) JACS
        """
        mf = self.mf
        mo_coeff_loc = self.mo_coeff_loc
        nocc = self.nocc
        nvir = self.nvir
        can_inv = numpy.linalg.inv(mf.mo_coeff.T)
        c_occ = (mo_coeff_loc[:,:nocc].T.dot(can_inv[:,:nocc])).T

        c_vir = (mo_coeff_loc[:,nocc:].T.dot(can_inv[:,nocc:])).T
        total = numpy.einsum('ij,ab->iajb',c_occ,c_vir)
        total = total.reshape(nocc*nvir,nocc*nvir)
        return c_occ, total, c_vir
    
    def pp(self,atom1, atom2, FC=False, PSO=False, FCSD=False):
        """Fuction that localize perturbators and principal propagator inverse
        of a chosen mechanism
        Args:
            atom1 (str): atom1 label
            atom2 (str): atom2 label
            FC (bool, optional): If true, returns elements from FC mechanisms
            PSO (bool, optional): If true, returns elements for PSO mechanisms
            FCSD (bool, optional): If true, returns elements for FC+SD mechanisms

        Returns:
            h1_loc, m_loc, h2_loc: numpy.ndarrays with perturbators and principal 
            propagator inverse in the localized basis
        """
        atom1_ = [self.rpa_obj.obtain_atom_order(atom1)]
        atom2_ = [self.rpa_obj.obtain_atom_order(atom2)]
        obj = self.obj
        if FC:
            h1, m, h2 = obj.elements(atom1_, atom2_, FC=True )
        if FCSD:
            h1, m, h2 = obj.elements(atom1_, atom2_, FCSD=True )
        if PSO:
            h1, m, h2 = obj.elements(atom1_, atom2_, PSO=True)
        c_occ, total, c_vir = self.inv_mat
        h1_loc = c_occ.T@h1@c_vir
        h2_loc = c_occ.T@h2@c_vir
        m_loc = total.T @ m @ total
        return h1_loc, m_loc, h2_loc


    def ssc(self, atom1=None, atom2=None, FC=False, PSO=False, FCSD=False ):
        """Function that obtains ssc mechanism for two chosen atoms in the localized
        basis

        Args:
            atom1 (str): atom1 label
            atom2 (str): atom2 label
            FC (bool, optional): If true, returs fc-ssc. Defaults to False.
            PSO (bool, optional): If true, returs pso-ssc. Defaults to False
            FCSD (bool, optional): If true, returs fcsd-ssc. Defaults to False

        Returns:
            real: ssc mechanism
        """
        nocc = self.nocc
        nvir = self.nvir

        if FC:
            h1, m, h2 = self.pp(FC=True,atom1=atom1,atom2=atom2)
            p = -numpy.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = lib.einsum("ia,iajb,jb", h1, p, h2)
            para.append(e/4)
            prop = lib.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))
        if PSO:
            h1, m, h2 = self.pp(atom1=atom1, atom2=atom2, PSO=True)
            p = -numpy.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = lib.einsum('xia,iajb,yjb->xy', h1, p , h2)
            para.append(e)
            prop = numpy.asarray(para) * nist.ALPHA ** 4
        elif FCSD:
            h1, m, h2 = self.pp(FCSD=True,atom1=atom1,atom2=atom2)
            p = -numpy.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            e = numpy.einsum("wxia,iajb,wyjb->xy", h1, p, h2)
            para.append(e)
            prop = numpy.asarray(para) * nist.ALPHA ** 4
        
        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton ** 2
        iso_ssc = unit * lib.einsum("kii->k", prop) / 3
        atom1_ = [self.rpa_obj.obtain_atom_order(atom1)]
        atom2_ = [self.rpa_obj.obtain_atom_order(atom2)]
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        jtensor = lib.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]
    
    def ssc_pathways(self, atom1=None, atom2=None, 
                              h1=None, m=None, h2=None, 
                              occ_atom1=None, vir_atom1=None,
                              occ_atom2=None, vir_atom2=None):
        """Function that obtains coupling pathways between two couple of exitations
        or a set of them. The shape of perturbator claims which mechanism is.
        For this function, you must introduce the perturbators and principal 
        propagators previously calculated with "pp_loc" function, in order to only
        calculate it once, and then evaluate each coupling pathway.

        Args:
            atom1 (str): atom1 label
            atom2 (str): atom2 label
            h1 (np.array): perturbator centered in atom1
            m (np.array): principal propagator inverse
            h2 (np.array): perturbator centeder in atom2
            occ_atom1 (list): list with occupied LMOs centered on atom1
            vir_atom1 (list): list with virtual LMOs centered on atom1
            occ_atom2 (list): list with occupied LMOs centered on atom2
            vir_atom2 (list): list with virtual LMOs centered on atom2

        Returns:
            real: ssc mechanism for the coupling pathway defined for the LMOs
        """
        nocc = self.nocc
        nvir = self.nvir
        atom1_ = [self.rpa_obj.obtain_atom_order(atom1)]
        atom2_ = [self.rpa_obj.obtain_atom_order(atom2)]
        para = []
        if len(h1.shape)==2:
            h1_pathway = numpy.zeros(h1.shape)
            h2_pathway = numpy.zeros(h1.shape)

            p = -numpy.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            
            if vir_atom1 is None:
                h1_pathway[occ_atom1,:] += h1[occ_atom1,: ]
                h2_pathway[occ_atom2,: ] += h2[ occ_atom2,:]
            else:
                vir_atom1 = [i-nocc for i in vir_atom1]
                vir_atom2 = [i-nocc for i in vir_atom2]
                for i,a in list(product(occ_atom1,vir_atom1)):
                    h1_pathway[i, a] += h1[i, a]
                for j,b in list(product(occ_atom1,vir_atom1)):
                    h2_pathway[j,b] += h2[j, b]
            e = lib.einsum("ia,iajb,jb", h1_pathway, p, h2_pathway)
            para.append(e/4)
            prop = lib.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))
        if len(h1.shape)==3:
            h1_pathway = numpy.zeros(h1.shape)
            h2_pathway = numpy.zeros(h1.shape)

            p = -numpy.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            if vir_atom1 is None:
                h1_pathway[:,occ_atom1,:] += h1[:,occ_atom1,: ]
                h2_pathway[:,occ_atom2,: ] += h2[:,occ_atom2,:]
            else:
                vir_atom1 = [i-nocc for i in vir_atom1]
                vir_atom2 = [i-nocc for i in vir_atom2]
                for i,a in list(product(occ_atom1,vir_atom1)):
                    h1_pathway[:,i, a] += h1[:,i, a]
                for j,b in list(product(occ_atom1,vir_atom1)):
                    h2_pathway[:,j,b] += h2[:,j, b]
                       
            e = lib.einsum('xia,iajb,yjb->xy', h1_pathway, p , h2_pathway)
            para.append(e)
            prop = numpy.asarray(para) * nist.ALPHA ** 4
        if len(h1.shape)==4:
            h1_pathway = numpy.zeros(h1.shape)
            h2_pathway = numpy.zeros(h1.shape)

            p = -numpy.linalg.inv(m)
            p = p.reshape(nocc, nvir, nocc, nvir)
            para = []
            if vir_atom1 is None:
                h1_pathway[:,:,occ_atom1,:] += h1[:,:,occ_atom1,: ]
                h2_pathway[:,:,occ_atom2,: ] += h2[:,:,occ_atom2,:]
            else:
                vir_atom1 = [i-nocc for i in vir_atom1]
                vir_atom2 = [i-nocc for i in vir_atom2]
                for i,a in list(product(occ_atom1,vir_atom1)):
                    h1_pathway[:,:,i, a] += h1[:,:,i, a]
                for j,b in list(product(occ_atom1,vir_atom1)):
                    h2_pathway[:,:,j,b] += h2[:,:,j, b]
            e = lib.einsum("wxia,iajb,wyjb->xy", h1_pathway, p, h2_pathway)
            para.append(e)
            prop = numpy.asarray(para) * nist.ALPHA ** 4


        nuc_magneton = 0.5 * (nist.E_MASS / nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton ** 2
        iso_ssc = unit * lib.einsum("kii->k", prop) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atom1_[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atom2_[0]))]
        jtensor = lib.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)
        return jtensor[0]