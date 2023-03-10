from pyscf import gto, scf
from pyscf.gto import Mole
import numpy 
from pyscf import lib
import attr
from pyscf import ao2mo
from pyscf.dft import numint
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
from pyscf import tools
from pyscf.lib import logger
import sys
from functools import reduce

def uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

@attr.s
class Prop_pol:
    """_summary_
    """
    mf = attr.ib(default=None, type=scf.hf.RHF, validator=attr.validators.instance_of(scf.hf.RHF))

    def __attrs_post_init__(self):
        self.mo_occ = self.mf.mo_occ
        self.mo_energy = self.mf.mo_energy
        self.mo_coeff = self.mf.mo_coeff
        self.mol = self.mf.mol    
        self.nuc_pair = [(i,j) for i in range(self.mol.natm) for j in range(i)]
        self.occidx = numpy.where(self.mo_occ>0)[0]
        self.viridx = numpy.where(self.mo_occ==0)[0]
        self.orbv = self.mo_coeff[:,self.viridx]
        self.orbo = self.mo_coeff[:,self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]

        self.atm1dic, self.atm2dic = uniq_atoms(nuc_pair=self.nuc_pair)

    #@property
    def M(self, triplet=True):
        r'''A and B matrices for TDDFT response function.

        A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)
        '''
        #nao, nmo = self.mo_coeff.shape
        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir


        e_ia = lib.direct_sum('a-i->ia', self.mo_energy[self.viridx], self.mo_energy[self.occidx])
        a = numpy.diag(e_ia.ravel()).reshape(self.nocc,self.nvir,self.nocc,self.nvir)
        b = numpy.zeros_like(a)

        eri_mo = ao2mo.general(self.mol, [self.orbo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(self.nocc,nmo,nmo,nmo)
        a -= numpy.einsum('ijba->iajb', eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
        if triplet: 
            b -= numpy.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
        elif not triplet:
            b += numpy.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
        m = a + b
        m = m.reshape(self.nocc*self.nvir,self.nocc*self.nvir, order='C')
        
        return m

    def pert_fc(self,atmlst):
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mol = self.mol
        coords = mol.atom_coords()
        ao = numint.eval_ao(mol, coords)
        mo = ao.dot(mo_coeff)
        orbo = mo[:,mo_occ> 0]
        orbv = mo[:,mo_occ==0]
        fac = 8*numpy.pi/3 *.5  # *.5 due to s = 1/2 * pauli-matrix
        h1 = []
        for ia in atmlst:
            h1.append(fac * numpy.einsum('p,i->pi', orbv[ia], orbo[ia]))
        return h1



    @property
    def pp_ssc_fc(self):
        
        nvir = self.nvir
        nocc = self.nocc

        h1 = self.pert_fc(sorted(self.atm1dic))
        h2 = self.pert_fc(sorted(self.atm2dic))    
        m = self.M(triplet=True)
        p = numpy.linalg.inv(m)
        p = -p.reshape(nocc,nvir,nocc,nvir)
        para = []
        for i,j in self.nuc_pair:
            at1 = self.atm1dic[i]
            at2 = self.atm2dic[j]
            e = numpy.einsum('ia,iajb,jb', h1[at1].T, p , h2[at2].T)
            #print(e)
            para.append(e*4)  # *4 for +c.c. and for double occupancy
            
        fc = numpy.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))    
        return fc

    def pp_ssc_fc_select(self,atom1,atom2):
        
        nvir = self.nvir
        nocc = self.nocc

        h1 = self.pert_fc(atom1)
        h2 = self.pert_fc(atom2)    
        m = self.M(triplet=True)
        p = numpy.linalg.inv(m)
        p = -p.reshape(nocc,nvir,nocc,nvir)
        para = []
        e = numpy.einsum('ia,iajb,jb', h1[0].T, p , h2[0].T)
        #print(e)
        para.append(e*4)  # *4 for +c.c. and for double occupancy
            
        fc = numpy.einsum(',k,xy->kxy', nist.ALPHA**4, para, numpy.eye(3))    
        return fc

    def pert_fcsd(self, atmlst):
        '''MO integrals for FC + SD'''
        orbo = self.mo_coeff[:,self.mo_occ> 0]
        orbv = self.mo_coeff[:,self.mo_occ==0]

        h1 = []
        for ia in atmlst:
            h1ao = self._get_integrals_fcsd(ia)
            for i in range(3):
                for j in range(3):
                    h1.append(orbv.T.conj().dot(h1ao[i,j]).dot(orbo) * .5)
        return h1

    def _get_integrals_fcsd(self, atm_id):
        '''AO integrals for FC + SD'''
        mol = self.mol
        nao = mol.nao
        with mol.with_rinv_origin(mol.atom_coord(atm_id)):
            # Note the fermi-contact part is different to the fermi-contact
            # operator in HFC, as well as the FC operator in EFG.
            # FC here is associated to the the integrals of
            # (-\nabla \nabla 1/r + I_3x3 \nabla\dot\nabla 1/r), which includes the
            # contribution of Poisson equation twice, i.e. 8\pi rho.
            # Therefore, -1./3 * (8\pi rho) is used as the contact contribution in
            # function _get_integrals_fc to remove the FC part.
            # In HFC or EFG, the factor of FC part is 4\pi/3.
            a01p = mol.intor('int1e_sa01sp', 12).reshape(3,4,nao,nao)
            h1ao = -(a01p[:,:3] + a01p[:,:3].transpose(0,1,3,2))
        return h1ao

    def pert_pso(self, atmlst):
        # Imaginary part of H01 operator
        # 1/2(A01 dot p + p dot A01) => (a01p + c.c.)/2 ~ <a01p>
        # Im[A01 dot p] = Im[vec{r}/r^3 x vec{p}] = Im[-i p (1/r) x p] = -p (1/r) x p
        orbo = self.orbo
        orbv = self.orbv

        h1 = []
        for ia in atmlst:
            self.mol.set_rinv_origin(self.mol.atom_coord(ia))
            h1ao = -self.mol.intor_asymmetric('int1e_prinvxp', 3)
            h1 += [reduce(numpy.dot, (orbv.T.conj(), x, orbo)) for x in h1ao]
        return h1

    @property
    def pp_ssc_pso(self):
        nuc_pair = self.nuc_pair
        para = []
        nocc = numpy.count_nonzero(self.mo_occ> 0)
        nvir = numpy.count_nonzero(self.mo_occ==0)
        atm1lst = sorted(set([i for i,j in nuc_pair]))
        atm2lst = sorted(set([j for i,j in nuc_pair]))
        atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
        atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
        #mo1 = mo1.reshape(len(atm1lst),3,nvir,nocc)
        m = self.M(triplet=False)
        p = numpy.linalg.inv(m)
        p = -p.reshape(nocc,nvir,nocc,nvir)
        h1 = self.pert_pso(atm1lst)
        h1 = numpy.asarray(h1).reshape(len(atm1lst),3,nvir,nocc)
        h2 = self.pert_pso(atm2lst)
        h2 = numpy.asarray(h2).reshape(len(atm2lst),3,nvir,nocc)
        for i,j in nuc_pair:
            # PSO = -Tr(Im[h1_ov], Im[mo1_vo]) + cc = 2 * Tr(Im[h1_vo], Im[mo1_vo])
            e = numpy.einsum('iax,iajb,jby->xy', h1[atm1dic[i]].T, p, h2[atm2dic[j]].T)
            para.append(e*4)  # *4 for +c.c. and double occupnacy
        pso = numpy.asarray(para) * nist.ALPHA**4
        return pso
        #return h1[0].T.shape
    
    def pp_ssc_pso_select(self, atom1, atom2):
        para = []
        nvir = self.nvir
        nocc = self.nocc
        #mo1 = mo1.reshape(len(atm1lst),3,nvir,nocc)
        m = self.M(triplet=False)
        p = numpy.linalg.inv(m)
        
        p = -p.reshape(nocc,nvir,nocc,nvir)
        h1 = self.pert_pso(atom1)
        h1 = numpy.asarray(h1).reshape(1,3,nvir,nocc)
        h2 = self.pert_pso(atom2)
        h2 = numpy.asarray(h2).reshape(1,3,nvir,nocc)
        print(h1.shape)
        print(h1[0][2][0,0])
        print(p[0,0,0,0])
        print(h2[0][2][0,0])
            # PSO = -Tr(Im[h1_ov], Im[mo1_vo]) + cc = 2 * Tr(Im[h1_vo], Im[mo1_vo])
        e = numpy.einsum('iax,iajb,jby->xy', h1[0].T, p, h2[0].T)
        para.append(e*4)  # *4 for +c.c. and double occupnacy
        pso = numpy.asarray(para) * nist.ALPHA**4
        return pso

    @property
    def pp_ssc_fcsd(self):
        nvir = self.nvir
        nocc = self.nocc

        h1 = self.pert_fcsd(sorted(self.atm1dic.keys()))
        h1  = numpy.asarray(h1).reshape(-1,3,3,nvir,nocc)
        h2 = self.pert_fcsd(sorted(self.atm2dic.keys()))
        h2  = numpy.asarray(h2).reshape(-1,3,3,nvir,nocc)    
        m = self.M(triplet=True)
        p = numpy.linalg.inv(m)
        p = -p.reshape(nocc,nvir,nocc,nvir)
        para = []
        for i,j in self.nuc_pair:
            at1 = self.atm1dic[i]
            at2 = self.atm2dic[j]
            e = numpy.einsum('iawx,iajb,jbwy->xy', h1[at1].T, p , h2[at2].T)
            para.append(e*4)
        fcsd = numpy.asarray(para) * nist.ALPHA**4
        return fcsd

    def pp_ssc_fcsd_select(self,atom1,atom2):
        nvir = self.nvir
        nocc = self.nocc

        h1 = self.pert_fcsd(atom1)
        h1  = numpy.asarray(h1).reshape(-1,3,3,nvir,nocc)
        h2 = self.pert_fcsd(atom2)
        h2  = numpy.asarray(h2).reshape(-1,3,3,nvir,nocc)    
        m = self.M(triplet=True)
        p = numpy.linalg.inv(m)
        p = -p.reshape(nocc,nvir,nocc,nvir)
        para = []
        e = numpy.einsum('iawx,iajb,jbwy->xy', h1[0].T, p , h2[0].T)
        para.append(e*4)
        fcsd = numpy.asarray(para) * nist.ALPHA**4
        return fcsd





    def _atom_gyro_list(self,mol):
        gyro = []
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            if symb in mol.nucprop:
                prop = mol.nucprop[symb]
                mass = prop.get('mass', None)
                gyro.append(get_nuc_g_factor(symb, mass))
            else:
                # Get default isotope
                gyro.append(get_nuc_g_factor(symb))
        return numpy.array(gyro)

    def _atom_gyro_list_2(self, num_atom):
        gyro = []
        symb = self.mol.atom_symbol(num_atom)
                # Get default isotope
        gyro.append(get_nuc_g_factor(symb))
        return numpy.array(gyro)


    #@property
    def kernel(self, FC=True, FCSD=False, PSO=False):
        
        if FC:
            prop = self.pp_ssc_fc
        if PSO:
            prop = self.pp_ssc_pso
        elif FCSD:
            prop = self.pp_ssc_fcsd
        #return prop
            
        nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton ** 2
        iso_ssc = unit * numpy.einsum('kii->k', prop) / 3
        
        #print(iso_ssc)
        
        natm = self.mol.natm
        ktensor = numpy.zeros((natm,natm))
        for k, (i, j) in enumerate(self.nuc_pair):
            ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
        
        gyro = self._atom_gyro_list(self.mol)
        jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
        label = ['%2d %-2s'%(ia, self.mol.atom_symbol(ia)) for ia in range(natm)]
        #log.info( '\nNuclear g factor %s', gyro)
        #log.note(self, 'Spin-spin coupling constant J (Hz)')
        tools.dump_mat.dump_tri(self.mol.stdout, jtensor, label)
        #return ssc

    def kernel_select(self, FC=True, FCSD=False, PSO=False, atom1=None, atom2=None):
        
        if FC:
            prop = self.pp_ssc_fc_select(atom1,atom2)
        if PSO:
            prop = self.pp_ssc_pso_select(atom1, atom2)
        elif FCSD:
            prop = self.pp_ssc_fcsd_select(atom1, atom2)
        
            
        nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
        au2Hz = nist.HARTREE2J / nist.PLANCK
        unit = au2Hz * nuc_magneton ** 2
        iso_ssc = unit * numpy.einsum('kii->k', prop) / 3 
        natm = self.mol.natm
        gyro1 = get_nuc_g_factor(atom1[0])
        gyro2 = self._atom_gyro_list_2(atom2[0])
        jtensor = numpy.einsum('i,i,j->i', iso_ssc, gyro1, gyro2)
        return jtensor

