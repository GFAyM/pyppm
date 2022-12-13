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

        self.eri_mo = ao2mo.general(self.mol, [self.orbo,mo,mo,mo], compact=False)
        self.eri_mo = self.eri_mo.reshape(self.nocc,nmo,nmo,nmo)
        a -= numpy.einsum('ijba->iajb', self.eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
        if triplet: 
            b -= numpy.einsum('jaib->iajb', self.eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
        elif not triplet:
            b += numpy.einsum('jaib->iajb', self.eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
        
        #a = a + self.part_a2()
        m = a + b
        
        m = m.reshape(self.nocc*self.nvir,self.nocc*self.nvir, order='C')
                
        return m


    def kappa_full(self,I):
        '''
        K_{ij}^{a,b} = [(1-\delta_{ij})(1-\delta_ab]^{I-1}(2I-1)^.5 * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]

        for i \noteq j, a \noteq b

        K_{ij}^{a,b}=1^{I-1}(2I-1)^.5 * [[(ab|bj) -(-1)^I (aj|bi)]/ [e_i+e_j-e_a-e_b]]

        Oddershede 1984, eq C.7
        '''

        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir

        e_iajb = lib.direct_sum('i+j-a-b->iajb',
        self.mo_energy[self.occidx], self.mo_energy[self.occidx],
        self.mo_energy[self.viridx], self.mo_energy[self.viridx],               
                                    )
        #print(e_iajb.shape)
        self.eri_mo = ao2mo.general(self.mol, [mo,mo,mo,mo], compact=False)
        self.eri_mo = self.eri_mo.reshape(nmo,nmo,nmo,nmo)
        int1 = numpy.einsum('aibj->iajb', self.eri_mo[self.nocc:,:self.nocc,self.nocc:,:self.nocc])
        int2 = numpy.einsum('ajbi->iajb', self.eri_mo[self.nocc:,:self.nocc,self.nocc:,:self.nocc])

        K = 1**(I-1) * (2*I - 1)**.5 * ((int1)-(-1)**I*int2)/e_iajb

        for i in range(self.nocc):
            for j in range(self.nocc):
                if i==j:
                    K[i,:,j,:]=0
        
        for a in range(self.nvir):
            for b in range(self.nvir):
                if a==b:
                    K[:,a,:,b]=0
        return K

    @property
    def part_a2(self):
        '''
        method for obtain the correction to the A matrix for the HRPA nivel of approach
        equation C.13 in Oddershede 1984
        '''
        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir
        self.eri_mo = ao2mo.general(self.mol, [mo,mo,mo,mo], compact=False)
        eri_mo = self.eri_mo.reshape(nmo,nmo,nmo,nmo)
        int1 = numpy.einsum('iajb->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
        k_1 = self.kappa_full(1)
        k_2 = self.kappa_full(2)
        A = numpy.zeros_like(int1)
        
        for alfa in range(self.nocc):
            for beta in range(self.nocc):
                for m in range(self.nvir):
                    for n in range(self.nvir):
                        if n==m:
                            k = k_1[alfa,:,:,:] + numpy.sqrt(3)*k_2[alfa,:,:,:]
                            A[alfa,m,beta,n] -= .5*numpy.einsum('adb,adb->',int1[beta,:,:,:],k)
                        if alfa==beta:
                            k = k_1[:,m,:,:] + numpy.sqrt(3)*k_2[:,m,:,:]
                            A[alfa,m,beta,n] -= .5*numpy.einsum('dbp,pdb->', int1[:,:,:,n],k)
        return A

    def part_b2(self,S):
        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir
        self.eri_mo = ao2mo.general(self.mol, [mo,mo,mo,mo], compact=False)
        eri_mo = self.eri_mo.reshape(nmo,nmo,nmo,nmo)
        int1 = eri_mo[:self.nocc,self.nocc:,self.nocc:,:self.nocc]
        int2 = eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:]
        int3 = eri_mo[:self.nocc,:self.nocc,:self.nocc,:self.nocc]
        int4 = eri_mo[self.nocc:,self.nocc:,self.nocc:,self.nocc:]
        k_1 = self.kappa_full(1)
        k_2 = self.kappa_full(2)
        B = numpy.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
        for alfa in range(self.nocc):
            for beta in range(self.nocc):
                for m in range(self.nvir):
                    for n in range(self.nvir):
                        k_b1 = k_1[beta,m,:,:] + numpy.sqrt(3)*k_2[beta,m,:,:]
                        k_b2 = k_1[alfa,n,:,:] + numpy.sqrt(3)*k_2[alfa,n,:,:]
                        k_b3 = k_1[:,m,beta,:] + (numpy.sqrt(3)/(1-4*S))*k_2[:,m,beta,:]
                        k_b4 = k_1[:,n,alfa,:] + (numpy.sqrt(3)/(1-4*S))*k_2[:,n,alfa,:] 
                        k_b5 = k_1[:,m,:,n] + (numpy.sqrt(3)/(1-4*S))*k_2[:,m,:,n]
                        k_b6 = k_1[beta,:,alfa,:] + (numpy.sqrt(3)/(1-4*S))*k_1[beta,:,alfa,:]
                        B[alfa,m,beta,n] += .5*(numpy.einsum('rp,pr->',int1[alfa,n,:,:],k_b1) +  
                                                numpy.einsum('rp,pr->',int1[beta,m,:,:],k_b2) +
                                                -1**S*((numpy.einsum('pr,pr->',int2[alfa,:,:,n],k_b3)
                                                + numpy.einsum('pr,pr->',int2[beta,:,:,m],k_b4))     
                                                - numpy.einsum('pd,dp->',int3[beta,:,alfa,:],k_b5 ) 
                                                - numpy.einsum('qp,pq->',int4[:,m,:,n],k_b6)
                                                ) )  
        return B

    @property
    def S2(self):
        '''
        equation C.27 of Oddershede 1984.
        This Matrix will be multiplicated by the energy
        '''
        e_iajb = lib.direct_sum('i+j-a-b->iajb',
        self.mo_energy[self.occidx], self.mo_energy[self.occidx],
        self.mo_energy[self.viridx], self.mo_energy[self.viridx],               
                                    )

        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir
        self.eri_mo = ao2mo.general(self.mol, [mo,mo,mo,mo], compact=False)
        eri_mo = self.eri_mo.reshape(nmo,nmo,nmo,nmo)
        k_1 = self.kappa_full(1)
        k_2 = self.kappa_full(2)
        S2 = numpy.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
        int1 = eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:]
        for alfa in range(self.nocc):
            for beta in range(self.nocc):
                for m in range(self.nvir):
                    for n in range(self.nvir):
                        if m == m :
                            k_s2_1 = k_1[alfa,:,:,:] + numpy.sqrt(3)*k_2[alfa,:,:,:]
                            S2[alfa,m,beta,n] -= .5*numpy.einsum('apb,apb->',int1[beta,:,:,:]/e_iajb[beta,:,:,:],k_s2_1)              
                        if alfa == beta:
                            k_s2_2 = k_1[:,m,:,:] + + numpy.sqrt(3)*k_2[:,m,:,:]
                            S2[alfa,m,beta,n] -= .5*numpy.einsum('dap,pda->',int1[:,:,:,n]/e_iajb[:,:,:,n],k_s2_2)
        return S2


    def kappa_2(self,alfa,m):
        '''
        equation C.24 of 1984 Oddershede Paper, ready for multiplicate for 
        the Propagator in |SCF>
        '''
        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir
        nocc = self.nocc
        nvir = self.nvir
        k_1 = self.kappa_full(1)
        k_2 = self.kappa_full(2)
        e_ia = lib.direct_sum('i-a->ia',
                                self.mo_energy[self.occidx],
                                self.mo_energy[self.viridx])
        eri_mo = ao2mo.general(self.mol, [mo,mo,mo,mo], compact=False)
        eri_mo = self.eri_mo.reshape(nmo,nmo,nmo,nmo)
        int1 = eri_mo[:nocc,nocc:,nocc:,nocc:]
        int2 = eri_mo[:nocc,nocc:,:nocc,:nocc]
        kappa = numpy.einsum('pab,pab->',int1[:,:,m,:],
                                    (k_1[:,:,alfa,:] + numpy.sqrt(3)*k_2[:,:,alfa,:])  )

        kappa -= numpy.einsum('pad,pad->', int2[:,:,:,alfa],
                                    (k_1[:,:,:,m] + numpy.sqrt(3)*k_2[:,:,:,m]) )

        kappa = kappa*numpy.sqrt(2)/2 / e_ia[alfa,m]
        return kappa

    @property
    def kappa_2_full(self):
        '''
        equation C.24 of 1984 Oddershede Paper, in a matrix form,
        ready for multiplicate for the Propagator in |SCF>
        '''
        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir
        nocc = self.nocc
        nvir = self.nvir
        k_1 = self.kappa_full(1)
        k_2 = self.kappa_full(2)
        e_ia = lib.direct_sum('i-a->ia',
                                self.mo_energy[self.occidx],
                                self.mo_energy[self.viridx])
        eri_mo = ao2mo.general(self.mol, [mo,mo,mo,mo], compact=False)
        eri_mo = self.eri_mo.reshape(nmo,nmo,nmo,nmo)
        int1 = eri_mo[:nocc,nocc:,nocc:,nocc:]
        int2 = eri_mo[:nocc,nocc:,:nocc,:nocc]
        kappa = numpy.zeros((nocc,nvir))
        for alfa in range(nocc):
            for m in range(nvir):
                kappa[alfa,m] = numpy.einsum('pab,pab->',int1[:,:,m,:],
                                            (k_1[:,:,alfa,:] + numpy.sqrt(3)*k_2[:,:,alfa,:])  )

                kappa[alfa,m] -= numpy.einsum('pad,pad->', int2[:,:,:,alfa],
                                            (k_1[:,:,:,m] + numpy.sqrt(3)*k_2[:,:,:,m]) )

                kappa[alfa,m] = kappa[alfa,m]*numpy.sqrt(2)/2 / e_ia[alfa,m]
        return kappa

    def correction_pert(self,atmlst):
        '''
        equation C.25, the first correction to the perturbator
        
        '''
        h1 = self.pert_fc(atmlst)[0]
        kappa = self.kappa_2_full
        nocc = self.nocc
        pert = numpy.zeros((self.nocc,self.nvir))
        for alfa in range(self.nocc):
            for m in range(self.nvir):
                p_virt = h1[nocc:,nocc:]
                pert[alfa,m] = numpy.einsum('n,n->', kappa[alfa,:],p_virt[m,:])
                p_occ = h1[:nocc,:nocc]
                pert[alfa,m] -= numpy.einsum('b,b->', kappa[:,m],p_occ[:,alfa])

        return pert


    def correction_pert_2(self,atmlst):
        mo = numpy.hstack((self.orbo,self.orbv))
        nmo = self.nocc + self.nvir
        nocc = self.nocc
        nvir = self.nvir
        eri_mo = ao2mo.general(self.mol, [mo,mo,mo,mo], compact=False)
        eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
        int1 = eri_mo[:nocc,nocc:,:nocc,nocc:]
        c1 = numpy.sqrt(3)
        k_1 = self.kappa_full(1)
        k_2 = self.kappa_full(2)
        e_iajb = lib.direct_sum('i+j-a-b->iajb',
                                self.mo_energy[self.occidx], 
                                self.mo_energy[self.occidx],
                                self.mo_energy[self.viridx], 
                                self.mo_energy[self.viridx])
        h1 = self.pert_fc(atmlst)[0]
        pert = numpy.zeros((self.nocc,self.nvir))
        h1 = h1[nocc:,:nocc]
        for alfa in range(nocc):
            for m in range(nvir):    
                t = numpy.einsum('dapb,d,apb->',int1[:,:,:,:]/e_iajb[:,:,:,:], h1[m,:],
                                             (k_1[alfa,:,:,:]+c1*k_2[alfa,:,:,:]))
                t += numpy.einsum('dapb,b,pda',int1[:,:,:,:]/e_iajb[:,:,:,:],h1[:,alfa],
                                    (k_1[:,m,:,:]+c1*k_2[:,m,:,:]))
                t = -t*numpy.sqrt(2)/2  
                pert[alfa,m] = t
        return pert

    def pert_fc(self,atmlst):
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        mol = self.mol
        coords = mol.atom_coords()
        ao = numint.eval_ao(mol, coords)
        mo = ao.dot(mo_coeff)
        orbo = mo[:,:]
        #orbo = mo[:,mo_occ> 0]
        orbv = mo[:,:]
        #orbv = mo[:,mo_occ==0]
        fac = 8*numpy.pi/3 *.5  
        h1 = []
        for ia in atmlst:
            h1.append(fac * numpy.einsum('p,i->pi', orbv[ia], orbo[ia]))
        return h1



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
        gyro1 = self._atom_gyro_list_2(atom1[0])
        gyro2 = self._atom_gyro_list_2(atom2[0])
        jtensor = numpy.einsum('i,i,j->i', iso_ssc, gyro1, gyro2)
        return jtensor