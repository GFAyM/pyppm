from pyscf import gto, scf
from pyscf.gto import Mole
import numpy as np
import attr
from src.help_functions import extra_functions
from pyscf import ao2mo
from functools import reduce
from pyscf.data import nist
from pyscf.dft import numint
from pyscf.data.gyro import get_nuc_g_factor
from pyscf import tools

def uniq_atoms(nuc_pair):
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    return atm1dic, atm2dic

@attr.s
class Cloppa:
        """
        Cloppa method for the calculation of NMR J-coupling using localized molecular orbitals of any kind
        e.g Foster-Boys, Pipek-Mezey, etc. 
        This method follows the Giribet et. al article: Molecular Physics 91: 1, 105-112
        
        This class pretends calculate the full M matrix, i.e, using all the occupied LMO and all virtuals LMO
        """
        mo_coeff_loc = attr.ib(default=None, type=np.array)
        mol_loc = attr.ib(default=None)
        mo_occ_loc = attr.ib(default=None)

        def __attrs_post_init__(self):
                #if self.occ != None: 
                #        self.orbv = self.mo_coeff_loc[:,self.vir]
                #        self.orbo = self.mo_coeff_loc[:,self.occ]
                #else:
            self.occidx = np.where(self.mo_occ_loc>0)[0]
            self.viridx = np.where(self.mo_occ_loc==0)[0]

            self.orbv = self.mo_coeff_loc[:,self.viridx]
            self.orbo = self.mo_coeff_loc[:,self.occidx]

            self.nvir = self.orbv.shape[1]
            self.nocc = self.orbo.shape[1]
            self.mo = np.hstack((self.orbo,self.orbv))
            self.nmo = self.nocc + self.nvir
                
            self.nuc_pair = [(i,j) for i in range(self.mol_loc.natm) for j in range(i)]

            self.atm1dic, self.atm2dic = uniq_atoms(nuc_pair=self.nuc_pair)
            #here we made a SCF (in the canonical basis) calculation of the molecule
            self.mf = scf.RHF(self.mol_loc).run()

        @property
        def fock_matrix_canonical(self):
                self.fock_canonical = self.mf.get_fock()
                return self.fock_canonical


        def M(self,triplet=True, energy_m=True, pzoa=False):
                self.m = np.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
                if energy_m == False:
                    self.m = np.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
                elif energy_m == True:
                    fock = self.fock_matrix_canonical
                    for i in range(self.nocc):
                        for j in range(self.nocc):
                            for a in range(self.nvir):
                                for b in range(self.nvir):
                                    if a==b:
                                        self.m[i,a,j,b] -= self.orbo[:,i].T @ fock @ self.orbo[:,j]
                                    if i==j:
                                        self.m[i,a,j,b] += self.orbv[:,a].T @ fock @ self.orbv[:,b]
                if pzoa==True:
                    return self.m.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
                eri_mo = ao2mo.general(self.mol_loc, 
                        [self.mo,self.mo,self.mo,self.mo], compact=False)
                eri_mo = eri_mo.reshape(self.nmo,self.nmo,self.nmo,self.nmo)
                self.m -= np.einsum('ijba->iajb', eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
                if triplet:
                    self.m -= np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
                elif not triplet:
                    self.m += np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
                
                self.m = self.m.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
                return self.m



        def pert_fc(self,atmlst):
                mo_coeff = self.mo_coeff_loc
                mo_occ = self.mo_occ_loc
                mol = self.mol_loc
                coords = mol.atom_coords()
                ao = numint.eval_ao(mol, coords)
                mo = ao.dot(mo_coeff)
                orbo = mo[:,mo_occ> 0]
                orbv = mo[:,mo_occ==0]
                fac = 8*np.pi/3 *.5  # *.5 due to s = 1/2 * pauli-matrix
                h1 = []
                for ia in atmlst:
                        h1.append(fac * np.einsum('p,i->pi', orbv[ia], orbo[ia]))
                return h1

        def pert_fcsd(self, atmlst):
            '''MO integrals for FC + SD'''
            orbo = self.orbo
            orbv = self.orbv

            h1 = []
            for ia in atmlst:
                h1ao = self._get_integrals_fcsd( ia)
                for i in range(3):
                    for j in range(3):
                        h1.append(orbv.T.conj().dot(h1ao[i,j]).dot(orbo) * .5)
            return h1

        def pert_pso(self, atmlst):
        # Imaginary part of H01 operator
        # 1/2(A01 dot p + p dot A01) => (a01p + c.c.)/2 ~ <a01p>
        # Im[A01 dot p] = Im[vec{r}/r^3 x vec{p}] = Im[-i p (1/r) x p] = -p (1/r) x p
            orbo = self.orbo
            orbv = self.orbv

            h1 = []
            for ia in atmlst:
                self.mol_loc.set_rinv_origin(self.mol_loc.atom_coord(ia))
                h1ao = -self.mol_loc.intor_asymmetric('int1e_prinvxp', 3)
                h1 += [reduce(np.dot, (orbv.T.conj(), x, orbo)) for x in h1ao]
            return h1        


        def _get_integrals_fcsd(self, atm_id):
            '''AO integrals for FC + SD'''
            mol = self.mol_loc
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



        @property
        def pp_ssc_fc(self):
            nvir = self.nvir
            nocc = self.nocc
            h1 = self.pert_fc(sorted(self.atm1dic.keys()))
            h2 = self.pert_fc(sorted(self.atm2dic.keys()))    
            m = self.M(triplet=True)
            p = np.linalg.inv(m)
            p = -p.reshape(nocc,nvir,nocc,nvir)
            para = []
            for i,j in self.nuc_pair:
                    at1 = self.atm1dic[i]
                    at2 = self.atm2dic[j]
                    e = np.einsum('ia,iajb,jb', h1[at1].T, p , h2[at2].T)
                    para.append(e*4)  # *4 for +c.c. and for double occupancy
            fc = np.einsum(',k,xy->kxy', nist.ALPHA**4, para, np.eye(3))    
            return fc        

        @property
        def pp_ssc_fcsd(self):
            nvir = self.nvir
            nocc = self.nocc

            h1 = self.pert_fcsd(sorted(self.atm1dic.keys()))
            h1  = np.asarray(h1).reshape(-1,3,3,nvir,nocc)
            h2 = self.pert_fcsd(sorted(self.atm2dic.keys()))
            h2  = np.asarray(h2).reshape(-1,3,3,nvir,nocc)
            m = self.M(triplet=True)
            p = np.linalg.inv(m)
            p = -p.reshape(nocc,nvir,nocc,nvir)
            para = []
            for i,j in self.nuc_pair:
                at1 = self.atm1dic[i]
                at2 = self.atm2dic[j]
                e = np.einsum('iawx,iajb,jbwy->xy', h1[at1].T, p , h2[at2].T)
                para.append(e*4)
            fcsd = np.asarray(para) * nist.ALPHA**4
            return fcsd

        @property
        def pp_ssc_pso(self):
            nuc_pair = self.nuc_pair
            para = []
            nocc = np.count_nonzero(self.mo_occ_loc> 0)
            nvir = np.count_nonzero(self.mo_occ_loc==0)
            atm1lst = sorted(set([i for i,j in nuc_pair]))
            atm2lst = sorted(set([j for i,j in nuc_pair]))
            atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
            atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
            #mo1 = mo1.reshape(len(atm1lst),3,nvir,nocc)
            m = self.M(triplet=False)
            p = np.linalg.inv(m)
            p = -p.reshape(nocc,nvir,nocc,nvir)
            h1 = self.pert_pso(atm1lst)
            h1 = np.asarray(h1).reshape(len(atm1lst),3,nvir,nocc)
            h2 = self.pert_pso(atm2lst)
            h2 = np.asarray(h2).reshape(len(atm2lst),3,nvir,nocc)
            for i,j in nuc_pair:
                # PSO = -Tr(Im[h1_ov], Im[mo1_vo]) + cc = 2 * Tr(Im[h1_vo], Im[mo1_vo])
                e = np.einsum('iax,iajb,jby->xy', h1[atm1dic[i]].T, p, h2[atm2dic[j]].T)
                para.append(e*4)  # *4 for +c.c. and double occupnacy
            pso = np.asarray(para) * nist.ALPHA**4
            return pso

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
            return np.array(gyro)
                                        
        def kernel(self, FC=True, FCSD=False, PSO=False):
            if FC:
                prop = self.pp_ssc_fc
            if PSO:
                prop = self.pp_ssc_pso
            elif FCSD:
                prop = self.pp_ssc_fcsd
            
            nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
            au2Hz = nist.HARTREE2J / nist.PLANCK
            unit = au2Hz * nuc_magneton ** 2
            iso_ssc = unit * np.einsum('kii->k', prop) / 3 
            #print(iso_ssc)
            natm = self.mol_loc.natm
            ktensor = np.zeros((natm,natm))
            for k, (i, j) in enumerate(self.nuc_pair):
                ktensor[i,j] = ktensor[j,i] = iso_ssc[k]

            gyro = self._atom_gyro_list(self.mol_loc)
            jtensor = np.einsum('ij,i,j->ij', ktensor, gyro, gyro)
            label = ['%2d %-2s'%(ia, self.mol_loc.atom_symbol(ia)) for ia in range(natm)]
            ssc = tools.dump_mat.dump_tri(self.mol_loc.stdout, jtensor, label)
            
        def pp_ssc_pso_pathways(self,princ_prop,n_atom1,occ_atom1,vir_atom1,n_atom2,occ_atom2,vir_atom2,all_pathways,elements):
            nvir = self.nvir
            nocc = self.nocc

            if princ_prop.all() == None:
                m = self.M(triplet=False)
                p = np.linalg.inv(m)
                p = -p.reshape(nocc,nvir,nocc,nvir)
            else:
                p=princ_prop
                p = -p.reshape(nocc,nvir,nocc,nvir)

            h1 = self.pert_pso(n_atom1)
            h1 = np.asarray(h1).reshape(1,3,nvir,nocc)
            h1_pathway = np.zeros(h1.shape)
            h2 = self.pert_pso(n_atom2)
            h2 = np.asarray(h2).reshape(1,3,nvir,nocc)
            h2_pathway = np.zeros(h2.shape)

            if all_pathways == True:
                h1_pathway[0,:,:,:] += h1[0,:,:,:]
                h2_pathway[0,:,:,:] += h2[0,:,:,:]

            elif vir_atom1 == None:
                h1_pathway[0,:,:,occ_atom1] += h1[0,:,:,occ_atom1]
                h2_pathway[0,:,:,occ_atom2] += h2[0,:,:,occ_atom2]
            
            else: 
                h1_pathway[0,:,vir_atom1-nocc,occ_atom1] += h1[0,:,vir_atom1-nocc,occ_atom1]
                h2_pathway[0,:,vir_atom2-nocc,occ_atom2] += h2[0,:,vir_atom2-nocc,occ_atom2]

            para = []
            e = np.einsum('iax,iajb,jby->xy', h1_pathway[0].T, p, h2_pathway[0].T)
            para.append(e*4)  # *4 for +c.c. and double occupnacy
            pso = np.asarray(para) * nist.ALPHA**4
            if elements==False:
                return pso
            elif elements==True:
                return h1_pathway[0].T, p, h2_pathway[0].T

        def pp_ssc_fcsd_pathways(self,princ_prop,n_atom1,occ_atom1,vir_atom1,n_atom2,occ_atom2,vir_atom2,all_pathways,elements):
            nvir = self.nvir
            nocc = self.nocc

            h1 = self.pert_fcsd(n_atom1)
            h1  = np.asarray(h1).reshape(-1,3,3,nvir,nocc)
            h1_pathway = np.zeros((1,3,3,nvir,nocc))
            h2 = self.pert_fcsd(n_atom2)
            h2  = np.asarray(h2).reshape(-1,3,3,nvir,nocc)
            h2_pathway = np.zeros((1,3,3,nvir,nocc))
            
            if all_pathways == True:
                h1_pathway[0,:,:,:] += h1[0,:,:,:]
                h2_pathway[0,:,:,:] += h2[0,:,:,:]
            elif vir_atom1 == None:
                h1_pathway[0,:,:,:,occ_atom1] += h1[0,:,:,:,occ_atom1]
                h2_pathway[0,:,:,:,occ_atom2] += h2[0,:,:,:,occ_atom2]
            else: 
                h1_pathway[0,:,:,vir_atom1-nocc,occ_atom1] += h1[0,:,:,vir_atom1-nocc,occ_atom1]
                h2_pathway[0,:,:,vir_atom2-nocc,occ_atom2] += h2[0,:,:,vir_atom2-nocc,occ_atom2]    
            
            

            p=princ_prop
            p = -p.reshape(nocc,nvir,nocc,nvir)
            para = []
            e = np.einsum('iawx,iajb,jbwy->xy', h1_pathway[0].T, p , h2_pathway[0].T)
            para.append(e*4)            
            fcsd = np.asarray(para) * nist.ALPHA**4
            if elements==False:
                return fcsd
            elif elements==True:
                return h1_pathway[0].T, p, h2_pathway[0].T

        def pp_ssc_fc_pathways(self,princ_prop,n_atom1,occ_atom1,vir_atom1,n_atom2,occ_atom2,vir_atom2,all_pathways, elements):
            nvir = self.nvir
            nocc = self.nocc
            h1 = self.pert_fc(n_atom1)
            h2 = self.pert_fc(n_atom2)
            h1_pathway = np.zeros(h1[0].shape)
            h2_pathway = np.zeros(h2[0].shape)
            if all_pathways == True:
                h1_pathway[:,:] += h1[0][:,:]
                h2_pathway[:,:] += h2[0][:,:]
            elif vir_atom1 == None:    
                h1_pathway[:,occ_atom1] += h1[0][:,occ_atom1]
                h2_pathway[:,occ_atom2] += h2[0][:,occ_atom2]
            else:
                h1_pathway[vir_atom1-nocc,occ_atom1] += h1[0][vir_atom1-nocc,occ_atom1]
                h2_pathway[vir_atom2-nocc,occ_atom2] += h2[0][vir_atom2-nocc,occ_atom2]
            
            
            
            p=princ_prop
            p = -p.reshape(nocc,nvir,nocc,nvir)    
            para = []
            e = np.einsum('ia,iajb,jb', h1_pathway.T, p , h2_pathway.T)
            para.append(e*4)  # *4 for +c.c. and for double occupancy
            fc = np.einsum(',k,xy->kxy', nist.ALPHA**4, para, np.eye(3))
            if elements==False:
                return fc
            elif elements==True:
                return h1_pathway.T, p, h2_pathway.T
            
        def _atom_gyro_list_2(self, num_atom):
            gyro = []
            symb = self.mol_loc.atom_symbol(num_atom)
                    # Get default isotope
            gyro.append(get_nuc_g_factor(symb))
            return np.array(gyro)

        def kernel_pathway(self, FC=False, FCSD=True, PSO=False, princ_prop=None,
                                n_atom1=None, occ_atom1=None, vir_atom1=None, 
                                n_atom2=None, occ_atom2=None, vir_atom2=None,
                                all_pathways=False, elements=False):

            if FC:
                prop = self.pp_ssc_fc_pathways(princ_prop=princ_prop,
                                                     n_atom1=n_atom1,occ_atom1=occ_atom1, vir_atom1=vir_atom1,
                                                     n_atom2=n_atom2,occ_atom2=occ_atom2, vir_atom2=vir_atom2,
                                                     all_pathways=all_pathways,elements=elements)
            if PSO:
                prop = self.pp_ssc_pso_pathways(princ_prop=princ_prop,
                                                     n_atom1=n_atom1,occ_atom1=occ_atom1, vir_atom1=vir_atom1,
                                                     n_atom2=n_atom2,occ_atom2=occ_atom2, vir_atom2=vir_atom2,
                                                     all_pathways=all_pathways,elements=elements)
            if FCSD:
                prop = self.pp_ssc_fcsd_pathways(princ_prop=princ_prop,
                                                     n_atom1=n_atom1,occ_atom1=occ_atom1, vir_atom1=vir_atom1,
                                                     n_atom2=n_atom2,occ_atom2=occ_atom2, vir_atom2=vir_atom2,
                                                     all_pathways=all_pathways, elements=elements)
            
            nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
            au2Hz = nist.HARTREE2J / nist.PLANCK
            unit = au2Hz * nuc_magneton ** 2
            iso_ssc = unit * np.einsum('kii->k', prop) / 3 
            natm = self.mol_loc.natm
            gyro1 = self._atom_gyro_list_2(n_atom1[0])
            gyro2 = self._atom_gyro_list_2(n_atom2[0])
            jtensor = np.einsum('i,i,j->i', iso_ssc, gyro1, gyro2)
            return jtensor

        def kernel_pathway_elements(self, FC=False, FCSD=False, PSO=False, princ_prop=None,
                                n_atom1=None, occ_atom1=None, vir_atom1=None, 
                                n_atom2=None, occ_atom2=None, vir_atom2=None,
                                all_pathways=False, elements=False):

            if FC:
                p1, m, p2 = self.pp_ssc_fc_pathways(princ_prop=princ_prop,
                                                     n_atom1=n_atom1,occ_atom1=occ_atom1, vir_atom1=vir_atom1,
                                                     n_atom2=n_atom2,occ_atom2=occ_atom2, vir_atom2=vir_atom2,
                                                     all_pathways=all_pathways, elements=elements)
                p1_pathway = p1[occ_atom1,vir_atom1 - self.nocc] 
                m_pathway = m[occ_atom1,vir_atom1 - self.nocc,occ_atom2,vir_atom2 - self.nocc]
                p2_pathway = p2[occ_atom2,vir_atom2 - self.nocc]
                return p1_pathway, m_pathway, p2_pathway
            if PSO:
                p1, m, p2 = self.pp_ssc_pso_pathways(princ_prop=princ_prop,
                                                     n_atom1=n_atom1,occ_atom1=occ_atom1, vir_atom1=vir_atom1,
                                                     n_atom2=n_atom2,occ_atom2=occ_atom2, vir_atom2=vir_atom2,
                                                     all_pathways=all_pathways, elements=elements)
                p1_pathway = p1[occ_atom1,vir_atom1 - self.nocc,:] 
                m_pathway = m[occ_atom1,vir_atom1 - self.nocc,occ_atom2,vir_atom2 - self.nocc]
                p2_pathway = p2[occ_atom2,vir_atom2 - self.nocc,:]
                
                return np.sum(p1)/3, m_pathway, np.sum(p2)/3
            if FCSD:
                p1, m, p2 = self.pp_ssc_fcsd_pathways(princ_prop=princ_prop,
                                                     n_atom1=n_atom1,occ_atom1=occ_atom1, vir_atom1=vir_atom1,
                                                     n_atom2=n_atom2,occ_atom2=occ_atom2, vir_atom2=vir_atom2,
                                                     all_pathways=all_pathways, elements=elements)            
                p1_pathway = p1[occ_atom1,vir_atom1 - self.nocc,:,:] 
                m_pathway = m[occ_atom1,vir_atom1 - self.nocc,occ_atom2,vir_atom2 - self.nocc]
                p2_pathway = p2[occ_atom2,vir_atom2 - self.nocc,:,:]
                
                return np.trace(p1_pathway)/3, m_pathway, np.trace(p2_pathway)/3
                