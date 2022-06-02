from pyscf import gto, scf
from pyscf.gto import Mole
import numpy as np
import attr
from src.help_functions import extra_functions
from pyscf import ao2mo
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
        mol_input = attr.ib(default=None, type=str, validator=attr.validators.instance_of(str))
        basis = attr.ib(default='cc-pvdz', type=str, validator=attr.validators.instance_of(str))
        mo_coeff_loc = attr.ib(default=None, type=np.array)
        mol_loc = attr.ib(default=None)
        mo_occ_loc = attr.ib(default=None)
        #for testing
        #occ = attr.ib(default=None)
        #vir = attr.ib(default=None)
        #cart = attr.ib(default=False, type=bool, validator=attr.validators.instance_of(bool))
        

        def __attrs_post_init__(self):
                #if self.occ != None: 
                #        self.orbv = self.mo_coeff_loc[:,self.vir]
                #        self.orbo = self.mo_coeff_loc[:,self.occ]
                #else:
                self.occidx = np.where(self.mo_occ_loc==1)[0]
                self.viridx = np.where(self.mo_occ_loc==0)[0]
                
                self.orbv = self.mo_coeff_loc[:,self.viridx]
                self.orbo = self.mo_coeff_loc[:,self.occidx]
                
                self.nvir = self.orbv.shape[1]
                self.nocc = self.orbo.shape[1]
                self.mo = np.hstack((self.orbo,self.orbv))
                self.nmo = self.nocc + self.nvir

                self.mol = gto.M(atom=str(self.mol_input), basis=self.basis, verbose=0)#, cart=self.cart)
                
                self.nuc_pair = [(i,j) for i in range(self.mol.natm) for j in range(i)]
                self.nvir = self.orbv.shape[1]
                self.nocc = self.orbo.shape[1]
                self.atm1dic, self.atm2dic = uniq_atoms(nuc_pair=self.nuc_pair)
                #here we made a SCF (in the canonical basis) calculation of the molecule
                self.mf = scf.RHF(self.mol).run()
                


        @property
        def fock_matrix_canonical(self):
                self.fock_canonical = self.mf.get_fock()
                return self.fock_canonical

        @property
        def M(self):
                self.m = np.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
                #here we calculate the fock matrix in the localized molecular basis set
                fock = self.fock_matrix_canonical
                for i in range(self.nocc):
                        for j in range(self.nocc):
                                for a in range(self.nvir):
                                        for b in range(self.nvir):
                                                if a==b:
                                                        self.m[i,a,j,b] -= self.orbo[:,i].T @ fock @ self.orbo[:,j]
                                                if i==j:
                                                        self.m[i,a,j,b] += self.orbv[:,a].T @ fock @ self.orbv[:,b]
                #here, the 2e part of the M matrix 
                eri_mo = ao2mo.general(self.mol_loc, 
                        [self.mo,self.mo,self.mo,self.mo], compact=False)
                eri_mo = eri_mo.reshape(self.nmo,self.nmo,self.nmo,self.nmo)
                self.m -= np.einsum('ijba->iajb', eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
                self.m -= np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
                self.m = self.m.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
                
                return self.m

        def P(self):
                """
                This function calculate the P matrix, i.e, the inverse of the M matrix
                """
                m = self.M
                self.P = np.linalg.inv(m)
                return self.P                

        def elements_m(self, i,a,j,b):
                m = self.M
                m_resheped = m.reshape((self.nocc,self.nvir,self.nocc,self.nvir))
                #m_resheped = m_resheped.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
                return m_resheped[i,a-self.nocc,j,b-self.nocc]

        def elements_p(self, i,a,j,b):
                m = self.M
                p = np.linalg.inv(m)
                p_resheped = p.reshape((self.nocc,self.nvir,self.nocc,self.nvir))
                #m_resheped = m_resheped.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
                return p_resheped[i,a-self.nocc,j,b-self.nocc]

        def h1_fc_pyscf(self,atmlst):
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
        
        @property
        def pp_ssc_fc(self):
                nvir = self.nvir
                nocc = self.nocc
                h1 = self.h1_fc_pyscf(sorted(self.atm1dic))
                h2 = self.h1_fc_pyscf(sorted(self.atm2dic))    
                m = self.M
                p = np.linalg.inv(m)
                p = p.reshape(nocc,nvir,nocc,nvir)
                para = []
                for i,j in self.nuc_pair:
                        at1 = self.atm1dic[i]
                        at2 = self.atm2dic[j]
                        e = np.einsum('ia,iajb,jb', h1[at1].T, p , h2[at2].T)
                        para.append(e*4)  # *4 for +c.c. and for double occupancy
                fc = np.einsum(',k,xy->kxy', nist.ALPHA**4, para, np.eye(3))    
                return fc        

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
                                
        @property
        def kernel(self):
                fc = self.pp_ssc_fc
                nuc_magneton = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
                au2Hz = nist.HARTREE2J / nist.PLANCK
                unit = au2Hz * nuc_magneton ** 2
                iso_ssc = unit * np.einsum('kii->k', fc) / 3 
                #print(iso_ssc)
                natm = self.mol.natm
                ktensor = np.zeros((natm,natm))
                for k, (i, j) in enumerate(self.nuc_pair):
                        ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
                
                gyro = self._atom_gyro_list(self.mol)
                jtensor = np.einsum('ij,i,j->ij', ktensor, gyro, gyro)
                label = ['%2d %-2s'%(ia, self.mol.atom_symbol(ia)) for ia in range(natm)]
                ssc = tools.dump_mat.dump_tri(self.mol.stdout, jtensor, label)
                return ssc