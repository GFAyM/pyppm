from pyscf import gto, scf
from pyscf.gto import Mole
import numpy as np
import attr
from src.help_functions import extra_functions
from pyscf import ao2mo


@attr.s
class Cloppa_specific_pathways:
    """
    Cloppa method for the calculation of NMR J-coupling using localized molecular orbitals of any kind
    e.g Foster-Boys, Pipek-Mezey, etc. 
    This method follows the Giribet et. al article: Molecular Physics 91: 1, 105-112
    """
    mol_input = attr.ib(default=None, type=str, validator=attr.validators.instance_of(str))
    basis = attr.ib(default='cc-pvdz', type=str, validator=attr.validators.instance_of(str))
    mo_coeff_loc = attr.ib(default=None, type=np.array)
    mol_loc = attr.ib(default=None)
    o1 = attr.ib(default=None, type=list, validator=attr.validators.instance_of(list))
    o2 = attr.ib(default=None, type=list, validator=attr.validators.instance_of(list))
    v1 = attr.ib(default=None, type=list, validator=attr.validators.instance_of(list))
    v2 = attr.ib(default=None, type=list, validator=attr.validators.instance_of(list))

    def __attrs_post_init__(self):
        self.mol = gto.M(atom=str(self.mol_input), basis=self.basis, verbose=0)
        self.mf = scf.RHF(self.mol).run()

    @property
    def fock_matrix_canonical(self):
        self.fock_canonical = self.mf.get_fock()
        return self.fock_canonical

    @property
    def M(self):
        self.virt = self.mo_coeff_loc[:,self.v1+self.v2]
        self.fock_matrix_canonical

        
        block11 = np.zeros((len(self.v1 + self.v2),len(self.v1 + self.v2)))
        #now, we sum the F_{ab}\delta_{ij} localized Fock matrix 
        block11 += self.virt.T @ self.fock_canonical @ self.virt

        # this is the F_{ij}\delta_{ab} localized Fock matrix
        F_ii = self.i.T @ self.fock_canonical @ self.i
        np.fill_diagonal(block11, block11.diagonal() - F_ii)

        block11 -= ao2mo.general(self.mol_loc, 
                [self.virt,self.virt,self.i,self.i],
                compact=False).reshape(len(self.v1 +self.v2),len(self.v1 + self.v2))    
        
        block11 -= ao2mo.general(self.mol_loc, 
                [self.virt,self.i,self.virt,self.i],
                compact=False).reshape(len(self.v1 +self.v2),len(self.v1 + self.v2))    


        block12 = np.zeros((len(self.v1 + self.v2),len(self.v1 + self.v2)))
        #now, the block12 matrix only contain the F_{ij} element in all the diagonal
        F_ij = self.i.T @ self.fock_canonical @ self.j
        np.fill_diagonal(block12, -F_ij)
        block12 -= ao2mo.general(self.mol_loc, [self.virt,self.virt,self.j,self.i],
         compact=False).reshape(len(self.v1 + self.v2),len(self.v1 + self.v2))
        block12 -= ao2mo.general(self.mol_loc, 
                [self.virt,self.j,self.virt,self.i],
                compact=False).reshape(len(self.v1 +self.v2),len(self.v1 + self.v2))    


        block21 = np.zeros((len(self.v1 + self.v2),len(self.v1 + self.v2)))
        #now, the block21 matrix only contain the F_{ij} element in all the diagonal
        F_ji = self.j.T @ self.fock_canonical @ self.i
        np.fill_diagonal(block21, -F_ji)
        block21 -= ao2mo.general(self.mol_loc, [self.virt,self.virt,self.i,self.j],
            compact=False).reshape(len(self.v1 + self.v2),len(self.v1 + self.v2))
        block21 -= ao2mo.general(self.mol_loc, 
                [self.virt,self.i,self.virt,self.j],
                compact=False).reshape(len(self.v1 +self.v2),len(self.v1 + self.v2))    


        block22 = np.zeros((len(self.v1 + self.v2),len(self.v1 + self.v2)))
        block22 += self.virt.T @ self.fock_canonical @ self.virt
        F_jj = self.j.T @ self.fock_canonical @ self.j
        np.fill_diagonal(block22, block22.diagonal() - F_jj) 
        block22 -= ao2mo.general(self.mol_loc,
                [self.virt,self.virt,self.j,self.j],
                compact=False).reshape(len(self.v1 + self.v2),len(self.v1 + self.v2))
        block22 -= ao2mo.general(self.mol_loc, 
                [self.virt,self.j,self.virt,self.j],
                compact=False).reshape(len(self.v1 +self.v2),len(self.v1 + self.v2))    
        
        
        superblock1 = np.concatenate((block11, block12), axis=1)
        superblock2 = np.concatenate((block21, block22), axis=1)
        M = np.concatenate((superblock1, superblock2), axis=0)
        #M_diag = np.zeros(M.shape)
        #np.fill_diagonal(M_diag, np.diag(M))


        return M

    @property
    def M_cruzada(self):
        self.i = self.mo_coeff_loc[:, self.o1]
        self.a = self.mo_coeff_loc[:, self.v1]
        self.j = self.mo_coeff_loc[:, self.o2]
        self.b = self.mo_coeff_loc[:, self.v2]

        self.virt = self.mo_coeff_loc[:,self.v1+self.v2]
        self.fock_matrix_canonical

        
        block11 = np.zeros((len(self.v1 + self.v2),len(self.v1 + self.v2)))

        block12 = np.zeros((len(self.v1 + self.v2),len(self.v1 + self.v2)))
        #now, the block12 matrix only contain the F_{ij} element in all the diagonal
        F_ij = self.i.T @ self.fock_canonical @ self.j
        np.fill_diagonal(block12, -F_ij)
        block12 -= ao2mo.general(self.mol_loc, [self.virt,self.virt,self.j,self.i],
         compact=False).reshape(len(self.v1 + self.v2),len(self.v1 + self.v2))
        block12 -= ao2mo.general(self.mol_loc, 
                [self.virt,self.j,self.virt,self.i],
                compact=False).reshape(len(self.v1 +self.v2),len(self.v1 + self.v2))    

        block21 = np.zeros((len(self.v1 + self.v2),len(self.v1 + self.v2)))
        #now, the block21 matrix only contain the F_{ij} element in all the diagonal
        F_ji = self.j.T @ self.fock_canonical @ self.i
        np.fill_diagonal(block21, -F_ji)
        block21 -= ao2mo.general(self.mol_loc, [self.virt,self.virt,self.i,self.j],
            compact=False).reshape(len(self.v1 + self.v2),len(self.v1 + self.v2))
        block21 -= ao2mo.general(self.mol_loc, 
                [self.virt,self.i,self.virt,self.j],
                compact=False).reshape(len(self.v1 +self.v2),len(self.v1 + self.v2))    

        block22 = np.zeros((len(self.v1 + self.v2),len(self.v1 + self.v2)))
        
        superblock1 = np.concatenate((block11, block12), axis=1)
        superblock2 = np.concatenate((block21, block22), axis=1)
        M = np.concatenate((superblock1, superblock2), axis=0)

        return M



@attr.s
class Cloppa_full:
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
        occ = attr.ib(default=None)
        vir = attr.ib(default=None)


        def __attrs_post_init__(self):
                if self.occ != None: 
                        self.orbv = self.mo_coeff_loc[:,self.vir]
                        self.orbo = self.mo_coeff_loc[:,self.occ]
                else:
                        self.occidx = np.where(self.mo_occ_loc==2)[0]
                        self.viridx = np.where(self.mo_occ_loc==0)[0]
                        
                        self.orbv = self.mo_coeff_loc[:,self.viridx]
                        self.orbo = self.mo_coeff_loc[:,self.occidx]
                        
                self.nvir = self.orbv.shape[1]
                self.nocc = self.orbo.shape[1]
                self.mo = np.hstack((self.orbo,self.orbv))
                self.nmo = self.nocc + self.nvir
                #here we made a SCF calculation of the molecule
                self.mol = gto.M(atom=str(self.mol_input), basis=self.basis, verbose=0)
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

        def  P(self):
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


@attr.s
class Cloppa_pathways:
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
        o1 = attr.ib(default=None, type=list, validator=attr.validators.instance_of(list))
        o2 = attr.ib(default=None, type=list, validator=attr.validators.instance_of(list))
        v1 = attr.ib(default=None, type=list, validator=attr.validators.instance_of(list))
        v2 = attr.ib(default=None, type=list, validator=attr.validators.instance_of(list))

        def __attrs_post_init__(self):
                self.vir = self.v1+self.v2
                self.occ = self.o1+self.o2

                self.orbv = self.mo_coeff_loc[:,self.vir]
                self.orbo = self.mo_coeff_loc[:,self.occ]

                self.nvir = self.orbv.shape[1]
                self.nocc = self.orbo.shape[1]
                self.mo = np.hstack((self.orbo,self.orbv))
                self.nmo = self.nocc + self.nvir
                #here we made a SCF calculation of the molecule
                self.mol = gto.M(atom=str(self.mol_input), basis=self.basis, verbose=0)
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

        @property
        def elements(self, i,a,j,b):
                m = self.M
                m_resheped = m.reshape((self.nocc,self.nvir,self.nocc,self.nvir))
                #m_resheped = m_resheped.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
                return m_resheped[i,a-self.nocc,j,b-self.nocc]
