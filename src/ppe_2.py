import numpy as np
from pyscf import gto, ao2mo, scf
from os import remove, path
from scipy.linalg import expm
import attr


@attr.s
class M_matrix:
    """[summary]
    Clase para calcular la matriz inversa del propagador principal utilizando
    orbitales moleculares localizados previamente. No se tiene en cuenta
    el término A(0) (que tiene que ver con las energías). La ecuación
    es M = (A(1) +- B(1)). Donde A y B pueden ser las matrices triplete o
    singlete.
    Ref: Aucar G.A., Concepts in Magnetic Resonance,2008,doi:10.1002/cmr.a.20108

    o1 y o2 son los números de "orden" de los LMOs ocupados, a uno y otro lado
    de la molécula.
    v1 y v2 son los números de "orden" de los LMOs desocupados.
    Los LMOs van a estar dados por los coeficientes de constracción mo_coeff,
    donde i,j son los LMOs ocupados y a,b los LMOs desocupados.

    Todos los elementos matriciales son calculados cuando llamamos a la clase,
    luego, debemos llamar a las distintas funciones para obtener las distintas
    funcionalidades.

    La diferencia con la clase Inverse_principal_propagator es que aquí calcula
    solamente el elemento matricial correspondiente a M_{ia,jb}
    Returns
    -------
    [type]
        [description]
    """

    occ = attr.ib(default=None, type=list)
    vir = attr.ib(default=None, type=list)
    mo_coeff = attr.ib(default=None, type=np.ndarray)
    mol = attr.ib(default=None, type=gto.Mole)
    triplet = attr.ib(default=True, type=bool)
    mo_occ = attr.ib(default=None)
    classical = attr.ib(default=False, type=bool)
    mo_occ = attr.ib(default=None)


    @property
    def fock_matrix_canonical(self):
        self.fock_canonical = self.mf.get_fock()
        return self.fock_canonical

    def __attrs_post_init__(self):
        #if self.occ == None:
        #    self.occidx = np.where(self.mo_occ>0)[0]
        #    self.viridx = np.where(self.mo_occ==0)[0]

        #    self.orbv = self.mo_coeff[:,self.viridx]
        #    self.orbo = self.mo_coeff[:,self.occidx]
        #else:
        self.orbo = self.mo_coeff[:,self.occ]
        self.orbv = self.mo_coeff[:,self.vir]
    

        self.nocc = self.orbo.shape[1]        
        self.nvir = self.orbv.shape[1]
        
        self.mo = np.hstack((self.orbo,self.orbv))
        
        self.nmo = self.nocc + self.nvir

        if self.classical == True:
            self.mf = scf.RHF(self.mol).run()
            self.m = np.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
            fock = self.fock_matrix_canonical
            for i in range(self.nocc):
                for j in range(self.nocc):
                    for a in range(self.nvir):
                        for b in range(self.nvir):
                            if a==b:
                                self.m[i,a,j,b] -= self.orbo[:,i].T @ fock @ self.orbo[:,j]
                            if i==j:
                                self.m[i,a,j,b] += self.orbv[:,a].T @ fock @ self.orbv[:,b]
        
        elif self.classical == False:
            eri_mo = ao2mo.general(self.mol, 
                [self.mo,self.mo,self.mo,self.mo], compact=False)
            eri_mo = eri_mo.reshape(self.nmo,self.nmo,self.nmo,self.nmo)
            self.m = np.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
            self.m -= np.einsum('ijba->iajb', eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
            if self.triplet:
                self.m -= np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
            elif not self.triplet:
                self.m += np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])

        self.m = self.m.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
        return self.m

    @property
    def rho(self):
        self.occidx = np.where(self.mo_occ>0)[0]
        self.viridx = np.where(self.mo_occ==0)[0]

        self.orbv = self.mo_coeff[:,self.viridx]
        self.orbo = self.mo_coeff[:,self.occidx]
        self.nocc = self.orbo.shape[1]        
        self.nvir = self.orbv.shape[1]
        
        self.mo = np.hstack((self.orbo,self.orbv))
        
        self.nmo = self.nocc + self.nvir
        
        eri_mo = ao2mo.general(self.mol, 
                [self.mo,self.mo,self.mo,self.mo], compact=False)
        eri_mo = eri_mo.reshape(self.nmo,self.nmo,self.nmo,self.nmo)
        self.m_full = np.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
        self.m_full -= np.einsum('ijba->iajb', eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
        if self.triplet:
            self.m_full -= np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
        elif not self.triplet:
            self.m_full += np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])

        self.m_full = self.m_full.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
        exp = expm(self.m_full)
        self.Z = np.trace(exp)
        self.rho_ = exp/self.Z
        return self.rho_


    @property
    def entropy_iaia(self):
        """Entanglement of the M_{ia,jb} matrix:
        M = (M_{ia,ia}  )
            
        Returns
        -------
        [real]
            [value of entanglement]
        """
        m = self.m 
        self.m_iaia = m[:m.shape[0]//4, :m.shape[0]//4] 
        #rho = self.rho
        
        orb_a = self.vir[:(len(self.vir)//2)]
        orb_i = self.occ[:(len(self.occ)//2)]
        #rho_reshaped = rho.reshape(self.nocc,self.nvir,self.nocc,self.nvir)
        #for i in orb_i:
        #    for a in orb_a:
        #        rho_reshaped[i,a-self.nocc,i,a-self.nocc] = 0
        #rho = rho_reshaped.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
        #tr_rho = np.trace(rho)
        
        rho_iaia = expm(self.m_iaia)#*tr_rho/self.Z
        
        eigenvalues = np.linalg.eigvals(rho_iaia)
        
        ent_ia = 0
        for eig in eigenvalues:    
            ent_ia += -eig*np.log(eig)
        return ent_ia

    @property
    def entropy_iajb(self):
        """Entanglement of the M_{ia,jb} matrix:
        M = (M_{ia,ia}  )
            
        Returns
        -------
        [real]
            [value of entanglement]
        """
        m = self.m 
        self.m_iajb = np.zeros((m.shape[0]//2,m.shape[0]//2))
        self.m_iajb[:self.m_iajb.shape[0]//2, :self.m_iajb.shape[0]//2] += m[int(m.shape[0]*3/4):, :int(m.shape[0]*1/4)]
        self.m_iajb[self.m_iajb.shape[0]//2:, self.m_iajb.shape[0]//2:] += m[:int(m.shape[0]*1/4), int(m.shape[0]*3/4):]
        #self.m_iajb[self.m_iajb.shape[0]//2:, :self.m_iajb.shape[0]//2] += m[:int(m.shape[0]*1/4):, :int(m.shape[0]*1/4)]
        #self.m_iajb[:self.m_iajb.shape[0]//2, self.m_iajb.shape[0]//2:] += m[int(m.shape[0]*3/4):, int(m.shape[0]*3/4):]

        #rho = self.rho_
        orb_a = self.vir[:(len(self.vir)//2)]
        orb_i = self.occ[:(len(self.occ)//2)]
        orb_b = self.vir[(len(self.vir)//2):]
        orb_j = self.occ[(len(self.occ)//2):]
        #rho_reshaped = rho.reshape(self.nocc,self.nvir,self.nocc,self.nvir)
        #for i in orb_i:
        #    for a in orb_a:
        #        rho_reshaped[i,a-self.nocc,i,a-self.nocc] = 0
        
        #for j in orb_j:
        #    for b in orb_b:
        #        rho_reshaped[j,b-self.nocc,j,b-self.nocc] = 0
        
        #rho = rho_reshaped.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
        #tr_rho = np.trace(rho)
        rho_iajb = expm(self.m_iajb)#*tr_rho/self.Z
        print(rho_iajb)
        eigenvalues = np.linalg.eigvals(rho_iajb)
        print(eigenvalues)
        ent_iajb = 0
        for eig in eigenvalues:    
            ent_iajb += -eig*np.log(eig)
        return ent_iajb


    @property
    def entropy_jbjb(self):
        """Entanglement of the M_{ia,jb} matrix:
        M = (M_{ia,ia}  )
            
        Returns
        -------
        [real]
            [value of entanglement]
        """
        m = self.m 
        self.m_jbjb = m[int(m.shape[0]*3/4):, int(m.shape[0]*3/4):]
        #rho = self.rho_
        orb_b = self.vir[(len(self.vir)//2):]
        orb_j = self.occ[(len(self.occ)//2):]
        #rho_reshaped = rho.reshape(self.nocc,self.nvir,self.nocc,self.nvir)
        #for j in orb_j:
        #    for b in orb_b:
        #        rho_reshaped[j,b-self.nocc,j,b-self.nocc] = 0
        
        #rho = rho_reshaped.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
        #tr_rho = np.trace(rho)
        
        rho_jbjb = expm(self.m_jbjb)#*tr_rho/self.Z
        eigenvalues = np.linalg.eigvals(rho_jbjb)
        ent_jb = 0
        for eig in eigenvalues:    
            ent_jb += -eig*np.log(eig)
        return ent_jb

    @property
    def entropy_iajb_1(self):
        """Entanglement of the M_{ia,jb} matrix:
        M = (M_{ia,ia}  )
            
        Returns
        -------
        [real]
            [value of entanglement]
        """
        m = self.m 
        self.m_iajb = m[int(m.shape[0]*3/4):, :int(m.shape[0]*1/4)]
        #print(self.m_iajb)
        rho = np.exp(self.m_iajb)
        #print(rho)
        ent_jb = -rho*np.log(rho)
        #print(ent_jb)
        return ent_jb[0][0]
