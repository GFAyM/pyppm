import numpy as np
from pyscf import gto, ao2mo, scf
from os import remove, path
from functools import reduce
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
        m = self.m
        self.M = np.zeros((m.shape[0]//2,m.shape[0]//2))
        self.M[self.M.shape[0]//2:, :self.M.shape[0]//2] += m[int(m.shape[0]*3/4):, :int(m.shape[0]*1/4)]
        self.M[:self.M.shape[0]//2, self.M.shape[0]//2:] += m[:int(m.shape[0]*1/4), int(m.shape[0]*3/4):]
        self.M[:self.M.shape[0]//2, :self.M.shape[0]//2] += m[:int(m.shape[0]*1/4), :int(m.shape[0]*1/4)]
        self.M[self.M.shape[0]//2:, self.M.shape[0]//2:] += m[int(m.shape[0]*3/4):, int(m.shape[0]*3/4):]
        exp_ = expm(self.M)
        self.Z = np.trace(exp_)
        #self.Z = 1

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
        orb_a = self.vir[:(len(self.vir)//2)]
        orb_i = self.occ[:(len(self.occ)//2)]
        
        exp_ = expm(self.m_iaia)/self.Z
        eigenvalues = np.linalg.eigvals(exp_)
        
        ent_iajb = 0
        rho_ = eigenvalues
        rho_ = rho_
        for eig in rho_:    
                ent_iajb += -eig*np.log(eig)
        return ent_iajb
        

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
        self.m_iajb[self.m_iajb.shape[0]//2:, :self.m_iajb.shape[0]//2] += m[int(m.shape[0]*3/4):, :int(m.shape[0]*1/4)]
        self.m_iajb[:self.m_iajb.shape[0]//2, self.m_iajb.shape[0]//2:] += m[:int(m.shape[0]*1/4), int(m.shape[0]*3/4):]
        orb_a = self.vir[:(len(self.vir)//2)]
        orb_i = self.occ[:(len(self.occ)//2)]
        orb_b = self.vir[(len(self.vir)//2):]
        orb_j = self.occ[(len(self.occ)//2):]
        exp_ = expm(self.m_iajb)/self.Z
        eigenvalues = np.linalg.eigvals(exp_)

        ent_iajb = 0
        rho_ = eigenvalues
        rho_ = rho_
        for eig in rho_:    
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
        exp_ = expm(self.m_jbjb)/self.Z
        eigenvalues = np.linalg.eigvals(exp_)
        ent_iajb = 0
        rho_ = eigenvalues
        rho_ = rho_
        for eig in rho_:    
                ent_iajb += -eig*np.log(eig)
        return ent_iajb
