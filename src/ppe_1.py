import numpy as np
from pyscf import gto, ao2mo, scf
from os import remove, path
from functools import reduce
import math

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

    def __attrs_post_init__(self):
        self.orbo = self.mo_coeff[:,self.occ]
        self.orbv = self.mo_coeff[:,self.vir]
        self.nocc = self.orbo.shape[1]        
        self.nvir = self.orbv.shape[1]        
        self.mo = np.hstack((self.orbo,self.orbv))        
        self.nmo = self.nocc + self.nvir
        
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
        eigenvalues = np.linalg.eigvals(np.exp(self.m_iaia)) # e^(m) da autovalores negativos
        diagonals = np.diag(np.exp(self.m_iaia)) # e^(m) da autovalores negativos
        
        #print(eigenvalues)
        #print(np.exp(self.m_iaia))
        #print(np.diag(np.exp(self.m_iaia)))
        #eigenvalues = np.exp(eigenvalues)
        ent = 0
        for i in diagonals:
            ent += -i*np.log(i)
        #print(ent)
        return ent


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
        #print(self.m_iajb)
        eigenvalues = np.diag(self.m_iajb)
        eigenvalues = np.exp(eigenvalues)
        ent = 0
        for i in eigenvalues:
            ent += -i*np.log(i)
        #    ent += -i*np.log(i)
        return ent

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
        eigenvalues = np.linalg.eigvals(np.exp(-self.m_jbjb))
        return eigenvalues
        #ent = 0
        #for i in eigenvalues:
            #ent += -np.exp(i)/Z*np.log(np.exp(i)/Z)
        #    ent += -i*np.log(i)    
        #return ent