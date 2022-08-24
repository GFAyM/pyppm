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
        self.z = np.zeros((self.nocc,self.nvir,self.nocc,self.nvir))
        self.z -= np.einsum('ijba->iajb', eri_mo[:self.nocc,:self.nocc,self.nocc:,self.nocc:])
        if self.triplet:
            self.z -= np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])
        elif not self.triplet:
            self.z += np.einsum('jaib->iajb', eri_mo[:self.nocc,self.nocc:,:self.nocc,self.nocc:])

        self.z = self.z.reshape((self.nocc*self.nvir,self.nocc*self.nvir))
        z = self.z
        self.m_iajb = np.zeros((z.shape[0]//2,z.shape[0]//2))
        self.m_iajb[self.m_iajb.shape[0]//2:, :self.m_iajb.shape[0]//2] += z[int(z.shape[0]*3/4):, :int(z.shape[0]*1/4)]
        self.m_iajb[:self.m_iajb.shape[0]//2, self.m_iajb.shape[0]//2:] += z[:int(z.shape[0]*1/4), int(z.shape[0]*3/4):]
        eigenvalue_z = np.linalg.eigvals(self.m_iajb)
        #eigenvalue_z = np.linalg.eigvals(self.z)
        self.Z = 0
        for i in eigenvalue_z:
            self.Z += np.exp(i)
        return self.Z

    def orb_m(self,orb_occ,orb_vir):
        """Function for the orbital entanglement \rho_iajb 

        Args:
            orb_occ (_type_): _description_ occupied orbitals ij
            orb_vir (_type_): _description_ virtual orbitals ab

        Returns:
            M(1) (triplet or singlet) for the selected orbitals
        """
        self.orbo = self.mo_coeff[:,orb_occ]
        self.orbv = self.mo_coeff[:,orb_vir]
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
        return self.m

    @property
    def Z_partition(self):
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
        tr = np.trace(self.m_full)
        Z=0
        for i in tr:
            Z += np.exp(i)
        return Z



    def entropy_iaia(self,m):
        """Entanglement of the M_{ia,jb} matrix:
        M = (M_{ia,ia}  )
            
        Returns
        -------
        [real]
            [value of entanglement]
        """ 
        self.m_iaia = m[:m.shape[0]//4, :m.shape[0]//4]
        eigenvalues = np.linalg.eigvals(self.m_iaia)
        
        Z = self.Z
        ent = 0
        for i in eigenvalues:
            ent += -np.exp(i)/Z*np.log(np.exp(i)/Z)
        
        return ent



    def entropy_iajb(self,m):
        """Entanglement of the M_{ia,jb} matrix:
        M = (M_{ia,ia}  )
            
        Returns
        -------
        [real]
            [value of entanglement]
        """
        self.m_iajb = np.zeros((m.shape[0]//2,m.shape[0]//2))
        self.m_iajb[self.m_iajb.shape[0]//2:, :self.m_iajb.shape[0]//2] += m[int(m.shape[0]*3/4):, :int(m.shape[0]*1/4)]
        self.m_iajb[:self.m_iajb.shape[0]//2, self.m_iajb.shape[0]//2:] += m[:int(m.shape[0]*1/4), int(m.shape[0]*3/4):]
        eigenvalues = np.linalg.eigvals(self.m_iajb)
        Z = self.Z        
        ent = 0
        for i in eigenvalues:
            ent += -np.exp(i)/Z*np.log(np.exp(i)/Z)
            #ent += -np.exp(i)*np.log(np.exp(i))
        return ent


    def entropy_jbjb(self,m):
        """Entanglement of the M_{ia,jb} matrix:
        M = (M_{ia,ia}  )
            
        Returns
        -------
        [real]
            [value of entanglement]
        """
        m = self.m 
        self.m_jbjb = m[int(m.shape[0]*3/4):, int(m.shape[0]*3/4):]
        eigenvalues = np.linalg.eigvals(self.m_jbjb)
        Z = self.Z
        ent = 0
        for i in eigenvalues:
            ent += -np.exp(i)/Z*np.log(np.exp(i)/Z)
            #ent += -np.exp(i)*np.log(np.exp(i))    
        return ent