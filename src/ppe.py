import numpy as np
from pyscf.gto.mole import M
from pyscf.tools import mo_mapping
from pyscf import tools, gto, ao2mo
from os import remove, path
from functools import reduce
import math
import pandas as pd
import attr


@attr.s
class inverse_principal_propagator:
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
    Returns
    -------
    [type]
        [description]
    """

    o1 = attr.ib(default=None, type=list)
    o2 = attr.ib(default=None, type=list)
    v1 = attr.ib(default=None, type=list)
    v2 = attr.ib(default=None, type=list)
    mo_coeff = attr.ib(default=None, type=np.ndarray)
    mol = attr.ib(default=None, type=gto.Mole)
    spin_dependence = attr.ib(default="triplet", type=str)



    def __attrs_post_init__(self):
        if self.spin_dependence == "singlet":
            factor = -1
        elif self.spin_dependence == "triplet":
            factor = 1
        self.i = self.mo_coeff[:, self.o1]
        self.a = self.mo_coeff[:, self.v1]
        self.j = self.mo_coeff[:, self.o2]
        self.b = self.mo_coeff[:, self.v2]

        self.block11 = -ao2mo.general(self.mol, [self.a, self.a, self.i, self.i], compact=False)
        self.block11 = self.block11.reshape(len(self.a[0]), len(self.a[0]))
        self.block11 -= factor * ao2mo.general(self.mol, [self.i, self.a, self.i, self.a], compact=False)

        self.block22 = -ao2mo.general(self.mol, [self.b, self.b, self.j, self.j], compact=False)
        self.block22 = self.block22.reshape(len(self.a[0]), len(self.a[0]))
        self.block22 -= factor * ao2mo.general(self.mol, [self.j, self.b, self.j, self.b], compact=False)

        self.block12 = -ao2mo.general(self.mol, [self.a, self.b, self.j, self.i], compact=False)
        self.block12 = self.block12.reshape(len(self.a[0]), len(self.a[0]))
        self.block12 -= factor * ao2mo.general(self.mol, [self.i, self.b, self.j, self.a], compact=False)

        self.block21 = -ao2mo.general(self.mol, [self.b, self.a, self.i, self.j], compact=False)
        self.block21 = self.block21.reshape(len(self.a[0]), len(self.a[0]))
        self.block21 -= factor * ao2mo.general(self.mol, [self.j, self.a, self.i, self.b], compact=False)

        block1 = np.concatenate((self.block11, self.block12), axis=1)
        block2 = np.concatenate((self.block21, self.block22), axis=1)
        self.M_total = np.concatenate((block1, block2), axis=0)
        
    @property
    def entropy_ia(self):
        """Entrelazamiento entre LMOs de un lado de la molécula, utilizando
        el esquema de Millán et. al JCCP 2018

        Returns
        -------
        [real]
            [valor de entrelazamiento]
        """
        eigenvalues = np.linalg.eigvals(self.block11)
        totalent = []
        global ent
        Z = 0
        val = []
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a / Z * np.log(a / Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]

    @property
    def entropy_jb(self):
        """Similar a entanglement_ia, pero utilizando los LMOs del otro lado
        de la molécula

        Returns
        -------
        [real]
            [value of the entanglement]
        """
        eigenvalues = np.linalg.eigvals(self.block22)
        totalent = []
        global ent
        Z = 0
        val = []
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a / Z * np.log(a / Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]

    @property
    def entropy_iajb_mixedstate(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = (0         M_{ia,jb} )
            (M_{jb,ia}          0)
        Returns
        -------
        [real]
            [value of entanglement]
        """
        zero_matrix = np.zeros((len(self.a[0]), len(self.a[0])))
        bloq1_a = np.concatenate((zero_matrix, self.block12), axis=1)
        bloq2_a = np.concatenate((self.block21, zero_matrix), axis=1)
        M = np.concatenate((bloq1_a, bloq2_a), axis=0)
        eigenvalues = np.linalg.eigvals(M)
        totalent = []
        global ent
        Z=0
        val=[]
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a/Z*np.log(a/Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]

    @property
    def entropy_iajb_mixedstate_2(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = (M_{ia,jb}       0   )
            (    0      M_{jb,ia})
        Returns
        -------
        [real]
            [value of entanglement]
        """
        zero_matrix = np.zeros((len(self.a[0]), len(self.a[0])))
        bloq1_a = np.concatenate((self.block12, zero_matrix), axis=1)
        bloq2_a = np.concatenate((zero_matrix, self.block21), axis=1)
        M = np.concatenate((bloq1_a, bloq2_a), axis=0)
        eigenvalues = np.linalg.eigvals(M)
        totalent = []
        global ent
        Z=0
        val=[]
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a/Z*np.log(a/Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]

    @property
    def entropy_iajb(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = (M_{ia,jb})
        Returns
        -------
        [real]
            [value of entanglement]
        """

        eigenvalues = np.linalg.eigvals(self.block12)
        totalent = []
        global ent
        Z=0
        val=[]
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a/Z*np.log(a/Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]


    @property
    def m_iajb_mixedstate(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = (0         M_{ia,jb} )
            (M_{jb,ia}          0)
        Returns
        -------
        [real]
            [value of entanglement]
        """
        zero_matrix = np.zeros((len(self.a[0]), len(self.a[0])))
        bloq1_a = np.concatenate((zero_matrix, self.block12), axis=1)
        bloq2_a = np.concatenate((self.block21, zero_matrix), axis=1)
        M = np.concatenate((bloq1_a, bloq2_a), axis=0)
        #eigenvalues = np.linalg.eigvals(M)
        return M

    @property
    def m_iajb_mixedstate_2(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = (M_{ia,jb}         0 )
            (0          M_{jb,ia})
        Returns
        -------
        [real]
            [value of entanglement]
        """
        zero_matrix = np.zeros((len(self.a[0]), len(self.a[0])))
        bloq1_a = np.concatenate((self.block12, zero_matrix), axis=1)
        bloq2_a = np.concatenate((zero_matrix, self.block21), axis=1)
        M = np.concatenate((bloq1_a, bloq2_a), axis=0)
        return M



    @property
    def m_iajb(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = M_{ia,jb}
        Returns
        -------
        [real]
            [value of entanglement]
        """
        M = self.block12
        return M

    @property
    def m_iaia(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = M_{ia,ia}
        Returns
        -------
        [real]
            [value of entanglement]
        """
        M = self.block11
        return M

    @property
    def entropy_iajb(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = (M_{ia,ia})
        Returns
        -------
        [real]
            [value of entanglement]
        """
        eigenvalues = np.linalg.eigvals(self.m_iajb)
        totalent = []
        global ent
        Z = 0
        val = []
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a / Z * np.log(a / Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]

    @property
    def entropy_iajb_purestate(self):
        """Entanglement between LMOs on diferents places of the molecule using
        the Inverse of the PP like:
        M = (M_{ia,ia}        0    )
            (   0         M_{jb,jb})
        Returns
        -------
        [real]
            [value of entanglement]
        """
        zero_matrix = np.zeros((len(self.a[0]), len(self.a[0])))
        bloq1_a = np.concatenate((self.block11, zero_matrix), axis=1)
        bloq2_a = np.concatenate((zero_matrix, self.block22), axis=1)
        self.M = np.concatenate((bloq1_a, bloq2_a), axis=0)
        eigenvalues = np.linalg.eigvals(self.M)
        totalent = []
        global ent
        Z = 0
        val = []
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a / Z * np.log(a / Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]


    @property
    def mutual_information(self):
        I = self.entropy_ia + self.entropy_jb - self.entropy_iajb_mixedstate
        return I

    @property
    def mutual_information_2(self):
        I = self.entropy_ia + self.entropy_jb - self.entropy_iajb_mixedstate_2
        return I


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



    def __attrs_post_init__(self):
        if self.occ == None:
            self.occidx = np.where(self.mo_occ>0)[0]
            self.viridx = np.where(self.mo_occ==0)[0]

            self.orbv = self.mo_coeff[:,self.viridx]
            self.orbo = self.mo_coeff[:,self.occidx]
        else:
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
        return self.m


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
        eigenvalues = np.linalg.eigvals(self.m_iaia)
        totalent = []
        global ent
        Z = 0
        val = []
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a / Z * np.log(a / Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]

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
        eigenvalues = np.linalg.eigvals(self.m_iajb)
        totalent = []
        global ent
        Z = 0
        val = []
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a / Z * np.log(a / Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]

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
        eigenvalues = np.linalg.eigvals(self.m_jbjb)
        totalent = []
        global ent
        Z = 0
        val = []
        for i in eigenvalues:
            Z += math.exp(i)
            val.append(math.exp(i))
            ent = [-a / Z * np.log(a / Z) for a in val]
        totalent.append(sum(ent))
        return totalent[0]
