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
        self.M = np.concatenate((bloq1_a, bloq2_a), axis=0)
        eigenvalues = np.linalg.eigvals(self.M)
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
    def M(self):
        """Sum of the matrix elements of the inverse of the Principal Propagator

        Returns
        -------
        [type]
            [description]
        """
        
        return self.M_total


