import numpy as np
from pyscf import gto, ao2mo
import attr


@attr.s
class entropy:
    """[summary]
    Class to compute the entropy of systems formed by virtual excitations.
    Density matrix corresponding to those systems are formed by
    the inverse matrix of the principal propagator using a set of previously
    localized occupied and virtual molecular orbitals.
    Is not taken into account the term A(0) (which contains molecular energies)
    Ref: Millan et. al Phys. Chem. Chem. Phys., 2018, DOI: 10.1039/C8CP03480J
    It can be use the single or the triplet inverse of principal propagator.
    Ref: Aucar G.A., Concepts in Magnetic Resonance,2008,doi:10.1002/cmr.a.20108

    mo_coeff [np.array] = Localized molecular coefficients
    mol [gto.Mole] = mole of the molecule
    triplet [bool] = if True, it use the triplet principal propagator inverse,
    if false, use the singlet.
    occ [list] = Order number of the set of occupied LMO in the localized
    mo_coeff coefficient matrix whit which you want to form the system.

    vir [list] = Order number of the set of virtual LMOs in the localized
    mo_coeff coefficient matrix with which you want to form the system

    The order is very important in order to calculate the quantum entanglement
    between two virtual excitations that are in diferent bonds.
    In both list you must put the number order of LMOs centered in one bond,
    and then in the other bond, in such a way that both list are divided in two,
    with number orders that correspond to one bond, and another.


    -------
    [type]
        [description]
    """

    occ = attr.ib(default=None, type=list)
    vir = attr.ib(default=None, type=list)
    mo_coeff = attr.ib(default=None, type=np.ndarray)
    mol = attr.ib(default=None, type=gto.Mole)
    triplet = attr.ib(default=True, type=bool)

    def __attrs_post_init__(self):
        self.orbo = self.mo_coeff[:, self.occ]
        self.orbv = self.mo_coeff[:, self.vir]

        self.nocc = self.orbo.shape[1]
        self.nvir = self.orbv.shape[1]

        self.mo = np.hstack((self.orbo, self.orbv))

        self.nmo = self.nocc + self.nvir

        eri_mo = ao2mo.general(
            self.mol, [self.mo, self.mo, self.mo, self.mo], compact=False
        )
        eri_mo = eri_mo.reshape(self.nmo, self.nmo, self.nmo, self.nmo)
        self.m = np.zeros((self.nocc, self.nvir, self.nocc, self.nvir))
        self.m -= np.einsum(
            "ijba->iajb", eri_mo[: self.nocc, : self.nocc, self.nocc :, self.nocc :]
        )
        if self.triplet:
            self.m -= np.einsum(
                "jaib->iajb", eri_mo[: self.nocc, self.nocc :, : self.nocc, self.nocc :]
            )
        elif not self.triplet:
            self.m += np.einsum(
                "jaib->iajb", eri_mo[: self.nocc, self.nocc :, : self.nocc, self.nocc :]
            )

        self.m = self.m.reshape((self.nocc * self.nvir, self.nocc * self.nvir))
        m = self.m
        m_iajb = np.zeros((m.shape[0] // 2, m.shape[0] // 2))
        m_iajb[m_iajb.shape[0] // 2 :, : m_iajb.shape[0] // 2] += m[
            int(m.shape[0] * 3 / 4) :, : int(m.shape[0] * 1 / 4)
        ]
        m_iajb[: m_iajb.shape[0] // 2, m_iajb.shape[0] // 2 :] += m[
            : int(m.shape[0] * 1 / 4), int(m.shape[0] * 3 / 4) :
        ]
        m_iajb[: m_iajb.shape[0] // 2, : m_iajb.shape[0] // 2] += m[
            : int(m.shape[0] * 1 / 4), : int(m.shape[0] * 1 / 4)
        ]
        m_iajb[m_iajb.shape[0] // 2 :, m_iajb.shape[0] // 2 :] += m[
            int(m.shape[0] * 3 / 4) :, int(m.shape[0] * 3 / 4) :
        ]

        self.eigenvalues = np.linalg.eigvals(m_iajb)
        self.Z = 0
        for i in self.eigenvalues:
            self.Z += np.exp(i)

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
        self.m_iaia = m[: m.shape[0] // 4, : m.shape[0] // 4]
        eigenvalues = np.linalg.eigvals(self.m_iaia)
        Z = 0
        for i in eigenvalues:
            Z += np.exp(i)
        Z = self.Z
        ent = 0
        for i in eigenvalues:
            ent += -np.exp(i) / Z * np.log(np.exp(i) / Z)
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
        self.m_jbjb = m[
            int(m.shape[0] * 3 / 4) :, int(m.shape[0] * 3 / 4) :
        ]  # * np.sum(np.diag(self.m_iaia))
        eigenvalues = np.linalg.eigvals(self.m_jbjb)
        Z = 0
        for i in eigenvalues:
            Z += np.exp(i)
        ent = 0
        Z = self.Z
        for i in eigenvalues:
            ent += -np.exp(i) / Z * np.log(np.exp(i) / Z)
        return ent

    @property
    def entropy_ab(self):
        """Entanglement of the M_{ia,jb} matrix:
            M = (M_{ia,ia}   M_{ia,jb} )
                (M_{jb,ia}   M_{jb,jb} )
        Returns
        -------
        [real]
            [value of entanglement]
        """

        ent = 0
        for i in self.eigenvalues:
            ent += -(np.exp(i) / self.Z) * np.log(np.exp(i) / self.Z)
        return ent
