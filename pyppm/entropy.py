import numpy as np
from pyscf import scf

from pyppm.hrpa import HRPA
from pyppm.rpa import RPA


class entropy:
    """Class to compute the entropy of systems formed by virtual excitations.
    Density matrix corresponding to those systems are formed by
    the inverse matrix of the principal propagator using a set of previously
    localized occupied and virtual molecular orbitals.
    Is not taken into account the term A(0) (which contains molecular energies)
    Ref: Millan et. al Phys. Chem. Chem. Phys., 2018, DOI: 10.1039/C8CP03480J
    It can be use the single or the triplet inverse of principal propagator.
    Ref: Aucar G.A. Concepts in Magnetic Resonance,2008,doi:10.1002/cmr.a.20108

    mf = RHF object
    m = communicator matrix at any level of approach (RPA of HRPA). This matrix
    is the principal propagator inverse without the A(0) term.

    triplet [bool]. If True, it use the triplet principal propagator inverse,
    if false, use the singlet.

    occ [list] = Order number of the set of occupied LMO in the localized
    mo_coeff coefficient matrix whit which you want to form the system.

    vir [list] = Order number of the set of virtual LMOs in the localized
    mo_coeff coefficient matrix with which you want to form the system

    The order of occ and vir list is very important in order to calculate the
    quantum entanglement
    between two virtual excitations that are in diferent bonds.
    In both list you must put the number order of LMOs centered in one bond,
    and then in the other bond, in such a way that both list are divided in two
    with number orders that correspond to one bond, and another.

    Returns:
        numpy.ndarray: Communicator matrix of the chosen system
    """

    def __init__(
        self,
        occ,
        vir,
        mo_coeff_loc,
        elec_corr="RPA",
        mf=None,
        triplet=True,
        z_allexc=True,
    ):
        self.occ = occ
        self.vir = vir
        self.mo_coeff_loc = mo_coeff_loc
        self.elec_corr = elec_corr
        self.mf = mf if mf is not None else scf.hf.RHF()
        self.triplet = triplet
        self.z_allexc = z_allexc
        self.__post_init__()

    def __post_init__(self):

        occ = self.occ
        vir = self.vir
        nocc_loc = len(occ)
        nvir_loc = len(vir)
        mf = self.mf
        mo_coeff_loc = self.mo_coeff_loc
        nocc = np.count_nonzero(mf.mo_occ > 0)
        nvir = np.count_nonzero(mf.mo_occ == 0)
        if self.elec_corr == "RPA":
            m = RPA(mf=self.mf).Communicator(triplet=self.triplet)
        elif self.elec_corr == "HRPA":
            m = HRPA(mf=self.mf).Communicator(triplet=self.triplet)
        else:
            raise Exception("Only RPA and HRPA are implemented in this code")
        can_inv = np.linalg.inv(mf.mo_coeff.T)
        c_occ = (mo_coeff_loc[:, :nocc].T.dot(can_inv[:, :nocc])).T

        c_vir = (mo_coeff_loc[:, nocc:].T.dot(can_inv[:, nocc:])).T
        total = np.einsum("ij,ab->iajb", c_occ, c_vir)
        total = total.reshape(nocc * nvir, nocc * nvir)
        m_loc = total.T @ m @ total
        m_loc = m_loc.reshape(nocc, nvir, nocc, nvir)
        m_loc_red = np.zeros((nocc_loc, nvir_loc, nocc_loc, nvir_loc))
        for i, ii in enumerate(occ):
            for j, jj in enumerate(occ):
                for a, aa in enumerate(vir):
                    for b, bb in enumerate(vir):
                        m_loc_red[i, a, j, b] = m_loc[
                            ii, aa - nocc, jj, bb - nocc
                        ]
        m_loc_red = m_loc_red.reshape(
            (nocc_loc * nvir_loc, nocc_loc * nvir_loc)
        )
        m_iajb = np.zeros((m_loc_red.shape[0] // 2, m_loc_red.shape[0] // 2))
        m_iajb[m_iajb.shape[0] // 2 :, : m_iajb.shape[0] // 2] += m_loc_red[
            int(m_loc_red.shape[0] * 3 / 4) :,
            : int(m_loc_red.shape[0] * 1 / 4),
        ]
        m_iajb[: m_iajb.shape[0] // 2, m_iajb.shape[0] // 2 :] += m_loc_red[
            : int(m_loc_red.shape[0] * 1 / 4),
            int(m_loc_red.shape[0] * 3 / 4) :,
        ]
        m_iajb[: m_iajb.shape[0] // 2, : m_iajb.shape[0] // 2] += m_loc_red[
            : int(m_loc_red.shape[0] * 1 / 4),
            : int(m_loc_red.shape[0] * 1 / 4),
        ]
        m_iajb[m_iajb.shape[0] // 2 :, m_iajb.shape[0] // 2 :] += m_loc_red[
            int(m_loc_red.shape[0] * 3 / 4) :,
            int(m_loc_red.shape[0] * 3 / 4) :,
        ]

        self.eigenvalues = np.linalg.eigvals(m_iajb)
        m_loc = m_loc.reshape(nocc * nvir, nocc * nvir)
        self.Z = 0
        if self.z_allexc is True:
            eig = np.linalg.eigvals(m_loc)
            for i in eig:
                self.Z += np.exp(np.real(i))
        else:
            for i in self.eigenvalues:
                self.Z += np.exp(np.real(i))
        self.m = m_loc_red
        return self.m

    @property
    def entropy_iaia(self):
        """Entanglement of the M_{ia,ia} matrix:
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
        self.m_jbjb = m[int(m.shape[0] * 3 / 4) :, int(m.shape[0] * 3 / 4) :]
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
