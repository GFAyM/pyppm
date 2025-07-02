
import h5py
import numpy as np
from pyscf import lib

from pyppm.hrpa import HRPA
from pyppm.rpa import RPA


class entropy:
    """
    Compute the entropy of systems formed by virtual excitations.

    The density matrix is constructed from the inverse of the principal
    propagator matrix, using a set of localized occupied and virtual
    molecular orbitals. The A(0) term (which includes molecular orbital
    energies) is not included in this formulation.

    This class supports singlet and triplet inverses of the principal
    propagator, depending on the `triplet` flag.

    References:
        - Millán et al., *Phys. Chem. Chem. Phys.*, 2018,
          DOI: 10.1039/C8CP03480J
        - Aucar G.A., *Concepts in Magnetic Resonance*, 2008,
          DOI: 10.1002/cmr.a.20108

    Args:
        occ1 (list[int]): Indices of the first set of occupied LMOs in the
            localized MO basis.
        occ2 (list[int]): Indices of the second set of occupied LMOs.
        vir1 (list[int]): Indices of the first set of virtual LMOs.
        vir2 (list[int]): Indices of the second set of virtual LMOs.
        mo_coeff_loc (numpy.ndarray): Coefficient matrix of the localized
            molecular orbitals.
        elec_corr (str, optional): Type of electronic correlation to use,
            either "RPA" or "HRPA". Default is "RPA".
        mol (pyscf.gto.Mole, optional): PySCF Mole object representing the
            molecule.
        chkfile (str, optional): Path to a PySCF checkpoint file containing
            MOs and occupation numbers.
        triplet (bool, optional): If True, use the triplet communicator
            matrix; otherwise, use singlet. Default is True.
        z_allexc (bool, optional): If True, all eigenvalues of the
            communicator matrix are used to compute the partition function Z;
            if False, only reduced eigenvalues are used. Default is True.
        h5_m (str, optional): Path to an HDF5 file containing the communicator
            matrix. If None, the matrix is computed and saved to a file.
        label (str, optional): Label used for the output file when saving the
            communicator matrix.
    """

    def __init__(
        self,
        occ1,
        occ2,
        vir1,
        vir2,
        mo_coeff_loc,
        elec_corr="RPA",
        mol=None,
        chkfile=None,
        triplet=True,
        z_allexc=True,
        h5_m=None,
        label=None,
    ):
        self.occ1 = occ1
        self.occ2 = occ2
        self.vir1 = vir1
        self.vir2 = vir2
        self.mo_coeff_loc = mo_coeff_loc
        self.elec_corr = elec_corr
        self.mol = mol
        self.chkfile = chkfile
        self.triplet = triplet
        self.z_allexc = z_allexc
        self.label = label
        self.h5_m = h5_m
        self.__post_init__()

    def __post_init__(self):

        occ1 = self.occ1
        occ2 = self.occ2

        vir1 = self.vir1
        vir2 = self.vir2
        mo_coeff_loc = self.mo_coeff_loc
        self.mo_coeff = lib.chkfile.load(self.chkfile, "scf/mo_coeff")
        self.mo_occ = lib.chkfile.load(self.chkfile, "scf/mo_occ")
        self.occidx = np.where(self.mo_occ > 0)[0]
        self.viridx = np.where(self.mo_occ == 0)[0]
        self.orbv = self.mo_coeff[:, self.viridx]
        self.orbo = self.mo_coeff[:, self.occidx]
        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]
        nocc = np.count_nonzero(self.mo_occ > 0)
        nvir = np.count_nonzero(self.mo_occ == 0)
        if self.elec_corr == "RPA":
            if self.h5_m is None:
                m = RPA(mol=self.mol, chkfile=self.chkfile).Communicator(
                    triplet=self.triplet
                )
                with h5py.File(f"m_RPA_{self.label}.h5", "w") as f:
                    f.create_dataset("m", data=m)
            else:
                with h5py.File(self.h5_m, "r") as f:
                    m = f["m"][()]
        elif self.elec_corr == "HRPA":
            if self.h5_m is None:
                m = HRPA(
                    mol=self.mol,
                    chkfile=self.chkfile,
                    calc_int=True,
                ).Communicator(triplet=self.triplet)
                with h5py.File(
                    f"m_HRPA_{self.label}_{self.triplet}.h5", "w"
                ) as f:
                    f.create_dataset("m", data=m)
            else:
                with h5py.File(self.h5_m, "r") as f:
                    m = f["m"][()]
        else:
            raise Exception("Only RPA and HRPA are implemented in this code")
        can_inv = np.linalg.inv(self.mo_coeff.T)
        c_occ = (mo_coeff_loc[:, :nocc].T.dot(can_inv[:, :nocc])).T
        c_vir = (mo_coeff_loc[:, nocc:].T.dot(can_inv[:, nocc:])).T
        total = np.einsum("ij,ab->iajb", c_occ, c_vir)
        total = total.reshape(nocc * nvir, nocc * nvir)
        m_loc = total.T @ m @ total
        m_loc = m_loc.reshape(nocc, nvir, nocc, nvir)
        m_iaia = np.zeros((len(occ1), len(vir1), len(occ1), len(vir1)))
        for i, ii in enumerate(occ1):
            for a, aa in enumerate(vir1):
                for j, jj in enumerate(occ1):
                    for b, bb in enumerate(vir1):
                        m_iaia[i, a, j, b] += m_loc[
                            ii, aa - nocc, jj, bb - nocc
                        ]

        m_jbjb = np.zeros((len(occ2), len(vir2), len(occ2), len(vir2)))
        for i, ii in enumerate(occ2):
            for a, aa in enumerate(vir2):
                for j, jj in enumerate(occ2):
                    for b, bb in enumerate(vir2):
                        m_jbjb[i, a, j, b] += m_loc[
                            ii, aa - nocc, jj, bb - nocc
                        ]

        m_iajb = np.zeros((len(occ1), len(vir1), len(occ2), len(vir2)))
        m_jbia = np.zeros((len(occ2), len(vir2), len(occ1), len(vir1)))
        for i, ii in enumerate(occ1):
            for a, aa in enumerate(vir1):
                for j, jj in enumerate(occ2):
                    for b, bb in enumerate(vir2):
                        m_iajb[i, a, j, b] += m_loc[
                            ii, aa - nocc, jj, bb - nocc
                        ]
                        m_jbia[j, b, i, a] += m_loc[
                            jj, bb - nocc, ii, aa - nocc
                        ]

        m_iajb = m_iajb.reshape(len(occ1) * len(vir1), len(occ2) * len(vir2))
        m_jbia = m_jbia.reshape(len(occ2) * len(vir2), len(occ1) * len(vir1))
        m_iaia = m_iaia.reshape(len(occ1) * len(vir1), len(occ1) * len(vir1))
        m_jbjb = m_jbjb.reshape(len(occ2) * len(vir2), len(occ2) * len(vir2))

        self.m_iaia = m_iaia
        self.m_jbjb = m_jbjb
        m_1 = np.hstack((m_iaia, m_iajb))
        m_2 = np.hstack((m_jbia, m_jbjb))
        m_red = np.vstack((m_1, m_2))
        self.eigenvalues = np.linalg.eigvals(m_red)
        m_loc = m_loc.reshape(nocc * nvir, nocc * nvir)
        self.Z = 0
        if self.z_allexc is True:
            eig = np.linalg.eigvals(m_loc)
            for i in eig:
                self.Z += np.exp(np.real(i))
        else:
            for i in self.eigenvalues:
                self.Z += np.exp(np.real(i))

    @property
    def entropy_iaia(self):
        """Entanglement of the M_{ia,ia} matrix:
        M = (M_{ia,ia}  )

        Returns
        -------
        [real]
            [value of entanglement]
        """
        eigenvalues = np.linalg.eigvals(self.m_iaia)
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
        eigenvalues = np.linalg.eigvals(self.m_jbjb)
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
