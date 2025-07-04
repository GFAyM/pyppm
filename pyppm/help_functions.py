import numpy as np
from pyscf import tools
from pyscf.tools import mo_mapping


class extra_functions:
    def __init__(self, molden_file=None):
        self.molden_file = molden_file
        self.__post_init__()

    def __post_init__(self):
        (
            self.mol,
            mo_energy,
            self.mo_coeff,
            self.mo_occ,
            irrep_labels,
            spins,
        ) = tools.molden.load(self.molden_file)
        self.occidx = np.where(self.mo_occ > 0)[0]
        self.viridx = np.where(self.mo_occ == 0)[0]

        self.orbv = self.mo_coeff[:, self.viridx]
        self.orbo = self.mo_coeff[:, self.occidx]

        self.nvir = self.orbv.shape[1]
        self.nocc = self.orbo.shape[1]

    @property
    def extraer_coeff(self):
        """Function that extractes mol, mo_coeff and mo_occ of a molden file

        Returns:
            gto.mole, np.array, np.array: mole object, mo_coeff and mo_occ
        """
        (
            self.mol,
            mo_energy,
            self.mo_coeff,
            self.mo_occ,
            irrep_labels,
            spins,
        ) = tools.molden.load(self.molden_file)
        return self.mol, self.mo_coeff, self.mo_occ

    def mo_hibridization(self, mo_label, lim1, lim2):
        """This function gives the orbital index of the MOs that are in the
        hibridization range and the contribution

        Args:
            mo_label (str): Which AO i watn to know the composition
            lim1 (int): inferior limit of the hibridization range
            lim2 (int): superior limit of the hibridization range
            cart (bool, optional): If the MOs are in cartesian coordinates,
            use True in cart. Otherwise, let unchange this value.
            Defaults to False.
            orth_method (str): The localization method to generated orthogonal
            AO upon which the AO contribution
            are computed. It can be one of ‘meta_lowdin’, ‘lowdin’ or ‘nao’.
        Returns:
            list : atm-id, hibridization coeff
        """
        comp = mo_mapping.mo_comps(
            mo_label, self.mol, self.mo_coeff, orth_method="meta_lowdin"
        )
        orbital = np.array([])
        for i, c in enumerate(comp):
            if lim1 < c < lim2:
                orbital = np.append(orbital, (i, c))
        return orbital

    def mo_hibridization_2(self, mo_label, lim1, lim2, vir):
        """This function gives a lit with a set of orbital index of the MOs
        that are in the hibridization range.

        Args:
            mo_label (str): Which AO label you want to know the composition
            lim1 (int): inferior limit of the hibridization range
            lim2 (int): superior limit of the hibridization range
            vir (bool, optional): If true, analize virtual orbitals
            Use False for analyze the occupied set.
        Returns:
            list : atm-ids
        """
        orbital = []
        if vir:
            mo_coeff = self.mo_coeff[:, self.nocc :]
            comp = mo_mapping.mo_comps(
                mo_label, self.mol, mo_coeff, orth_method="meta_lowdin"
            )

            for i, c in enumerate(comp):
                if lim1 < c < lim2:
                    orbital.append(i + self.nocc)
            return orbital
        else:
            mo_coeff = self.mo_coeff[:, : self.nocc]
            comp = mo_mapping.mo_comps(
                mo_label, self.mol, mo_coeff, orth_method="meta_lowdin"
            )
            for i, c in enumerate(comp):
                if lim1 < c < lim2:
                    orbital.append(i)
            return orbital

    def mo_hibridization_fixed(self, mo_label, fixed_orbital, lim1, lim2):
        """Evaluate the 'mo_label' composition of a fixed orbital of the molden
          file between two limits

        Args:
            mo_label (str): In terms of wich AO want to know the composition
            fixed_orbital (int): in which MO wan to know the composition
            lim1 (real): inferior limit of the hibridization range
            lim2 (real): superior limit of the hibridization range

        Returns:
            real: composition of 'mo_label' in a definite fixed_orbital
        """
        self.mol, self.mo_coeff, self.mo_occ = self.extraer_coeff
        comp = mo_mapping.mo_comps(
            mo_label, self.mol, self.mo_coeff[:, [fixed_orbital]]
        )
        orbital = np.array([])
        for i, c in enumerate(comp):
            if lim1 < c < lim2:
                orbital = np.append(orbital, c)
        return orbital[0]

    def mo_hibridization_fixed_2(self, fixed_orbital, threeshold):
        """Evaluate the 'mo_label' composition of a fixed orbital of the molden
          file between two limits

        Args:
            mo_label (str): In terms of wich AO want to know the composition
            fixed_orbital (int): in which MO wan to know the composition
            lim1 (real): inferior limit of the hibridization range
            lim2 (real): superior limit of the hibridization range

        Returns:
            real: composition of 'mo_label' in a definite fixed_orbital
        """
        self.mol, self.mo_coeff, self.mo_occ = self.extraer_coeff
        labels = self.mol.ao_labels()
        for i in range(len(labels)):
            comp = mo_mapping.mo_comps(
                labels[i], self.mol, self.mo_coeff[:, [fixed_orbital]]
            )
            if comp > threeshold:
                print(labels[i], comp)

    def mo_hibridization_2_double_filter(
        self, mo_label1, lim1_1, lim2_1, mo_label2, lim1_2, lim2_2, vir
    ):
        """This function gives a list with a set of orbital indices that 
            satisfy two hybridization range conditions.

        Args:
            mo_label1 (str): First AO label for composition filtering
            lim1_1 (int): Inferior limit of the first hybridization range
            lim2_1 (int): Superior limit of the first hybridization range
            mo_label2 (str): Second AO label for additional filtering
            lim1_2 (int): Inferior limit of the second hybridization range
            lim2_2 (int): Superior limit of the second hybridization range
            vir (bool): If True, analyze virtual orbitals; if False, analyze 
                        occupied orbitals.

        Returns:
            list: List of orbital indices that satisfy both conditions.
        """
        orbital = []
        if vir:
            mo_coeff = self.mo_coeff[:, self.nocc :]
        else:
            mo_coeff = self.mo_coeff[:, : self.nocc]

        # First filter
        comp1 = mo_mapping.mo_comps(
            mo_label1, self.mol, mo_coeff, orth_method="meta_lowdin"
        )
        first_filtered = [
            i for i, c in enumerate(comp1) if lim1_1 < c < lim2_1
        ]

        # Second filter applied to the already filtered orbitals
        comp2 = mo_mapping.mo_comps(
            mo_label2, self.mol, mo_coeff, orth_method="meta_lowdin"
        )
        for i in first_filtered:
            if lim1_2 < comp2[i] < lim2_2:
                orbital.append(i + self.nocc if vir else i)

        return orbital
