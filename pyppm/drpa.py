import itertools
import os

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import scipy as sp
from pyscf import gto, lib, scf
from pyscf.ao2mo import r_outcore
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor


class DRPA:
    """This class calculates J-coupling between two nuclei and NMR shielding
        in the 4-component framework, as well as the dynamic polarizability.

    Need, as attribute, the mole file of the molecule, a chkfile object,
    the rotatons, that set if you want to evaluate the ee contribution or
    the pp.

    Returns:
        object: DRPA object
    """

    def __init__(
        self,
        mol=None,
        chkfile=None,
        rotations=None,
        mole_name=None,
        calc_int=False,
    ):
        if not isinstance(mol, gto.mole.Mole):
            raise TypeError("mol must be an instance of gto.mole.Mole")
        if not isinstance(chkfile, str):
            raise TypeError("chkfile must be a string")

        self.mol = mol
        self.chkfile = chkfile
        self.rotations = rotations
        self.mole_name = mole_name
        self.calc_int = calc_int
        c = lib.param.LIGHT_SPEED
        if c < 200:
            self.rel = "4c"
        else:
            self.rel = "NR"
        self.mo_coeff = lib.chkfile.load(self.chkfile, "scf/mo_coeff")
        self.mo_occ = lib.chkfile.load(self.chkfile, "scf/mo_occ")
        n4c, nmo = self.mo_coeff.shape
        self.nmo = nmo
        self.n2c = nmo // 2
        self.occidx = np.where(self.mo_occ == 1)[0]
        if self.rotations == "ee":
            self.viridx = self.n2c + np.where(self.mo_occ[self.n2c :] == 0)[0]
        elif self.rotations == "pp":
            self.viridx = np.where(self.mo_occ[: self.n2c] == 0)[0]
        else:
            self.rotations = "eepp"
            self.viridx = np.where(self.mo_occ == 0)[0]
        orbv = self.mo_coeff[:, self.viridx]
        orbo = self.mo_coeff[:, self.occidx]
        self.nvir = orbv.shape[1]
        self.nocc = orbo.shape[1]
        self.nmo = self.nocc + self.nvir
        self.mo = np.hstack((orbo, orbv))
        self.mo_energy = lib.chkfile.load(self.chkfile, "scf/mo_energy")
        self.scratch_dir = os.getenv("SCRATCH", os.getcwd())
        if self.calc_int:
            self.eri_mo(mole_name=self.mole_name)

    def mulliken_pop(self, mo_order, perc=0.1):
        """Function that calculates the mulliken population of a given mo_order
        works fine for occupied and virtual positive MOs.
        Function under development. Don't tested in negative MOs.
        Args:
            mo_order (int): order of the molecular orbital
            perc (float): threeshold of contribution
        Returns:
            str: mulliken population of the given mo_order
        """
        mo_energy = lib.chkfile.load(self.chkfile, "scf/mo_energy")
        mo_coeff = lib.chkfile.load(self.chkfile, "scf/mo_coeff")
        mo_occ = lib.chkfile.load(self.chkfile, "scf/mo_occ")
        c1 = 0.5 / lib.param.LIGHT_SPEED
        moL = np.asarray(mo_coeff[: self.n2c, :], order="F")
        moS = np.asarray(mo_coeff[self.n2c :, :], order="F") * c1
        mo_coeff = np.vstack((moL, moS))
        dhf = scf.dhf.DHF(self.mol)
        s = dhf.get_ovlp(self.mol)
        mo1 = mo_coeff[:, [mo_order]]
        dm1 = np.dot(mo1, mo1.conj().T)
        pop = np.einsum("ij,ji->i", dm1, s).real
        report = f"Energy: {mo_energy[mo_order]}, Occ: {mo_occ[mo_order]}\n"
        p = 0
        for i, s in enumerate(self.mol.spinor_labels()):
            if abs(pop[i]) > perc:
                p += pop[i]
                report += f"Pop (Large): {s}, {round(pop[i], 5)}\n"
        for i, s in enumerate(self.mol.spinor_labels()):
            if abs(pop[i + self.n2c]) > perc:
                p += pop[i + self.n2c]
                report += f"Pop (Small): {s}, {round(pop[i + self.n2c], 5)}\n"
        return report

    def eri_mo(self, with_ssss=False, mole_name=None):
        """Function that generates two electron integrals for A and B matrices
        including LLLL, LLSS and SSSS integrals, if with_sss is True.
        The integrals are stored in a h5 file saved in the scratch directory
        Args:
            with_sss (bool): if True, the SSSS integrals are included
        """
        mol = self.mol
        mo = self.mo
        n2c = self.n2c
        c1 = 0.5 / lib.param.LIGHT_SPEED
        moL = np.asarray(mo[:n2c, :], order="F")
        moS = np.asarray(mo[n2c:, :], order="F") * c1
        orboL = moL[:, : self.nocc]
        orboS = moS[:, : self.nocc]
        orbvL = moL[:, self.nocc :]
        orbvS = moS[:, self.nocc :]
        erifile = os.path.join(
            self.scratch_dir, f"{mole_name}_{self.rel}_{self.rotations}.h5"
        )
        self.erifile = erifile

        def run(mos, intor, dataname):
            r_outcore.general(mol, mos, erifile, dataname="tmp", intor=intor)
            blksize = 200
            nij = mos[0].shape[1] * mos[1].shape[1]
            with h5py.File(erifile, "a") as feri:
                for i0, i1 in lib.prange(0, nij, blksize):
                    buf = feri[dataname][i0:i1]
                    buf += feri["tmp"][i0:i1]
                    feri[dataname][i0:i1] = buf
                if "tmp" in feri:
                    del feri["tmp"]

        r_outcore.general(
            mol,
            (orbvL, orboL, orboL, orbvL),
            erifile,
            dataname="A1",
            intor="int2e_spinor",
            verbose=3,
        )
        run((orbvS, orboS, orboL, orbvL), "int2e_spsp1_spinor", dataname="A1")
        run((orbvL, orboL, orboS, orbvS), "int2e_spsp2_spinor", dataname="A1")
        if with_ssss:
            run(
                (orbvS, orboS, orboS, orbvS),
                "int2e_spsp1spsp2_spinor",
                dataname="A1",
            )

        r_outcore.general(
            mol,
            (orbvL, orbvL, orboL, orboL),
            erifile,
            dataname="A2",
            intor="int2e_spinor",
            verbose=3,
        )
        run((orbvS, orbvS, orboL, orboL), "int2e_spsp1_spinor", dataname="A2")
        run((orbvL, orbvL, orboS, orboS), "int2e_spsp2_spinor", dataname="A2")
        if with_ssss:
            run(
                (orbvS, orbvS, orboS, orboS),
                "int2e_spsp1spsp2_spinor",
                dataname="A2",
            )

        r_outcore.general(
            mol,
            (orboL, orbvL, orboL, orbvL),
            erifile,
            dataname="B",
            intor="int2e_spinor",
            verbose=3,
        )
        run((orboS, orbvS, orboL, orbvL), "int2e_spsp1_spinor", dataname="B")
        run((orboL, orbvL, orboS, orbvS), "int2e_spsp2_spinor", dataname="B")
        if with_ssss:
            run(
                (orboS, orbvS, orboS, orbvS),
                "int2e_spsp1spsp2_spinor",
                dataname="B",
            )

    def pert_r_alpha(self, atmlst):
        """
        Perturbator \alpha \times r/r^3  in 4component framework
        Extracted from properties module, ssc/dhf.py

        Args:
            atmlst (list): The atom number in which is centered the
            perturbator

        Returns:
            list: list with perturbator in molecular basis
        """
        mol = self.mol
        mo = self.mo
        n4c = self.n2c * 2
        n2c = self.n2c
        h1 = []
        for ia in atmlst:
            mol.set_rinv_origin(mol.atom_coord(ia))
            a01int = mol.intor("int1e_sa01sp_spinor", 3)
            h01 = np.zeros((n4c, n4c), np.complex128)
            for k in range(3):
                h01[:n2c, n2c:] = 0.5 * a01int[k]
                h01[n2c:, :n2c] = 0.5 * a01int[k].conj().T
                h1.append(
                    mo[:, self.nocc :]
                    .conj()
                    .T.dot(h01)
                    .dot(mo[:, : self.nocc])
                )
        return h1

    def pert_alpha_rg(self, atmlst):
        """Perturbator \alpha \times r_g

        Args:
            atmlst (list): nuclei in which the perturbator is centered

        Returns:
            list with perturbator in molecular basis
        """
        mol = self.mol
        mo = self.mo
        n4c = self.n2c * 2
        n2c = self.n2c
        n2c = n4c // 2
        mol.set_common_origin(mol.atom_coord(atmlst[0]))
        t1 = mol.intor("int1e_cg_sa10sp_spinor", 3)
        h1 = []
        for i in range(3):
            h01 = np.zeros((n4c, n4c), complex)
            h01[:n2c, n2c:] += 0.5 * t1[i]
            h01[n2c:, :n2c] += 0.5 * t1[i].conj().T
            h1.append(
                mo[:, self.nocc :].conj().T.dot(h01).dot(mo[:, : self.nocc])
            )
        return h1

    def pert_r(self):
        """
        Perturbator R in 4component framework

        Returns:
            list: list with perturbator in molecular basis
        """
        mol = self.mol
        mo = self.mo
        n2c = self.n2c
        n4c = self.n2c * 2
        h1 = []
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        charge_center = np.einsum("i,ix->x", charges, coords) / charges.sum()
        with mol.with_common_orig(charge_center):
            a01int = mol.intor_symmetric("int1e_r_spinor")
            h01 = np.zeros((n4c, n4c), np.complex128)
            for k in range(3):
                h01[:n2c, n2c:] = 0.5 * a01int[k]
                h01[n2c:, :n2c] = 0.5 * a01int[k].conj().T
                h1.append(
                    mo[:, self.nocc :]
                    .conj()
                    .T.dot(h01)
                    .dot(mo[:, : self.nocc])
                )
        return h1

    def principal_propagator(self, freq=None, lu=False, corr=None, M=False):
        """Explicit calculation of the principal propagator in the 4-component 
        framework using the A and B matrices and inverting the matrix with 
        scipy.linalg.inv and save it in a h5 file in the scratch directory.
        A and B matrices for PP inverse
        A[i,a,j,b] = delta_{ab} delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)
        This matrices was extracted from the tdscf pyscf module.

        Args:
            atm1lst (list): atom1 order list
            atm2lst (list): atom2 order list
            corr (str): level of correlation to calculate the response
            freq (float): frequency to calculate the response,
                            if None, static response is calculated
            M (bool): if True, the function returns the A and B matrices
                       without inverting them
            lu (bool): if True, the function returns the lu factorization

        Returns:

            np.array: SSC response
        """
        mo_energy = self.mo_energy
        nocc = self.nocc
        nvir = self.nvir
        viridx = self.viridx
        occidx = self.occidx
        erifile = os.path.join(
            self.scratch_dir,
            f"{self.mole_name}_{self.rel}_{self.rotations}.h5",
        )
        with h5py.File(str(erifile), "r") as f:
            if corr == "PZOA":
                a = np.zeros((nocc, nvir, nocc, nvir), dtype=complex)
                b = np.zeros_like(a)
            else:
                a1 = (
                    np.array(f["A1"])
                    .reshape(nvir, nocc, nocc, nvir)
                    .transpose(1, 0, 2, 3)
                )
                a2 = (
                    np.array(f["A2"])
                    .reshape(nvir, nvir, nocc, nocc)
                    .transpose(3, 0, 2, 1)
                )
                a = a1 - a2

                if corr == "TDA":
                    b = np.zeros_like(a)
                else:
                    b = (
                        np.array(f["B"])
                        .reshape(nocc, nvir, nocc, nvir)
                        .transpose(2, 1, 0, 3)
                    )

        b -= b.transpose(0, 3, 2, 1)
        e_ia = lib.direct_sum("a-i->ia", mo_energy[viridx], mo_energy[occidx])
        a += np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
        a = a.reshape(nocc * nvir, nocc * nvir)
        b = b.reshape(nocc * nvir, nocc * nvir)
        n = a.shape[1]
        if freq is None:
            m1 = np.hstack((a, b.conj()))
            del a, b
            m2 = np.hstack((m1[:, n:].conj(), m1[:, :n].conj()))
        else:
            a -= np.eye(nocc * nvir) * freq
            m1 = np.hstack((a, b.conj()))
            del a, b
            m2 = np.hstack(
                (
                    m1[:, n:].conj(),
                    (m1[:, :n].conj()) + (2 * (np.eye(nocc * nvir) * freq)),
                )
            )

        m = np.vstack((m1, m2))
        del m1, m2
        if M:
            return m

        elif lu:
            m, piv = sp.linalg.lu_factor(m, overwrite_a=True)
            return m, piv

        p = sp.linalg.inv(m, overwrite_a=True)
        del m
        return np.negative(p, out=p)

    def pp_j(self, atm1lst, atm2lst, corr):
        """4C-SSC between atm1lst and atm2lst using Polarization propagator
        saved in h5 files or generated in the function.
        Then, multiplyes the principal propagator with the perturbators
        contracting the indices to calculate the J-coupling.
        One of the products uses dask for prevent memory leak.
        Args:
            atm1lst (list): atom1 order list
            atm2lst (list): atom2 order list
            corr (str): level of correlation to calculate the response

        Returns:
            np.array: SSC response
        """
        nocc = self.nocc
        nvir = self.nvir
        h1 = np.asarray(self.pert_r_alpha(atm1lst)).reshape(1, 3, nvir, nocc)[
            0
        ]
        h1 = np.concatenate((h1.conj(), -h1), axis=2)
        h2 = np.asarray(self.pert_r_alpha(atm2lst)).reshape(1, 3, nvir, nocc)[
            0
        ]
        h2 = np.concatenate((h2, -h2.conj()), axis=2)

        p = self.principal_propagator(corr=corr)
        p = p.reshape(2 * nocc, nvir, 2 * nocc, nvir)
        h1 = da.from_array(h1, chunks=(3, nvir // 2, nocc // 2))
        p = da.from_array(
            p, chunks=(nocc // 2, nvir // 2, nocc // 2, nvir // 2)
        )
        e1 = da.tensordot(h1, p, axes=([1, 2], [1, 0])).compute()
        del p
        e = np.tensordot(e1, h2, axes=([1, 2], [2, 1]))
        return e.real * nist.ALPHA**4

    def pp_shield(self, atmlst, corr):
        """NMR 4C-shielding response of atmlst with gauge origin in atmlst 
        nuclei using Polarization Propagator saved in h5 files or generated 
        in the function. Then, multiplyes the principal pro by the 
        perturbators

        Args:
            atmlst (list): atom1 order list
            corr (str): level of correlation to calculate the response

        Returns:
            np.array: NMR shielding response
        """
        nocc = self.nocc
        nvir = self.nvir

        h1 = np.asarray(self.pert_r_alpha(atmlst)).reshape(1, 3, nvir, nocc)[0]
        h1 = np.concatenate((h1.conj(), -h1), axis=2)
        h2 = np.asarray(self.pert_alpha_rg(atmlst)).reshape(1, 3, nvir, nocc)[
            0
        ]
        h2 = np.concatenate((h2, -h2.conj()), axis=2)
        p = self.principal_propagator(corr=corr)
        p = p.reshape(2 * nocc, nvir, 2 * nocc, nvir)
        h1 = da.from_array(h1, chunks=(3, nvir // 2, nocc // 2))
        p = da.from_array(
            p, chunks=(nocc // 2, nvir // 2, nocc // 2, nvir // 2)
        )

        e1 = da.tensordot(h1, p, axes=([1, 2], [1, 0])).compute()
        del p
        e = np.tensordot(e1, h2, axes=([1, 2], [2, 1]))
        return e.real

    def pp_polarizability(self, freq=None, corr="RPA"):
        """Polarizability in 4component framework
        using Polarization Propagator saved in h5 files or generated in the 
        function.
        Then, multiplyes the principal propagator with the perturbators.

        Args:
            corr (str): level of correlation to calculate the response
            freq (float): frequency to calculate the response,
                         if None, static polarizability is calculated
        Returns:
            np.array: polarizability response
        """
        nocc = self.nocc
        nvir = self.nvir

        h1 = np.asarray(self.pert_r())
        h1 = np.concatenate((h1.conj(), -h1), axis=2)
        h2 = np.asarray(self.pert_r())
        h2 = np.concatenate((h2, -h2.conj()), axis=2)
        p = self.principal_propagator(freq=freq, corr=corr)
        p = p.reshape(2 * nocc, nvir, 2 * nocc, nvir)
        h1 = da.from_array(h1, chunks=(3, nvir // 2, nocc // 2))
        p = da.from_array(
            p, chunks=(nocc // 2, nvir // 2, nocc // 2, nvir // 2)
        )

        e1 = da.tensordot(h1, p, axes=([1, 2], [1, 0])).compute()
        del p
        e = np.tensordot(e1, h2, axes=([1, 2], [2, 1]))
        return e.real

    def pp_polarizability_lu(self, corr="RPA", freq=None):
        """Polarizability in 4component framework with lu factorization
        using Polarization Propagator saved in h5 files or generated in the 
        function.
        Then, multiplyes the principal propagator with the perturbators.

        Args:
            corr (str): level of correlation to calculate the response
            freq (float): frequency to calculate the response,
                         if None, static polarizability is calculated
        Returns:
            np.array: polarizability response
        """
        nocc = self.nocc
        nvir = self.nvir

        h1 = np.asarray(self.pert_r())
        h1 = np.concatenate((h1.conj(), -h1), axis=2)
        h2 = np.asarray(self.pert_r())
        h2 = np.concatenate((h2, -h2.conj()), axis=2)
        lu, piv = self.principal_propagator(freq=freq, lu=True, corr=corr)
        h2 = h2.transpose(0, 2, 1)
        X = np.zeros_like(h2, dtype=complex)
        for i in range(h2.shape[0]):
            b = h2[i].ravel()
            X[i] = sp.linalg.lu_solve((lu, piv), b).reshape(2 * nocc, nvir)

        e = np.tensordot(h1, -X, axes=([2, 1], [1, 2]))
        return e.real

    def obtain_atom_order(self, atom):
        """Function that return the atom order in the molecule input
        given the atom label

        Args:
            atom (str): atom label

        Returns:
            int: atom orden in the mol
        """
        atoms = []
        for i in range(self.mol.natm):
            atoms.append(self.mol._atom[i][0])
        if atom not in atoms:
            raise Exception(f"{atom} must be one of the labels {atoms}")
        for i in range(self.mol.natm):
            atom_ = self.mol.atom_symbol(i)
            if atom_ == atom:
                return i

    def ssc(self, atom1, atom2, corr):
        """This function multiplicates the response by the constants
        in order to get the isotropic J-coupling J between atom1 and atom2 
        nuclei

        Args:
            atom1 (str): atom1 nuclei
            atom2 (str): atom2 nuclei
            corr (str): level of correlation to calculate the response

        Returns:
            real: isotropic J
        """
        atm1lst = [self.obtain_atom_order(atom1)]
        atm2lst = [self.obtain_atom_order(atom2)]
        e11 = self.pp_j(atm1lst, atm2lst, corr)
        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag**2 * np.trace(e11) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atm1lst[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atm2lst[0]))]
        jtensor = iso_ssc * gyro1[0] * gyro2[0]
        return jtensor

    def shield(self, atom, corr):
        """This function multiplicates the response by the constants
        in order to get Shielding tensor of a nuclei

        Args:
            atom (str): atom nuclei
            corr (str): level of correlation to calculate the response

        Returns:
            real: isotropic Shielding
        """
        atmlst = [self.obtain_atom_order(atom)]
        e11 = self.pp_shield(atmlst, corr)  # [0]
        unit_ppm = nist.ALPHA**2 * 1e6
        return unit_ppm * np.trace(e11) / 3

    def ssc_pathways(
        self, atom1, atom2, double=True, threshold=0.1, corr="RPA"
    ):
        """Coupling pathways of 4C-SSC between atm1lst and atm2lst using
        Polarization propagator. If double is True, the function returns a sum
        over all coupling pathways with for loops and create an excel
        with all the pair of virtual excitations maioring than 0.1 Hz.
        Otherwise, the function returns an excel with single excitations,
        having contracted two indices.
        Args:
            atm1lst (list): atom1 order list
            atm2lst (list): atom2 order list
            double (bool): if True, all the pair of virtual excitations are 
                            calculated
            threshold (float): threshold to filter the excitations

        Returns:
            np.array: SSC response"""
        nocc = self.nocc
        nvir = self.nvir
        nocc = self.nocc
        nvir = self.nvir
        p = self.principal_propagator(corr="RPA")
        p = p.reshape(2 * nocc, nvir, 2 * nocc, nvir)
        m = self.principal_propagator(corr="RPA", M=True)
        m = m.reshape(2 * nocc, nvir, 2 * nocc, nvir)
        atm1lst = [self.obtain_atom_order(atom1)]
        atm2lst = [self.obtain_atom_order(atom2)]
        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_cte = au2Hz * nuc_mag**2
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atm1lst[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atm2lst[0]))]
        nocc = self.nocc
        nvir = self.nvir
        h1 = np.asarray(self.pert_r_alpha(atm1lst)).reshape(1, 3, nvir, nocc)[
            0
        ]
        h1 = np.concatenate((h1.conj(), -h1), axis=2)
        h2 = np.asarray(self.pert_r_alpha(atm2lst)).reshape(1, 3, nvir, nocc)[
            0
        ]
        h2 = np.concatenate((h2, -h2.conj()), axis=2)
        data = []
        e = 0
        gyros = gyro1[0] * gyro2[0] * nist.ALPHA**4
        if double:
            for x, i, a, j, b in itertools.product(
                range(3),
                range(2 * nocc),
                range(nvir),
                range(2 * nocc),
                range(nvir),
            ):
                e_1 = iso_cte * h1[x, a, i] * p[i, a, j, b] * h2[x, b, j] / 3
                e_1 = e_1.real * gyros
                e += e_1
                if abs(e_1) > threshold:

                    p_i = p[i, a, j, b].real
                    m_i = m[i, a, j, b].real
                    data.append(
                        [
                            e_1,
                            i,
                            a,
                            j,
                            b,
                            h1[x, a, i],
                            p_i,
                            h2[x, b, j],
                            m_i,
                            1 / m_i,
                        ]
                    )

            df = pd.DataFrame(
                data,
                columns=[
                    "e",
                    "i",
                    "a",
                    "j",
                    "b",
                    "b_ia",
                    "P_iajb",
                    "b_jb",
                    "m_i",
                    "1/m_i",
                ],
            )
            df = df.reindex(df["e"].abs().sort_values(ascending=False).index)
            file_name = f"{self.mole_name}_{self.rel}_{self.rotations}.xlsx"
            df.to_excel(file_name, index=False)

        else:
            h1 = da.from_array(h1, chunks=(3, nvir // 2, nocc // 2))
            p = da.from_array(
                p, chunks=(nocc // 2, nvir // 2, nocc // 2, nvir // 2)
            )
            e1 = da.tensordot(h1, p, axes=([1, 2], [1, 0])).compute()
            del p
            for x, i, a in itertools.product(
                range(3), range(2 * nocc), range(nvir)
            ):
                e_1 = iso_cte * e1[x, i, a] * h2[x, a, i] / 3
                e_1 = e_1.real * gyros
                if abs(e_1) > threshold:
                    e += e_1

            df = pd.DataFrame(data, columns=["e", "i", "a"])
            df = df.reindex(df["e"].abs().sort_values(ascending=False).index)
            file_name = f"{self.mole_name}_{self.rel}_{self.rotations}.xlsx"
            df.to_excel(file_name, index=False)
        return e
