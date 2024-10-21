from pyscf import gto, lib, ao2mo, scf
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
import numpy as np
import os
from pyscf.ao2mo import r_outcore
import h5py
import scipy as sp
import dask.array as da
import pandas as pd
import itertools


class DRPA:
    """This class calculates J-coupling  between two nuclei and NMR shielding
        in the 4-component framework

    Need, as attribute, the mole file of the molecule, a chkfile object,
    the rotatons, that set if you want to evaluate the ee contribution or
    the pp, and the rel, that set the speed of light of the Wave Function
    saved in the chkfile.

    Returns:
        object: DRPA object
    """

    def __init__(self, mol=None, chkfile=None, rotations=None, rel=None):
        if not isinstance(mol, gto.mole.Mole):
            raise TypeError("mol must be an instance of gto.mole.Mole")
        if not isinstance(chkfile, str):
            raise TypeError("chkfile must be a string")
        if not isinstance(rotations, str):
            raise TypeError("rotations must be a string")
        if not isinstance(rel, bool):
            raise TypeError("rel must be a boolean")

        self.mol = mol
        self.chkfile = chkfile
        self.rotations = rotations
        if rel:
            self.rel = "rel"
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
            self.viridx = np.where(self.mo_occ == 0)[0]
        orbv = self.mo_coeff[:, self.viridx]
        orbo = self.mo_coeff[:, self.occidx]
        self.nvir = orbv.shape[1]
        self.nocc = orbo.shape[1]
        self.nmo = self.nocc + self.nvir
        self.mo = np.hstack((orbo, orbv))
        self.mo_energy = lib.chkfile.load(self.chkfile, "scf/mo_energy")
        self.scratch_dir = os.getenv("SCRATCH", os.getcwd())

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
        moL = np.asarray(mo_coeff[:self.n2c, :], order="F")
        moS = np.asarray(mo_coeff[self.n2c:, :], order="F") * c1
        mo_coeff = np.vstack((moL, moS))
        dhf = scf.dhf.DHF(self.mol)
        s = dhf.get_ovlp(self.mol)
        mo1 = mo_coeff[:, [mo_order]]
        dm1 = np.dot(mo1, mo1.conj().T)

        pop = np.einsum('ij,ji->i', dm1, s).real
        report = f'Energy: {round(mo_energy[mo_order],6)}, Occ: {mo_occ[mo_order]}\n'
        p = 0
        for i, s in enumerate(self.mol.spinor_labels()):
            if abs(pop[i]) > perc:
                p += pop[i]
                report += f'Pop (Large): {s}, {round(pop[i],5)}\n'
        for i, s in enumerate(self.mol.spinor_labels()):
            if abs(pop[i + self.n2c]) > perc:
                p += pop[i + self.n2c]
                report += f'Pop (Small): {s}, {round(pop[i + self.n2c],5)}\n'
        # report += f'Total occupancy: {round(p,5)}\n'
        return report

    def eri_mo_mem(self, integral, with_ssss=False):
        """Function that generates two electron integrals, including LLLL, LLSS and
        SSSS integrals, if with_sss is True.
        Args:
            integral (str): integral to calculate, A1, A2 or B
            with_sss (bool): if True, the SSSS integrals are included

        Returns:
            np.ndarray: array with two electron integrals
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
        nocc = self.nocc
        nvir = self.nvir
        if integral == "A1":
            eri_mo_a1 = ao2mo.kernel(
                mol, [orboL, orbvL, orbvL, orboL], intor="int2e_spinor"
            )
            eri_mo_a1 = eri_mo_a1 + ao2mo.kernel(
                mol, [orboS, orbvS, orbvL, orboL], intor="int2e_spsp1_spinor"
            )
            eri_mo_a1 = eri_mo_a1 + ao2mo.kernel(
                mol, [orboL, orbvL, orbvS, orboS], intor="int2e_spsp2_spinor"
            )
            if with_ssss:
                eri_mo_a1 = eri_mo_a1 + ao2mo.kernel(
                    mol, [orboS, orbvS, orbvS, orboS], intor="int2e_spsp1spsp2_spinor"
                )

            return eri_mo_a1.reshape(nocc, nvir, nvir, nocc).conj()
        elif integral == "A2":
            eri_mo_a2 = ao2mo.kernel(
                mol, [orboL, orboL, orbvL, orbvL], intor="int2e_spinor"
            )
            eri_mo_a2 = eri_mo_a2 + ao2mo.kernel(
                mol, [orboS, orboS, orbvL, orbvL], intor="int2e_spsp1_spinor"
            )
            eri_mo_a2 = eri_mo_a2 + ao2mo.kernel(
                mol, [orboL, orboL, orbvS, orbvS], intor="int2e_spsp2_spinor"
            )
            if with_ssss:
                eri_mo_a2 = eri_mo_a2 + ao2mo.kernel(
                    mol, [orboS, orboS, orbvS, orbvS], intor="int2e_spsp1spsp2_spinor"
                )
            return eri_mo_a2.reshape(nocc, nocc, nvir, nvir).conj()

        elif integral == "B":
            eri_mo_b = ao2mo.kernel(
                mol, [orboL, orbvL, orboL, orbvL], intor="int2e_spinor"
            )
            eri_mo_b = eri_mo_b + ao2mo.kernel(
                mol, [orboS, orbvS, orboL, orbvL], intor="int2e_spsp1_spinor"
            )
            eri_mo_b = eri_mo_b + ao2mo.kernel(
                mol, [orboL, orbvL, orboS, orbvS], intor="int2e_spsp2_spinor"
            )
            if with_ssss:
                eri_mo_b = eri_mo_b + ao2mo.kernel(
                    mol, [orboS, orbvS, orboS, orbvS], intor="int2e_spsp1spsp2_spinor"
                )

            return eri_mo_b.reshape(nocc, nvir, nocc, nvir).conj()

    def eri_mo_2(self, with_ssss=False, mole_name=None):
        """Function that generates two electron integrals for A and B matrices,
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
            (orboL, orbvL, orbvL, orboL),
            erifile,
            dataname="A1",
            intor="int2e_spinor",
            verbose=4,
        )
        run((orboS, orbvS, orbvL, orboL), "int2e_spsp1_spinor", dataname="A1")
        run((orboL, orbvL, orbvS, orboS), "int2e_spsp2_spinor", dataname="A1")
        if with_ssss:
            run((orboS, orbvS, orbvS, orboS), "int2e_spsp1spsp2_spinor", dataname="A1")

        r_outcore.general(
            mol,
            (orboL, orboL, orbvL, orbvL),
            erifile,
            dataname="A2",
            intor="int2e_spinor",
            verbose=4,
        )
        run((orboS, orboS, orbvL, orbvL), "int2e_spsp1_spinor", dataname="A2")
        run((orboL, orboL, orbvS, orbvS), "int2e_spsp2_spinor", dataname="A2")
        if with_ssss:
            run((orboS, orboS, orbvS, orbvS), "int2e_spsp1spsp2_spinor", dataname="A2")

        r_outcore.general(
            mol,
            (orboL, orbvL, orboL, orbvL),
            erifile,
            dataname="B",
            intor="int2e_spinor",
            verbose=4,
        )
        run((orboS, orbvS, orboL, orbvL), "int2e_spsp1_spinor", dataname="B")
        run((orboL, orbvL, orboS, orbvS), "int2e_spsp2_spinor", dataname="B")
        if with_ssss:
            run((orboS, orbvS, orboS, orbvS), "int2e_spsp1spsp2_spinor", dataname="B")

    def pert_ssc(self, atmlst):
        """
        Perturbator SSC in 4component framework
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
                h1.append(mo[:, self.nocc :].conj().T.dot(h01).dot(mo[:, : self.nocc]))
        return h1

    def pert_alpha_rg(self, atmlst):
        """Perturbator \alpha \times r_g

        Returns:


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
        mol.set_common_origin(atmlst)
        t1 = mol.intor("int1e_cg_sa10sp_spinor", 3)
        h1 = []
        for i in range(3):
            h01 = np.zeros((n4c, n4c), complex)
            h01[:n2c, n2c:] += 0.5 * t1[i]
            h01[n2c:, :n2c] += 0.5 * t1[i].conj().T
            h1.append(mo[:, self.nocc :].conj().T.dot(h01).dot(mo[:, : self.nocc]))
        return h1

    def principal_propagator(self, mole_name=None):
        """Explicit calculation of the principal propagator in the 4-component framework
        using the A and B matrices and inverting the matrix with scipy.linalg.inv
        and save it in a h5 file in the scratch directory.
        A and B matrices for PP inverse
        A[i,a,j,b] = delta_{ab} delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)
        This matrices was extracted from the tdscf pyscf module.

        Args:
            atm1lst (list): atom1 order list
            atm2lst (list): atom2 order list

        Returns:

            np.array: SSC response
        """
        mo_energy = self.mo_energy
        nocc = self.nocc
        nvir = self.nvir
        viridx = self.viridx
        occidx = self.occidx
        if mole_name is None:
            a = self.eri_mo_mem(integral="A1").transpose(0, 1, 3, 2)
            a -= self.eri_mo_mem(integral="A2").transpose(0, 3, 1, 2)
            b = self.eri_mo_mem(integral="B")
        else:
            erifile = os.path.join(
                self.scratch_dir, f"{mole_name}_{self.rel}_{self.rotations}.h5"
            )
            with h5py.File(str(erifile), "r") as f:
                a = (
                    np.array((f["A1"]), dtype=np.complex64)
                    .reshape(nocc, nvir, nvir, nocc)
                    .conj()
                    .transpose(0, 1, 3, 2)
                )
                a -= (
                    np.array(f["A2"], dtype=np.complex64)
                    .reshape(nocc, nocc, nvir, nvir)
                    .conj()
                    .transpose(0, 3, 1, 2)
                )
                b = (
                    np.array((f["B"]), dtype=np.complex64)
                    .reshape(nocc, nvir, nocc, nvir)
                    .conj()
                )

        b -= b.transpose(2, 1, 0, 3)
        e_ia = lib.direct_sum("a-i->ia", mo_energy[viridx], mo_energy[occidx])
        a += np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
        a = a.reshape(nocc * nvir, nocc * nvir)
        b = b.reshape(nocc * nvir, nocc * nvir)
        n = a.shape[1]
        m1 = np.hstack((a, b))
        del a, b
        m2 = np.hstack((m1[:, n:].conj(), m1[:, :n].conj()))
        m = np.vstack((m1, m2))
        del m1, m2
        m = -sp.linalg.inv(m, overwrite_a=True).copy()
        if mole_name is None:
            return m
        else:
            pp_path = os.path.join(
                self.scratch_dir, f"pp_{mole_name}_{self.rel}_{self.rotations}.h5"
            )
            with h5py.File(pp_path, "w") as f:
                f.create_dataset("inverse_matrix", data=m, compression="gzip")

    def pp_j(self, atm1lst, atm2lst, mole_name):
        """4C-SSC between atm1lst and atm2lst using Polarization propagator
        saved in h5 files or generated in the function.
        Then, multiplyes the principal propagator with the perturbators
        contracting the indices to calculate the J-coupling.
        One of the products uses dask for prevent memory leak.
        Args:
            atm1lst (list): atom1 order list
            atm2lst (list): atom2 order list
            mole_name (str): molecule name of the h5 file, if None, 
            uses the principal propagator generated in the function

        Returns:
            np.array: SSC response
        """
        nocc = self.nocc
        nvir = self.nvir
        h1 = np.asarray(self.pert_ssc(atm1lst)).reshape(1, 3, nvir, nocc)[0]
        h1 = np.concatenate((h1, h1.conj()), axis=2)
        h2 = np.asarray(self.pert_ssc(atm2lst)).reshape(1, 3, nvir, nocc)[0]
        h2 = np.concatenate((h2, h2.conj()), axis=2)
        if mole_name is None:
            p = self.principal_propagator()
            p = p.reshape(2 * nocc, nvir, 2 * nocc, nvir).copy()
        else:
            self.principal_propagator(mole_name)
            pp_path = os.path.join(
                self.scratch_dir, f"pp_{mole_name}_{self.rel}_{self.rotations}.h5"
            )
            with h5py.File(pp_path, "r") as f:
                p = f["inverse_matrix"][:].reshape(2 * nocc, nvir, 2 * nocc, nvir)
        para = []
        p = p.copy()
        h1 = da.from_array(h1, chunks=(3, nvir // 2, nocc // 2)).copy()
        p = da.from_array(p, chunks=(nocc // 2, nvir // 2, nocc // 2, nvir // 2))
        e1 = da.tensordot(h1, p.conj(), axes=([1, 2], [1, 0])).compute()
        del p
        e = np.tensordot(e1, h2.conj(), axes=([1, 2], [2, 1]))
        para.append(e.real)
        resp = np.asarray(para)
        return resp * nist.ALPHA**4

    def pp_shield(self, atmlst, mole_name):
        """NMR 4C-shielding response of atmlst with gauge origin in atmlst nuclei
        using Polarization Propagator saved in h5 files or generated in the function.
        Then, multiplyes the principal propagator with the perturbators. 

        Args:
            atmlst (list): atom1 order list
            mole_name (str): molecule name of the h5 file, if None,
            uses the principal propagator generated in the function.
        Returns:
            np.array: NMR shielding response
        """
        nocc = self.nocc
        nvir = self.nvir

        h2 = np.asarray(self.pert_alpha_rg(atmlst)).reshape(1, 3, nvir, nocc)[0]
        h2 = np.concatenate((h2, h2.conj()), axis=2)
        h1 = np.asarray(self.pert_ssc(atmlst)).reshape(1, 3, nvir, nocc)[0]
        h1 = np.concatenate((h1, h1.conj()), axis=2)
        if mole_name is None:
            p = self.principal_propagator()
            p = p.reshape(2 * nocc, nvir, 2 * nocc, nvir).copy()
        else:
            self.principal_propagator(mole_name)
            pp_path = os.path.join(
                self.scratch_dir, f"pp_{mole_name}_{self.rel}_{self.rotations}.h5"
            )
            with h5py.File(pp_path, "r") as f:
                p = f["inverse_matrix"][:].reshape(2 * nocc, nvir, 2 * nocc, nvir)
        para = []
        p = p.copy()
        h1 = da.from_array(h1, chunks=(3, nvir // 2, nocc // 2)).copy()
        p = da.from_array(p, chunks=(nocc // 2, nvir // 2, nocc // 2, nvir // 2))
        e1 = da.tensordot(h1, p.conj(), axes=([1, 2], [1, 0])).compute()
        del p
        e = np.tensordot(e1, h2.conj(), axes=([1, 2], [2, 1]))

        para.append(e.real)
        resp = np.asarray(para)
        return resp

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

    def ssc(self, atom1, atom2, mole_name=None, calc_integrals=False, with_ssss=False):
        """This function multiplicates the response by the constants
        in order to get the isotropic J-coupling J between atom1 and atom2 nuclei

        Args:
            atom1 (str): atom1 nuclei
            atom2 (str): atom2 nuclei
            mole_name(str): if None, use the integrals calculated in the function 
            eri_mo. Otherwise, use the integrals saved in the h5 file.
            calc_integrals (bool): if True, calculate the integrals and save in a h5 
            file with mole_name name
            with_ssss (bool): if True, the SSSS integrals are included in the
             calculation of the integrals.

        Returns:
            real: isotropic J
        """
        atm1lst = [self.obtain_atom_order(atom1)]
        atm2lst = [self.obtain_atom_order(atom2)]
        if mole_name is None:
            e11 = self.pp_j(atm1lst, atm2lst, mole_name)
        else:
            if calc_integrals:
                self.eri_mo_2(with_ssss=with_ssss, mole_name=mole_name)
            e11 = self.pp_j(atm1lst, atm2lst, mole_name)
        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag**2 * lib.einsum("kii->k", e11) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atm1lst[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atm2lst[0]))]
        jtensor = lib.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)[0]
        return jtensor

    def shield(self, atom, mole_name=None, calc_integrals=False, with_ssss=False):
        """This function multiplicates the response by the constants
        in order to get Shielding tensor of a nuclei

        Args:
            atom (str): atom nuclei
            mole_name(str): if None, use the integrals calculated in the function 
            eri_mo. Otherwise, use the integrals saved in the h5 file.
            calc_integrals (bool): if True, calculate the integrals and save in a h5 
            file with mole_name name
            with_ssss (bool): if True, the SSSS integrals are included in the 
            calculation of the integrals.

        Returns:
            real: isotropic Shielding
        """
        atmlst = [self.obtain_atom_order(atom)]
        if mole_name is None:
            e11 = self.pp_shield(atmlst, mole_name)[0]
        else:
            if calc_integrals:
                self.eri_mo_2(with_ssss=with_ssss, mole_name=mole_name)
            e11 = self.pp_shield(atmlst, mole_name)[0]
        return np.trace(e11 / 3)

    def ssc_pathways(self, atom1, atom2, mole_name, double=True, threshold=0.1):
        """Coupling pathways of 4C-SSC between atm1lst and atm2lst using
        Polarization propagator. If double is True, the function returns the sum
        over all coupling pathways with for loops and create an excel
        with all the pair of virtual excitations maioring than 0.1 Hz.
        Otherwise, the function returns an excel with single excitations,
        having contracted two indices.
        Args:
            atm1lst (list): atom1 order list
            atm2lst (list): atom2 order list
            mole_name (str): molecule name of the h5 file
            double (bool): if True, all the pair of virtual excitations are calculated
            threshold (float): threshold to filter the excitations

        Returns:
            np.array: SSC response"""
        nocc = self.nocc
        nvir = self.nvir
        mo_energy = self.mo_energy
        nocc = self.nocc
        nvir = self.nvir
        viridx = self.viridx
        occidx = self.occidx
        e_ia = lib.direct_sum("a-i->ia", mo_energy[viridx], mo_energy[occidx])
        erifile = os.path.join(
            self.scratch_dir, f"{mole_name}_{self.rel}_{self.rotations}.h5"
        )
        atm1lst = [self.obtain_atom_order(atom1)]
        atm2lst = [self.obtain_atom_order(atom2)]
        with h5py.File(str(erifile), "r") as f:
            a = (
                np.array((f["A1"]), dtype=np.complex64)
                .reshape(nocc, nvir, nvir, nocc)
                .conj()
                .transpose(0, 1, 3, 2)
            )
            a -= (
                np.array(f["A2"], dtype=np.complex64)
                .reshape(nocc, nocc, nvir, nvir)
                .conj()
                .transpose(0, 3, 1, 2)
            )
            b = (
                np.array((f["B"]), dtype=np.complex64)
                .reshape(nocc, nvir, nocc, nvir)
                .conj()
            )
        b -= b.transpose(2, 1, 0, 3)
        a += np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
        a = a.reshape(nocc * nvir, nocc * nvir)
        b = b.reshape(nocc * nvir, nocc * nvir)
        n = a.shape[1]
        m1 = np.hstack((a, b)).copy()
        del a, b
        m2 = np.hstack((m1[:, n:].copy().conj(), m1[:, :n].copy().conj()))
        m = np.vstack((m1.copy(), m2.copy()))
        del m1, m2
        m = m.copy().reshape(2 * nocc, nvir, 2 * nocc, nvir)
        pp_path = os.path.join(
            self.scratch_dir, f"pp_{mole_name}_{self.rel}_{self.rotations}.h5"
        )
        with h5py.File(pp_path, "r") as f:
            p = f["inverse_matrix"][:].reshape(2 * nocc, nvir, 2 * nocc, nvir)
        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_cte = au2Hz * nuc_mag**2
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atm1lst[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atm2lst[0]))]
        h1 = np.asarray(self.pert_ssc(atm1lst)).reshape(1, 3, nvir, nocc)[0]
        h1 = np.concatenate((h1, h1.conj()), axis=2)
        h2 = np.asarray(self.pert_ssc(atm2lst)).reshape(1, 3, nvir, nocc)[0]
        h2 = np.concatenate((h2, h2.conj()), axis=2)
        p = p.copy()
        e = np.zeros((3, 3), dtype=np.complex128)
        data = []
        e = 0
        gyros = gyro1[0] * gyro2[0] * nist.ALPHA**4
        if double:
            for x, i, a, j, b in itertools.product(
                range(3), range(2 * nocc), range(nvir), range(2 * nocc), range(nvir)
            ):
                e_1 = iso_cte * h1[x, a, i].conj() * p[i, a, j, b] * h2[x, b, j] / 3
                e_1 = e_1.real * gyros
                e += e_1
                if abs(e_1) > threshold:
                    if i >= nocc:
                        comment_i = self.mulliken_pop(self.n2c + i - self.nocc)
                    else:
                        comment_i = self.mulliken_pop(self.n2c + i)
                    if j >= nocc:
                        comment_j = self.mulliken_pop(self.n2c + j - self.nocc)
                    else:
                        comment_j = self.mulliken_pop(self.n2c + j)

                    comment_a = self.mulliken_pop(self.n2c + self.nocc + a)
                    comment_b = self.mulliken_pop(self.n2c + self.nocc + b)
                    p_i = p[i, a, j, b].real
                    m_i = m[i, a, j, b].real
                    data.append(
                        [
                            e_1,
                            i,
                            a,
                            j,
                            b,
                            comment_i,
                            comment_a,
                            comment_j,
                            comment_b,
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
                    "comment_i",
                    "comment_a",
                    "comment_j",
                    "comment_b",
                    "b_ia",
                    "P_iajb",
                    "b_jb",
                    "m_i",
                    "1/m_i",
                ],
            )
            df = df.reindex(df["e"].abs().sort_values(ascending=False).index)
            file_name = f"{mole_name}_{self.rel}_{self.rotations}.xlsx"
            df.to_excel(file_name, index=False)

        else:
            h1 = da.from_array(h1, chunks=(3, nvir // 2, nocc // 2)).copy()
            p = da.from_array(p, chunks=(nocc // 2, nvir // 2, nocc // 2, nvir // 2))
            e1 = da.tensordot(h1, p.conj(), axes=([1, 2], [1, 0])).compute()
            for x, i, a in itertools.product(range(3), range(2 * nocc), range(nvir)):
                e_1 = iso_cte * e1[x, i, a] * h2[x, a, i].conj() / 3
                e_1 = e_1.real * gyros
                e += e_1
                if abs(e_1) > threshold:
                    if i >= nocc:
                        comment_i = self.mulliken_pop(self.n2c + i - self.nocc)
                    else:
                        comment_i = self.mulliken_pop(self.n2c + i)
                    comment_a = self.mulliken_pop(self.n2c + self.nocc + a)
                    data.append([e_1.real, i, a, comment_i, comment_a])

            df = pd.DataFrame(data, columns=["e", "i", "a", "comment_i", "comment_a"])
            df = df.reindex(df["e"].abs().sort_values(ascending=False).index)
            file_name = f"{mole_name}_{self.rel}_{self.rotations}.xlsx"
            df.to_excel(file_name, index=False)
        return e
