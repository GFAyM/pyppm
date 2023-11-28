from pyscf import scf
import numpy
from pyscf import lib
import attr
from pyscf import ao2mo
from pyscf.ao2mo import r_outcore
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
import scipy
import h5py


@attr.s
class DRPA:
    """This class Calculates the J-coupling between two nuclei in the 4-component
        framework

    Need, as attribute, a DHF object and a boolean, ee, that set if you want to
    evaluate the ee contribution

    Returns:
        object: DRPA object
    """

    mf = attr.ib(
        default=None,
        type=scf.dhf.DHF,
        validator=attr.validators.instance_of(scf.dhf.DHF),
    )
    ee = attr.ib(default=False, type=bool)

    def __attrs_post_init__(self):
        self.mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        n4c, nmo = mo_coeff.shape
        self.nmo = nmo
        self.n2c = nmo // 2
        self.occidx = numpy.where(mo_occ == 1)[0]
        if self.ee is True:
            self.viridx = numpy.where(mo_occ[self.n2c :] == 0)[0]
        else:
            self.viridx = numpy.where(mo_occ == 0)[0]

        orbv = mo_coeff[:, self.viridx]
        orbo = mo_coeff[:, self.occidx]
        self.nvir = orbv.shape[1]
        self.nocc = orbo.shape[1]
        self.nmo = self.nocc + self.nvir
        self.mo = numpy.hstack((orbo, orbv))

    @property
    def eri_mo(self):
        """Function that generates two electron integrals, including LLLL, LLSS and
        SSSS integrals

        Returns:
            numpy.ndarray: array with two electron integrals 
        """
        mol = self.mol
        mo = self.mo
        n2c = self.n2c
        nmo = self.nmo
        c1 = 0.5 / lib.param.LIGHT_SPEED
        moL = numpy.asarray(mo[:n2c, :], order="F")
        moS = numpy.asarray(mo[n2c:, :], order="F") * c1
        eri_mo = ao2mo.kernel(mol, [moL, moL, moL, moL], intor="int2e_spinor")
        eri_mo = eri_mo + ao2mo.kernel(
            mol, [moS, moS, moS, moS], intor="int2e_spsp1spsp2_spinor"
        )
        eri_mo = eri_mo + ao2mo.kernel(
            mol, [moS, moS, moL, moL], intor="int2e_spsp1_spinor"
        )
        eri_mo = (
            eri_mo
            + ao2mo.kernel(mol, [moS, moS, moL, moL], intor="int2e_spsp1_spinor").T
        )
        eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo).conj()
        return eri_mo

    @property
    def eri_mo_2(self):
        """Function that generates two electron integrals, including LLLL, LLSS and
        SSSS integrals, but more eficiently, saving eri_mos in a h5 file
        """
        mol = self.mol
        mo = self.mo
        n2c = self.n2c
        c1 = 0.5 / lib.param.LIGHT_SPEED
        moL = numpy.asarray(mo[:n2c, :], order="F")
        moS = numpy.asarray(mo[n2c:, :], order="F") * c1
        erifile = "dhf_ovov.h5"
        dataname = "dhf_ovov"

        def run(mos, intor):
            # Ajustar el valor de blksize: El valor de blksize controla cuántos 
            # elementos
            # de las integrales se calculan y almacenan en la memoria a la vez. Puedes
            # ajustar este valor según la cantidad de memoria disponible en tu sistema.
            # Un valor más pequeño reducirá la huella de memoria, pero también puede 
            # aumentar el tiempo de cálculo debido a más escrituras en disco
            r_outcore.general(mol, mos, erifile, dataname="tmp", intor=intor)
            blksize = 400
            nij = mos[0].shape[1] * mos[1].shape[1]
            with h5py.File(erifile, "a") as feri:
                for i0, i1 in lib.prange(0, nij, blksize):
                    buf = feri[dataname][i0:i1]
                    buf += feri["tmp"][i0:i1]
                    feri[dataname][i0:i1] = buf

        r_outcore.general(
            mol, (moL, moL, moL, moL), erifile, dataname=dataname, intor="int2e_spinor"
        )
        run((moS, moS, moS, moS), "int2e_spsp1spsp2_spinor")
        run((moS, moS, moL, moL), "int2e_spsp1_spinor")
        run((moL, moL, moS, moS), "int2e_spsp2_spinor")

    def M(self, eri_m):
        """
        A and B matrices for PP inverse
        A[i,a,j,b] = delta_{ab} delta_{ij}(E_a - E_i) + (ia||bj)
        B[i,a,j,b] = (ia||jb)
        This matrices was extracted from the tdscf pyscf module.

        Returns:

            Numpy.array: Inverse of Principal Propagator
        """
        mo_energy = self.mf.mo_energy
        nocc = self.nocc
        nvir = self.nvir
        viridx = self.viridx
        occidx = self.occidx
        nmo = self.nmo
        e_ia = lib.direct_sum("a-i->ia", mo_energy[viridx], mo_energy[occidx])
        a = numpy.zeros((nocc, nvir, nocc, nvir), dtype="complex")
        b = numpy.zeros_like(a)
        if eri_m is True:
            print("ao2mo")
            eri_mo = self.eri_mo
            a += lib.einsum("iabj->iajb", eri_mo[:nocc, nocc:, nocc:, :nocc])
            a -= lib.einsum("ijba->iajb", eri_mo[:nocc, :nocc, nocc:, nocc:])
            b += lib.einsum("iajb->iajb", eri_mo[:nocc, nocc:, :nocc, nocc:])
            b -= lib.einsum("jaib->iajb", eri_mo[:nocc, nocc:, :nocc, nocc:])

        else:
            print("r_outcore")
            self.eri_mo_2
            with h5py.File("dhf_ovov.h5", "r") as h5file:
                eri_mo = (
                    h5file["dhf_ovov"][()]
                    .reshape(nmo, nmo, nmo, nmo)[:nocc, :, :, :]
                    .conj()
                )

                a += lib.einsum("iabj->iajb", eri_mo[:, nocc:, nocc:, :nocc])
                a -= lib.einsum("ijba->iajb", eri_mo[:, :nocc, nocc:, nocc:])
                b += lib.einsum("iajb->iajb", eri_mo[:, nocc:, :nocc, nocc:])
                b -= lib.einsum("jaib->iajb", eri_mo[:, nocc:, :nocc, nocc:])
                eri_mo = 0
        a = a + numpy.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
        a = a.reshape(nocc * nvir, nocc * nvir, order="C")
        b = b.reshape(nocc * nvir, nocc * nvir, order="C")
        m1 = numpy.concatenate((a, b), axis=1)
        m2 = numpy.concatenate((b.conj(), a.conj()), axis=1)
        m = numpy.concatenate((m1, m2), axis=0)
        return m

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
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        mo = self.mo
        n4c = mo_coeff.shape[0]
        n2c = n4c // 2
        h1 = []
        for ia in atmlst:
            mol.set_rinv_origin(mol.atom_coord(ia))
            a01int = mol.intor("int1e_sa01sp_spinor", 3)
            h01 = numpy.zeros((n4c, n4c), numpy.complex128)
            for k in range(3):
                h01[:n2c, n2c:] = 0.5 * a01int[k]
                h01[n2c:, :n2c] = 0.5 * a01int[k].conj().T
                h1.append(mo.conj().T.dot(h01).dot(mo))
        return h1

    def pp(self, atm1lst, atm2lst, eri_m):
        """In this Function generate de Response << ; >>, i.e,
        multiplicate the perturbators centered in nuclei1 and nuclei2 with the
        principal propagator matrix.

        Args:
            atm1lst (list): list with atm1 id
            atm2lst (list): list with atm2 id

        Returns:
            numpy.array: J tensor
        """
        nocc = self.nocc
        nvir = self.nvir
        nmo = nocc + nvir
        m = self.M(eri_m=eri_m)
        h1 = numpy.asarray(self.pert_ssc(atm1lst)).reshape(1, 3, nmo, nmo)[0]
        h1 = h1[:, nocc:, :nocc]
        h1 = numpy.concatenate((h1, h1.conj()), axis=2)
        h2 = numpy.asarray(self.pert_ssc(atm2lst)).reshape(1, 3, nmo, nmo)[0]
        h2 = h2[:, nocc:, :nocc]
        h2 = numpy.concatenate((h2, h2.conj()), axis=2)
        p = -scipy.linalg.inv(m)
        p = p.reshape(2 * nocc, nvir, 2 * nocc, nvir)
        para = []
        e = lib.einsum("xai,iajb,ybj->xy", h1, p.conj(), h2.conj())
        para.append(e.real)
        resp = numpy.asarray(para)
        return resp * nist.ALPHA ** 4

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

    def ssc(self, atom1, atom2, eri_m):
        """This function multiplicates the response by the constants
        in order to get the isotropic J-coupling J between atom1 and atom2 nuclei

        Args:
            atom1 (str): atom1 nuclei
            atom2 (str): atom2 nuclei

        Returns:
            real: isotropic J
        """
        atm1lst = [self.obtain_atom_order(atom1)]
        atm2lst = [self.obtain_atom_order(atom2)]
        e11 = self.pp(atm1lst, atm2lst, eri_m)
        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag ** 2 * lib.einsum("kii->k", e11) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atm1lst[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atm2lst[0]))]
        jtensor = lib.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)[0]
        return jtensor
