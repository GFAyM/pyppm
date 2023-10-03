from pyscf import scf
import numpy
from pyscf import lib
import attr
from pyscf import ao2mo
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor
import scipy
from pyppm.rpa import RPA


@attr.s
class DRPA(RPA):
    """This class Calculates the J-coupling between two nuclei in the 4-component
        framework

        Need, as attribute, a DHF object

    Returns:
        _type_: _description_
    """

    mf = attr.ib(
        default=None,
        type=scf.dhf.DHF,
        validator=attr.validators.instance_of(scf.dhf.DHF),
    )

    @property
    def eri_mo(self):

        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        mo_occ = self.mf.mo_occ
        n4c, nmo = mo_coeff.shape
        n2c = nmo // 2
        self.occidx = numpy.where(mo_occ == 1)[0]
        self.viridx = numpy.where(mo_occ == 0)[0]

        orbv = mo_coeff[:, self.viridx]
        orbo = mo_coeff[:, self.occidx]
        self.nvir = orbv.shape[1]
        self.nocc = orbo.shape[1]
        nmo = self.nocc + self.nvir
        mo = numpy.hstack((orbo, orbv))
        c1 = 0.5 / lib.param.LIGHT_SPEED
        moL = numpy.asarray(mo[:n2c, :], order="F")
        moS = numpy.asarray(mo[n2c:, :], order="F") * c1
        eri_mo = ao2mo.kernel(mol, [moL, moL, moL, moL], intor="int2e_spinor")
        eri_mo += ao2mo.kernel(
            mol, [moS, moS, moS, moS], intor="int2e_spsp1spsp2_spinor"
        )
        eri_mo += ao2mo.kernel(mol, [moS, moS, moL, moL], intor="int2e_spsp1_spinor")
        eri_mo += ao2mo.kernel(mol, [moS, moS, moL, moL], intor="int2e_spsp1_spinor").T
        eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo).conj()
        return eri_mo

    @property
    def M(self):
        """
        A and B matrices for TDDFT response function.


        A[i,a,j,b] = \delta_{ab}\delta_{ij}(E_a - E_i) + (ia||bj)
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
        e_ia = lib.direct_sum("a-i->ia", mo_energy[viridx], mo_energy[occidx])
        a = numpy.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
        b = numpy.zeros_like(a)

        eri_mo = self.eri_mo
        a = a + numpy.einsum("iabj->iajb", eri_mo[:nocc, nocc:, nocc:, :nocc])
        a = a - numpy.einsum("ijba->iajb", eri_mo[:nocc, :nocc, nocc:, nocc:])
        b = b + numpy.einsum("iajb->iajb", eri_mo[:nocc, nocc:, :nocc, nocc:])
        b = b - numpy.einsum("jaib->iajb", eri_mo[:nocc, nocc:, :nocc, nocc:])

        a = a.reshape(nocc * nvir, nocc * nvir, order="C")
        b = b.reshape(nocc * nvir, nocc * nvir, order="C")
        m1 = numpy.concatenate((a, b), axis=1)
        m2 = numpy.concatenate((b.conj(), a.conj()), axis=1)
        m = numpy.concatenate((m1, m2), axis=0)
        return m

    def pert_ssc(self, atmlst):
        """
        Perturbator in 4component framework
        Extracted from properties module, ssc/dhf.py

        Args:
            atmlst (list): The atom number in which is centered the
            perturbator

        Returns:
            list: list with perturbator in molecular basis
        """
        mol = self.mf.mol
        mo_coeff = self.mf.mo_coeff
        orbo = mo_coeff[:, self.occidx]
        orbv = mo_coeff[:, self.viridx]
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
                h1.append(orbv.conj().T.dot(h01).dot(orbo))
        return h1

    def pp(self, atm1lst, atm2lst):
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
        h1 = numpy.asarray(self.pert_ssc(atm1lst)).reshape(1, 3, nvir, nocc)[0]
        h1 = numpy.concatenate((h1, h1.conj()), axis=2)
        h2 = numpy.asarray(self.pert_ssc(atm2lst)).reshape(1, 3, nvir, nocc)[0]
        h2 = numpy.concatenate((h2, h2.conj()), axis=2)
        m = self.M
        p = -numpy.linalg.inv(m)
        p = p.reshape(2 * nocc, nvir, 2 * nocc, nvir)
        para = []
        e = numpy.einsum("xai,iajb,ybj->xy", h1, p.conj(), h2.conj())
        para.append(e.real)
        resp = numpy.asarray(para)
        return resp * nist.ALPHA ** 4

    def ssc(self, atom1, atom2):
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
        e11 = self.pp(atm1lst, atm2lst)
        nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
        au2Hz = nist.HARTREE2J / nist.PLANCK
        iso_ssc = au2Hz * nuc_mag ** 2 * numpy.einsum("kii->k", e11) / 3
        gyro1 = [get_nuc_g_factor(self.mol.atom_symbol(atm1lst[0]))]
        gyro2 = [get_nuc_g_factor(self.mol.atom_symbol(atm2lst[0]))]
        jtensor = numpy.einsum("i,i,j->i", iso_ssc, gyro1, gyro2)[0]
        return jtensor
