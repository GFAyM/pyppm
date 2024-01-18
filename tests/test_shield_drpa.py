import pytest
from pyscf import gto, scf
from pyppm.shield_drpa import Shielding as shield4c
import numpy as np
import h5py


@pytest.mark.parametrize("eri_mo_sum", [(938.5446901)])
def test_eri_mo(eri_mo_sum):
    """test for eri_mo property

    Args:
        eri_mo_2_sum (real): sum of eri_mo squared
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = shield4c(mf)
    eri_mo = pp.eri_mo
    assert abs((eri_mo * eri_mo.conj()).sum() - eri_mo_sum) < 1e-4


@pytest.mark.parametrize("eri_mo_sum", [(938.5446901)])
def test_eri_mo_2(eri_mo_sum):
    """test for eri_mo property

    Args:
        eri_mo_2_sum (real): sum of eri_mo squared
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = shield4c(mf)
    pp.eri_mo_2
    with h5py.File("dhf_ovov.h5", "r") as h5file:
        eri_mo = h5file["dhf_ovov"][()]
    assert abs((eri_mo * eri_mo.conj()).sum() - eri_mo_sum) < 1e-4


@pytest.mark.parametrize("M_sum, eri_m", [(395189665209.0902, True)])
def test_M(M_sum, eri_m):
    """
    Test for Inverse of the Principal Propagator Matrix sum in 4-component formalism

    Args:
        M_sum (real) : value of the M squared sum
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = shield4c(mf)
    m = pp.M(eri_m=eri_m)
    assert abs((m * m.conj()).sum() - M_sum) < 1e-2


@pytest.mark.parametrize("M_sum, eri_m", [(395189665209.0902, False)])
def test_M(M_sum, eri_m):
    """
    Test for Inverse of the Principal Propagator Matrix sum in 4-component formalism

    Args:
        M_sum (real) : value of the M squared sum
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = shield4c(mf)
    m = pp.M(eri_m=eri_m)
    assert abs((m * m.conj()).sum() - M_sum) < 1e-2


@pytest.mark.parametrize("pert_sum", [(2.0090821038)])
def test_pert_alpha_nabla_r(pert_sum):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = shield4c(mf, ee=True)
    nocc = pp.nocc
    pert = pp.pert_alpha_nabla_r([0])[0][nocc:, :nocc]
    assert abs(((pert * pert.conj())).sum() - pert_sum) < 1e-2


@pytest.mark.parametrize("pert_sum", [(0.152041265)])
def test_pert_alpha_rg(pert_sum):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = shield4c(mf, ee=True)
    nocc = pp.nocc
    pert = pp.pert_alpha_rg([0], False)[0][nocc:, :nocc]
    assert abs(((pert * pert.conj())).sum() - pert_sum) < 1e-2


@pytest.mark.parametrize("resp_trace, eri_m, giao", [(-0.55216783, False, False)])
def test_pp(resp_trace, eri_m, giao):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = shield4c(mf)
    resp = pp.pp([0], eri_m, giao)[0]
    assert abs(np.trace(resp) - resp_trace) < 1e-5


@pytest.mark.parametrize("shield_trace, eri_m, giao", [(766.29444140, False, False)])
def test_shield(shield_trace, eri_m, giao):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = shield4c(mf)
    resp = pp.shield("O", eri_m, giao)[0]
    assert abs(np.trace(resp) - shield_trace) < 1e-5


@pytest.mark.parametrize("Element, I", [("F1", 1)])
def test_obtain_atom_order(Element, I):
    """Test for the function that gives the id of the element chosen

    Args:
        Element (str): Element
        ID (int): id number of the atom in the molecule
    """
    HF_mol = gto.M(atom="""H 0 0 0; F1 1 0 0""", basis="sto-3g")
    mf = scf.dhf.DHF(HF_mol)
    mf.kernel()
    prop = shield4c(mf)
    id = prop.obtain_atom_order(Element)
    assert id == I
