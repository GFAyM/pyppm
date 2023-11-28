import pytest
from pyscf import gto, scf
from pyppm.drpa import DRPA
import numpy as np
import h5py

@pytest.mark.parametrize("eri_mo_sum", [(938.5446901)])
def test_eri_mo(eri_mo_sum):
    """test for eri_mo property

    Args:
        eri_mo_2_sum (real): sum of eri_mo squared
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    eri_mo = pp.eri_mo
    assert abs((eri_mo*eri_mo.conj()).sum() - eri_mo_sum) < 1e-4

@pytest.mark.parametrize("eri_mo_sum", [(938.5446901)])
def test_eri_mo_2(eri_mo_sum):
    """test for eri_mo property

    Args:
        eri_mo_2_sum (real): sum of eri_mo squared
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    pp.eri_mo_2
    with h5py.File('dhf_ovov.h5', 'r') as h5file:
        eri_mo = h5file['dhf_ovov'][()]
    assert abs((eri_mo*eri_mo.conj()).sum() - eri_mo_sum) < 1e-4

@pytest.mark.parametrize("M_sum, eri_m", [(395189665209.0902, True)])
def test_M(M_sum, eri_m):
    """
    Test for Inverse of the Principal Propagator Matrix sum in 4-component formalism

    Args:
        M_sum (real) : value of the M squared sum
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    m = pp.M(eri_m=eri_m)
    assert abs((m*m.conj()).sum() - M_sum) < 1e-2

@pytest.mark.parametrize("M_sum, eri_m", [(395189665209.0902, False)])
def test_M(M_sum, eri_m):
    """
    Test for Inverse of the Principal Propagator Matrix sum in 4-component formalism

    Args:
        M_sum (real) : value of the M squared sum
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    m = pp.M(eri_m=eri_m)
    assert abs((m*m.conj()).sum() - M_sum) < 1e-2

@pytest.mark.parametrize("pert_sum", [(41025.50691)])
def test_pert_ssc(pert_sum):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    nocc = pp.nocc
    pert = pp.pert_ssc([0])[0][nocc:,:nocc]
    assert abs(((pert*pert.conj())).sum() - pert_sum) < 1e-2

@pytest.mark.parametrize("resp_trace, eri_m", [(2.749370523140595e-08, True)])
def test_pp(resp_trace, eri_m):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    resp = pp.pp([0],[1], eri_m=eri_m)[0]
    assert abs(np.trace(resp) - resp_trace) < 1e-10

@pytest.mark.parametrize("resp_trace, eri_m", [(2.749370523140595e-08, False)])
def test_pp(resp_trace,eri_m):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    resp = pp.pp([0],[1], eri_m=eri_m)[0]
    assert abs(np.trace(resp) - resp_trace) < 1e-10

@pytest.mark.parametrize("ssc_iso, eri_mo", [(139.50631398, True)])
def test_ssc(ssc_iso, eri_mo):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    ssc = pp.ssc(atom1='H1',atom2='H2', eri_m=eri_mo)
    assert abs(ssc - ssc_iso) < 1e-4

@pytest.mark.parametrize("ssc_iso, eri_mo", [(139.50631398, False)])
def test_ssc(ssc_iso, eri_mo):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.dhf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    ssc = pp.ssc(atom1='H1',atom2='H2', eri_m=eri_mo)
    assert abs(ssc - ssc_iso) < 1e-4




    