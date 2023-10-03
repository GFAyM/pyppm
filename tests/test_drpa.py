import pytest
from pyscf import gto, scf
from pyppm.drpa import DRPA
import numpy as np

@pytest.mark.parametrize("eri_mo_sum", [(938.5446901)])
def test_eri_mo(eri_mo_sum):
    """test for eri_mo property

    Args:
        eri_mo_2_sum (real): sum of eri_mo squared
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    eri_mo = pp.eri_mo
    assert abs((eri_mo*eri_mo.conj()).sum() - eri_mo_sum) < 1e-4

@pytest.mark.parametrize("M_sum", [(395189665209.0902)])
def test_M(M_sum):
    """
    Test for Inverse of the Principal Propagator Matrix sum in 4-component formalism

    Args:
        M_sum (real) : value of the M squared sum
    """
    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    m = pp.M
    assert abs((m*m.conj()).sum() - M_sum) < 1e-2

@pytest.mark.parametrize("pert_sum", [(41025.50691)])
def test_pert_ssc(pert_sum):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    pert = pp.pert_ssc([0])[0]
    assert abs(((pert*pert.conj())).sum() - pert_sum) < 1e-2

@pytest.mark.parametrize("resp_trace", [(2.749370523140595e-08)])
def test_pp(resp_trace):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    resp = pp.pp([0],[1])[0]
    assert abs(np.trace(resp) - resp_trace) < 1e-10

@pytest.mark.parametrize("ssc_iso", [(139.50631398)])
def test_ssc(ssc_iso):

    mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
                    basis="sto-3g", unit="angstrom")
    mf = scf.DHF(mol)
    mf.kernel()
    pp = DRPA(mf)
    ssc = pp.ssc(atom1='H1',atom2='H2')
    assert abs(ssc - ssc_iso) < 1e-4




    