import pytest
from pyscf import gto, scf
from pyppm.drpa import DRPA
import numpy as np
import h5py
import os

mol = gto.M(atom="""H1 0 2 0; H2 0 0 0; O 0 1 0""", 
            basis="sto-3g", unit="angstrom")
mf = scf.dhf.DHF(mol)
mf.chkfile = 'test.chk'
mf.kernel()

@pytest.mark.parametrize("mo_order, expected_energy, expected_populations", [
    (20, -0.289983, 0.66638)
])
def test_mulliken_pop(mo_order, expected_energy, expected_populations):
    """Test for mulliken_pop function

    Args:
        mo_order (int): order of the molecular orbital
        expected_energy (float): expected energy of the molecular orbital
        expected_occupancy (float): expected occupancy of the molecular orbital
        expected_populations (float): expected mulliken populations
    """
    # Configuración de la molécula

    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    report = pp.mulliken_pop(mo_order, perc=0.3)
    lines = report.split('\n')
    energy_line = lines[0]
    occupancy_line = lines[1]
    energy = float(energy_line.split(':')[1].split(',')[0])
    pop = float(occupancy_line.split(':')[1].split()[4])
    assert abs(energy - expected_energy) < 1e-6, f"Energia esperada: {expected_energy}, obtenida: {energy}"
    assert abs(pop - expected_populations) < 1e-6, f"Ocupacion esperada: {expected_populations}, obtenida: {occupancy}"

@pytest.mark.parametrize("integral, square_sum", [
    ('A1', 0.2968921402057)
])
def test_eri_mo_mem(integral, square_sum):
    """Test for eri_mo_mem function

    Args:
        integral (str): type of integral
        square_sum (float): expected sum of the square of the integral
        with_ssss (bool): if the integral has ssss
    """
    # Configuración de la molécula
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    m = pp.eri_mo_mem(integral=integral)
    a = (m.conj()*m).sum()
    assert abs(a - square_sum) < 1e-6, f"Suma de cuadrados esperada: {square_sum}, obtenida: {a}"

@pytest.mark.parametrize("integral, square_sum", [
    ('A2', 16.09276665265)
])
def test_eri_mo_mem(integral, square_sum):
    """Test for eri_mo_mem function

    Args:
        integral (str): type of integral
        square_sum (float): expected sum of the square of the integral
        with_ssss (bool): if the integral has ssss
    """
    # Configuración de la molécula
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    m = pp.eri_mo_mem(integral=integral)
    a = (m.conj()*m).sum().real
    assert abs(a - square_sum) < 1e-6, f"Suma de cuadrados esperada: {square_sum}, obtenida: {a}"

@pytest.mark.parametrize("integral, square_sum", [
    ('B', 0.2968921105483253)
])
def test_eri_mo_mem(integral, square_sum):
    """Test for eri_mo_mem function

    Args:
        integral (str): type of integral
        square_sum (float): expected sum of the square of the integral
        with_ssss (bool): if the integral has ssss
    """
    # Configuración de la molécula
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    m = pp.eri_mo_mem(integral=integral)
    a = (m.conj()*m).sum().real
    assert abs(a - square_sum) < 1e-6, f"Suma de cuadrados esperada: {square_sum}, obtenida: {a}"

@pytest.mark.parametrize("A1, A2, B", [
    (0.2968921402057, 16.09276665265, 0.2968921105483253)
])
def test_eri_mo_2(A1, A2, B):
    """Test for eri_mo_2 property

    Args:
        A1 (float): expected sum of the square of the A1 integral
        A2 (float): expected sum of the square of the A2 integral
        B (float): expected sum of the square of the B integral
    """
    # Configuración de la molécula
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    pp.eri_mo_2(mole_name='test')
    scratch_dir = os.getenv("SCRATCH", os.getcwd())
    erifile = os.path.join(
        scratch_dir, "test_rel_ee.h5"
    )
    with h5py.File(str(erifile), "r") as f:
        a1 = (
            np.array((f["A1"])))
        a2 = (
            np.array((f["A2"])))
        b = (
            np.array((f["B"])))
    a1 = (a1*a1.conj()).sum().real
    a2 = (a2*a2.conj()).sum().real
    b = (b*b.conj()).sum().real
    assert abs(a1 - A1) < 1e-6, f"Suma de cuadrados esperada: {A1}, obtenida: {a1}"
    assert abs(a2 - A2) < 1e-6, f"Suma de cuadrados esperada: {A2}, obtenida: {a2}"
    assert abs(b - B) < 1e-6, f"Suma de cuadrados esperada: {B}, obtenida: {b}"

@pytest.mark.parametrize("pert_sum", [(2.009082103838383)])
def test_pert_ssc(pert_sum):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    nocc = pp.nocc
    pert = pp.pert_ssc([0])[0]
    assert abs(((pert*pert.conj())).sum().real - pert_sum) < 1e-6

@pytest.mark.parametrize("pert_sum", [(0.15204126508440774)])
def test_pert_alpha_rg(pert_sum):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    nocc = pp.nocc
    pert = pp.pert_alpha_rg([0])[0]
    assert abs(((pert*pert.conj())).sum().real - pert_sum) < 1e-6

@pytest.mark.parametrize("p_sum", [(603.0874071179228)])
def test_pprincipal_propagator(p_sum):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    p = pp.principal_propagator(mole_name=None)
    assert abs((p*p.conj()).sum().real - p_sum) < 1e-6

@pytest.mark.parametrize("p_sum", [(603.0874)])
def test_pprincipal_propagator(p_sum):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    pp.principal_propagator(mole_name='test')
    scratch_dir = os.getenv("SCRATCH", os.getcwd())
    erifile = os.path.join(
        scratch_dir, "pp_test_rel_ee.h5"
    )
    with h5py.File(str(erifile), "r") as f:
        eri = f["inverse_matrix"][()]
    assert abs((eri*eri.conj()).sum().real - p_sum) < 1e-4

@pytest.mark.parametrize("pp_j_sum", [(2.8901811363119265e-08)])
def test_pp_j(pp_j_sum):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.pp_j(atm1lst=[0], atm2lst=[1], mole_name='test')
    assert abs(x.sum() - pp_j_sum) < 1e-5

@pytest.mark.parametrize("pp_j_sum", [(2.8901811363119265e-08)])
def test_pp_j(pp_j_sum):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.pp_j(atm1lst=[0], atm2lst=[1], mole_name=None)
    assert abs(x.sum() - pp_j_sum) < 1e-5

@pytest.mark.parametrize("pp_shield_sum", [(0.797086790423847)])
def test_pp_shield(pp_shield_sum):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.pp_shield(atmlst=[0], mole_name='test')
    assert abs(x.sum() - pp_shield_sum) < 1e-6

@pytest.mark.parametrize("pp_shield_sum", [(0.797086790423847)])
def test_pp_shield(pp_shield_sum):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.pp_shield(atmlst=[0], mole_name=None)
    assert abs(x.sum() - pp_shield_sum) < 1e-6

@pytest.mark.parametrize("ssc", [(146.65121114276525)]) 
def test_ssc(ssc):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.ssc(atom1='H1', atom2='H2', mole_name='test', calc_integrals=False, with_ssss=False)
    assert abs(x - ssc) < 1e-6

@pytest.mark.parametrize("ssc", [(146.65121114276525)]) 
def test_ssc(ssc):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.ssc(atom1='H1', atom2='H2', mole_name=None, calc_integrals=False, with_ssss=False)
    assert abs(x - ssc) < 1e-4

@pytest.mark.parametrize("shield", [(0.265695596882299)])
def test_shield(shield):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.shield(atom='H1', mole_name=None, calc_integrals=False, with_ssss=False)
    assert abs(x - shield) < 1e-4

@pytest.mark.parametrize("shield", [(0.265695596882299)])
def test_shield(shield):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.shield(atom='H1', mole_name='test', calc_integrals=False, with_ssss=False)
    assert abs(x - shield) < 1e-4

@pytest.mark.parametrize("ssc_path", [(146.65121114276525)])
def test_ssc_pathways(ssc_path):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.ssc_pathways(atom1='H1', atom2='H2', mole_name='test', double=False)
    assert abs(x - ssc_path) < 1e-4

@pytest.mark.parametrize("ssc_path", [(146.65121114276525)])
def test_ssc_pathways(ssc_path):
    pp = DRPA(mol=mol, chkfile=mf.chkfile, rotations='ee', rel=True)
    x = pp.ssc_pathways(atom1='H1', atom2='H2', mole_name='test', double=True)
    assert abs(x - ssc_path) < 1e-4