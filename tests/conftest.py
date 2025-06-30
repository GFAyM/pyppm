import os

def pytest_sessionfinish(session, exitstatus):
    # Archivos a eliminar después de todos los tests
    FILES_TO_CLEANUP = [
        "11.h5", "19.h5", "pp_test_rel_ee.h5", "test.chk", "test_rel_ee.h5", "test_rel_ee.xlsx",
        "tests/pp_test_rel_ee.h5", "tests/test.chk", "tests/test_rel_ee.h5", "tests/test_rel_ee.xlsx",
        "HF_test.chk", "HF_test_loc.chk", "full_eri_None.h5", "HF_test_hrpa.chk",
        "m_HRPA_None_True.h5"
    ]
    for file in FILES_TO_CLEANUP:
        if os.path.exists(file):
            os.remove(file)
