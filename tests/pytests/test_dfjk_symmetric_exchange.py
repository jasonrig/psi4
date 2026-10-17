import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]


@pytest.mark.parametrize("scf_type", ["MEM_DF", "DISK_DF"])
@pytest.mark.parametrize("subtype", ["INCORE", "OUT_OF_CORE"])
def test_dfjk_symmetric_exchange(scf_type, subtype):
    """Compare symmetric exchange with the general left/right contraction."""
    mol = psi4.geometry("O\nH 1 1.00\nH 1 1.00 2 103.1\nsymmetry c1")
    psi4.set_options({"scf_type": scf_type, "scf_subtype": subtype})
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ")
    auxiliary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ-jkfit")
    jk = psi4.core.JK.build_JK(primary, auxiliary)
    jk.set_memory(1000000 if subtype == "INCORE" else 62000)
    jk.initialize()
    rng = np.random.default_rng(17)
    try:
        for occupations in [(5,), (3, 0), (1,), (5, 2)]:
            orbitals = [psi4.core.Matrix.from_array(rng.random((primary.nbf(), n))) for n in occupations]
            results = []
            for symmetric in [True, False]:
                jk.C_clear()
                for coefficients in orbitals:
                    jk.C_left_add(coefficients)
                    if not symmetric:
                        jk.C_right_add(coefficients)
                jk.compute()
                assert len(jk.K()) == len(occupations)
                results.append([matrix.np.copy() for matrix in jk.K()])
            for actual, expected in zip(*results):
                np.testing.assert_array_equal(actual, actual.T)
                np.testing.assert_allclose(actual, expected, atol=1.e-10, rtol=1.e-12)
    finally:
        jk.finalize()
