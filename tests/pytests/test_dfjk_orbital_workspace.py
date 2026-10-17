import numpy as np
import pytest
import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]


@pytest.mark.parametrize("subtype,do_wK", [("INCORE", False), ("OUT_OF_CORE", False), ("INCORE", True)])
@pytest.mark.parametrize("symmetric", [True, False])
def test_dfjk_orbital_workspace(subtype, do_wK, symmetric):
    """Check changing orbital spans against the independent DISK_DF backend."""
    mol = psi4.geometry("O\nH 1 1.00\nH 1 1.00 2 103.1\nsymmetry c1")
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ")
    auxiliary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ-jkfit")
    builders = []
    rng = np.random.default_rng(23)
    try:
        for scf_type in ["MEM_DF", "DISK_DF"]:
            psi4.set_options({"scf_type": scf_type, "scf_subtype": subtype})
            jk = psi4.core.JK.build_JK(primary, auxiliary)
            jk.set_memory(1000000 if subtype == "INCORE" else 62000)
            jk.set_do_wK(do_wK)
            jk.set_omega(0.3)
            jk.initialize()
            builders.append(jk)
        for occupations in [(0,), (1, 5), (primary.nbf() + 2,), (3,)]:
            left = [psi4.core.Matrix.from_array(rng.random((primary.nbf(), n))) for n in occupations]
            right = left if symmetric else [
                psi4.core.Matrix.from_array(rng.random((primary.nbf(), n))) for n in occupations
            ]
            for jk in builders:
                jk.C_clear()
                for cl, cr in zip(left, right):
                    jk.C_left_add(cl)
                    if not symmetric:
                        jk.C_right_add(cr)
                jk.compute()
            for component in ["J", "K"] + (["wK"] if do_wK else []):
                actual = getattr(builders[0], component)()
                expected = getattr(builders[1], component)()
                assert len(actual) == len(expected) == len(occupations)
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a.np, e.np, atol=1.e-10, rtol=1.e-12)
    finally:
        for jk in builders:
            jk.finalize()
