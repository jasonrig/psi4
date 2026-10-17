"""Check DIRECT integral reuse across changes to basis and density inputs."""

import numpy as np
import pytest

import psi4

pytestmark = [pytest.mark.psi, pytest.mark.api]


def _build_directjk(primary: psi4.core.BasisSet) -> psi4.core.JK:
    """Return an initialized DIRECT builder with range-separated exchange."""
    jk = psi4.core.JK.build_JK(primary, psi4.core.BasisSet.zero_ao_basis_set())
    jk.set_do_wK(True)
    jk.set_omega(0.3)
    jk.set_memory(1000000)
    jk.initialize()
    return jk


def test_directjk_basis_centers() -> None:
    """Match a fresh JK object after moving basis centers and moving them back."""
    mol = psi4.geometry("""
    H 0 0 0
    H 0 0 30
    units bohr
    symmetry c1
    no_reorient
    no_com
    """)
    psi4.set_options(
        {
            "scf_type": "direct",
            "screening": "density",
            "ints_tolerance": 1.0e-12,
            "incfock": False,
        }
    )
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ")
    coefficients = psi4.core.Matrix.from_array(
        np.arange(2 * primary.nbf(), dtype=float).reshape(primary.nbf(), 2) / 10.0
    )
    reused = _build_directjk(primary)
    reused.C_left_add(coefficients)
    try:
        for displacement in (0.0, -28.6, 0.0, 28.6):
            primary.move_atom(1, psi4.core.Vector3(0.0, 0.0, displacement))
            fresh = _build_directjk(primary)
            fresh.C_left_add(coefficients)
            try:
                reused.compute()
                fresh.compute()
                for cached, reference in [
                    (reused.J()[0], fresh.J()[0]),
                    (reused.K()[0], fresh.K()[0]),
                    (reused.wK()[0], fresh.wK()[0]),
                ]:
                    np.testing.assert_allclose(
                        cached.np, reference.np, atol=1.0e-11, rtol=0.0
                    )
            finally:
                fresh.finalize()
    finally:
        reused.finalize()


def test_directjk_density_count() -> None:
    """Refresh density-screening storage when the number of densities changes."""
    mol = psi4.geometry("""
    H 0 0 0
    H 0 0 1.4
    units bohr
    symmetry c1
    """)
    psi4.set_options(
        {
            "scf_type": "direct",
            "screening": "density",
            "ints_tolerance": 1.0e-12,
            "incfock": False,
        }
    )
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ")
    values = np.arange(2 * primary.nbf(), dtype=float).reshape(primary.nbf(), 2) / 10.0
    spaces = [
        psi4.core.Matrix.from_array(values),
        psi4.core.Matrix.from_array(0.5 * values),
    ]
    reused = _build_directjk(primary)
    try:
        for count in (1, 2, 1):
            fresh = _build_directjk(primary)
            reused.C_clear()
            for space in spaces[:count]:
                reused.C_left_add(space)
                fresh.C_left_add(space)
            try:
                reused.compute()
                fresh.compute()
                for cached, reference in [
                    (reused.J(), fresh.J()),
                    (reused.K(), fresh.K()),
                    (reused.wK(), fresh.wK()),
                ]:
                    for density in range(count):
                        np.testing.assert_allclose(
                            cached[density].np,
                            reference[density].np,
                            atol=1.0e-11,
                            rtol=0.0,
                        )
            finally:
                fresh.finalize()
    finally:
        reused.finalize()


def test_directjk_integral_package() -> None:
    """Honor an integral-package change after caching range-separated engines."""
    mol = psi4.geometry("""
    H 0 0 0
    H 0 0 1.4
    units bohr
    symmetry c1
    """)
    psi4.set_options(
        {
            "scf_type": "direct",
            "integral_package": "libint2",
            "screening": "schwarz",
            "ints_tolerance": 1.0e-12,
            "incfock": False,
        }
    )
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ")
    coefficients = psi4.core.Matrix.from_array(
        np.arange(primary.nbf(), dtype=float).reshape(primary.nbf(), 1) / 10.0
    )
    reused = _build_directjk(primary)
    fresh = _build_directjk(primary)
    reused.C_left_add(coefficients)
    fresh.C_left_add(coefficients)
    try:
        reused.compute()
        psi4.set_options({"integral_package": "simint"})
        # Range-separated integrals require Libint even in Simint-enabled builds.
        factory = psi4.core.IntegralFactory(primary, primary, primary, primary)
        with pytest.raises(RuntimeError, match="No ERI object to return"):
            factory.erf_eri(0.3)
        with pytest.raises(RuntimeError, match="No ERI object to return"):
            reused.compute()
        psi4.core.clean_timers()

        psi4.set_options({"integral_package": "libint2"})
        reused.compute()
        fresh.compute()
        for cached, reference in [
            (reused.J()[0], fresh.J()[0]),
            (reused.K()[0], fresh.K()[0]),
            (reused.wK()[0], fresh.wK()[0]),
        ]:
            np.testing.assert_allclose(cached.np, reference.np, atol=1.0e-11, rtol=0.0)
    finally:
        psi4.core.clean_timers()
        psi4.set_options({"integral_package": "libint2"})
        fresh.finalize()
        reused.finalize()
