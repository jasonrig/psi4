#! compare MemJK and DiskJK

import psi4
import pytest
import numpy as np
import random
from utils import *

pytestmark = [pytest.mark.psi, pytest.mark.api]

@pytest.fixture(scope="module", params=["spherical", "cartesian"])
def build_system(request):
    mol = psi4.geometry("""
    O
    H 1 1.00
    H 1 1.00 2 103.1
    """)

    memory = 50000

    puream = request.param == "spherical"
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ", puream=puream)
    aux = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ-jkfit", puream=puream)

    nbf = primary.nbf()
    naux = aux.nbf()

    # construct spaces
    names = ['C1', 'C2', 'C3', 'C4', 'C5']
    sizes = [16, 16, 20, 20, 30]
    spaces = {names[ind]: psi4.core.Matrix.from_array(np.random.rand(nbf, size)) for ind, size in enumerate(sizes)}
    space_pairs = [[0, 0], [0, 1], [1, 1], [2, 2], [3, 2], [3, 3], [4, 4]]

    # space vectors
    C_vectors = [[spaces[names[left]], spaces[names[right]]] for left, right in space_pairs]

    # DiskJK
    psi4.set_options({"SCF_TYPE": "DISK_DF"})
    DiskJK = psi4.core.JK.build_JK(primary, aux)
    DiskJK.initialize()
    DiskJK.print_header()

    # symm_JK
    psi4.set_options({"SCF_TYPE": "MEM_DF"})
    MemJK = psi4.core.JK.build_JK(primary, aux)
    MemJK.initialize()
    MemJK.print_header()

    # add C matrices
    for Cleft, Cright in C_vectors:
        DiskJK.C_left_add(Cleft)
        MemJK.C_left_add(Cleft)
        DiskJK.C_right_add(Cright)
        MemJK.C_right_add(Cright)

    # compute
    DiskJK.compute()
    MemJK.compute()

    # get integrals
    DiskJK_ints = [DiskJK.J(), DiskJK.K()]
    MemJK_ints = [MemJK.J(), MemJK.K()]

    return (DiskJK_ints, MemJK_ints)


def test_dfjk_compare(build_system):

    disk, mem = build_system

    # compare
    for j, t in enumerate(['J', 'K']):
        for i in range(len(disk[0])):
            assert compare_arrays(np.asarray(disk[j][i]), np.asarray(mem[j][i]), 9, t + str(i))


@pytest.mark.parametrize("scf_type", ["DISK_DF", "MEM_DF"])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("do_K", [False, True])
def test_dfjk_blocked_repeated(scf_type, symmetric, do_K):
    mol = psi4.geometry("""
    O
    H 1 1.00
    H 1 1.00 2 103.1
    """)
    psi4.set_options({"scf_type": scf_type, "scf_subtype": "OUT_OF_CORE"})
    primary = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ")
    aux = psi4.core.BasisSet.build(mol, "ORBITAL", "cc-pVDZ-jkfit")
    rng = np.random.default_rng(9)
    left = [rng.random((primary.nbf(), 5)) for _ in range(2)]
    right = left if symmetric else [rng.random((primary.nbf(), 5)) for _ in range(2)]
    builders = []
    try:
        for memory in ([50000, 1000000] if scf_type == "DISK_DF" else [60000, 62000, 1000000]):
            jk = psi4.core.JK.build_JK(primary, aux)
            jk.set_memory(memory)
            jk.set_do_K(do_K)
            jk.initialize()
            builders.append(jk)
        # The small workspace requires multiple auxiliary batches. Rebuild with new densities.
        for scale, ndensity in [(1.0, 1), (0.5, 2), (0.25, 1)]:
            for jk in builders:
                jk.C_clear()
                for cl, cr in zip(left[:ndensity], right[:ndensity]):
                    jk.C_left_add(psi4.core.Matrix.from_array(scale * cl))
                    if not symmetric:
                        jk.C_right_add(psi4.core.Matrix.from_array(cr))
                jk.compute()
            for jk in builders:
                assert len(jk.J()) == ndensity
                if do_K:
                    assert len(jk.K()) == ndensity
            for jk in builders[:-1]:
                for blocked, unblocked in zip(jk.J(), builders[-1].J()):
                    np.testing.assert_allclose(blocked.np, unblocked.np, atol=1.e-10, rtol=1.e-12)
                if do_K:
                    for blocked, unblocked in zip(jk.K(), builders[-1].K()):
                        np.testing.assert_allclose(blocked.np, unblocked.np, atol=1.e-10, rtol=1.e-12)
    finally:
        for jk in builders:
            jk.finalize()
