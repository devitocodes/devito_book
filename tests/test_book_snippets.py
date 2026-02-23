import pytest


def _devito_importable() -> bool:
    try:
        import devito  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.devito,
    pytest.mark.skipif(not _devito_importable(), reason="Devito not importable in this environment"),
]


def test_what_is_devito_diffusion_runs():
    import runpy

    ns = runpy.run_path("src/book_snippets/what_is_devito_diffusion.py")
    max_u = ns["RESULT"]
    assert 0.0 < max_u < 1.0


def test_first_pde_wave1d_runs_and_is_bounded():
    import runpy

    ns = runpy.run_path("src/book_snippets/first_pde_wave1d.py")
    max_u = ns["RESULT"]
    assert 0.0 < max_u < 10.0


def test_boundary_dirichlet_wave_enforces_boundaries():
    import runpy

    ns = runpy.run_path("src/book_snippets/boundary_dirichlet_wave.py")
    boundary_mag = ns["RESULT"]
    assert boundary_mag == pytest.approx(0.0, abs=1e-12)


def test_verification_convergence_wave_rates_reasonable():
    import runpy

    ns = runpy.run_path("src/book_snippets/verification_convergence_wave.py")
    rates = ns["RESULT"]
    assert len(rates) >= 2
    assert all(1.5 < r < 2.5 for r in rates[-2:])


def test_neumann_bc_diffusion_1d_runs():
    import runpy

    ns = runpy.run_path("src/book_snippets/neumann_bc_diffusion_1d.py")
    grad = ns["RESULT"]
    assert 0.0 <= grad < 1.0


def test_mixed_bc_diffusion_1d_runs():
    import runpy

    ns = runpy.run_path("src/book_snippets/mixed_bc_diffusion_1d.py")
    result = ns["RESULT"]
    assert result["left_boundary"] == pytest.approx(0.0, abs=1e-12)
    assert result["right_copy_error"] == pytest.approx(0.0, abs=1e-12)


def test_bc_2d_dirichlet_wave_edges_zero():
    import runpy

    ns = runpy.run_path("src/book_snippets/bc_2d_dirichlet_wave.py")
    edge_max = ns["RESULT"]
    assert edge_max == pytest.approx(0.0, abs=1e-12)


def test_time_dependent_bc_sine_is_nonzero():
    import runpy

    ns = runpy.run_path("src/book_snippets/time_dependent_bc_sine.py")
    left_max = ns["RESULT"]
    assert left_max > 0.0


def test_absorbing_bc_right_wave_runs_and_bounded():
    import runpy

    ns = runpy.run_path("src/book_snippets/absorbing_bc_right_wave.py")
    max_u = ns["RESULT"]
    assert 0.0 < max_u < 10.0


def test_damping_layer_2d_wave_absorbs():
    import runpy

    ns = runpy.run_path("src/book_snippets/damping_layer_2d_wave.py")
    max_u = ns["RESULT"]
    # After damping, interior should have small residual
    assert 0.0 <= max_u < 1.0


def test_pml_wave_2d_absorbs():
    import runpy

    ns = runpy.run_path("src/book_snippets/pml_wave_2d.py")
    max_u = ns["RESULT"]
    # After PML absorption, interior should have small residual
    assert 0.0 <= max_u < 1.0


def test_higdon_abc_2d_wave_absorbs():
    import runpy

    ns = runpy.run_path("src/book_snippets/higdon_abc_2d_wave.py")
    max_u = ns["RESULT"]
    # After Higdon ABC, interior should have small residual
    assert 0.0 <= max_u < 1.0


def test_habc_wave_2d_absorbs():
    import runpy

    ns = runpy.run_path("src/book_snippets/habc_wave_2d.py")
    max_u = ns["RESULT"]
    # After HABC, interior should have small residual
    assert 0.0 <= max_u < 1.0


def test_periodic_bc_advection_1d_matches_endpoints():
    import runpy

    ns = runpy.run_path("src/book_snippets/periodic_bc_advection_1d.py")
    diff = ns["RESULT"]
    assert diff == pytest.approx(0.0, abs=1e-12)


def test_verification_mms_symbolic_computes_source():
    import runpy

    ns = runpy.run_path("src/book_snippets/verification_mms_symbolic.py")
    result = ns["RESULT"]
    assert "u_mms" in result
    assert "f_mms" in result
    assert "sin" in result["u_mms"]


def test_verification_mms_diffusion_converges():
    import runpy

    ns = runpy.run_path("src/book_snippets/verification_mms_diffusion.py")
    rates = ns["RESULT"]
    assert len(rates) >= 2
    # Expect second-order convergence
    assert all(1.5 < r < 2.5 for r in rates)


def test_verification_quick_checks_pass():
    import runpy

    ns = runpy.run_path("src/book_snippets/verification_quick_checks.py")
    result = ns["RESULT"]
    assert result["mass_change"] < 0.1  # Mass approximately conserved
    assert result["symmetry_error"] < 1e-10  # Symmetry preserved


def test_burgers_first_derivative_creates_stencils():
    import runpy

    ns = runpy.run_path("src/book_snippets/burgers_first_derivative.py")
    result = ns["RESULT"]
    assert "u_dx" in result
    assert "h_x" in result["u_dx"]  # Contains grid spacing


def test_burgers_equations_bc_creates_operator():
    import runpy

    ns = runpy.run_path("src/book_snippets/burgers_equations_bc.py")
    result = ns["RESULT"]
    assert result["num_equations"] == 10  # 2 updates + 8 BCs
    assert result["grid_shape"] == (41, 41)


def test_advec_upwind_runs_and_bounded():
    import runpy

    ns = runpy.run_path("src/book_snippets/advec_upwind.py")
    result = ns["RESULT"]
    assert 0.0 < result["max_u"] < 1.0
    assert result["u_shape"] == (101,)


def test_advec_lax_wendroff_runs_and_bounded():
    import runpy

    ns = runpy.run_path("src/book_snippets/advec_lax_wendroff.py")
    result = ns["RESULT"]
    assert 0.0 < result["max_u"] < 1.0
    assert result["u_shape"] == (101,)


# Non-Devito tests (no pytest.mark.devito needed)
def test_nonlin_logistic_be_solver():
    import runpy

    ns = runpy.run_path("src/book_snippets/nonlin_logistic_be_solver.py")
    result = ns["RESULT"]
    # CN should be most accurate
    assert result["cn_error"] < result["picard_error"]
    assert result["cn_error"] < result["newton_error"]
    # Newton should converge faster than Picard
    assert result["newton_avg_iters"] <= result["picard_avg_iters"]


def test_nonlin_split_logistic():
    import runpy

    ns = runpy.run_path("src/book_snippets/nonlin_split_logistic.py")
    result = ns["RESULT"]
    # FE on full equation should be more accurate than splitting
    assert result["FE_error"] < result["ordinary_split_error"]
    # Strange splitting should be better than ordinary splitting
    assert result["strange_split_error"] < result["ordinary_split_error"]


# Tests for src/nonlin/ module implementations (not Devito-dependent)
def test_nonlin_split_logistic_module():
    """Test the split_logistic.py module implementation."""
    import runpy

    ns = runpy.run_path("src/nonlin/split_logistic.py")
    result = ns["RESULT"]
    # FE on full equation should be more accurate than splitting
    assert result["FE_error"] < result["ordinary_split_error"]
    # Strange splitting should be better than ordinary splitting
    assert result["strange_split_error"] < result["ordinary_split_error"]
    # Strange with exact f_0 should be best splitting method
    assert result["strange_exact_error"] < result["strange_split_error"]
    # All errors should be reasonable (less than 20%)
    assert all(err < 0.2 for err in result.values())


def test_nonlin_split_diffu_react():
    """Test the split_diffu_react.py module implementation."""
    import runpy

    ns = runpy.run_path("src/nonlin/split_diffu_react.py")
    result = ns["RESULT"]
    # Should show first-order convergence in dt
    assert result["converges"]
    # Errors should decrease with refinement
    assert result["errors"][0] > result["errors"][1] > result["errors"][2]
    # Convergence rates should be close to 1.0 (first-order in dt)
    assert all(0.8 < r < 1.2 for r in result["rates"])
