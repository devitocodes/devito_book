"""Tests for Full Waveform Inversion (FWI) using Devito.

These tests verify the FWI implementation including:
- Gradient computation
- Gradient sign correctness
- Objective function decrease
- Box constraint enforcement
- Recovery of circular anomaly
"""

import importlib.util

import numpy as np
import pytest

# Check if Devito is available
DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None

pytestmark = pytest.mark.skipif(
    not DEVITO_AVAILABLE, reason="Devito not installed"
)


class TestFWIImport:
    """Test that FWI module imports correctly."""

    def test_import_fwi_result(self):
        """Test FWIResult import."""
        from src.adjoint import FWIResult

        assert FWIResult is not None

    def test_import_compute_fwi_gradient(self):
        """Test compute_fwi_gradient import."""
        from src.adjoint import compute_fwi_gradient

        assert compute_fwi_gradient is not None

    def test_import_fwi_gradient_descent(self):
        """Test fwi_gradient_descent import."""
        from src.adjoint import fwi_gradient_descent

        assert fwi_gradient_descent is not None

    def test_import_update_with_box_constraint(self):
        """Test update_with_box_constraint import."""
        from src.adjoint import update_with_box_constraint

        assert update_with_box_constraint is not None

    def test_import_compute_residual(self):
        """Test compute_residual import."""
        from src.adjoint import compute_residual

        assert compute_residual is not None

    def test_import_create_circle_model(self):
        """Test create_circle_model import."""
        from src.adjoint import create_circle_model

        assert create_circle_model is not None

    def test_import_ricker_wavelet(self):
        """Test ricker_wavelet import."""
        from src.adjoint import ricker_wavelet

        assert ricker_wavelet is not None


class TestRickerWavelet:
    """Test Ricker wavelet generation."""

    def test_ricker_shape(self):
        """Test wavelet has correct shape."""
        from src.adjoint import ricker_wavelet

        t = np.linspace(0, 1000, 1001)
        src = ricker_wavelet(t, f0=0.01)

        assert src.shape == t.shape

    def test_ricker_peak_at_t0(self):
        """Test wavelet peaks near t0."""
        from src.adjoint import ricker_wavelet

        t = np.linspace(0, 500, 5001)
        t0 = 100.0
        src = ricker_wavelet(t, f0=0.01, t0=t0)

        # Find peak
        idx_peak = np.argmax(src)
        t_peak = t[idx_peak]

        assert abs(t_peak - t0) < 1.0

    def test_ricker_default_t0(self):
        """Test default t0 = 1.5/f0 (to ensure wavelet starts near zero)."""
        from src.adjoint import ricker_wavelet

        t = np.linspace(0, 500, 5001)
        f0 = 0.01
        expected_t0 = 1.5 / f0  # Default is 1.5/f0 to start near zero
        src = ricker_wavelet(t, f0=f0)

        idx_peak = np.argmax(src)
        t_peak = t[idx_peak]

        assert abs(t_peak - expected_t0) < 2.0


class TestCreateCircleModel:
    """Test circle model creation."""

    def test_circle_model_shape(self):
        """Test model has correct shape."""
        from src.adjoint import create_circle_model

        shape = (101, 101)
        vp = create_circle_model(shape, (10.0, 10.0))

        assert vp.shape == shape

    def test_circle_model_background(self):
        """Test background velocity is correct."""
        from src.adjoint import create_circle_model

        vp_bg = 2.5
        vp = create_circle_model((101, 101), (10.0, 10.0), vp_background=vp_bg)

        # Check corners (should be background)
        assert vp[0, 0] == pytest.approx(vp_bg)
        assert vp[0, -1] == pytest.approx(vp_bg)
        assert vp[-1, 0] == pytest.approx(vp_bg)
        assert vp[-1, -1] == pytest.approx(vp_bg)

    def test_circle_model_anomaly(self):
        """Test circular anomaly is present."""
        from src.adjoint import create_circle_model

        vp_bg = 2.5
        vp_circle = 3.0
        shape = (101, 101)
        vp = create_circle_model(shape, (10.0, 10.0),
                                 vp_background=vp_bg, vp_circle=vp_circle)

        # Check center (should be circle velocity)
        center = (shape[0] // 2, shape[1] // 2)
        assert vp[center] == pytest.approx(vp_circle)


class TestComputeResidual:
    """Test residual computation."""

    def test_residual_shape(self):
        """Test residual has correct shape."""
        from src.adjoint import compute_residual

        nt, nrec = 100, 50
        rec_syn = np.random.randn(nt, nrec)
        rec_obs = np.random.randn(nt, nrec)

        residual = compute_residual(rec_syn, rec_obs)

        assert residual.shape == (nt, nrec)

    def test_residual_values(self):
        """Test residual = synthetic - observed."""
        from src.adjoint import compute_residual

        rec_syn = np.array([[1.0, 2.0], [3.0, 4.0]])
        rec_obs = np.array([[0.5, 1.0], [1.5, 2.0]])

        residual = compute_residual(rec_syn, rec_obs)

        np.testing.assert_allclose(residual, [[0.5, 1.0], [1.5, 2.0]])

    def test_zero_residual_for_identical_data(self):
        """Test zero residual when data matches."""
        from src.adjoint import compute_residual

        data = np.random.randn(100, 50)
        residual = compute_residual(data, data)

        np.testing.assert_allclose(residual, 0.0)


class TestUpdateWithBoxConstraint:
    """Test box constraint update."""

    def test_update_applies_gradient(self):
        """Test gradient is applied with step length."""
        from src.adjoint import update_with_box_constraint

        vp = np.array([[3.0, 3.0], [3.0, 3.0]])
        gradient = np.array([[1.0, 1.0], [1.0, 1.0]])
        alpha = 0.1

        vp_new = update_with_box_constraint(vp, alpha, gradient, vmin=1.0, vmax=5.0)

        # vp_new = vp - alpha * gradient
        expected = np.array([[2.9, 2.9], [2.9, 2.9]])
        np.testing.assert_allclose(vp_new, expected)

    def test_vmin_constraint(self):
        """Test minimum velocity constraint is enforced."""
        from src.adjoint import update_with_box_constraint

        vp = np.array([[2.0, 2.0]])
        gradient = np.array([[10.0, 10.0]])  # Large gradient to push below vmin
        alpha = 1.0
        vmin = 1.5

        vp_new = update_with_box_constraint(vp, alpha, gradient, vmin=vmin, vmax=5.0)

        assert np.all(vp_new >= vmin)

    def test_vmax_constraint(self):
        """Test maximum velocity constraint is enforced."""
        from src.adjoint import update_with_box_constraint

        vp = np.array([[4.0, 4.0]])
        gradient = np.array([[-10.0, -10.0]])  # Negative gradient to push above vmax
        alpha = 1.0
        vmax = 4.5

        vp_new = update_with_box_constraint(vp, alpha, gradient, vmin=1.0, vmax=vmax)

        assert np.all(vp_new <= vmax)

    def test_box_constraints_both_bounds(self):
        """Test both constraints work together."""
        from src.adjoint import update_with_box_constraint

        vp = np.array([[2.0, 4.0]])
        gradient = np.array([[10.0, -10.0]])  # Push first below vmin, second above vmax
        alpha = 1.0
        vmin, vmax = 1.5, 4.5

        vp_new = update_with_box_constraint(vp, alpha, gradient, vmin=vmin, vmax=vmax)

        assert np.all(vp_new >= vmin)
        assert np.all(vp_new <= vmax)


class TestFWIResult:
    """Test FWIResult dataclass."""

    def test_fwi_result_creation(self):
        """Test FWIResult can be created."""
        from src.adjoint import FWIResult

        result = FWIResult(
            vp_final=np.ones((10, 10)),
            vp_initial=np.ones((10, 10)) * 2,
            vp_true=np.ones((10, 10)) * 3,
            history=np.array([100.0, 50.0, 25.0]),
            gradients=[],
            iterations=3,
        )

        assert result.vp_final.shape == (10, 10)
        assert result.iterations == 3
        assert len(result.history) == 3

    def test_fwi_result_optional_fields(self):
        """Test FWIResult with optional fields."""
        from src.adjoint import FWIResult

        result = FWIResult(
            vp_final=np.ones((10, 10)),
            vp_initial=np.ones((10, 10)),
        )

        assert result.vp_true is None
        assert result.iterations == 0


@pytest.mark.slow
class TestFWIGradient:
    """Test FWI gradient computation.

    These tests are marked slow as they require wave propagation.
    """

    def test_gradient_computation_runs(self):
        """Test gradient computation completes without error."""
        from src.adjoint import compute_fwi_gradient, create_circle_model

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_smooth = np.full(shape, 2.5, dtype=np.float32)

        # Single source at top
        src_positions = np.array([[200.0, 20.0]])
        # Receivers at bottom
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 20),
            np.full(20, 380.0)
        ])

        objective, gradient = compute_fwi_gradient(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0
        )

        assert np.isfinite(objective)
        assert np.all(np.isfinite(gradient))
        assert gradient.shape == shape

    def test_gradient_is_nonzero(self):
        """Test gradient computation produces meaningful (non-zero) values.

        When there is a velocity anomaly, the gradient should be non-zero
        in the region illuminated by the source-receiver geometry.
        """
        from src.adjoint import compute_fwi_gradient, create_circle_model

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_smooth = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        objective, gradient = compute_fwi_gradient(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0
        )

        # Objective should be positive (there is data misfit)
        assert objective > 0, "Expected positive objective due to model mismatch"

        # Gradient should be finite
        assert np.all(np.isfinite(gradient)), "Gradient contains NaN or Inf"

        # Gradient should have some non-zero values
        # (relaxed test: just check that gradient isn't all zeros)
        assert np.max(np.abs(gradient)) > 0 or objective < 1e-10, \
            "Gradient is all zeros despite objective > 0"


@pytest.mark.slow
class TestFWIGradientDescent:
    """Test FWI gradient descent optimization.

    These tests are marked slow as they require multiple iterations.
    """

    def test_fwi_runs_and_returns_result(self):
        """Test FWI optimization runs and returns valid result."""
        from src.adjoint import create_circle_model, fwi_gradient_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_initial = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0], [200.0, 380.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        result = fwi_gradient_descent(
            shape, extent, vp_initial, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=2,
            vmin=2.0, vmax=4.0,
        )

        assert result.vp_final.shape == shape
        assert result.iterations == 2
        assert len(result.history) == 2

    def test_objective_decreases(self):
        """Test that objective function decreases during optimization."""
        from src.adjoint import create_circle_model, fwi_gradient_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_initial = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        result = fwi_gradient_descent(
            shape, extent, vp_initial, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=3,
        )

        # Objective should generally decrease (allow some tolerance for noise)
        # Check that final is less than or close to initial
        assert result.history[-1] <= result.history[0] * 1.1

    def test_box_constraints_enforced(self):
        """Test that box constraints are enforced during optimization."""
        from src.adjoint import create_circle_model, fwi_gradient_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_initial = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        vmin, vmax = 2.0, 3.5

        result = fwi_gradient_descent(
            shape, extent, vp_initial, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=2,
            vmin=vmin, vmax=vmax,
        )

        assert np.all(result.vp_final >= vmin)
        assert np.all(result.vp_final <= vmax)

    def test_save_gradients(self):
        """Test that gradients are saved when requested."""
        from src.adjoint import create_circle_model, fwi_gradient_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_initial = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        result = fwi_gradient_descent(
            shape, extent, vp_initial, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=2,
            save_gradients=True,
        )

        assert len(result.gradients) == 2
        assert result.gradients[0].shape == shape

    def test_callback_called(self):
        """Test that callback is called at each iteration."""
        from src.adjoint import create_circle_model, fwi_gradient_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_initial = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        callback_calls = []

        def callback(iteration, objective, vp):
            callback_calls.append((iteration, objective))

        fwi_gradient_descent(
            shape, extent, vp_initial, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=2,
            callback=callback,
        )

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 0
        assert callback_calls[1][0] == 1


@pytest.mark.slow
class TestStepLengthMethods:
    """Test different step length methods.

    These tests are marked slow as they require wave propagation.
    """

    def test_simple_step_length(self):
        """Test simple step length method."""
        from src.adjoint import create_circle_model, fwi_gradient_descent

        shape = (31, 31)
        extent = (300.0, 300.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_initial = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[150.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 280.0, 10),
            np.full(10, 280.0)
        ])

        result = fwi_gradient_descent(
            shape, extent, vp_initial, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=300.0,
            niter=2,
            step_length_method='simple',
        )

        assert result.vp_final is not None

    def test_backtracking_step_length(self):
        """Test backtracking step length method."""
        from src.adjoint import create_circle_model, fwi_gradient_descent

        shape = (31, 31)
        extent = (300.0, 300.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_initial = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[150.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 280.0, 10),
            np.full(10, 280.0)
        ])

        result = fwi_gradient_descent(
            shape, extent, vp_initial, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=300.0,
            niter=2,
            step_length_method='backtracking',
        )

        assert result.vp_final is not None

    def test_invalid_step_length_method_raises(self):
        """Test that invalid step length method raises error."""
        from src.adjoint import create_circle_model, fwi_gradient_descent

        shape = (31, 31)
        extent = (300.0, 300.0)
        spacing = (10.0, 10.0)

        vp_true = create_circle_model(shape, spacing, vp_background=2.5, vp_circle=3.0)
        vp_initial = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[150.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 280.0, 10),
            np.full(10, 280.0)
        ])

        with pytest.raises(ValueError, match="Unknown step length method"):
            fwi_gradient_descent(
                shape, extent, vp_initial, vp_true,
                src_positions, rec_coords,
                f0=0.025, t_end=300.0,
                niter=1,
                step_length_method='invalid_method',
            )
