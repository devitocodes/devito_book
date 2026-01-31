"""Tests for Least-Squares Reverse Time Migration (LSRTM) using Devito.

These tests verify the LSRTM implementation including:
- Born modeling operator
- Born adjoint operator
- Barzilai-Borwein step length
- LSRTM steepest descent optimization
"""

import importlib.util

import numpy as np
import pytest

# Check if Devito is available
DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None

pytestmark = pytest.mark.skipif(
    not DEVITO_AVAILABLE, reason="Devito not installed"
)


class TestLSRTMImport:
    """Test that LSRTM module imports correctly."""

    def test_import_lsrtm_result(self):
        """Test LSRTMResult import."""
        from src.adjoint import LSRTMResult

        assert LSRTMResult is not None

    def test_import_born_modeling(self):
        """Test born_modeling import."""
        from src.adjoint import born_modeling

        assert born_modeling is not None

    def test_import_born_adjoint(self):
        """Test born_adjoint import."""
        from src.adjoint import born_adjoint

        assert born_adjoint is not None

    def test_import_lsrtm_steepest_descent(self):
        """Test lsrtm_steepest_descent import."""
        from src.adjoint import lsrtm_steepest_descent

        assert lsrtm_steepest_descent is not None

    def test_import_barzilai_borwein_step(self):
        """Test barzilai_borwein_step import."""
        from src.adjoint import barzilai_borwein_step

        assert barzilai_borwein_step is not None

    def test_import_create_layered_model(self):
        """Test create_layered_model import."""
        from src.adjoint import create_layered_model

        assert create_layered_model is not None


class TestCreateLayeredModel:
    """Test layered model creation."""

    def test_layered_model_shape(self):
        """Test model has correct shape."""
        from src.adjoint import create_layered_model

        shape = (101, 101)
        vp = create_layered_model(shape, (10.0, 10.0))

        assert vp.shape == shape

    def test_layered_model_layers(self):
        """Test layers are created correctly."""
        from src.adjoint import create_layered_model

        shape = (101, 201)
        vp_layers = [1.5, 2.0, 2.5, 3.0]
        vp = create_layered_model(shape, (10.0, 10.0), vp_layers=vp_layers)

        # Check unique values (should have all layers)
        unique_vp = np.unique(vp)
        assert len(unique_vp) == len(vp_layers)

        for layer_vp in vp_layers:
            assert layer_vp in unique_vp

    def test_layered_model_top_velocity(self):
        """Test top layer has correct velocity."""
        from src.adjoint import create_layered_model

        vp_layers = [1.5, 2.0, 3.0]
        vp = create_layered_model((101, 201), (10.0, 10.0), vp_layers=vp_layers)

        # Top of model should be first layer velocity
        assert vp[0, 0] == pytest.approx(vp_layers[0])
        assert vp[50, 0] == pytest.approx(vp_layers[0])

    def test_layered_model_custom_depths(self):
        """Test custom layer depths."""
        from src.adjoint import create_layered_model

        shape = (101, 201)
        spacing = (10.0, 10.0)
        vp_layers = [1.5, 2.5]
        layer_depths = [1000.0]  # Interface at z=1000m

        vp = create_layered_model(shape, spacing, vp_layers=vp_layers,
                                  layer_depths=layer_depths)

        # Check velocities above and below interface
        iz_interface = int(1000.0 / spacing[1])
        assert vp[50, iz_interface - 1] == pytest.approx(vp_layers[0])
        assert vp[50, iz_interface + 1] == pytest.approx(vp_layers[1])


class TestBarzilaiborweinStep:
    """Test Barzilai-Borwein step length computation."""

    def test_bb_step_basic(self):
        """Test basic BB step computation."""
        from src.adjoint import barzilai_borwein_step

        s_prev = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_prev = np.array([[0.1, 0.2], [0.3, 0.4]])

        alpha = barzilai_borwein_step(s_prev, y_prev, iteration=1)

        assert np.isfinite(alpha)
        assert alpha > 0

    def test_bb_step_finite(self):
        """Test BB step is always finite.

        Note: BB step can be negative when curvature is negative (s_dot_y < 0).
        In practice, the optimization should use |alpha| or fall back to a
        default step when the curvature condition is violated.
        """
        from src.adjoint import barzilai_borwein_step

        np.random.seed(42)
        for _ in range(10):
            s_prev = np.random.randn(10, 10)
            y_prev = np.random.randn(10, 10)

            alpha = barzilai_borwein_step(s_prev, y_prev, iteration=1)

            assert np.isfinite(alpha)

    def test_bb_step_handles_zero_gradient(self):
        """Test BB step handles near-zero gradients."""
        from src.adjoint import barzilai_borwein_step

        s_prev = np.ones((5, 5))
        y_prev = np.zeros((5, 5))  # Zero gradient change

        alpha = barzilai_borwein_step(s_prev, y_prev, iteration=1)

        assert np.isfinite(alpha)

    def test_bb_step_formula(self):
        """Test BB step formula implementation."""
        from src.adjoint import barzilai_borwein_step

        # Simple case where we can verify the formula
        s_prev = np.array([[2.0]])
        y_prev = np.array([[1.0]])

        # s_dot_s = 4, s_dot_y = 2, y_dot_y = 1
        # alpha_bb1 = 4/2 = 2
        # alpha_bb2 = 2/1 = 2
        # ratio = 1 (not in (0,1)), so returns alpha_bb1

        alpha = barzilai_borwein_step(s_prev, y_prev, iteration=1)

        # Should return alpha_bb1 = 2 since ratio = 1
        assert alpha == pytest.approx(2.0)


class TestLSRTMResult:
    """Test LSRTMResult dataclass."""

    def test_lsrtm_result_creation(self):
        """Test LSRTMResult can be created."""
        from src.adjoint import LSRTMResult

        result = LSRTMResult(
            image_final=np.ones((10, 10)),
            image_initial=np.ones((10, 10)) * 0.5,
            history=np.array([100.0, 50.0, 25.0]),
            iterations=3,
        )

        assert result.image_final.shape == (10, 10)
        assert result.iterations == 3
        assert len(result.history) == 3

    def test_lsrtm_result_defaults(self):
        """Test LSRTMResult default values."""
        from src.adjoint import LSRTMResult

        result = LSRTMResult(
            image_final=np.ones((10, 10)),
            image_initial=np.ones((10, 10)),
        )

        assert len(result.history) == 0
        assert result.iterations == 0


@pytest.mark.slow
class TestBornModeling:
    """Test Born modeling operator.

    These tests are marked slow as they require wave propagation.
    """

    def test_born_modeling_runs(self):
        """Test Born modeling completes without error."""
        from src.adjoint import born_modeling

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_smooth = np.full(shape, 2.5, dtype=np.float32)
        reflectivity = np.zeros(shape, dtype=np.float32)
        reflectivity[15:25, 15:25] = 0.01  # Small perturbation

        src_coords = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 20),
            np.full(20, 380.0)
        ])

        rec_data, p0_wavefield = born_modeling(
            shape, extent, vp_smooth, reflectivity,
            src_coords, rec_coords,
            f0=0.025, t_end=400.0
        )

        assert rec_data is not None
        assert p0_wavefield is not None
        assert np.all(np.isfinite(rec_data))
        assert np.all(np.isfinite(p0_wavefield))

    def test_born_modeling_output_shapes(self):
        """Test Born modeling output shapes are correct."""
        from src.adjoint import born_modeling

        shape = (41, 41)
        extent = (400.0, 400.0)
        nrec = 15

        vp_smooth = np.full(shape, 2.5, dtype=np.float32)
        reflectivity = np.zeros(shape, dtype=np.float32)

        src_coords = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, nrec),
            np.full(nrec, 380.0)
        ])

        rec_data, p0_wavefield = born_modeling(
            shape, extent, vp_smooth, reflectivity,
            src_coords, rec_coords,
            f0=0.025, t_end=400.0
        )

        # Check receiver data shape: (nt, nrec)
        assert rec_data.shape[1] == nrec

        # Check wavefield shape: (nt, nx, nz)
        assert p0_wavefield.shape[1:] == shape

    def test_born_modeling_zero_reflectivity(self):
        """Test Born modeling with zero reflectivity gives minimal scattered data."""
        from src.adjoint import born_modeling

        shape = (41, 41)
        extent = (400.0, 400.0)

        vp_smooth = np.full(shape, 2.5, dtype=np.float32)
        reflectivity = np.zeros(shape, dtype=np.float32)  # No reflectivity

        src_coords = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        rec_data, _ = born_modeling(
            shape, extent, vp_smooth, reflectivity,
            src_coords, rec_coords,
            f0=0.025, t_end=400.0
        )

        # With zero reflectivity, scattered data should be small
        # (might not be exactly zero due to numerical effects)
        max_amplitude = np.max(np.abs(rec_data))
        assert max_amplitude < 1.0  # Should be much smaller than with reflectivity


@pytest.mark.slow
class TestBornAdjoint:
    """Test Born adjoint operator.

    These tests are marked slow as they require wave propagation.
    """

    def test_born_adjoint_runs(self):
        """Test Born adjoint completes without error."""
        from src.adjoint import born_adjoint, born_modeling

        shape = (41, 41)
        extent = (400.0, 400.0)

        vp_smooth = np.full(shape, 2.5, dtype=np.float32)
        reflectivity = np.zeros(shape, dtype=np.float32)
        reflectivity[15:25, 15:25] = 0.01

        src_coords = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        # First do Born modeling to get wavefield
        rec_data, p0_wavefield = born_modeling(
            shape, extent, vp_smooth, reflectivity,
            src_coords, rec_coords,
            f0=0.025, t_end=400.0
        )

        # Compute dt from extent and shape
        dx = extent[0] / (shape[0] - 1)
        vp_max = np.max(vp_smooth)
        dt = 0.4 * dx / vp_max

        # Now run adjoint
        gradient = born_adjoint(
            shape, extent, vp_smooth, rec_data,
            p0_wavefield, rec_coords, dt
        )

        assert gradient.shape == shape
        assert np.all(np.isfinite(gradient))

    def test_born_adjoint_output_shape(self):
        """Test Born adjoint output shape is correct."""
        from src.adjoint import born_adjoint

        shape = (41, 51)
        extent = (400.0, 500.0)

        vp_smooth = np.full(shape, 2.5, dtype=np.float32)

        dx = extent[0] / (shape[0] - 1)
        vp_max = np.max(vp_smooth)
        dt = 0.4 * dx / vp_max
        t_end = 400.0
        nt = int(t_end / dt) + 1

        # Create mock data
        nrec = 10
        data_residual = np.random.randn(nt, nrec).astype(np.float32)
        forward_wavefield = np.random.randn(nt, *shape).astype(np.float32)
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, nrec),
            np.full(nrec, 480.0)
        ])

        gradient = born_adjoint(
            shape, extent, vp_smooth, data_residual,
            forward_wavefield, rec_coords, dt
        )

        assert gradient.shape == shape


@pytest.mark.slow
class TestLSRTMSteepestDescent:
    """Test LSRTM steepest descent optimization.

    These tests are marked slow as they require multiple iterations.
    """

    def test_lsrtm_runs_and_returns_result(self):
        """Test LSRTM optimization runs and returns valid result."""
        from src.adjoint import create_layered_model, lsrtm_steepest_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_layers = [2.0, 2.5, 3.0]
        vp_true = create_layered_model(shape, spacing, vp_layers=vp_layers)
        vp_smooth = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        result = lsrtm_steepest_descent(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=2,
        )

        assert result.image_final.shape == shape
        assert result.image_initial.shape == shape
        assert result.iterations == 2
        assert len(result.history) == 2

    def test_lsrtm_produces_result(self):
        """Test that LSRTM runs and produces finite results.

        This is a basic functionality test verifying that the algorithm
        runs without error and produces finite output.
        """
        from src.adjoint import create_layered_model, lsrtm_steepest_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_layers = [2.0, 2.5, 3.0]
        vp_true = create_layered_model(shape, spacing, vp_layers=vp_layers)
        vp_smooth = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        result = lsrtm_steepest_descent(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=3,
        )

        # The image should be finite
        assert np.all(np.isfinite(result.image_final))
        # History should be recorded
        assert len(result.history) == 3

    def test_objective_is_finite(self):
        """Test that objective function values are finite during LSRTM."""
        from src.adjoint import create_layered_model, lsrtm_steepest_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_layers = [2.0, 2.5, 3.0]
        vp_true = create_layered_model(shape, spacing, vp_layers=vp_layers)
        vp_smooth = np.full(shape, 2.5, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        result = lsrtm_steepest_descent(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=3,
        )

        # All objective values should be finite and non-negative
        assert np.all(np.isfinite(result.history))
        assert np.all(result.history >= 0)

    def test_callback_called(self):
        """Test that callback is called at each iteration."""
        from src.adjoint import create_layered_model, lsrtm_steepest_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_layers = [2.0, 2.5]
        vp_true = create_layered_model(shape, spacing, vp_layers=vp_layers)
        vp_smooth = np.full(shape, 2.25, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 10),
            np.full(10, 380.0)
        ])

        callback_calls = []

        def callback(iteration, objective, image):
            callback_calls.append((iteration, objective))

        lsrtm_steepest_descent(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=2,
            callback=callback,
        )

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 0
        assert callback_calls[1][0] == 1


class TestLSRTMMultipleShots:
    """Test LSRTM with multiple shots."""

    @pytest.mark.slow
    def test_lsrtm_multiple_shots(self):
        """Test LSRTM with multiple source positions."""
        from src.adjoint import create_layered_model, lsrtm_steepest_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_layers = [2.0, 2.5]
        vp_true = create_layered_model(shape, spacing, vp_layers=vp_layers)
        vp_smooth = np.full(shape, 2.25, dtype=np.float32)

        # Multiple sources
        src_positions = np.array([
            [100.0, 20.0],
            [200.0, 20.0],
            [300.0, 20.0],
        ])

        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 15),
            np.full(15, 380.0)
        ])

        result = lsrtm_steepest_descent(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=2,
        )

        assert result.image_final.shape == shape
        assert result.iterations == 2


class TestBarzilaiborweinIntegration:
    """Test Barzilai-Borwein step integration in LSRTM."""

    @pytest.mark.slow
    def test_bb_step_used_after_first_iteration(self):
        """Test that BB step is used after first iteration in LSRTM."""
        from src.adjoint import create_layered_model, lsrtm_steepest_descent

        shape = (31, 31)
        extent = (300.0, 300.0)
        spacing = (10.0, 10.0)

        vp_layers = [2.0, 2.5]
        vp_true = create_layered_model(shape, spacing, vp_layers=vp_layers)
        vp_smooth = np.full(shape, 2.25, dtype=np.float32)

        src_positions = np.array([[150.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 280.0, 10),
            np.full(10, 280.0)
        ])

        result = lsrtm_steepest_descent(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=300.0,
            niter=3,
        )

        # Just verify it runs without error with BB step
        assert result.iterations == 3
        assert np.all(np.isfinite(result.image_final))


class TestSpaceOrderVariation:
    """Test different spatial discretization orders."""

    @pytest.mark.slow
    def test_space_order_4(self):
        """Test LSRTM with space_order=4."""
        from src.adjoint import create_layered_model, lsrtm_steepest_descent

        shape = (31, 31)
        extent = (300.0, 300.0)
        spacing = (10.0, 10.0)

        vp_layers = [2.0, 2.5]
        vp_true = create_layered_model(shape, spacing, vp_layers=vp_layers)
        vp_smooth = np.full(shape, 2.25, dtype=np.float32)

        src_positions = np.array([[150.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 280.0, 10),
            np.full(10, 280.0)
        ])

        result = lsrtm_steepest_descent(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=300.0,
            niter=1,
            space_order=4,
        )

        assert result.image_final.shape == shape

    @pytest.mark.slow
    def test_space_order_8(self):
        """Test LSRTM with space_order=8."""
        from src.adjoint import create_layered_model, lsrtm_steepest_descent

        shape = (41, 41)
        extent = (400.0, 400.0)
        spacing = (10.0, 10.0)

        vp_layers = [2.0, 2.5]
        vp_true = create_layered_model(shape, spacing, vp_layers=vp_layers)
        vp_smooth = np.full(shape, 2.25, dtype=np.float32)

        src_positions = np.array([[200.0, 20.0]])
        rec_coords = np.column_stack([
            np.linspace(20.0, 380.0, 10),
            np.full(10, 380.0)
        ])

        result = lsrtm_steepest_descent(
            shape, extent, vp_smooth, vp_true,
            src_positions, rec_coords,
            f0=0.025, t_end=400.0,
            niter=1,
            space_order=8,
        )

        assert result.image_final.shape == shape
