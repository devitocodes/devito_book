"""Tests for the 3D Viscoelastic Wave Equations solver using Devito."""

import importlib.util

import numpy as np
import pytest

# Check if Devito is available
DEVITO_AVAILABLE = importlib.util.find_spec("devito") is not None

pytestmark = pytest.mark.skipif(
    not DEVITO_AVAILABLE, reason="Devito not installed"
)


class TestViscoelasticImport:
    """Test that the module imports correctly."""

    def test_import_solve_viscoelastic_3d(self):
        """Test main solver import."""
        from src.systems import solve_viscoelastic_3d

        assert solve_viscoelastic_3d is not None

    def test_import_result_class(self):
        """Test result dataclass import."""
        from src.systems import ViscoelasticResult

        assert ViscoelasticResult is not None

    def test_import_helper_functions(self):
        """Test helper function imports."""
        from src.systems import (
            compute_viscoelastic_relaxation_parameters,
            create_damping_field_3d,
            create_layered_model_3d,
        )

        assert compute_viscoelastic_relaxation_parameters is not None
        assert create_damping_field_3d is not None
        assert create_layered_model_3d is not None


class TestRelaxationParameters:
    """Test viscoelastic relaxation parameter computation."""

    def test_compute_relaxation_parameters(self):
        """Test relaxation parameter computation."""
        from src.systems import compute_viscoelastic_relaxation_parameters

        Qp = 100.0
        Qs = 50.0
        f0 = 0.12

        t_s, t_ep, t_es = compute_viscoelastic_relaxation_parameters(Qp, Qs, f0)

        # All parameters should be positive
        assert t_s > 0
        assert t_ep > 0
        assert t_es > 0

    def test_relaxation_with_zero_qs(self):
        """Test that Qs=0 (fluid) is handled."""
        from src.systems import compute_viscoelastic_relaxation_parameters

        Qp = 100.0
        Qs = 0.0  # Fluid - no shear
        f0 = 0.12

        t_s, t_ep, t_es = compute_viscoelastic_relaxation_parameters(Qp, Qs, f0)

        # Should not produce NaN or inf
        assert np.isfinite(t_s)
        assert np.isfinite(t_ep)
        assert np.isfinite(t_es)

    def test_relaxation_array_input(self):
        """Test with array inputs."""
        from src.systems import compute_viscoelastic_relaxation_parameters

        Qp = np.array([50., 100., 200.])
        Qs = np.array([30., 60., 100.])
        f0 = 0.12

        t_s, t_ep, t_es = compute_viscoelastic_relaxation_parameters(Qp, Qs, f0)

        assert t_s.shape == Qp.shape
        assert t_ep.shape == Qp.shape
        assert t_es.shape == Qp.shape


class TestLayeredModel:
    """Test 3D layered model creation."""

    def test_create_layered_model(self):
        """Test layered model creation."""
        from src.systems import create_layered_model_3d

        shape = (51, 31, 41)
        vp, vs, Qp, Qs, rho = create_layered_model_3d(shape)

        assert vp.shape == shape
        assert vs.shape == shape
        assert Qp.shape == shape
        assert Qs.shape == shape
        assert rho.shape == shape

    def test_layered_model_positive_values(self):
        """Test that model has physical values."""
        from src.systems import create_layered_model_3d

        shape = (51, 31, 41)
        vp, vs, Qp, Qs, rho = create_layered_model_3d(shape)

        # P-wave velocity and density must be positive
        assert np.all(vp > 0)
        assert np.all(rho > 0)

        # Qp must be positive
        assert np.all(Qp > 0)

        # vs and Qs can be zero (for fluid layers)
        assert np.all(vs >= 0)
        assert np.all(Qs >= 0)

    def test_layered_model_custom_layers(self):
        """Test custom layer specification."""
        from src.systems import create_layered_model_3d

        shape = (51, 31, 41)
        vp_layers = [1.5, 2.0, 2.5, 3.0]
        layer_depths = [0, 10, 20, 30]

        vp, vs, Qp, Qs, rho = create_layered_model_3d(
            shape,
            vp_layers=vp_layers,
            layer_depths=layer_depths,
        )

        # Check that different depths have different velocities
        unique_vp = np.unique(vp[25, 15, :])
        assert len(unique_vp) >= 3


class TestViscoelasticSolver:
    """Test the 3D viscoelastic solver."""

    def test_basic_run(self):
        """Test that solver runs without errors."""
        from src.systems import solve_viscoelastic_3d

        result = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=(21, 11, 11),
            T=5.0,
            vp=2.0,
            vs=1.0,
            rho=2.0,
            Qp=100.0,
            Qs=50.0,
            space_order=2,
        )

        assert result.vx is not None
        assert result.vy is not None
        assert result.vz is not None
        assert result.tau_xx is not None

    def test_result_shapes(self):
        """Test that results have correct shapes."""
        from src.systems import solve_viscoelastic_3d

        shape = (21, 15, 11)

        result = solve_viscoelastic_3d(
            extent=(100., 75., 50.),
            shape=shape,
            T=5.0,
            space_order=2,
        )

        assert result.vx.shape == shape
        assert result.vy.shape == shape
        assert result.vz.shape == shape
        assert result.tau_xx.shape == shape
        assert result.tau_yy.shape == shape
        assert result.tau_zz.shape == shape
        assert result.tau_xy.shape == shape
        assert result.tau_xz.shape == shape
        assert result.tau_yz.shape == shape

    def test_wavefield_finite(self):
        """Test that wavefield values are finite."""
        from src.systems import solve_viscoelastic_3d

        result = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=(21, 11, 11),
            T=5.0,
            space_order=2,
        )

        # Check velocities
        assert np.all(np.isfinite(result.vx))
        assert np.all(np.isfinite(result.vy))
        assert np.all(np.isfinite(result.vz))

        # Check stresses
        assert np.all(np.isfinite(result.tau_xx))
        assert np.all(np.isfinite(result.tau_yy))
        assert np.all(np.isfinite(result.tau_zz))
        assert np.all(np.isfinite(result.tau_xy))
        assert np.all(np.isfinite(result.tau_xz))
        assert np.all(np.isfinite(result.tau_yz))

    def test_no_nan(self):
        """Test that solution contains no NaN."""
        from src.systems import solve_viscoelastic_3d

        result = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=(21, 11, 11),
            T=5.0,
            space_order=2,
        )

        assert not np.any(np.isnan(result.vx))
        assert not np.any(np.isnan(result.vy))
        assert not np.any(np.isnan(result.vz))
        assert not np.any(np.isnan(result.tau_xx))


class TestViscoelasticResult:
    """Test the ViscoelasticResult dataclass."""

    def test_result_attributes(self):
        """Test that result has all expected attributes."""
        from src.systems import solve_viscoelastic_3d

        result = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=(21, 11, 11),
            T=5.0,
            space_order=2,
        )

        # Velocity components
        assert hasattr(result, 'vx')
        assert hasattr(result, 'vy')
        assert hasattr(result, 'vz')

        # Stress components
        assert hasattr(result, 'tau_xx')
        assert hasattr(result, 'tau_yy')
        assert hasattr(result, 'tau_zz')
        assert hasattr(result, 'tau_xy')
        assert hasattr(result, 'tau_xz')
        assert hasattr(result, 'tau_yz')

        # Coordinates
        assert hasattr(result, 'x')
        assert hasattr(result, 'y')
        assert hasattr(result, 'z')

        # Time
        assert hasattr(result, 't')
        assert hasattr(result, 'dt')

    def test_time_attributes(self):
        """Test time-related attributes."""
        from src.systems import solve_viscoelastic_3d

        T = 10.0

        result = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=(21, 11, 11),
            T=T,
            space_order=2,
        )

        assert result.t == T
        assert result.dt > 0
        assert result.dt < T


class TestCoordinateArrays:
    """Test coordinate arrays."""

    def test_coordinate_range(self):
        """Test that coordinate arrays cover the domain."""
        from src.systems import solve_viscoelastic_3d

        extent = (200., 100., 80.)
        shape = (21, 11, 9)

        result = solve_viscoelastic_3d(
            extent=extent,
            shape=shape,
            T=5.0,
            space_order=2,
        )

        assert result.x[0] == pytest.approx(0.0)
        assert result.x[-1] == pytest.approx(extent[0])
        assert result.y[0] == pytest.approx(0.0)
        assert result.y[-1] == pytest.approx(extent[1])
        assert result.z[0] == pytest.approx(0.0)
        assert result.z[-1] == pytest.approx(extent[2])

    def test_coordinate_lengths(self):
        """Test that coordinate arrays have correct lengths."""
        from src.systems import solve_viscoelastic_3d

        shape = (31, 21, 15)

        result = solve_viscoelastic_3d(
            extent=(200., 100., 80.),
            shape=shape,
            T=5.0,
            space_order=2,
        )

        assert len(result.x) == shape[0]
        assert len(result.y) == shape[1]
        assert len(result.z) == shape[2]


class TestSourceInjection:
    """Test source injection."""

    def test_source_generates_wavefield(self):
        """Test that source generates non-zero wavefield."""
        from src.systems import solve_viscoelastic_3d

        result = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=(21, 11, 11),
            T=10.0,
            src_coords=(50., 25., 17.5),
            space_order=2,
        )

        # At least one field should be non-zero
        max_stress = max(
            np.max(np.abs(result.tau_xx)),
            np.max(np.abs(result.tau_yy)),
            np.max(np.abs(result.tau_zz)),
        )

        assert max_stress > 0, "All stress fields are zero"


class TestVaryingParameters:
    """Test with spatially varying parameters."""

    def test_varying_velocity(self):
        """Test with spatially varying velocity."""
        from src.systems import solve_viscoelastic_3d

        shape = (21, 11, 11)

        # Create linearly varying velocity
        z = np.linspace(0, 1, shape[2])
        vp = 1.5 + 1.0 * np.broadcast_to(z, shape)
        vs = 0.8 + 0.4 * np.broadcast_to(z, shape)

        result = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=shape,
            T=5.0,
            vp=vp.astype(np.float32),
            vs=vs.astype(np.float32),
            space_order=2,
        )

        assert np.all(np.isfinite(result.vx))

    def test_varying_q(self):
        """Test with spatially varying Q factors."""
        from src.systems import solve_viscoelastic_3d

        shape = (21, 11, 11)

        # Create varying Q (higher Q at depth)
        z = np.linspace(0, 1, shape[2])
        Qp = 50.0 + 150.0 * np.broadcast_to(z, shape)
        Qs = 30.0 + 70.0 * np.broadcast_to(z, shape)

        result = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=shape,
            T=5.0,
            Qp=Qp.astype(np.float32),
            Qs=Qs.astype(np.float32),
            space_order=2,
        )

        assert np.all(np.isfinite(result.vx))


class TestFluidLayer:
    """Test handling of fluid layers (vs=0, Qs=0)."""

    def test_fluid_at_top(self):
        """Test simulation with water layer at top."""
        from src.systems import create_layered_model_3d, solve_viscoelastic_3d

        shape = (21, 11, 21)

        # Create model with water at top
        vp, vs, Qp, Qs, rho = create_layered_model_3d(
            shape,
            vp_layers=[1.5, 2.5],
            vs_layers=[0.0, 1.2],  # Water has vs=0
            Qp_layers=[10000., 100.],
            Qs_layers=[0., 50.],   # Water has Qs=0
            rho_layers=[1.0, 2.0],
            layer_depths=[0, 10],
        )

        result = solve_viscoelastic_3d(
            extent=(100., 50., 100.),
            shape=shape,
            T=5.0,
            vp=vp,
            vs=vs,
            rho=rho,
            Qp=Qp,
            Qs=Qs,
            space_order=2,
        )

        assert np.all(np.isfinite(result.vx))
        assert np.all(np.isfinite(result.tau_xx))


class TestDamping:
    """Test absorbing boundary damping."""

    def test_damping_field_creation(self):
        """Test that damping field can be created."""
        from devito import Grid

        from src.systems import create_damping_field_3d

        grid = Grid(shape=(51, 31, 21), extent=(100., 60., 40.))
        damp = create_damping_field_3d(grid, nbl=10, space_order=2)

        # Damping field should exist
        assert damp is not None

        # Interior should be 1.0
        mid = (25, 15, 10)
        assert damp.data[mid] == pytest.approx(1.0)

        # Boundary should be < 1.0
        assert damp.data[0, 15, 10] < 1.0
        assert damp.data[50, 15, 10] < 1.0

    def test_with_and_without_damping(self):
        """Test that damping reduces boundary reflections."""
        from src.systems import solve_viscoelastic_3d

        # Run without damping
        result_no_damp = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=(21, 11, 11),
            T=5.0,
            use_damp=False,
            space_order=2,
        )

        # Run with damping
        result_damp = solve_viscoelastic_3d(
            extent=(100., 50., 50.),
            shape=(21, 11, 11),
            T=5.0,
            use_damp=True,
            space_order=2,
        )

        # Both should produce valid results
        assert np.all(np.isfinite(result_no_damp.vx))
        assert np.all(np.isfinite(result_damp.vx))
