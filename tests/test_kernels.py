import jax
import jax.numpy as jnp
from diff_epr.kernels import spin_distance, deer_trace, deer_trace_oriented


def test_epr_basic():
    c1 = jnp.array([[0.0, 0.0, 0.0]])
    c2 = jnp.array([[30.0, 0.0, 0.0]])
    dist = spin_distance(c1, c2)
    assert jnp.allclose(dist, 30.0)


def test_deer_trace():
    distances = jnp.array([30.0, 40.0])
    time = jnp.linspace(0, 2.0, 100)
    trace = deer_trace(distances, time)
    assert trace.shape == (100,)
    assert trace[0] == 1.0
    assert jnp.all(trace <= 1.0)


def test_epr_differentiable():
    distances = jnp.array([30.0, 40.0])
    time = jnp.linspace(0, 2.0, 100)

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(deer_trace(x, time))

    grads = jax.grad(loss)(distances)
    assert grads.shape == distances.shape
    assert not jnp.any(jnp.isnan(grads))


def test_deer_dipolar_parity():
    """
    Verify DEER coupling frequency against the standard 52040 MHz·A^3 constant.
    """
    # At r = 30.0 A:
    # nu_dd = 52040 / 30^3 = 52040 / 27000 = 1.9274 MHz
    distances = jnp.array([30.0])
    time = jnp.array([1.0])  # 1.0 microseconds

    # Trace V(t) = 1 - λ(1 - cos(2π * nu_dd * t))
    # omega = 2 * pi * 1.9274 = 12.1101 rad/us
    # cos(12.1101) = 0.89776
    # V(1.0) = 1 - 0.3(1 - 0.89776) = 1 - 0.3(0.10224) = 0.96933

    # We set background_decay=0 for parity check
    trace = deer_trace(distances, time, modulation_depth=0.3, background_decay=0.0)
    assert jnp.allclose(trace[0], 0.96933, atol=1e-4)


def test_deer_trace_oriented():
    """
    Verify that deer_trace_oriented computes the correct single-crystal DEER
    trace using theta_r1 from relative_orientation.

    At theta_r1 = pi/4:
      nu_dd = nu_max * (1 - 3*cos^2(pi/4)) = nu_max * (1 - 3/2) = -nu_max/2
    This is a signed frequency distinct from the equatorial (theta=pi/2) peak
    that dominates the powder average, making it a clean discriminating test.
    """
    dist = 30.0
    time = jnp.linspace(0, 2.0, 10)

    theta = jnp.pi / 4  # 45 degrees
    orientation = jnp.array([theta, 0.0, 0.0, 0.0, 0.0])
    trace = deer_trace_oriented(dist, orientation, time)

    assert trace.shape == (10,)
    # At t=0: V(0) = 1 - lambda*(1 - cos(0)) = 1
    assert jnp.allclose(trace[0], 1.0)

    # Analytical check at each time point using the expected nu_dd
    nu_max = 52040.0 / (dist**3)
    nu_dd = nu_max * (1.0 - 3.0 * jnp.cos(theta) ** 2)  # = -nu_max/2
    expected = 1.0 - 0.3 * (1.0 - jnp.cos(2.0 * jnp.pi * nu_dd * time))
    assert jnp.allclose(trace, expected, atol=1e-5)


def test_deer_oriented_uses_theta_r1():
    """
    Verify that different theta_r1 values produce different traces.

    The old bug: relative_orientation was accepted but never read, so ALL
    values of theta_r1 produced the SAME output (a fixed powder average).
    This test explicitly checks that the output changes with theta_r1.
    """
    dist = 40.0
    time = jnp.linspace(0, 3.0, 50)

    # Three distinct orientations
    theta_0   = jnp.array([0.0,        0.0, 0.0, 0.0, 0.0])  # along B0
    theta_pi4 = jnp.array([jnp.pi/4,   0.0, 0.0, 0.0, 0.0])  # 45 deg
    theta_pi2 = jnp.array([jnp.pi/2,   0.0, 0.0, 0.0, 0.0])  # perp to B0

    trace_0   = deer_trace_oriented(dist, theta_0,   time)
    trace_pi4 = deer_trace_oriented(dist, theta_pi4, time)
    trace_pi2 = deer_trace_oriented(dist, theta_pi2, time)

    # All three must differ from each other (at all time points beyond t=0)
    assert not jnp.allclose(trace_0[1:],   trace_pi4[1:], atol=1e-4), (
        "theta_r1=0 and theta_r1=pi/4 must produce different DEER traces"
    )
    assert not jnp.allclose(trace_pi4[1:], trace_pi2[1:], atol=1e-4), (
        "theta_r1=pi/4 and theta_r1=pi/2 must produce different DEER traces"
    )


def test_deer_oriented_magic_angle():
    """
    At the magic angle (arccos(1/sqrt(3)) ≈ 54.74°) nu_dd = 0, so V(t) = 1
    for all t (no dipolar modulation).

    This is a sharp physical prediction that would have been invisible to
    a powder-average implementation (which averages many angles).
    """
    dist = 35.0
    magic_angle = jnp.arccos(1.0 / jnp.sqrt(3.0))  # ≈ 54.74 deg
    orientation = jnp.array([magic_angle, 0.0, 0.0, 0.0, 0.0])
    time = jnp.linspace(0, 5.0, 100)

    trace = deer_trace_oriented(dist, orientation, time)

    # At magic angle: nu_dd = nu_max*(1 - 3*(1/3)) = 0 → no modulation
    # V(t) = 1 - lambda*(1 - cos(0)) = 1 - 0 = 1 for all t
    assert jnp.allclose(trace, 1.0, atol=1e-5), (
        "At magic angle nu_dd=0, DEER trace should be flat at 1.0"
    )


def test_deer_oriented_differentiable_wrt_orientation():
    """
    Verify that gradients flow through theta_r1.

    The old code did not use relative_orientation, so gradients w.r.t. it
    were always zero.  The corrected implementation should give non-zero
    gradients (except exactly at the magic angle).
    """
    dist = 30.0
    time = jnp.linspace(0.5, 2.0, 20)  # exclude t=0 (trivially 1 for all theta)

    def loss(orientation):
        return jnp.sum(deer_trace_oriented(dist, orientation, time))

    orientation = jnp.array([jnp.pi / 3, 0.0, 0.0, 0.0, 0.0])
    grads = jax.grad(loss)(orientation)

    # Gradient w.r.t. theta_r1 must be non-zero (nu_dd varies with theta)
    assert not jnp.isclose(grads[0], 0.0, atol=1e-6), (
        "Gradient w.r.t. theta_r1 must be non-zero (was always 0 in old code)"
    )
