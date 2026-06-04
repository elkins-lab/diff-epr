import jax
import jax.numpy as jnp

from diff_epr.kernels import deer_trace, spin_distance


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
