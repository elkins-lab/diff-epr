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

    def loss(x):
        return jnp.sum(deer_trace(x, time))

    grads = jax.grad(loss)(distances)
    assert grads.shape == distances.shape
    assert not jnp.any(jnp.isnan(grads))
