import jax.numpy as jnp

from diff_epr.kernels import deer_trace_rotamers


def test_deer_rotamers() -> None:
    """
    Verify that weighted rotamer simulation runs.
    """
    # 2 rotamers for label 1
    rot1 = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    w1 = jnp.array([0.5, 0.5])

    # 1 rotamer for label 2
    rot2 = jnp.array([[30.0, 0.0, 0.0]])
    w2 = jnp.array([1.0])

    time = jnp.linspace(0, 1.0, 10)

    trace = deer_trace_rotamers(rot1, w1, rot2, w2, time)
    assert trace.shape == (10,)
    assert trace[0] == 1.0


def test_deer_rotamers_differentiable() -> None:
    """
    Verify that gradients flow through the rotamer coordinates and weights
    without producing NaNs.
    """
    import jax

    rot1 = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    w1 = jnp.array([0.5, 0.5])
    rot2 = jnp.array([[30.0, 0.0, 0.0]])
    w2 = jnp.array([1.0])
    time = jnp.linspace(0, 1.0, 10)

    def loss(r1: jnp.ndarray, weights1: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(deer_trace_rotamers(r1, weights1, rot2, w2, time))

    grad_r1, grad_w1 = jax.grad(loss, argnums=(0, 1))(rot1, w1)

    assert grad_r1.shape == rot1.shape
    assert grad_w1.shape == w1.shape
    assert not jnp.any(jnp.isnan(grad_r1))
    assert not jnp.any(jnp.isnan(grad_w1))
