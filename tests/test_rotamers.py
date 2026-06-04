import jax.numpy as jnp
from diff_epr.kernels import deer_trace_rotamers


def test_deer_rotamers():
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
