# 📻 diff-epr

**diff-epr** provides differentiable kernels for simulating EPR distance distributions (DEER/PELDOR) from structural ensembles using JAX.

## Quick Start

```python
import jax
import jax.numpy as jnp
from diff_epr import deer_trace

# Distances in Angstroms from a multi-component distance distribution
distances = jnp.array([30.0, 35.0, 40.0])
# Time points in microseconds
time = jnp.linspace(0, 2.0, 100)

# Simulate DEER trace
trace = deer_trace(distances, time)

# Gradient of the sum of the trace w.r.t. inter-spin distances
grad_r = jax.grad(lambda r: jnp.sum(deer_trace(r, time)))(distances)
print(grad_r)  # use to refine distances against an experimental trace
```
