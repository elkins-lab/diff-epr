# 📻 diff-epr

**diff-epr** provides differentiable kernels for simulating EPR distance distributions (DEER/PELDOR) from structural ensembles using JAX.

## Quick Start

```python
import jax.numpy as jnp
from diff_epr.kernels import deer_trace

# Distances in Angstroms
distances = jnp.array([30.0, 35.0, 40.0])
# Time points in microseconds
time = jnp.linspace(0, 2.0, 100)

# Simulate DEER trace
trace = deer_trace(distances, time)
```
