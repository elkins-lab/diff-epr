import jax.numpy as jnp


def spin_distance(
    coords1: jnp.ndarray,
    coords2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute distance between spin centers.

    Args:
        coords1: (N, 3) coordinates of first spin label.
        coords2: (N, 3) coordinates of second spin label.

    Returns:
        Distances (N,).
    """
    dist_sq = jnp.sum((coords1 - coords2) ** 2, axis=-1)
    # Safe distance for gradients (avoids NaN at dist=0)
    dist = jnp.sqrt(jnp.where(dist_sq > 0, dist_sq, 1.0))
    return jnp.where(dist_sq > 0, dist, 0.0)


def deer_trace(
    distances: jnp.ndarray,
    time: jnp.ndarray,
    modulation_depth: float = 0.3,
    background_decay: float = 0.05,
) -> jnp.ndarray:
    """
    Simulate a DEER (Double Electron-Electron Resonance) time-domain trace.

    Args:
        distances: (M,) distribution of distances.
        time: (T,) time points.
        modulation_depth: Modulation depth λ.
        background_decay: Decay rate of the background signal.

    Returns:
        V(t) normalized DEER signal.
    """
    # Dipolar coupling frequency: ω ~ 1/r^3
    # Actually, V(t) = (1 - λ(1 - cos(ωt))) * background
    # This is a simplified kernel.
    omega = 52.04 / (distances**3)  # MHz/Angstrom^3 constant

    # Kernel matrix (T, M)
    kernel = 1.0 - modulation_depth * (1.0 - jnp.cos(omega[None, :] * time[:, None]))

    # Integrate over distance distribution (assumed uniform weight here for simplicity)
    signal = jnp.mean(kernel, axis=-1)

    # Background decay
    background = jnp.exp(-background_decay * time)

    return signal * background
