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
        time: (T,) time points in microseconds.
        modulation_depth: Modulation depth λ.
        background_decay: Decay rate of the background signal.

    Returns:
        V(t) normalized DEER signal.
    """
    # Dipolar coupling frequency: ω = 2π * ν_dd
    # ν_dd = 52040 / r^3 (MHz for Angstroms)
    nu_dd = 52040.0 / (distances**3)
    omega = 2.0 * jnp.pi * nu_dd

    # Kernel matrix (T, M)
    # V(t) = 1 - λ(1 - cos(ωt))
    kernel = 1.0 - modulation_depth * (1.0 - jnp.cos(omega[None, :] * time[:, None]))

    # Integrate over distance distribution (assumed uniform weight here for simplicity)
    signal = jnp.mean(kernel, axis=-1)

    # Background decay
    background = jnp.exp(-background_decay * time)

    return signal * background


def deer_trace_rotamers(
    rotamers1: jnp.ndarray,
    weights1: jnp.ndarray,
    rotamers2: jnp.ndarray,
    weights2: jnp.ndarray,
    time: jnp.ndarray,
    modulation_depth: float = 0.3,
    background_decay: float = 0.05,
) -> jnp.ndarray:
    """
    Simulate a DEER trace using weighted rotamer libraries for both labels.

    Args:
        rotamers1: (N1, 3) coordinates of label 1 rotamers.
        weights1: (N1,) weights for label 1 rotamers (sum to 1).
        rotamers2: (N2, 3) coordinates of label 2 rotamers.
        weights2: (N2,) weights for label 2 rotamers (sum to 1).
        time: (T,) time points.
        modulation_depth: Modulation depth λ.
        background_decay: Background decay rate.

    Returns:
        V(t) normalized DEER signal.
    """
    # 1. Compute all pairwise distances (N1, N2)
    diff = rotamers1[:, None, :] - rotamers2[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-9)

    # 2. Compute pairwise weights (N1, N2)
    p_ij = (
        weights1[:, None]
        * weights2[
            None,
            :,
        ]
    )

    # 3. Simulate DEER kernel for each distance
    # Reshape distances to 1D for deer_trace (N1*N2,)
    dist_flat = dist.flatten()
    p_flat = p_ij.flatten()

    # Dipolar frequencies
    nu_dd = 52040.0 / (dist_flat**3)
    omega = 2.0 * jnp.pi * nu_dd

    # Kernel matrix (T, N_pairs)
    kernel = 1.0 - modulation_depth * (1.0 - jnp.cos(omega[None, :] * time[:, None]))

    # 4. Weighted average over rotamer pairs
    signal = jnp.sum(kernel * p_flat[None, :], axis=-1)

    # Background decay
    background = jnp.exp(-background_decay * time)

    return signal * background
