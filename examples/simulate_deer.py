import jax.numpy as jnp

from diff_epr.kernels import deer_trace


def main():
    # 1. Distance distribution P(r) - two peaks
    distances = jnp.array([30.0, 32.0, 35.0, 45.0, 47.0])

    # 2. Time points (microseconds)
    time = jnp.linspace(0, 3.0, 128)

    # 3. Simulate DEER trace
    # Modulation depth 0.3, fast background decay
    trace = deer_trace(distances, time, modulation_depth=0.3, background_decay=0.1)

    print(f"Time-domain signal (first 5 points): {trace[:5]}")

    # In a real use case, you would use jax.grad to optimize 'distances'
    # to fit an experimental 'trace'.


if __name__ == "__main__":
    main()
