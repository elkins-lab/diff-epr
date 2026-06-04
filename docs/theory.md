# EPR Theory

## DEER Spectroscopy

Double Electron-Electron Resonance (DEER) measures the dipolar coupling between two spin labels (e.g., nitroxide labels). The coupling frequency $\omega$ is inversely proportional to the cube of the distance $r$:

$$\omega(\theta) = \frac{C}{r^3} (1 - 3\cos^2\theta)$$

where $C \approx 52,040$ MHz·Å³ is the dipolar coupling constant and $\theta$ is the angle between the inter-spin vector and the external magnetic field $B_0$.

## Orientation Selection

In high-field EPR, pulses excite only a subset of possible label orientations. **diff-epr** implements the **Polyhach (2007)** geometric model, which describes the mutual orientation of two labels using five independent angles ($\theta_{r1}, \phi_{r1}, \alpha, \beta, \gamma$).

This allows for the modeling of rigid complexes where the relative 3D orientation of domains, not just their distance, is of interest.
