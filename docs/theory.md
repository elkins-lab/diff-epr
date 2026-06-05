# EPR Theory

## DEER Spectroscopy

Double Electron-Electron Resonance (DEER) measures the dipolar coupling between two spin labels (e.g., nitroxide labels). The coupling frequency $\omega$ is inversely proportional to the cube of the distance $r$:

$$\omega(\theta) = \frac{C}{r^3} (1 - 3\cos^2\theta)$$

where $C \approx 52,040$ MHz·Å³ is the dipolar coupling constant and $\theta$ is the angle between the inter-spin vector and the external magnetic field $B_0$.

## Orientation Selection

In high-field EPR, pulses excite only a subset of possible label orientations. **diff-epr** implements the **Polyhach (2007)** geometric model framework, which describes the mutual orientation of two labels using five independent angles ($\theta_{r1}, \phi_{r1}, \alpha, \beta, \gamma$).

### Current implementation: single-crystal oriented DEER

`deer_trace_oriented` currently implements the **single-crystal** case, where the interspin vector **r** makes a fixed angle $\theta_{r1}$ with the external field $B_0$.  This is the first and most fundamental angle in the Polyhach model:

$$\nu_{dd}(\theta_{r1}) = \frac{C}{r^3}(1 - 3\cos^2\theta_{r1})$$

At the **magic angle** $\theta_{r1} = \arccos(1/\sqrt{3}) \approx 54.7°$, $\nu_{dd} = 0$ and the dipolar modulation vanishes — a sharp physical signature that is directly testable.

### Future: full orientation selection

The remaining four angles ($\phi_{r1}, \alpha, \beta, \gamma$) govern how a given $B_0$ orientation selects sub-ensembles of spin labels via their g-tensor anisotropy (the "orientation selection" effect). This allows modeling of rigid complexes where the relative 3D orientation of domains, not just their distance, is of interest. Full implementation is planned for a future release.
