# EPR Theory

## DEER Spectroscopy

Double Electron-Electron Resonance (DEER) measures the dipolar coupling between two spin labels (e.g., nitroxide labels). The coupling frequency $\omega$ is inversely proportional to the cube of the distance $r$:

$$\omega \propto \frac{1}{r^3}$$

## Time-Domain Signal

The normalized DEER signal $V(t)$ for a pair of spins is given by:

$$V(t) = V_{inter}(t) \cdot [1 - \lambda(1 - \cos(\omega t))]$$

where $\lambda$ is the modulation depth and $V_{inter}(t)$ represents the background decay from intermolecular interactions. **diff-epr** models this background as an exponential decay:

$$V_{inter}(t) = \exp(-k \cdot t)$$
