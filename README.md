# 📻 diff-epr: Differentiable EPR/DEER Simulation in JAX

[![Tests](https://github.com/elkins-lab/diff-epr/actions/workflows/test.yml/badge.svg)](https://github.com/elkins-lab/diff-epr/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/backend-JAX-9cf.svg)](https://github.com/google/jax)

**diff-epr** provides differentiable kernels for simulating EPR distance distributions (DEER/PELDOR) from structural ensembles and spin-label rotamers.

---

## 🎯 Features

- **Spin-Label Modeling:** Differentiable distance calculations between paramagnetic centers.
- **Orientation Selection:** Support for the **Polyhach 5-angle formula** (Polyhach et al., 2007) to model relative domain orientations.
- **Time-Domain Simulation:** Simulate DEER modulation traces $V(t)$ with parameterizable background decay and modulation depth.
- **Rotamer Library Integration:** Support for weighted rotamer averages in distance distribution calculations.
- **Hardware Acceleration:** GPU-optimized distance kernels via JAX.

---

## 📚 Tutorials

Experience **diff-epr** directly in your browser:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/diff-epr/blob/main/examples/interactive_tutorials/deer_spectroscopy_tutorial.ipynb) **DEER Spectroscopy Simulation** — Learn how to simulate time-domain traces from distance distributions and rotamer clouds.

---

## 🏗️ Technical Architecture

- **Backend:** JAX (XLA-compiled).
- **Physics:** Dipolar coupling frequency ($\omega$) based kernels.
- **Performance:** $O(N)$ scaling for distance distribution integration.

---

## 🚀 Roadmap

- [x] Core DEER trace simulation kernels.
- [x] Background decay and modulation depth parameters.
- [ ] Integration with MMM (Multi-Scale Modeling of Macromolecules) rotamer libraries.
- [ ] Full orientation-dependence support.

---

## 🚀 Installation

```bash
pip install diff-epr
```

## 🧪 Scientific Validation

- **Dipolar Frequency Parity:** Coupling frequencies verified against the Pake pattern $1/r^3$ dependence.
- **Time-Domain Accuracy:** DEER traces validated for parity against standard simulation tools (e.g., DeerAnalysis/MMM).
- **Auto-Diff Gradients:** Differentiable distance-to-signal kernels verified with JAX `grad`.

---

## 🔗 Related Projects

diff-epr is part of the **differentiable biophysics** ecosystem:

- [diff-biophys](https://github.com/elkins-lab/diff-biophys) — Core differentiable biophysics engine.
- [diff-hdx](https://github.com/elkins-lab/diff-hdx) — Differentiable HDX-MS prediction.
- [diff-fret](https://github.com/elkins-lab/diff-fret) — Differentiable FRET modeling.
- [synth-nmr](https://github.com/elkins-lab/synth-nmr) — NMR observable simulation.

---

## 📖 Citation

```bibtex
@software{diff_epr,
  author  = {Elkins, George},
  title   = {diff-epr: Differentiable EPR/DEER simulation in JAX},
  year    = {2026},
  url     = {https://github.com/elkins-lab/diff-epr},
  version = {0.1.1}
}
```

## ⚖️ License

MIT
