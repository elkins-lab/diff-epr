# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-06-30

### Added
- Added full PEP 561 compliance by introducing a `py.typed` marker file.
- Added 100% test coverage including gradient safety tests for rotamers.
- Added interactive Google Colab DEER spectroscopy tutorial.
- Added Dependabot configuration for GitHub Actions.

### Fixed
- Fixed gradient instability (NaNs) in `deer_trace_rotamers` when evaluated at distance exactly equal to zero.
- Fixed pip installation order in CI to prevent `numpy` silent upgrades breaking `jax` module compatibility.
- Fixed `mypy` syntax errors running on Python 3.12+ environments by separating linting from the test matrix and pinning `numpy<2` in pre-commit.
- Fixed Node.js deprecation warnings and API rate limits in CI by forcing Node 24 and using GitHub cache for Hugging Face models.

## [0.1.1] - 2026-06-07

### Security
- Removed compromised `polyfill.io` CDN script from MkDocs configuration to resolve supply-chain vulnerability.
