# Changelog

All notable changes to PhysioMetrics will be documented in this file.

**Developer:** Ryan Sean Phillips
**Institution:** Seattle Children's Research Institute
**Funding:** NIDA K01DA058543
**Repository:** https://github.com/RyanSeanPhillips/PhysioMetrics

---

## [1.0.14] - 2024-11-10

### Added
- **Rebranded to PhysioMetrics** - Professional name for broader scope
  - Updated all application branding, documentation, and metadata
  - New three-column About dialog layout with author and funding information
  - Added K01 funding attribution (NIDA K01DA058543)
  - Created CITATION.cff for proper software citation
- Probability metrics for breath classification
  - P(noise) and P(breath) calculated from auto-threshold model
  - Computed on all detected peaks (including noise) for ML training

### Fixed
- Fixed P(noise)/P(breath) metrics to use all peaks (including noise)
- Fixed model parameter storage for probability metric calculation

### Changed
- Hidden 22 experimental ML metrics from CSV/PDF exports for clarity and performance
  - Metrics still computed for future ML work, just not displayed to users
  - Phase 2.2: Sigh detection features (9 metrics)
  - Phase 2.3 Group A: Shape & ratio metrics (6 metrics)
  - Phase 2.3 Group B: Normalized metrics (7 metrics)
- Simplified application description for future extensibility

### Known Issues
- Zero crossing markers may appear slightly offset on recordings with large DC offset removed by high-pass filtering
  - Issue is cosmetic and does not affect breath detection accuracy
  - Will be addressed in future ML refactor

---

## [1.0.10] - 2024-XX-XX

### Added
- Multi-file ABF concatenation support
- Spike2 .smrx file format support via CED SON64 library
- EDF/EDF+ file format support
- GMM clustering for eupnea/sniffing classification
- Status bar with timing and message history

### Changed
- Enhanced modular architecture with separate managers
- Performance optimization for GMM refresh

---

## Release Notes Format

Each release includes:
- **Version number**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Date**: Release date
- **Changes**: Organized by category (Added, Fixed, Changed, Removed, Known Issues)
- **Author**: Ryan Sean Phillips
- **Funding**: NIDA K01DA058543

---

## How to Report Issues

Found a bug or have a feature request? Please open an issue on GitHub:
https://github.com/RyanSeanPhillips/PhysioMetrics/issues

---

**PhysioMetrics** is developed by Ryan Sean Phillips at Seattle Children's Research Institute and funded by NIDA K01DA058543.
