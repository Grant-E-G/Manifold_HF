# Benchmark Data Source

This project stores **generated** benchmark data in `data/benchmarks.json` (not committed).

The reference energies and geometry metrics are produced by:

- **Method**: Hartree-Fock (restricted, closed-shell)
- **Basis**: STO-3G
- **Program**: PySCF
- **Geometry**: RDKit ETKDGv3 embedding + UFF optimization

## How to Generate

Install dependencies (example):

```bash
python -m pip install pyscf rdkit-pypi numpy
```

Then run:

```bash
python scripts/generate_benchmarks.py
```

This writes `data/benchmarks.json` using the molecule list in `data/benchmarks.targets.json`.

## Notes

- D2O is treated as H2O electronically (deuterium has atomic number 1).
- Geometry metrics (bond lengths/angles) are computed from the generated geometry.
- The benchmark file is intentionally excluded from version control.
- The generator currently supports H/C/N/O; extend `ELEMENT_SYMBOLS` for other elements.
- `energy_total` includes nuclear repulsion; `energy_electronic` is `energy_total - energy_nuclear`.
