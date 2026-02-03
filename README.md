# Manifold_HF

Manifold optimization for Hartree-Fock quantum chemistry calculations implemented in Rust with a functional programming style.

## Overview

This library implements the Hartree-Fock (HF) method for quantum chemistry calculations, with a novel approach that uses **manifold optimization** techniques. Specifically, molecular orbitals are optimized on the **Stiefel manifold**, which naturally preserves the orthonormality constraints of quantum mechanical wavefunctions.

### Key Features

- **Pure Rust Implementation**: All core algorithms implemented in Rust without external BLAS/LAPACK dependencies
- **Functional Programming Style**: Immutable data structures, pure functions, and functional composition throughout
- **Manifold Optimization**: Direct optimization on the Stiefel manifold for molecular orbitals
- **Self-Consistent Field (SCF)**: Traditional Roothaan-Hall SCF algorithm
- **Minimal Basis Sets**: STO-3G style Gaussian basis functions with support for s and p orbitals
- **Example Molecules**: Pre-configured H₂, H₂O, and D₂O (heavy water) molecules

## Quick Start

### Build and Run

```bash
# Build the project
cargo build --release

# Run the main binary (demonstrates H2, H2O, and D2O calculations)
cargo run --release

# Run the H2 example
cargo run --release --example simple_h2

# Run the molecular comparison example
cargo run --release --example compare_molecules

# Run tests
cargo test
```

### Using as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
manifold_hf = { git = "https://github.com/Grant-E-G/Manifold_HF" }
```

Example usage:

```rust
use manifold_hf::{Molecule, HartreeFock};

fn main() {
    // Create a hydrogen molecule
    let h2 = Molecule::h2();
    
    // Initialize Hartree-Fock calculator
    let hf = HartreeFock::new(h2).unwrap();
    
    // Run standard SCF
    let result = hf.run_scf(50, 1e-6).unwrap();
    println!("Energy (SCF): {:.10} Hartree", result.energy);
    
    // Run manifold-optimized SCF
    let result = hf.run_scf_manifold(100, 1e-6).unwrap();
    println!("Energy (Manifold): {:.10} Hartree", result.energy);
    
    // Compare water and heavy water
    let h2o = Molecule::h2o();
    let d2o = Molecule::d2o();
    
    println!("H2O and D2O have identical electronic structures");
    println!("(deuterium has same atomic number as hydrogen)");
}
```

## Architecture

The library is organized into functional modules:

### Core Modules

- **`linalg`**: Pure functional linear algebra operations
  - Jacobi eigenvalue decomposition
  - Gram-Schmidt orthogonalization
  - Matrix operations (multiplication, transpose, trace, norm)

- **`molecule`**: Molecular system representation
  - Atomic structures
  - Nuclear repulsion calculations
  - Pre-defined molecules (H₂, H₂O)

- **`basis`**: Gaussian basis set functions
  - Contracted Gaussian basis functions (CGBF)
  - STO-3G style minimal basis sets
  - Overlap and core Hamiltonian matrices

- **`scf`**: Hartree-Fock SCF implementation
  - Roothaan-Hall equations solver
  - Density matrix construction
  - Fock matrix building
  - Energy calculations

- **`manifold`**: Stiefel manifold optimization
  - Riemannian gradient computation
  - Retraction operations
  - Conjugate gradient optimization
  - Manifold projections

## Mathematical Background

### Hartree-Fock Method

The Hartree-Fock method approximates the many-electron wavefunction as a single Slater determinant and minimizes the electronic energy:

```
E[C] = Tr[P·H_core] + 0.5·Tr[P·G[P]] + E_nuc
```

where:
- `P` is the density matrix
- `H_core` contains kinetic and nuclear attraction integrals
- `G[P]` contains electron-electron repulsion integrals
- `E_nuc` is the nuclear repulsion energy

### Stiefel Manifold Optimization

The molecular orbital coefficients C are constrained to the Stiefel manifold St(n,p), the set of n×p matrices with orthonormal columns:

```
St(n,p) = {C ∈ ℝ^{n×p} : C^T·C = I_p}
```

This formulation:
1. Automatically preserves orthonormality
2. Enables use of Riemannian optimization methods
3. Avoids explicit orthogonalization at each iteration
4. Provides geometric insights into the energy landscape

The Riemannian gradient is computed as:

```
grad_R = G - X·(X^T·G + G^T·X)/2
```

where G is the Euclidean gradient and X is the current point on the manifold.

## Functional Programming Principles

This codebase follows functional programming principles:

1. **Immutability**: All data structures are immutable; operations return new instances
2. **Pure Functions**: Functions have no side effects and return deterministic results
3. **Composition**: Complex operations built from simple, composable functions
4. **Type Safety**: Strong typing with Rust's type system
5. **Declarative Style**: Focus on *what* to compute rather than *how*

Example:

```rust
// Functional composition for SCF iteration
let density = build_density(&c_occ);
let fock = build_fock(&density);
let energy = compute_energy(&density, &fock);
```

## Implementation Details

### Linear Algebra

All linear algebra operations are implemented in pure Rust:

- **Eigenvalue decomposition**: Jacobi algorithm for symmetric matrices
- **Orthogonalization**: Gram-Schmidt process
- **Matrix operations**: Using ndarray for efficient array operations

### Basis Sets

Currently loads full STO-3G definitions from the Basis Set Exchange (s/p shells as defined).

**Basis data source**

STO-3G parameters are sourced from the Basis Set Exchange (BSE). The raw JSON is stored in
`data/sto-3g.json`, with citations in `data/sto-3g.references.txt` and provenance in
`data/sto-3g.SOURCE.txt`. The data is imported in code via `src/basis_data.rs`.

**Status note**: One‑ and two‑electron integrals are evaluated with analytic Cartesian
Gaussian formulas and transformed to real spherical functions. The d‑shell transform is
implemented; higher‑l transforms can be added as needed.

### Integral Evaluation

One‑ and two‑electron integrals are evaluated analytically for Cartesian Gaussian functions and
transformed to real spherical functions. The Boys function is computed via an erf‑based
approximation suitable for small systems. For production workloads, a dedicated integral engine
and higher‑l spherical transforms would be required.

## Benchmarks

Benchmark targets live in `data/benchmarks.targets.json`. Reference values are generated with
PySCF HF/STO‑3G on RDKit ETKDGv3 + UFF geometries and stored in `data/benchmarks.json`
(ignored by git). See `data/benchmarks.SOURCE.md` for generation details.
The benchmark file stores total energy (electronic + nuclear), electronic energy, and
bond length/angle measurements.

Generate benchmarks:

```bash
conda env create -f environment.yml
conda activate manifold-hf-bench
python scripts/generate_benchmarks.py
```

Run benchmark comparisons (quick set by default):

```bash
MANIFOLD_HF_BENCHMARKS=quick cargo test --test benchmarks
```

Quick runs the `small` tagged molecules (H2, H2O, D2O). Run the full set
(includes alcohols, benzene, polymers, peptide):

```bash
MANIFOLD_HF_BENCHMARKS=full cargo test --test benchmarks
```

If `data/benchmarks.json` is missing, the benchmark test will skip.
Energy comparisons currently use a 6% relative tolerance while the integral
engine is still being refined for heavier systems.

## Testing

Run the full test suite:

```bash
cargo test
```

Tests cover:
- Linear algebra operations
- Molecular properties
- Basis set construction
- Manifold operations
- SCF convergence

## Performance

The current implementation prioritizes clarity and functional style over raw performance. For production-level calculations:

- Consider using optimized BLAS/LAPACK libraries
- Implement more efficient integral evaluation
- Add parallelization for expensive operations
- Use larger, more accurate basis sets

## Future Enhancements

Potential areas for expansion:

- [ ] Post-Hartree-Fock methods (MP2, CCSD)
- [ ] Analytical gradient calculations
- [ ] Geometry optimization
- [ ] Excited state calculations (CIS, TD-DFT)
- [ ] Larger basis sets (6-31G, cc-pVDZ)
- [ ] Density Functional Theory (DFT)
- [ ] Parallelization with Rayon
- [ ] GPU acceleration

## References

### Quantum Chemistry

1. Szabo, A. & Ostlund, N. S. "Modern Quantum Chemistry" (1996)
2. Helgaker, T., Jorgensen, P., & Olsen, J. "Molecular Electronic-Structure Theory" (2000)

### Manifold Optimization

1. Absil, P.-A., Mahony, R., & Sepulchre, R. "Optimization Algorithms on Matrix Manifolds" (2008)
2. Edelman, A., Arias, T. A., & Smith, S. T. "The Geometry of Algorithms with Orthogonality Constraints" (1998)

### Computational Methods

1. Thom H. Dunning Jr. "Gaussian basis sets for use in correlated molecular calculations" (1989)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Areas of particular interest:

- Additional basis sets
- More sophisticated integral evaluation
- Additional optimization algorithms
- Performance improvements
- Documentation enhancements

Please ensure all contributions maintain the functional programming style and include appropriate tests.
