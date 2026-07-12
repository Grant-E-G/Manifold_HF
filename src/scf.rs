//! Self-Consistent Field (Hartree-Fock) implementation
//!
//! This module implements the Hartree-Fock method in a functional style,
//! with integration for manifold optimization.

use crate::basis::BasisSet;
use crate::linalg::{eig, frobenius_norm, matmul, trace, transpose, Matrix, Vector};
use crate::manifold::{ManifoldOptimizationOptions, StiefelManifold};
use crate::molecule::Molecule;
use ndarray::s;
use ndarray::Array1;

/// Hartree-Fock calculator
pub struct HartreeFock {
    /// Molecular system
    pub molecule: Molecule,
    /// Basis set
    pub basis: BasisSet,
    /// Overlap matrix
    pub overlap: Matrix,
    /// Core Hamiltonian matrix
    pub core_hamiltonian: Matrix,
    /// Two-electron repulsion integrals (ij|kl)
    pub eri: Vec<f64>,
}

impl HartreeFock {
    /// Creates a new Hartree-Fock calculator
    pub fn new(molecule: Molecule) -> Result<Self, String> {
        validate_restricted_closed_shell(&molecule)?;
        let basis = BasisSet::sto3g(&molecule)?;
        if molecule.num_occupied() > basis.size() {
            return Err(format!(
                "restricted HF needs {} occupied orbitals, but the basis has only {} functions",
                molecule.num_occupied(),
                basis.size()
            ));
        }
        let overlap = basis.overlap_matrix();
        let core_hamiltonian = basis.core_hamiltonian(&molecule);
        let eri = basis.two_electron_integrals();

        Ok(Self {
            molecule,
            basis,
            overlap,
            core_hamiltonian,
            eri,
        })
    }

    /// Performs standard SCF iteration (Roothaan-Hall equations)
    ///
    /// Returns (energy, coefficient matrix, density matrix, iterations)
    pub fn run_scf(&self, max_iter: usize, tol: f64) -> Result<SCFResult, String> {
        let _n_basis = self.basis.size();
        let n_occ = self.molecule.num_occupied();

        // Initial guess: use core Hamiltonian
        let mut c = self.initial_guess()?;
        let mut converged = false;
        let mut iterations = 0usize;
        let initial_occupied = c.slice(s![.., 0..n_occ]).to_owned();
        let initial_density = self.build_density(&initial_occupied);
        let mut previous_energy = self.total_energy_from_density(&initial_density);

        for iter in 0..max_iter {
            // Build density matrix: P = C_occ * C_occ^T
            let c_occ = c.slice(s![.., 0..n_occ]).to_owned();
            let (density, fock) = self.density_and_fock_from_occupied(&c_occ);

            // Solve generalized eigenvalue problem: FC = SCE
            let (new_c, _orbital_energies) = self.solve_fock(&fock)?;

            // Compare physical states rather than raw MO coefficients. Eigenvectors may
            // flip sign (or rotate inside a degenerate subspace) without changing P.
            let new_c_occ = new_c.slice(s![.., 0..n_occ]).to_owned();
            let new_density = self.build_density(&new_c_occ);
            let new_energy = self.total_energy_from_density(&new_density);

            // Check convergence
            let energy_diff = (new_energy - previous_energy).abs();
            let density_scale = frobenius_norm(&density).max(1.0);
            let density_diff = frobenius_norm(&(&new_density - &density)) / density_scale;
            iterations = iter + 1;

            if energy_diff < tol && density_diff < tol {
                converged = true;
                c = new_c;
                println!("SCF converged in {} iterations", iterations);
                println!("Final energy: {:.10} Hartree", new_energy);
                break;
            }

            if iter % 5 == 0 {
                println!(
                    "Iteration {}: E = {:.10} Hartree, ΔE = {:.2e}, ΔP = {:.2e}",
                    iter, new_energy, energy_diff, density_diff
                );
            }

            c = new_c;
            previous_energy = new_energy;
        }

        if !converged {
            println!("Warning: SCF did not converge in {} iterations", max_iter);
        }

        let c_occ = c.slice(s![.., 0..n_occ]).to_owned();
        let density = self.build_density(&c_occ);
        let energy = self.total_energy_from_density(&density);

        Ok(SCFResult {
            energy,
            coefficients: c,
            density,
            converged,
            iterations,
        })
    }

    /// Performs SCF with manifold optimization
    ///
    /// Instead of diagonalizing Fock matrix, optimizes orbitals directly
    /// on the Stiefel manifold
    pub fn run_scf_manifold(&self, max_iter: usize, tol: f64) -> Result<SCFResult, String> {
        self.run_scf_manifold_with_options(max_iter, tol, &ManifoldOptimizationOptions::default())
    }

    pub fn run_scf_manifold_with_options(
        &self,
        max_iter: usize,
        tol: f64,
        options: &ManifoldOptimizationOptions,
    ) -> Result<SCFResult, String> {
        let n_basis = self.basis.size();
        let n_occ = self.molecule.num_occupied();

        let s_inv_sqrt = symmetric_orthogonalizer(&self.overlap)?;
        let s_sqrt = symmetric_sqrt(&self.overlap)?;

        // Initialize manifold optimizer
        let manifold = StiefelManifold::new(n_basis, n_occ);

        // Initial guess for occupied orbitals
        let c_initial = self.initial_guess()?;
        let c_occ_initial = c_initial.slice(s![.., 0..n_occ]).to_owned();
        let y_initial = matmul(&s_sqrt, &c_occ_initial);

        println!("Running Hartree-Fock with Manifold optimization");
        println!("Basis functions: {}, Occupied orbitals: {}", n_basis, n_occ);

        // Energy function for manifold optimization
        let energy_fn = |y: &Matrix| -> f64 {
            let state = self.manifold_state(y, &s_inv_sqrt);
            self.compute_energy(&state.density, &state.fock)
        };

        // Gradient function for manifold optimization
        let grad_fn = |y: &Matrix| -> Matrix {
            let state = self.manifold_state(y, &s_inv_sqrt);
            // Gradient: 4 * Fock * C_occ
            let grad_c = &state.fock.dot(&state.c_occ) * 4.0;
            matmul(&transpose(&s_inv_sqrt), &grad_c)
        };

        // Optimize on manifold
        let optimization = manifold
            .optimize_cg_with_options(&y_initial, grad_fn, energy_fn, max_iter, tol, options)?;
        let y_opt = optimization.point;
        let state = self.manifold_state(&y_opt, &s_inv_sqrt);
        let energy = self.compute_energy(&state.density, &state.fock);

        // Complete the optimized occupied orbitals with an S-orthonormal virtual subspace.
        let y_full = complete_orthonormal_basis(&y_opt)?;
        debug_assert_eq!(y_full.nrows(), n_basis);
        debug_assert_eq!(y_full.ncols(), n_basis);
        let c_full = matmul(&s_inv_sqrt, &y_full);

        Ok(SCFResult {
            energy,
            coefficients: c_full,
            density: state.density,
            converged: optimization.converged,
            iterations: optimization.iterations,
        })
    }

    /// Initial guess for MO coefficients
    fn initial_guess(&self) -> Result<Matrix, String> {
        // Use core Hamiltonian for initial guess
        self.solve_fock(&self.core_hamiltonian).map(|(c, _)| c)
    }

    /// Builds density matrix from occupied orbitals: P = 2 * C_occ * C_occ^T
    fn build_density(&self, c_occ: &Matrix) -> Matrix {
        matmul(c_occ, &transpose(c_occ)) * 2.0
    }

    fn density_and_fock_from_occupied(&self, c_occ: &Matrix) -> (Matrix, Matrix) {
        let density = self.build_density(c_occ);
        let fock = self.build_fock(&density);
        (density, fock)
    }

    fn manifold_state(&self, y: &Matrix, s_inv_sqrt: &Matrix) -> ManifoldState {
        let c_occ = matmul(s_inv_sqrt, y);
        let (density, fock) = self.density_and_fock_from_occupied(&c_occ);
        ManifoldState {
            c_occ,
            density,
            fock,
        }
    }

    /// Builds Fock matrix: F = H_core + G(P)
    fn build_fock(&self, density: &Matrix) -> Matrix {
        let g = self.build_g_matrix(density);
        &self.core_hamiltonian + &g
    }

    /// Computes total energy from a density matrix.
    pub fn total_energy_from_density(&self, density: &Matrix) -> f64 {
        let fock = self.build_fock(density);
        self.compute_energy(density, &fock)
    }

    /// Computes physical consistency checks for a returned HF state.
    ///
    /// These diagnostics are invariant to orbital signs and occupied-orbital
    /// rotations, making them more useful than inspecting coefficients directly.
    pub fn diagnostics(&self, result: &SCFResult) -> Result<HFDiagnostics, String> {
        let n = self.basis.size();
        if result.density.dim() != (n, n) {
            return Err(format!(
                "density has shape {:?}; expected ({}, {})",
                result.density.dim(),
                n,
                n
            ));
        }
        if result.coefficients.dim() != (n, n) {
            return Err(format!(
                "coefficient matrix has shape {:?}; expected ({}, {})",
                result.coefficients.dim(),
                n,
                n
            ));
        }

        let density_transpose = transpose(&result.density);
        let density_symmetry_residual = frobenius_norm(&(&result.density - &density_transpose));

        let cts = matmul(&transpose(&result.coefficients), &self.overlap);
        let ctsc = matmul(&cts, &result.coefficients);
        let identity = Matrix::eye(n);
        let orbital_orthonormality_residual = frobenius_norm(&(&ctsc - &identity));

        let electron_count = trace(&matmul(&result.density, &self.overlap));
        let expected_electron_count = self.molecule.num_electrons() as f64;
        let electron_count_error = (electron_count - expected_electron_count).abs();

        // For a closed-shell Slater determinant with P = 2 C_occ C_occ^T,
        // non-orthogonal AO idempotency is P S P = 2 P.
        let psp = matmul(&matmul(&result.density, &self.overlap), &result.density);
        let density_idempotency_residual = frobenius_norm(&(&psp - &(&result.density * 2.0)));

        let fock = self.build_fock(&result.density);
        let fps = matmul(&matmul(&fock, &result.density), &self.overlap);
        let spf = matmul(&matmul(&self.overlap, &result.density), &fock);
        let scf_commutator_residual = frobenius_norm(&(&fps - &spf));

        let nuclear_energy = self.molecule.nuclear_repulsion();
        let recomputed_total_energy = self.compute_energy(&result.density, &fock);
        let energy_consistency_error = (result.energy - recomputed_total_energy).abs();
        Ok(HFDiagnostics {
            total_energy: result.energy,
            electronic_energy: result.energy - nuclear_energy,
            nuclear_energy,
            electron_count,
            expected_electron_count,
            electron_count_error,
            energy_consistency_error,
            density_symmetry_residual,
            density_idempotency_residual,
            orbital_orthonormality_residual,
            scf_commutator_residual,
        })
    }

    /// Builds two-electron part of Fock matrix (simplified)
    fn build_g_matrix(&self, density: &Matrix) -> Matrix {
        let n = self.basis.size();
        let mut g = Matrix::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        let dkl = density[[k, l]];
                        let ijkl = self.eri_index(i, j, k, l, n);
                        let ikjl = self.eri_index(i, k, j, l, n);
                        sum += dkl * (self.eri[ijkl] - 0.5 * self.eri[ikjl]);
                    }
                }
                g[[i, j]] = sum;
            }
        }

        g
    }

    /// Solves Fock equation: FC = SCE
    ///
    /// Uses Löwdin symmetric orthogonalization (S^{-1/2}) for FC = SCE.
    fn solve_fock(&self, fock: &Matrix) -> Result<(Matrix, Vector), String> {
        let x = symmetric_orthogonalizer(&self.overlap)?;
        let f_prime = matmul(&transpose(&x), &matmul(fock, &x));
        let (energies, c_prime) = eig(&f_prime)?;

        // Sort by energy
        let mut indices: Vec<usize> = (0..energies.len()).collect();
        indices.sort_by(|&i, &j| energies[i].partial_cmp(&energies[j]).unwrap());

        // Reorder columns
        let n = c_prime.nrows();
        let mut c_sorted = Matrix::zeros((n, n));
        for (new_j, &old_j) in indices.iter().enumerate() {
            for i in 0..n {
                c_sorted[[i, new_j]] = c_prime[[i, old_j]];
            }
        }

        let energies_sorted = indices.iter().map(|&i| energies[i]).collect::<Vec<_>>();
        let energies_vec = Vector::from_vec(energies_sorted);

        let c = matmul(&x, &c_sorted);

        Ok((c, energies_vec))
    }

    /// Computes electronic energy
    fn compute_energy(&self, density: &Matrix, fock: &Matrix) -> f64 {
        // E = 0.5 * Tr[P(H + F)] + E_nuc
        let h_plus_f = &self.core_hamiltonian + fock;
        let electronic = 0.5 * trace(&matmul(density, &h_plus_f));
        let nuclear = self.molecule.nuclear_repulsion();

        electronic + nuclear
    }
}

fn validate_restricted_closed_shell(molecule: &Molecule) -> Result<(), String> {
    if molecule.atoms.is_empty() {
        return Err("restricted HF requires at least one atom".to_string());
    }
    let nuclear_charge: i64 = molecule
        .atoms
        .iter()
        .map(|atom| i64::from(atom.atomic_number))
        .sum();
    let electrons = nuclear_charge - i64::from(molecule.charge);
    if electrons <= 0 {
        return Err(format!(
            "restricted HF requires a positive electron count; got {}",
            electrons
        ));
    }
    if molecule.multiplicity != 1 {
        return Err(format!(
            "restricted closed-shell HF supports only multiplicity 1; got {}",
            molecule.multiplicity
        ));
    }
    if electrons % 2 != 0 {
        return Err(format!(
            "restricted closed-shell HF requires an even electron count; got {}",
            electrons
        ));
    }
    Ok(())
}

impl HartreeFock {
    fn eri_index(&self, i: usize, j: usize, k: usize, l: usize, n: usize) -> usize {
        ((i * n + j) * n + k) * n + l
    }
}

fn symmetric_orthogonalizer(overlap: &Matrix) -> Result<Matrix, String> {
    let (eigenvalues, eigenvectors) = eig(overlap)?;
    let n = eigenvalues.len();
    let mut inv_sqrt = Matrix::zeros((n, n));
    for i in 0..n {
        let value = eigenvalues[i];
        if value <= 1e-12 {
            return Err(format!(
                "overlap matrix eigenvalue too small or negative: {}",
                value
            ));
        }
        inv_sqrt[[i, i]] = 1.0 / value.sqrt();
    }
    let x = matmul(&matmul(&eigenvectors, &inv_sqrt), &transpose(&eigenvectors));
    Ok(x)
}

fn symmetric_sqrt(overlap: &Matrix) -> Result<Matrix, String> {
    let (eigenvalues, eigenvectors) = eig(overlap)?;
    let n = eigenvalues.len();
    let mut sqrt = Matrix::zeros((n, n));
    for i in 0..n {
        let value = eigenvalues[i];
        if value <= 0.0 {
            return Err(format!(
                "overlap matrix eigenvalue too small or negative: {}",
                value
            ));
        }
        sqrt[[i, i]] = value.sqrt();
    }
    let s = matmul(&matmul(&eigenvectors, &sqrt), &transpose(&eigenvectors));
    Ok(s)
}

/// Results from SCF calculation
#[derive(Debug)]
pub struct SCFResult {
    /// Total energy (electronic + nuclear)
    pub energy: f64,
    /// MO coefficient matrix
    pub coefficients: Matrix,
    /// Density matrix
    pub density: Matrix,
    /// Whether SCF converged
    pub converged: bool,
    /// Number of optimization / SCF iterations performed
    pub iterations: usize,
}

/// Basis-independent and AO-matrix consistency checks for an HF result.
#[derive(Debug, Clone)]
pub struct HFDiagnostics {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_energy: f64,
    pub electron_count: f64,
    pub expected_electron_count: f64,
    pub electron_count_error: f64,
    /// Difference between the stored energy and energy recomputed from the density.
    pub energy_consistency_error: f64,
    pub density_symmetry_residual: f64,
    /// Frobenius norm of `P S P - 2 P` for a closed-shell density.
    pub density_idempotency_residual: f64,
    pub orbital_orthonormality_residual: f64,
    /// Frobenius norm of `F P S - S P F`; zero at an SCF stationary point.
    pub scf_commutator_residual: f64,
}

impl HFDiagnostics {
    /// A compact text report suitable for logs and visualization sidecars.
    pub fn human_readable_report(&self, converged: bool, iterations: usize) -> String {
        let status = if converged {
            "PASS (converged)"
        } else {
            "WARN (not converged)"
        };
        format!(
            "HF validation: {status}\n\
             iterations: {iterations}\n\
             total energy: {:.10} Ha\n\
             electronic energy: {:.10} Ha\n\
             nuclear repulsion: {:.10} Ha\n\
             electrons from Tr[P S]: {:.8} (expected {:.0}, error {:.3e})\n\
             energy/density consistency error: {:.3e} Ha\n\
             density symmetry ||P-P^T||_F: {:.3e}\n\
             density idempotency ||P S P-2P||_F: {:.3e}\n\
             MO orthonormality ||C^T S C-I||_F: {:.3e}\n\
             SCF stationarity ||F P S-S P F||_F: {:.3e}",
            self.total_energy,
            self.electronic_energy,
            self.nuclear_energy,
            self.electron_count,
            self.expected_electron_count,
            self.electron_count_error,
            self.energy_consistency_error,
            self.density_symmetry_residual,
            self.density_idempotency_residual,
            self.orbital_orthonormality_residual,
            self.scf_commutator_residual,
        )
    }
}

struct ManifoldState {
    c_occ: Matrix,
    density: Matrix,
    fock: Matrix,
}

fn complete_orthonormal_basis(occupied: &Matrix) -> Result<Matrix, String> {
    let n = occupied.nrows();
    let k = occupied.ncols();
    if k > n {
        return Err("occupied orbital block has more columns than rows".to_string());
    }

    let mut full = Matrix::zeros((n, n));
    for i in 0..n {
        for j in 0..k {
            full[[i, j]] = occupied[[i, j]];
        }
    }

    let mut next_col = k;
    for seed in 0..n {
        if next_col == n {
            break;
        }

        let mut candidate = Array1::zeros(n);
        candidate[seed] = 1.0;

        for j in 0..next_col {
            let basis_vec = full.column(j).to_owned();
            let projection = candidate.dot(&basis_vec);
            candidate = &candidate - &(basis_vec * projection);
        }

        let norm = candidate.dot(&candidate).sqrt();
        if norm <= 1e-10 {
            continue;
        }

        let normalized = candidate / norm;
        for i in 0..n {
            full[[i, next_col]] = normalized[i];
        }
        next_col += 1;
    }

    if next_col != n {
        return Err("failed to complete orbital basis from occupied subspace".to_string());
    }

    Ok(full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Atom;
    use approx::assert_abs_diff_eq;

    // Fixed geometries below are in Angstrom; reference energies were generated
    // independently with PySCF 2.11 RHF/STO-3G.
    const ANGSTROM_TO_BOHR: f64 = 1.889_725_988_6;

    fn atom_angstrom(atomic_number: u32, position: [f64; 3]) -> Atom {
        Atom::new(
            atomic_number,
            [
                position[0] * ANGSTROM_TO_BOHR,
                position[1] * ANGSTROM_TO_BOHR,
                position[2] * ANGSTROM_TO_BOHR,
            ],
        )
    }

    fn assert_physical_invariants(hf: &HartreeFock, result: &SCFResult, tolerance: f64) {
        let diagnostics = hf.diagnostics(result).expect("diagnostics failed");
        assert!(
            diagnostics.electron_count_error < tolerance,
            "wrong electron count:\n{}",
            diagnostics.human_readable_report(result.converged, result.iterations)
        );
        assert!(
            diagnostics.energy_consistency_error < tolerance,
            "stored energy does not match the returned density:\n{}",
            diagnostics.human_readable_report(result.converged, result.iterations)
        );
        assert!(
            diagnostics.density_symmetry_residual < tolerance,
            "density is not symmetric:\n{}",
            diagnostics.human_readable_report(result.converged, result.iterations)
        );
        assert!(
            diagnostics.density_idempotency_residual < tolerance,
            "density is not closed-shell idempotent:\n{}",
            diagnostics.human_readable_report(result.converged, result.iterations)
        );
        assert!(
            diagnostics.orbital_orthonormality_residual < tolerance,
            "orbitals are not S-orthonormal:\n{}",
            diagnostics.human_readable_report(result.converged, result.iterations)
        );
        assert!(
            diagnostics.scf_commutator_residual < tolerance,
            "Roothaan-Hall stationarity residual is too large:\n{}",
            diagnostics.human_readable_report(result.converged, result.iterations)
        );
    }

    fn assert_molecular_reference(
        name: &str,
        atoms: Vec<Atom>,
        charge: i32,
        expected_energy: f64,
        energy_tolerance: f64,
    ) {
        let hf = HartreeFock::new(Molecule::new(atoms, charge, 1))
            .unwrap_or_else(|error| panic!("{name} HF setup failed: {error}"));
        let result = hf
            .run_scf(150, 1e-7)
            .unwrap_or_else(|error| panic!("{name} SCF failed: {error}"));
        let diagnostics = hf.diagnostics(&result).expect("diagnostics failed");

        assert!(
            result.converged,
            "{name} did not converge:\n{}",
            diagnostics.human_readable_report(result.converged, result.iterations)
        );
        assert_abs_diff_eq!(result.energy, expected_energy, epsilon = energy_tolerance,);
        assert_physical_invariants(&hf, &result, 2e-5);
    }

    #[test]
    fn test_hf_h2_scf() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol).unwrap();
        let result = hf.run_scf(100, 1e-6).unwrap();

        assert!(result.converged);
        // Energy should be negative (binding)
        assert!(result.energy < 0.0);
        assert!(result.iterations > 0);
        // Independent STO-3G/RHF reference at R=1.4 Bohr.
        assert_abs_diff_eq!(result.energy, -1.116_714_225_3, epsilon = 2e-8);
        assert_physical_invariants(&hf, &result, 1e-7);
    }

    #[test]
    fn test_hf_h2_manifold() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol).unwrap();
        let result = hf.run_scf_manifold(100, 1e-6).unwrap();

        assert!(result.converged);
        assert!(result.energy < 0.0);
        assert_abs_diff_eq!(result.energy, -1.116_714_225_3, epsilon = 2e-8);
        assert_physical_invariants(&hf, &result, 1e-7);
    }

    #[test]
    fn test_helium_one_basis_function_reference_case() {
        // He/STO-3G is the smallest nontrivial closed-shell atom: one occupied
        // spatial orbital and no geometry or nuclear-repulsion ambiguity.
        let molecule = Molecule::new(vec![Atom::new(2, [0.0, 0.0, 0.0])], 0, 1);
        let hf = HartreeFock::new(molecule).unwrap();
        let result = hf.run_scf(20, 1e-10).unwrap();

        assert!(result.converged);
        assert_eq!(hf.basis.size(), 1);
        assert_abs_diff_eq!(result.energy, -2.807_783_957_5, epsilon = 2e-8);
        assert_physical_invariants(&hf, &result, 1e-9);
    }

    #[test]
    fn test_standard_and_manifold_h2_agree() {
        let hf = HartreeFock::new(Molecule::h2()).unwrap();
        let standard = hf.run_scf(100, 1e-8).unwrap();
        let manifold = hf.run_scf_manifold(100, 1e-8).unwrap();

        assert!(standard.converged && manifold.converged);
        assert_abs_diff_eq!(standard.energy, manifold.energy, epsilon = 1e-8);
        let density_gap = frobenius_norm(&(&standard.density - &manifold.density));
        assert!(density_gap < 1e-7, "density gap = {density_gap:.3e}");
    }

    #[test]
    fn test_water_multicenter_reference_and_invariants() {
        // Exercises s/p functions, five occupied orbitals, multicenter nuclear
        // attraction, and the full Coulomb/exchange contraction.
        let hf = HartreeFock::new(Molecule::h2o()).unwrap();
        let result = hf.run_scf(100, 1e-7).unwrap();

        assert!(result.converged);
        assert_abs_diff_eq!(result.energy, -74.963_074_544_5, epsilon = 3e-7);
        assert_physical_invariants(&hf, &result, 1e-6);
    }

    #[test]
    fn test_heh_plus_ionic_heteronuclear_reference() {
        assert_molecular_reference(
            "HeH+",
            vec![
                atom_angstrom(2, [0.0, 0.0, 0.0]),
                atom_angstrom(1, [0.0, 0.0, 0.774]),
            ],
            1,
            -2.841_779_241_3,
            2e-5,
        );
    }

    #[test]
    fn test_lih_second_row_reference() {
        assert_molecular_reference(
            "LiH",
            vec![
                atom_angstrom(3, [0.0, 0.0, 0.0]),
                atom_angstrom(1, [0.0, 0.0, 1.595]),
            ],
            0,
            -7.862_023_860_1,
            2e-5,
        );
    }

    #[test]
    fn test_carbon_monoxide_multiple_bond_reference() {
        assert_molecular_reference(
            "CO",
            vec![
                atom_angstrom(6, [0.0, 0.0, 0.0]),
                atom_angstrom(8, [0.0, 0.0, 1.128]),
            ],
            0,
            -111.224_558_695_6,
            2e-4,
        );
    }

    #[test]
    fn test_methane_tetrahedral_reference() {
        let h = 0.629_312;
        assert_molecular_reference(
            "CH4",
            vec![
                atom_angstrom(6, [0.0, 0.0, 0.0]),
                atom_angstrom(1, [h, h, h]),
                atom_angstrom(1, [h, -h, -h]),
                atom_angstrom(1, [-h, h, -h]),
                atom_angstrom(1, [-h, -h, h]),
            ],
            0,
            -39.726_700_035_4,
            3e-5,
        );
    }

    #[test]
    fn test_ammonia_pyramidal_reference() {
        assert_molecular_reference(
            "NH3",
            vec![
                atom_angstrom(7, [0.0, 0.0, 0.116_489]),
                atom_angstrom(1, [0.0, 0.939_731, -0.271_808]),
                atom_angstrom(1, [0.813_831, -0.469_865, -0.271_808]),
                atom_angstrom(1, [-0.813_831, -0.469_865, -0.271_808]),
            ],
            0,
            -55.454_560_879_5,
            2e-5,
        );
    }

    #[test]
    fn test_restricted_hf_rejects_open_shell_inputs() {
        let hydrogen_atom = Molecule::new(vec![Atom::new(1, [0.0, 0.0, 0.0])], 0, 2);
        let error = HartreeFock::new(hydrogen_atom)
            .err()
            .expect("open-shell input should be rejected");
        assert!(
            error.contains("multiplicity 1"),
            "unexpected error: {error}"
        );

        let odd_electron_singlet = Molecule::new(vec![Atom::new(1, [0.0, 0.0, 0.0])], 0, 1);
        let error = HartreeFock::new(odd_electron_singlet)
            .err()
            .expect("odd-electron RHF input should be rejected");
        assert!(
            error.contains("even electron count"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_hf_h2_scf_energy_matches_returned_density() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol).unwrap();
        let result = hf.run_scf(100, 1e-6).unwrap();
        let recomputed_energy = hf.total_energy_from_density(&result.density);

        assert_abs_diff_eq!(result.energy, recomputed_energy, epsilon = 1e-10);
    }

    #[test]
    fn test_hf_h2_manifold_coefficients_are_s_orthonormal() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol).unwrap();
        let result = hf.run_scf_manifold(100, 1e-6).unwrap();
        let cts = matmul(&transpose(&result.coefficients), &hf.overlap);
        let ctsc = matmul(&cts, &result.coefficients);

        for i in 0..ctsc.nrows() {
            for j in 0..ctsc.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(ctsc[[i, j]], expected, epsilon = 1e-8);
            }
        }
    }
}
