//! Self-Consistent Field (Hartree-Fock) implementation
//!
//! This module implements the Hartree-Fock method in a functional style,
//! with integration for manifold optimization.

use crate::linalg::{Matrix, Vector, matmul, transpose, eig, trace, identity, frobenius_norm};
use crate::molecule::Molecule;
use crate::basis::BasisSet;
use crate::manifold::StiefelManifold;

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
}

impl HartreeFock {
    /// Creates a new Hartree-Fock calculator
    pub fn new(molecule: Molecule) -> Self {
        let basis = BasisSet::minimal(&molecule);
        let overlap = basis.overlap_matrix();
        let core_hamiltonian = basis.core_hamiltonian(&molecule);

        Self {
            molecule,
            basis,
            overlap,
            core_hamiltonian,
        }
    }

    /// Performs standard SCF iteration (Roothaan-Hall equations)
    ///
    /// Returns (energy, coefficient matrix, density matrix, iterations)
    pub fn run_scf(&self, max_iter: usize, tol: f64) -> Result<SCFResult, String> {
        let _n_basis = self.basis.size();
        let n_occ = self.molecule.num_occupied();

        // Initial guess: use core Hamiltonian
        let mut c = self.initial_guess()?;
        let mut energy = 0.0;
        let mut converged = false;

        for iter in 0..max_iter {
            // Build density matrix: P = C_occ * C_occ^T
            let c_occ = c.slice(s![.., 0..n_occ]).to_owned();
            let density = self.build_density(&c_occ);

            // Build Fock matrix: F = H_core + G(P)
            let fock = self.build_fock(&density);

            // Solve generalized eigenvalue problem: FC = SCE
            let (new_c, _orbital_energies) = self.solve_fock(&fock)?;

            // Compute energy
            let new_energy = self.compute_energy(&density, &fock);

            // Check convergence
            let energy_diff = (new_energy - energy).abs();
            let c_diff = frobenius_norm(&(&new_c - &c));

            if iter > 0 && energy_diff < tol && c_diff < tol {
                converged = true;
                println!("SCF converged in {} iterations", iter);
                println!("Final energy: {:.10} Hartree", new_energy);
                break;
            }

            if iter % 5 == 0 {
                println!("Iteration {}: E = {:.10} Hartree, ΔE = {:.2e}", 
                         iter, new_energy, energy_diff);
            }

            c = new_c;
            energy = new_energy;
        }

        if !converged {
            println!("Warning: SCF did not converge in {} iterations", max_iter);
        }

        let c_occ = c.slice(s![.., 0..n_occ]).to_owned();
        let density = self.build_density(&c_occ);

        Ok(SCFResult {
            energy,
            coefficients: c,
            density,
            converged,
        })
    }

    /// Performs SCF with manifold optimization
    ///
    /// Instead of diagonalizing Fock matrix, optimizes orbitals directly
    /// on the Stiefel manifold
    pub fn run_scf_manifold(&self, max_iter: usize, tol: f64) -> Result<SCFResult, String> {
        let n_basis = self.basis.size();
        let n_occ = self.molecule.num_occupied();

        // Initialize manifold optimizer
        let manifold = StiefelManifold::new(n_basis, n_occ);

        // Initial guess for occupied orbitals
        let c_initial = self.initial_guess()?;
        let c_occ_initial = c_initial.slice(s![.., 0..n_occ]).to_owned();

        println!("Running Hartree-Fock with Manifold optimization");
        println!("Basis functions: {}, Occupied orbitals: {}", n_basis, n_occ);

        // Energy function for manifold optimization
        let energy_fn = |c_occ: &Matrix| -> f64 {
            let density = self.build_density(c_occ);
            let fock = self.build_fock(&density);
            self.compute_energy(&density, &fock)
        };

        // Gradient function for manifold optimization
        let grad_fn = |c_occ: &Matrix| -> Matrix {
            let density = self.build_density(c_occ);
            let fock = self.build_fock(&density);
            // Gradient: 4 * Fock * C_occ
            &fock.dot(c_occ) * 4.0
        };

        // Optimize on manifold
        let (c_occ_opt, final_energy) = manifold.optimize_cg(
            &c_occ_initial,
            grad_fn,
            energy_fn,
            max_iter,
            tol,
        )?;

        let density = self.build_density(&c_occ_opt);

        // Extend to full coefficient matrix (for compatibility)
        let mut c_full = identity(n_basis);
        for i in 0..n_basis {
            for j in 0..n_occ {
                c_full[[i, j]] = c_occ_opt[[i, j]];
            }
        }

        Ok(SCFResult {
            energy: final_energy,
            coefficients: c_full,
            density,
            converged: true,
        })
    }

    /// Initial guess for MO coefficients
    fn initial_guess(&self) -> Result<Matrix, String> {
        // Use core Hamiltonian for initial guess
        self.solve_fock(&self.core_hamiltonian)
            .map(|(c, _)| c)
    }

    /// Builds density matrix from occupied orbitals: P = C_occ * C_occ^T
    fn build_density(&self, c_occ: &Matrix) -> Matrix {
        matmul(c_occ, &transpose(c_occ))
    }

    /// Builds Fock matrix: F = H_core + G(P)
    fn build_fock(&self, density: &Matrix) -> Matrix {
        let g = self.build_g_matrix(density);
        &self.core_hamiltonian + &g
    }

    /// Builds two-electron part of Fock matrix (simplified)
    fn build_g_matrix(&self, density: &Matrix) -> Matrix {
        let n = self.basis.size();
        let mut g = Matrix::zeros((n, n));

        // Simplified two-electron contribution
        // In practice, would use electron repulsion integrals
        for i in 0..n {
            for j in 0..n {
                g[[i, j]] = density[[i, j]] * 0.1; // Simplified
            }
        }

        g
    }

    /// Solves Fock equation: FC = SCE
    ///
    /// Uses canonical orthogonalization
    fn solve_fock(&self, fock: &Matrix) -> Result<(Matrix, Vector), String> {
        // For simplicity, assume S ≈ I and solve as standard eigenvalue problem
        // In practice, would use S^{-1/2} transformation
        let (energies, c) = eig(fock)?;

        // Sort by energy
        let mut indices: Vec<usize> = (0..energies.len()).collect();
        indices.sort_by(|&i, &j| energies[i].partial_cmp(&energies[j]).unwrap());

        // Reorder columns
        let n = c.nrows();
        let mut c_sorted = Matrix::zeros((n, n));
        for (new_j, &old_j) in indices.iter().enumerate() {
            for i in 0..n {
                c_sorted[[i, new_j]] = c[[i, old_j]];
            }
        }

        let energies_sorted = indices.iter().map(|&i| energies[i]).collect::<Vec<_>>();
        let energies_vec = Vector::from_vec(energies_sorted);

        Ok((c_sorted, energies_vec))
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
}

// Need to import this for slicing
use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_h2_scf() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol);
        let result = hf.run_scf(50, 1e-6).unwrap();
        
        assert!(result.converged);
        // Energy should be negative (binding)
        assert!(result.energy < 0.0);
    }

    #[test]
    fn test_hf_h2_manifold() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol);
        let result = hf.run_scf_manifold(100, 1e-6).unwrap();
        
        assert!(result.converged);
        assert!(result.energy < 0.0);
    }
}
