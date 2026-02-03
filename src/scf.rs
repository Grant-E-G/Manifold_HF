//! Self-Consistent Field (Hartree-Fock) implementation
//!
//! This module implements the Hartree-Fock method in a functional style,
//! with integration for manifold optimization.

use ndarray::s;
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
    /// Two-electron repulsion integrals (ij|kl)
    pub eri: Vec<f64>,
}

impl HartreeFock {
    /// Creates a new Hartree-Fock calculator
    pub fn new(molecule: Molecule) -> Result<Self, String> {
        let basis = BasisSet::sto3g(&molecule)?;
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
            let c_occ = matmul(&s_inv_sqrt, y);
            let density = self.build_density(&c_occ);
            let fock = self.build_fock(&density);
            self.compute_energy(&density, &fock)
        };

        // Gradient function for manifold optimization
        let grad_fn = |y: &Matrix| -> Matrix {
            let c_occ = matmul(&s_inv_sqrt, y);
            let density = self.build_density(&c_occ);
            let fock = self.build_fock(&density);
            // Gradient: 4 * Fock * C_occ
            let grad_c = &fock.dot(&c_occ) * 4.0;
            matmul(&transpose(&s_inv_sqrt), &grad_c)
        };

        // Optimize on manifold
        let (y_opt, final_energy) = manifold.optimize_cg(
            &y_initial,
            grad_fn,
            energy_fn,
            max_iter,
            tol,
        )?;

        let c_occ_opt = matmul(&s_inv_sqrt, &y_opt);
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

    /// Builds density matrix from occupied orbitals: P = 2 * C_occ * C_occ^T
    fn build_density(&self, c_occ: &Matrix) -> Matrix {
        matmul(c_occ, &transpose(c_occ)) * 2.0
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_h2_scf() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol).unwrap();
        let result = hf.run_scf(100, 1e-6).unwrap();
        
        assert!(result.converged);
        // Energy should be negative (binding)
        assert!(result.energy < 0.0);
    }

    #[test]
    fn test_hf_h2_manifold() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol).unwrap();
        let result = hf.run_scf_manifold(100, 1e-6).unwrap();
        
        assert!(result.converged);
        assert!(result.energy < 0.0);
    }
}
