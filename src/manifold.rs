//! Manifold optimization on the Stiefel manifold
//!
//! This module implements optimization on the Stiefel manifold,
//! which is the set of n×p matrices with orthonormal columns.
//! This is crucial for maintaining orthonormality of molecular orbitals.

use crate::linalg::{Matrix, matmul, transpose, project_stiefel, frobenius_norm};

/// Represents the Stiefel manifold St(n, p) = {X ∈ ℝ^{n×p} : X^T X = I_p}
pub struct StiefelManifold {
    /// Dimension n (ambient space)
    pub n: usize,
    /// Dimension p (manifold dimension)
    pub p: usize,
}

impl StiefelManifold {
    /// Creates a new Stiefel manifold
    pub fn new(n: usize, p: usize) -> Self {
        assert!(p <= n, "p must be <= n for Stiefel manifold");
        Self { n, p }
    }

    /// Projects a matrix onto the Stiefel manifold
    ///
    /// Uses QR decomposition to ensure orthonormal columns
    pub fn project(&self, x: &Matrix) -> Result<Matrix, String> {
        project_stiefel(x)
    }

    /// Computes Riemannian gradient on the Stiefel manifold
    ///
    /// Given Euclidean gradient G and point X on manifold,
    /// computes the Riemannian gradient: G - X(X^T G)
    pub fn riemannian_gradient(&self, x: &Matrix, euclidean_grad: &Matrix) -> Matrix {
        let xt = transpose(x);
        let xtg = matmul(&xt, euclidean_grad);
        let sym = &xtg + &transpose(&xtg);
        let correction = matmul(x, &(&sym * 0.5));
        euclidean_grad - &correction
    }

    /// Performs a retraction onto the manifold
    ///
    /// Given point X and tangent vector V, retracts to manifold:
    /// R_X(tV) = qf(X + tV)
    pub fn retract(&self, x: &Matrix, tangent: &Matrix, step_size: f64) -> Result<Matrix, String> {
        let update = x + &(tangent * step_size);
        self.project(&update)
    }

    /// Performs gradient descent on the Stiefel manifold
    ///
    /// # Arguments
    /// * `initial` - Initial point on manifold
    /// * `grad_fn` - Function computing Euclidean gradient at a point
    /// * `max_iter` - Maximum iterations
    /// * `tol` - Convergence tolerance
    ///
    /// # Returns
    /// Optimized point on the manifold
    pub fn optimize<F>(
        &self,
        initial: &Matrix,
        grad_fn: F,
        max_iter: usize,
        tol: f64,
    ) -> Result<Matrix, String>
    where
        F: Fn(&Matrix) -> Matrix,
    {
        let mut x = self.project(initial)?;
        let step_size = 0.01;

        for iter in 0..max_iter {
            // Compute Euclidean gradient
            let euc_grad = grad_fn(&x);

            // Convert to Riemannian gradient
            let riem_grad = self.riemannian_gradient(&x, &euc_grad);

            // Check convergence
            let grad_norm = frobenius_norm(&riem_grad);
            if grad_norm < tol {
                println!("Converged at iteration {} with gradient norm {:.2e}", iter, grad_norm);
                break;
            }

            // Retract onto manifold
            x = self.retract(&x, &(-&riem_grad), step_size)?;

            if iter % 10 == 0 {
                println!("Iteration {}: gradient norm = {:.2e}", iter, grad_norm);
            }
        }

        Ok(x)
    }

    /// Conjugate gradient optimization on Stiefel manifold
    ///
    /// More efficient than steepest descent
    pub fn optimize_cg<F, E>(
        &self,
        initial: &Matrix,
        grad_fn: F,
        energy_fn: E,
        max_iter: usize,
        tol: f64,
    ) -> Result<(Matrix, f64), String>
    where
        F: Fn(&Matrix) -> Matrix,
        E: Fn(&Matrix) -> f64,
    {
        let mut x = self.project(initial)?;
        let mut energy = energy_fn(&x);
        let step_size = 0.01;

        for iter in 0..max_iter {
            let euc_grad = grad_fn(&x);
            let riem_grad = self.riemannian_gradient(&x, &euc_grad);

            let grad_norm = frobenius_norm(&riem_grad);
            if grad_norm < tol {
                println!("CG converged at iteration {} with gradient norm {:.2e}", iter, grad_norm);
                break;
            }

            // Simple line search
            x = self.retract(&x, &(-&riem_grad), step_size)?;
            energy = energy_fn(&x);

            if iter % 10 == 0 {
                println!("Iteration {}: energy = {:.6}, gradient norm = {:.2e}", 
                         iter, energy, grad_norm);
            }
        }

        Ok((x, energy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, s};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_stiefel_projection() {
        let manifold = StiefelManifold::new(4, 2);
        let x = Array2::from_shape_vec((4, 2), vec![
            1.0, 0.5,
            0.0, 1.0,
            0.5, 0.0,
            0.0, 0.5,
        ]).unwrap();

        let projected = manifold.project(&x).unwrap();
        
        // Check orthonormality: X^T X should be identity
        let xtx = matmul(&transpose(&projected), &projected);
        assert_abs_diff_eq!(xtx[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(xtx[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(xtx[[0, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_riemannian_gradient() {
        let manifold = StiefelManifold::new(3, 2);
        let x = Array2::eye(3).slice(s![.., 0..2]).to_owned();
        let grad = Array2::from_shape_vec((3, 2), vec![
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6,
        ]).unwrap();

        let riem_grad = manifold.riemannian_gradient(&x, &grad);
        
        // Riemannian gradient should be orthogonal to the space spanned by X
        let xtg = matmul(&transpose(&x), &riem_grad);
        
        // X^T * riem_grad should be skew-symmetric
        let sym_part = &xtg + &transpose(&xtg);
        assert!(frobenius_norm(&sym_part) < 1e-10);
    }
}
