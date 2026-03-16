//! Manifold optimization on the Stiefel manifold.
//!
//! This module implements optimization on the Stiefel manifold,
//! which is the set of n x p matrices with orthonormal columns.
//! This is crucial for maintaining orthonormality of molecular orbitals.

use crate::linalg::{frobenius_norm, matmul, project_stiefel, transpose, Matrix};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjugateGradientVariant {
    FletcherReeves,
    PolakRibierePlus,
}

#[derive(Debug, Clone)]
pub struct FixedStepLineSearch {
    pub step_size: f64,
}

impl Default for FixedStepLineSearch {
    fn default() -> Self {
        Self { step_size: 1e-2 }
    }
}

#[derive(Debug, Clone)]
pub struct BacktrackingArmijoLineSearch {
    pub initial_step_size: f64,
    pub contraction_factor: f64,
    pub sufficient_decrease: f64,
    pub min_step_size: f64,
    pub max_backtracks: usize,
}

impl Default for BacktrackingArmijoLineSearch {
    fn default() -> Self {
        Self {
            initial_step_size: 1.0,
            contraction_factor: 0.5,
            sufficient_decrease: 1e-4,
            min_step_size: 1e-8,
            max_backtracks: 24,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LineSearchMethod {
    FixedStep(FixedStepLineSearch),
    BacktrackingArmijo(BacktrackingArmijoLineSearch),
}

impl Default for LineSearchMethod {
    fn default() -> Self {
        Self::BacktrackingArmijo(BacktrackingArmijoLineSearch::default())
    }
}

#[derive(Debug, Clone)]
pub struct ManifoldOptimizationOptions {
    pub line_search: LineSearchMethod,
    pub cg_variant: ConjugateGradientVariant,
    pub log_interval: Option<usize>,
}

impl Default for ManifoldOptimizationOptions {
    fn default() -> Self {
        Self {
            line_search: LineSearchMethod::default(),
            cg_variant: ConjugateGradientVariant::PolakRibierePlus,
            log_interval: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ManifoldOptimizationResult {
    pub point: Matrix,
    pub energy: f64,
    pub converged: bool,
    pub iterations: usize,
    pub gradient_norm: f64,
    pub last_step_size: f64,
}

/// Represents the Stiefel manifold St(n, p) = {X in R^(n x p) : X^T X = I_p}
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

    /// Projects a matrix onto the Stiefel manifold.
    ///
    /// Uses Gram-Schmidt to ensure orthonormal columns.
    pub fn project(&self, x: &Matrix) -> Result<Matrix, String> {
        project_stiefel(x)
    }

    /// Projects an ambient matrix onto the tangent space at `x`.
    pub fn project_tangent(&self, x: &Matrix, ambient: &Matrix) -> Matrix {
        let xt = transpose(x);
        let xt_ambient = matmul(&xt, ambient);
        let sym = &xt_ambient + &transpose(&xt_ambient);
        let correction = matmul(x, &(&sym * 0.5));
        ambient - &correction
    }

    /// Computes the Riemannian gradient on the Stiefel manifold.
    pub fn riemannian_gradient(&self, x: &Matrix, euclidean_grad: &Matrix) -> Matrix {
        self.project_tangent(x, euclidean_grad)
    }

    /// Performs a retraction onto the manifold.
    ///
    /// Given point X and tangent vector V, retracts to the manifold via
    /// `R_X(tV) = qf(X + tV)`.
    pub fn retract(&self, x: &Matrix, tangent: &Matrix, step_size: f64) -> Result<Matrix, String> {
        let update = x + &(tangent * step_size);
        self.project(&update)
    }

    /// Uses tangent-space projection as a simple vector transport.
    pub fn transport(&self, x_next: &Matrix, tangent: &Matrix) -> Matrix {
        self.project_tangent(x_next, tangent)
    }

    /// Performs gradient descent on the Stiefel manifold with a fixed step.
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
        let step_size = FixedStepLineSearch::default().step_size;

        for _ in 0..max_iter {
            let euc_grad = grad_fn(&x);
            let riem_grad = self.riemannian_gradient(&x, &euc_grad);
            if frobenius_norm(&riem_grad) < tol {
                break;
            }
            x = self.retract(&x, &(-&riem_grad), step_size)?;
        }

        Ok(x)
    }

    /// Conjugate-gradient optimization on the Stiefel manifold using default options.
    pub fn optimize_cg<F, E>(
        &self,
        initial: &Matrix,
        grad_fn: F,
        energy_fn: E,
        max_iter: usize,
        tol: f64,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> Matrix,
        E: Fn(&Matrix) -> f64,
    {
        self.optimize_cg_with_options(
            initial,
            grad_fn,
            energy_fn,
            max_iter,
            tol,
            &ManifoldOptimizationOptions::default(),
        )
    }

    /// Conjugate-gradient optimization on the Stiefel manifold with explicit options.
    pub fn optimize_cg_with_options<F, E>(
        &self,
        initial: &Matrix,
        grad_fn: F,
        energy_fn: E,
        max_iter: usize,
        tol: f64,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> Matrix,
        E: Fn(&Matrix) -> f64,
    {
        let mut x = self.project(initial)?;
        let mut energy = energy_fn(&x);
        if !energy.is_finite() {
            return Err("initial manifold energy is not finite".to_string());
        }

        let euc_grad = grad_fn(&x);
        let mut riem_grad = self.riemannian_gradient(&x, &euc_grad);
        let mut grad_norm = frobenius_norm(&riem_grad);
        let mut direction = -&riem_grad;
        let mut iterations = 0usize;
        let mut last_step_size = 0.0;

        for iter in 0..max_iter {
            if grad_norm < tol {
                break;
            }

            if frobenius_inner(&riem_grad, &direction) >= 0.0 {
                direction = -&riem_grad;
            }

            let (x_next, energy_next, step_size) = self.line_search(
                &x,
                energy,
                &riem_grad,
                &direction,
                &energy_fn,
                &options.line_search,
            )?;

            let next_euc_grad = grad_fn(&x_next);
            let next_riem_grad = self.riemannian_gradient(&x_next, &next_euc_grad);
            let next_grad_norm = frobenius_norm(&next_riem_grad);

            let transported_grad = self.transport(&x_next, &riem_grad);
            let transported_direction = self.transport(&x_next, &direction);
            let beta = cg_beta(
                options.cg_variant,
                &next_riem_grad,
                &transported_grad,
                &riem_grad,
            );
            let mut next_direction = -&next_riem_grad + &(transported_direction * beta);
            if frobenius_inner(&next_riem_grad, &next_direction) >= 0.0 {
                next_direction = -&next_riem_grad;
            }

            x = x_next;
            energy = energy_next;
            riem_grad = next_riem_grad;
            grad_norm = next_grad_norm;
            direction = next_direction;
            iterations = iter + 1;
            last_step_size = step_size;

            if let Some(interval) = options.log_interval {
                if interval > 0 && iter % interval == 0 {
                    println!(
                        "Iteration {}: energy = {:.10}, gradient norm = {:.2e}, step = {:.2e}",
                        iter, energy, grad_norm, last_step_size
                    );
                }
            }
        }

        Ok(ManifoldOptimizationResult {
            point: x,
            energy,
            converged: grad_norm < tol,
            iterations,
            gradient_norm: grad_norm,
            last_step_size,
        })
    }

    fn line_search<E>(
        &self,
        x: &Matrix,
        energy: f64,
        riem_grad: &Matrix,
        direction: &Matrix,
        energy_fn: &E,
        method: &LineSearchMethod,
    ) -> Result<(Matrix, f64, f64), String>
    where
        E: Fn(&Matrix) -> f64,
    {
        match method {
            LineSearchMethod::FixedStep(config) => {
                let candidate = self.retract(x, direction, config.step_size)?;
                let candidate_energy = energy_fn(&candidate);
                if !candidate_energy.is_finite() {
                    return Err("fixed-step line search produced non-finite energy".to_string());
                }
                Ok((candidate, candidate_energy, config.step_size))
            }
            LineSearchMethod::BacktrackingArmijo(config) => {
                let directional_derivative = frobenius_inner(riem_grad, direction);
                let mut step_size = config.initial_step_size;
                let mut best_trial: Option<(Matrix, f64, f64)> = None;

                for _ in 0..config.max_backtracks {
                    let candidate = self.retract(x, direction, step_size)?;
                    let candidate_energy = energy_fn(&candidate);
                    if candidate_energy.is_finite() {
                        if best_trial
                            .as_ref()
                            .map(|(_, best_energy, _)| candidate_energy < *best_energy)
                            .unwrap_or(true)
                        {
                            best_trial = Some((candidate.clone(), candidate_energy, step_size));
                        }

                        if candidate_energy
                            <= energy
                                + config.sufficient_decrease * step_size * directional_derivative
                        {
                            return Ok((candidate, candidate_energy, step_size));
                        }
                    }

                    step_size *= config.contraction_factor;
                    if step_size < config.min_step_size {
                        break;
                    }
                }

                match best_trial {
                    Some((candidate, candidate_energy, step_size))
                        if candidate_energy <= energy =>
                    {
                        Ok((candidate, candidate_energy, step_size))
                    }
                    _ => Err("backtracking line search failed to reduce the energy".to_string()),
                }
            }
        }
    }
}

fn frobenius_inner(a: &Matrix, b: &Matrix) -> f64 {
    assert_eq!(a.dim(), b.dim(), "Frobenius inner product shape mismatch");
    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn cg_beta(
    variant: ConjugateGradientVariant,
    next_grad: &Matrix,
    transported_prev_grad: &Matrix,
    prev_grad: &Matrix,
) -> f64 {
    let denom = frobenius_inner(prev_grad, prev_grad).max(1e-16);
    let raw_beta = match variant {
        ConjugateGradientVariant::FletcherReeves => frobenius_inner(next_grad, next_grad) / denom,
        ConjugateGradientVariant::PolakRibierePlus => {
            let grad_diff = next_grad - transported_prev_grad;
            frobenius_inner(next_grad, &grad_diff) / denom
        }
    };

    if raw_beta.is_finite() {
        raw_beta.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::trace;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, s, Array2};

    #[test]
    fn test_stiefel_projection() {
        let manifold = StiefelManifold::new(4, 2);
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]).unwrap();

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
        let grad = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

        let riem_grad = manifold.riemannian_gradient(&x, &grad);

        // Riemannian gradient should be orthogonal to the space spanned by X
        let xtg = matmul(&transpose(&x), &riem_grad);

        // X^T * riem_grad should be skew-symmetric
        let sym_part = &xtg + &transpose(&xtg);
        assert!(frobenius_norm(&sym_part) < 1e-10);
    }

    #[test]
    fn test_cg_with_backtracking_armijo_decreases_energy() {
        let manifold = StiefelManifold::new(4, 2);
        let initial =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.2, 0.3, 1.0, 0.2, 0.4, 0.1, 0.8]).unwrap();
        let a = Array2::from_diag(&array![4.0, 3.0, 2.0, 1.0]);
        let projected_initial = manifold.project(&initial).unwrap();
        let initial_energy = rayleigh_energy(&a, &projected_initial);

        let result = manifold
            .optimize_cg_with_options(
                &initial,
                |x| -&(matmul(&a, x) * 2.0),
                |x| rayleigh_energy(&a, x),
                100,
                1e-8,
                &ManifoldOptimizationOptions::default(),
            )
            .unwrap();

        assert!(result.converged);
        assert!(result.energy < initial_energy);
        assert!(result.last_step_size > 0.0);
    }

    #[test]
    fn test_cg_accepts_fixed_step_line_search() {
        let manifold = StiefelManifold::new(4, 2);
        let initial =
            Array2::from_shape_vec((4, 2), vec![0.8, 0.3, 0.1, 1.0, 0.5, 0.4, 0.2, 0.7]).unwrap();
        let a = Array2::from_diag(&array![5.0, 4.0, 2.0, 1.0]);
        let options = ManifoldOptimizationOptions {
            line_search: LineSearchMethod::FixedStep(FixedStepLineSearch { step_size: 5e-2 }),
            cg_variant: ConjugateGradientVariant::FletcherReeves,
            log_interval: None,
        };

        let projected_initial = manifold.project(&initial).unwrap();
        let initial_energy = rayleigh_energy(&a, &projected_initial);
        let result = manifold
            .optimize_cg_with_options(
                &initial,
                |x| -&(matmul(&a, x) * 2.0),
                |x| rayleigh_energy(&a, x),
                200,
                1e-6,
                &options,
            )
            .unwrap();

        assert!(result.energy < initial_energy);
        assert_abs_diff_eq!(result.last_step_size, 5e-2, epsilon = 1e-12);
    }

    fn rayleigh_energy(a: &Matrix, x: &Matrix) -> f64 {
        let ax = matmul(a, x);
        let xtax = matmul(&transpose(x), &ax);
        -trace(&xtax)
    }
}
