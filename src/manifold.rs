//! Optimization algorithms for matrices with orthonormal columns.
//!
//! The optimizers share one value-and-gradient oracle so expensive objectives can
//! build their state once per trial point. This matters for Hartree--Fock, where
//! both the energy and gradient use the same density and Fock matrices.
//!
//! # Geometry
//!
//! A point `X` on the Stiefel manifold has shape `n × p` and satisfies
//! `X.t().dot(X) = I`. An ambient direction is projected onto the tangent space,
//! an optimizer chooses a tangent step, and a retraction restores exact
//! orthonormality. Momentum and quasi-Newton history are projection-transported
//! because consecutive iterates have different tangent spaces.
//!
//! # Objective API
//!
//! [`StiefelManifold::optimize_with_options`] accepts one callback that returns a
//! [`ManifoldEvaluation`]. Keeping energy and gradient together avoids repeating
//! shared objective work. In Hartree--Fock both values reuse the same density and
//! Fock matrices.

use crate::linalg::{frobenius_norm, matmul, project_stiefel, transpose, Matrix};
use std::collections::VecDeque;
use std::fmt;

/// Formula used for the nonlinear conjugate-gradient coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjugateGradientVariant {
    /// `beta = ||g_next||² / ||g||²`.
    FletcherReeves,
    /// Nonnegative Polak--Ribiere: `beta = max(0, <g_next, g_next-T(g)>/||g||²)`.
    PolakRibierePlus,
}

/// A line search that always accepts the configured step size.
#[derive(Debug, Clone)]
pub struct FixedStepLineSearch {
    /// Positive tangent step multiplier.
    pub step_size: f64,
}

impl Default for FixedStepLineSearch {
    fn default() -> Self {
        Self { step_size: 1e-2 }
    }
}

/// Parameters for Armijo sufficient-decrease backtracking.
#[derive(Debug, Clone)]
pub struct BacktrackingArmijoLineSearch {
    /// Step size tried before any contraction.
    pub initial_step_size: f64,
    /// Factor in `(0, 1)` applied after a rejected trial.
    pub contraction_factor: f64,
    /// Armijo sufficient-decrease constant in `(0, 1)`.
    pub sufficient_decrease: f64,
    /// Smallest permitted trial step.
    pub min_step_size: f64,
    /// Maximum number of objective trials in one search.
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

/// Policy for selecting a tangent step size.
#[derive(Debug, Clone)]
pub enum LineSearchMethod {
    /// Evaluate exactly one trial at a fixed step size.
    FixedStep(FixedStepLineSearch),
    /// Contract the step until Armijo sufficient decrease is reached.
    BacktrackingArmijo(BacktrackingArmijoLineSearch),
}

impl Default for LineSearchMethod {
    fn default() -> Self {
        Self::BacktrackingArmijo(BacktrackingArmijoLineSearch::default())
    }
}

/// Optimizers available through [`StiefelManifold::optimize_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldOptimizer {
    /// Nonlinear conjugate gradient with transported history and restart safeguards.
    RiemannianConjugateGradient,
    /// Steepest descent in the tangent space (RSGD in the stochastic setting).
    RiemannianGradientDescent,
    /// Transported first moment with one scalar second moment per matrix block.
    RiemannianAdam,
    /// The same block-adaptive method with a nondecreasing second moment.
    RiemannianAmsGrad,
    /// Momentum SGD using a low-rank Cayley feasibility-preserving update.
    CayleySgd,
    /// Limited-memory BFGS with projection-transported extrinsic curvature pairs.
    RiemannianLbfgs,
}

impl ManifoldOptimizer {
    /// Every optimizer in a stable display order, useful for comparisons.
    pub const ALL: [Self; 6] = [
        Self::RiemannianGradientDescent,
        Self::RiemannianConjugateGradient,
        Self::RiemannianAdam,
        Self::RiemannianAmsGrad,
        Self::CayleySgd,
        Self::RiemannianLbfgs,
    ];
}

impl fmt::Display for ManifoldOptimizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::RiemannianConjugateGradient => "Riemannian CG",
            Self::RiemannianGradientDescent => "Riemannian gradient descent",
            Self::RiemannianAdam => "Riemannian Adam",
            Self::RiemannianAmsGrad => "Riemannian AMSGrad",
            Self::CayleySgd => "Cayley momentum SGD",
            Self::RiemannianLbfgs => "Riemannian L-BFGS",
        };
        formatter.write_str(name)
    }
}

/// Hyperparameters for the block-adaptive Adam and AMSGrad variants.
///
/// The second moment is a scalar Frobenius-norm estimate for the complete matrix
/// block, not an elementwise accumulator. This avoids assigning coordinate-wise
/// learning rates to a representation whose tangent basis changes over time.
#[derive(Debug, Clone)]
pub struct RiemannianAdamOptions {
    /// Positive fixed retraction step.
    pub step_size: f64,
    /// First-moment decay in `[0, 1)`.
    pub beta1: f64,
    /// Scalar second-moment decay in `[0, 1)`.
    pub beta2: f64,
    /// Positive denominator regularizer.
    pub epsilon: f64,
}

impl Default for RiemannianAdamOptions {
    fn default() -> Self {
        Self {
            step_size: 1e-2,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Hyperparameters for momentum SGD with a Cayley update.
#[derive(Debug, Clone)]
pub struct CayleySgdOptions {
    /// Positive fixed Cayley step.
    pub step_size: f64,
    /// Transported momentum coefficient in `[0, 1)`.
    pub momentum: f64,
}

impl Default for CayleySgdOptions {
    fn default() -> Self {
        Self {
            step_size: 1e-2,
            momentum: 0.9,
        }
    }
}

/// Hyperparameters for transported-memory Riemannian L-BFGS.
#[derive(Debug, Clone)]
pub struct RiemannianLbfgsOptions {
    /// Maximum number of `(s, y)` curvature pairs retained.
    pub memory_size: usize,
    /// Relative positivity threshold for accepting a curvature pair.
    ///
    /// A pair is accepted only when
    /// `<s,y> > curvature_epsilon * ||s|| * ||y||`.
    pub curvature_epsilon: f64,
}

impl Default for RiemannianLbfgsOptions {
    fn default() -> Self {
        Self {
            memory_size: 10,
            curvature_epsilon: 1e-10,
        }
    }
}

/// Complete configuration for manifold optimization.
///
/// Only the option block used by [`Self::optimizer`] is validated. For example,
/// an unused Adam configuration cannot prevent an L-BFGS run.
#[derive(Debug, Clone)]
pub struct ManifoldOptimizationOptions {
    /// Algorithm to execute.
    pub optimizer: ManifoldOptimizer,
    /// Step policy for gradient descent, conjugate gradient, and L-BFGS.
    pub line_search: LineSearchMethod,
    /// Conjugate-gradient coefficient formula.
    pub cg_variant: ConjugateGradientVariant,
    /// Adam/AMSGrad-specific settings.
    pub adam: RiemannianAdamOptions,
    /// Cayley-momentum-specific settings.
    pub cayley_sgd: CayleySgdOptions,
    /// L-BFGS-specific settings.
    pub lbfgs: RiemannianLbfgsOptions,
    /// Print progress every `interval` iterations; `None` disables logging.
    pub log_interval: Option<usize>,
}

impl Default for ManifoldOptimizationOptions {
    fn default() -> Self {
        Self {
            optimizer: ManifoldOptimizer::RiemannianConjugateGradient,
            line_search: LineSearchMethod::default(),
            cg_variant: ConjugateGradientVariant::PolakRibierePlus,
            adam: RiemannianAdamOptions::default(),
            cayley_sgd: CayleySgdOptions::default(),
            lbfgs: RiemannianLbfgsOptions::default(),
            log_interval: None,
        }
    }
}

/// Objective information evaluated at one manifold point.
///
/// The gradient is the ambient Euclidean gradient. The optimizer validates its
/// shape and projects it onto the tangent space exactly once.
#[derive(Debug, Clone)]
pub struct ManifoldEvaluation {
    /// Scalar objective value.
    pub energy: f64,
    /// Ambient gradient with the same shape as the manifold point.
    pub euclidean_gradient: Matrix,
}

/// Final point and diagnostics returned by a manifold optimizer.
#[derive(Debug, Clone)]
pub struct ManifoldOptimizationResult {
    /// Final orthonormal matrix.
    pub point: Matrix,
    /// Objective value at [`Self::point`].
    pub energy: f64,
    /// Whether the final Riemannian gradient norm is below the requested tolerance.
    pub converged: bool,
    /// Number of accepted optimizer steps.
    pub iterations: usize,
    /// Frobenius norm of the final Riemannian gradient.
    pub gradient_norm: f64,
    /// Last accepted step size, or zero if no step was taken.
    pub last_step_size: f64,
    /// Number of value-and-gradient oracle calls, including the initial point.
    pub objective_evaluations: usize,
}

#[derive(Debug, Clone)]
struct EvaluatedPoint {
    energy: f64,
    gradient: Matrix,
}

/// Shared stopping criteria passed to each internal algorithm.
#[derive(Debug, Clone, Copy)]
struct StoppingCriteria {
    max_iterations: usize,
    tolerance: f64,
}

/// A trial returned by a line search, including all objective calls it consumed.
struct TrialPoint {
    point: Matrix,
    evaluated: EvaluatedPoint,
    step_size: f64,
    objective_evaluations: usize,
}

/// Bookkeeping common to every optimizer.
///
/// Keeping point/evaluation/counters together prevents the algorithm loops from
/// independently updating six related variables after each accepted step.
struct OptimizationRun {
    point: Matrix,
    evaluated: EvaluatedPoint,
    iterations: usize,
    last_step_size: f64,
    objective_evaluations: usize,
}

impl OptimizationRun {
    fn new(point: Matrix, evaluated: EvaluatedPoint) -> Self {
        Self {
            point,
            evaluated,
            iterations: 0,
            last_step_size: 0.0,
            objective_evaluations: 1,
        }
    }

    fn converged(&self, tolerance: f64) -> bool {
        frobenius_norm(&self.evaluated.gradient) < tolerance
    }

    fn accept(&mut self, trial: TrialPoint) {
        self.point = trial.point;
        self.evaluated = trial.evaluated;
        self.iterations += 1;
        self.last_step_size = trial.step_size;
        self.objective_evaluations += trial.objective_evaluations;
    }

    fn finish(self, tolerance: f64) -> ManifoldOptimizationResult {
        let gradient_norm = frobenius_norm(&self.evaluated.gradient);
        ManifoldOptimizationResult {
            point: self.point,
            energy: self.evaluated.energy,
            converged: gradient_norm < tolerance,
            iterations: self.iterations,
            gradient_norm,
            last_step_size: self.last_step_size,
            objective_evaluations: self.objective_evaluations,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AdamVariant {
    Adam,
    AmsGrad,
}

#[derive(Debug, Clone)]
struct LbfgsPair {
    s: Matrix,
    y: Matrix,
    rho: f64,
}

/// Bounded L-BFGS curvature history stored in the current tangent space.
struct LbfgsHistory {
    pairs: VecDeque<LbfgsPair>,
    capacity: usize,
    curvature_epsilon: f64,
}

impl LbfgsHistory {
    fn new(options: &RiemannianLbfgsOptions) -> Self {
        Self {
            pairs: VecDeque::with_capacity(options.memory_size),
            capacity: options.memory_size,
            curvature_epsilon: options.curvature_epsilon,
        }
    }

    fn clear(&mut self) {
        self.pairs.clear();
    }

    fn direction(&self, gradient: &Matrix) -> Matrix {
        lbfgs_direction(gradient, &self.pairs)
    }

    /// Re-express all pairs at `next_point` and discard lost positive curvature.
    fn transport_to(&mut self, manifold: &StiefelManifold, next_point: &Matrix) {
        for pair in &mut self.pairs {
            pair.s = manifold.transport(next_point, &pair.s);
            pair.y = manifold.transport(next_point, &pair.y);
            let curvature = frobenius_inner(&pair.s, &pair.y);
            pair.rho = if curvature > 0.0 {
                1.0 / curvature
            } else {
                0.0
            };
        }
        self.pairs
            .retain(|pair| pair.rho.is_finite() && pair.rho > 0.0);
    }

    /// Add a pair only when it satisfies the scale-aware curvature condition.
    fn push_if_valid(&mut self, s: Matrix, y: Matrix) {
        let curvature = frobenius_inner(&s, &y);
        let scale = frobenius_norm(&s).max(1e-30) * frobenius_norm(&y).max(1e-30);
        if !curvature.is_finite() || curvature <= self.curvature_epsilon * scale {
            return;
        }
        if self.pairs.len() == self.capacity {
            self.pairs.pop_front();
        }
        self.pairs.push_back(LbfgsPair {
            s,
            y,
            rho: 1.0 / curvature,
        });
    }
}

/// Matrices with `p` orthonormal columns in an `n`-dimensional ambient space.
///
/// ```text
/// St(n, p) = { X in R^(n×p) : X^T X = I_p }.
/// ```
///
/// Hartree--Fock energy is invariant to occupied-frame rotations, so its
/// physical search space is a Grassmann quotient. This type intentionally keeps
/// the explicit Stiefel representative used by the numerical implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StiefelManifold {
    /// Number of rows in a manifold point.
    pub n: usize,
    /// Number of orthonormal columns in a manifold point.
    pub p: usize,
}

impl StiefelManifold {
    /// Construct `St(n, p)`.
    ///
    /// # Panics
    ///
    /// Panics when `p > n`, because such a real matrix cannot have orthonormal
    /// columns.
    pub fn new(n: usize, p: usize) -> Self {
        assert!(p <= n, "p must be <= n for Stiefel manifold");
        Self { n, p }
    }

    /// Number of scalars in the ambient `n × p` matrix representation.
    #[must_use]
    pub fn ambient_dimension(&self) -> usize {
        self.n * self.p
    }

    /// Intrinsic dimension `np - p(p+1)/2` of `St(n, p)`.
    #[must_use]
    pub fn intrinsic_dimension(&self) -> usize {
        self.n * self.p - self.p * (self.p + 1) / 2
    }

    /// Dimension `p(n-p)` of the corresponding Grassmann quotient.
    #[must_use]
    pub fn grassmann_dimension(&self) -> usize {
        self.p * (self.n - self.p)
    }

    /// Orthonormalize the columns of an ambient matrix.
    ///
    /// The current linear-algebra backend uses modified Gram--Schmidt. An error
    /// is returned for the wrong shape or numerically dependent columns.
    pub fn project(&self, x: &Matrix) -> Result<Matrix, String> {
        self.validate_shape(x)?;
        project_stiefel(x)
    }

    /// Embedded-Euclidean projection onto the Stiefel tangent space at `x`.
    ///
    /// Computes `Z - X sym(X^T Z)`. The returned matrix `Xi` satisfies
    /// `X^T Xi + Xi^T X = 0` up to floating-point error.
    ///
    /// # Panics
    ///
    /// Panics if `x` and `ambient` do not have compatible matrix shapes.
    pub fn project_tangent(&self, x: &Matrix, ambient: &Matrix) -> Matrix {
        let xt_ambient = matmul(&transpose(x), ambient);
        let sym = &xt_ambient + &transpose(&xt_ambient);
        ambient - &matmul(x, &(&sym * 0.5))
    }

    /// Convert an ambient Euclidean gradient into a tangent Riemannian gradient.
    ///
    /// This implementation uses the embedded Euclidean metric, so this is the
    /// same operation as [`Self::project_tangent`].
    pub fn riemannian_gradient(&self, x: &Matrix, euclidean_grad: &Matrix) -> Matrix {
        self.project_tangent(x, euclidean_grad)
    }

    /// QR-like retraction `R_X(step * Xi) = qf(X + step * Xi)`.
    ///
    /// The orthonormal factor is computed with modified Gram--Schmidt. This is a
    /// simple reference implementation; Householder QR is preferable for large
    /// or poorly conditioned matrices.
    pub fn retract(&self, x: &Matrix, tangent: &Matrix, step_size: f64) -> Result<Matrix, String> {
        self.project(&(x + &(tangent * step_size)))
    }

    /// Projection-transport a tangent vector into `T_x_next St(n,p)`.
    ///
    /// Projection transport is inexpensive and tangent-valued, but it is not in
    /// general exact parallel transport or norm preserving.
    pub fn transport(&self, x_next: &Matrix, tangent: &Matrix) -> Matrix {
        self.project_tangent(x_next, tangent)
    }

    /// Minimize using fixed-step Riemannian gradient descent and return the point.
    ///
    /// This convenience API has no energy callback and therefore returns fewer
    /// diagnostics. New code should generally use [`Self::optimize_with_options`].
    ///
    /// # Errors
    ///
    /// Returns an error if the tolerance is not finite and positive, the initial
    /// point or a returned gradient has the wrong shape, a gradient is nonfinite,
    /// or orthonormalization fails.
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
        if !tol.is_finite() || tol <= 0.0 {
            return Err("tolerance must be finite and positive".to_string());
        }
        let mut x = self.project(initial)?;
        let step_size = FixedStepLineSearch::default().step_size;
        for _ in 0..max_iter {
            let ambient_gradient = grad_fn(&x);
            self.validate_shape(&ambient_gradient)?;
            if !matrix_is_finite(&ambient_gradient) {
                return Err("objective returned a non-finite gradient".to_string());
            }
            let grad = self.riemannian_gradient(&x, &ambient_gradient);
            if frobenius_norm(&grad) < tol {
                break;
            }
            x = self.retract(&x, &(-&grad), step_size)?;
        }
        Ok(x)
    }

    /// Minimize using a combined value-and-gradient oracle.
    ///
    /// The initial matrix is orthonormalized before the first objective call.
    /// The returned gradient from `oracle` must be the ambient Euclidean gradient;
    /// it is projected internally. Combining value and gradient is important for
    /// objectives such as Hartree--Fock where both reuse expensive intermediate
    /// state.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid active hyperparameters, shape mismatches,
    /// nonfinite objective values, failed retractions/linear solves, or a line
    /// search that cannot find a nonincreasing point.
    pub fn optimize_with_options<F>(
        &self,
        initial: &Matrix,
        oracle: F,
        max_iter: usize,
        tol: f64,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        self.validate_options(tol, options)?;
        let x = self.project(initial)?;
        let evaluated = self.evaluate(&x, &oracle)?;
        let run = OptimizationRun::new(x, evaluated);
        let stopping = StoppingCriteria {
            max_iterations: max_iter,
            tolerance: tol,
        };
        match options.optimizer {
            ManifoldOptimizer::RiemannianConjugateGradient => {
                self.optimize_rcg(run, &oracle, stopping, options)
            }
            ManifoldOptimizer::RiemannianGradientDescent => {
                self.optimize_rsgd(run, &oracle, stopping, options)
            }
            ManifoldOptimizer::RiemannianAdam => {
                self.optimize_adam(run, &oracle, stopping, options, AdamVariant::Adam)
            }
            ManifoldOptimizer::RiemannianAmsGrad => {
                self.optimize_adam(run, &oracle, stopping, options, AdamVariant::AmsGrad)
            }
            ManifoldOptimizer::CayleySgd => {
                self.optimize_cayley_sgd(run, &oracle, stopping, options)
            }
            ManifoldOptimizer::RiemannianLbfgs => {
                self.optimize_lbfgs(run, &oracle, stopping, options)
            }
        }
    }

    /// Minimize with conjugate gradient using default options.
    ///
    /// This compatibility API evaluates energy and gradient through separate
    /// callbacks. Prefer [`Self::optimize_with_options`] when they share work.
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

    /// Minimize with conjugate gradient and explicit options.
    ///
    /// The `optimizer` field is ignored and forced to
    /// [`ManifoldOptimizer::RiemannianConjugateGradient`]. New expensive
    /// objectives should use [`Self::optimize_with_options`] to avoid duplicate
    /// energy/gradient work.
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
        let mut cg_options = options.clone();
        cg_options.optimizer = ManifoldOptimizer::RiemannianConjugateGradient;
        self.optimize_with_options(
            initial,
            |x| ManifoldEvaluation {
                energy: energy_fn(x),
                euclidean_gradient: grad_fn(x),
            },
            max_iter,
            tol,
            &cg_options,
        )
    }

    fn optimize_rsgd<F>(
        &self,
        mut run: OptimizationRun,
        oracle: &F,
        stopping: StoppingCriteria,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        // Steepest descent is the baseline: negate the tangent gradient, choose
        // a step, then retract back to the manifold.
        for iteration in 0..stopping.max_iterations {
            if run.converged(stopping.tolerance) {
                break;
            }
            let direction = -&run.evaluated.gradient;
            let trial = self.line_search(
                &run.point,
                &run.evaluated,
                &direction,
                oracle,
                &options.line_search,
            )?;
            run.accept(trial);
            log_progress(options, iteration, &run.evaluated, run.last_step_size);
        }
        Ok(run.finish(stopping.tolerance))
    }

    fn optimize_rcg<F>(
        &self,
        mut run: OptimizationRun,
        oracle: &F,
        stopping: StoppingCriteria,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let mut direction = -&run.evaluated.gradient;

        for iteration in 0..stopping.max_iterations {
            if run.converged(stopping.tolerance) {
                break;
            }
            // Finite precision or an aggressive beta can destroy descent. A
            // restart makes the method fall back safely to steepest descent.
            if frobenius_inner(&run.evaluated.gradient, &direction) >= 0.0 {
                direction = -&run.evaluated.gradient;
            }
            let trial = self.line_search(
                &run.point,
                &run.evaluated,
                &direction,
                oracle,
                &options.line_search,
            )?;
            let transported_grad = self.transport(&trial.point, &run.evaluated.gradient);
            let transported_direction = self.transport(&trial.point, &direction);
            let beta = cg_beta(
                options.cg_variant,
                &trial.evaluated.gradient,
                &transported_grad,
                &run.evaluated.gradient,
            );
            let mut next_direction = -&trial.evaluated.gradient + &(transported_direction * beta);
            if frobenius_inner(&trial.evaluated.gradient, &next_direction) >= 0.0 {
                next_direction = -&trial.evaluated.gradient;
            }

            run.accept(trial);
            direction = next_direction;
            log_progress(options, iteration, &run.evaluated, run.last_step_size);
        }
        Ok(run.finish(stopping.tolerance))
    }

    fn optimize_adam<F>(
        &self,
        mut run: OptimizationRun,
        oracle: &F,
        stopping: StoppingCriteria,
        options: &ManifoldOptimizationOptions,
        variant: AdamVariant,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let config = &options.adam;
        let mut momentum = Matrix::zeros((self.n, self.p));
        let mut second_moment = 0.0_f64;
        let mut max_second_moment = 0.0_f64;

        for iteration in 0..stopping.max_iterations {
            if run.converged(stopping.tolerance) {
                break;
            }
            let t = (iteration + 1) as i32;
            momentum = &momentum * config.beta1 + &run.evaluated.gradient * (1.0 - config.beta1);
            // A scalar second moment is invariant to changes of tangent basis.
            // It adapts the whole matrix block instead of individual entries.
            let norm_sq = frobenius_inner(&run.evaluated.gradient, &run.evaluated.gradient);
            second_moment = config.beta2 * second_moment + (1.0 - config.beta2) * norm_sq;
            if variant == AdamVariant::AmsGrad {
                max_second_moment = max_second_moment.max(second_moment);
            } else {
                max_second_moment = second_moment;
            }
            let m_hat = &momentum / (1.0 - config.beta1.powi(t));
            let v_hat = max_second_moment / (1.0 - config.beta2.powi(t));
            let direction = -m_hat / (v_hat.sqrt() + config.epsilon);
            let point = self.retract(&run.point, &direction, config.step_size)?;
            let evaluated = self.evaluate(&point, oracle)?;
            momentum = self.transport(&point, &momentum);
            run.accept(TrialPoint {
                point,
                evaluated,
                step_size: config.step_size,
                objective_evaluations: 1,
            });
            log_progress(options, iteration, &run.evaluated, run.last_step_size);
        }
        Ok(run.finish(stopping.tolerance))
    }

    fn optimize_cayley_sgd<F>(
        &self,
        mut run: OptimizationRun,
        oracle: &F,
        stopping: StoppingCriteria,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let config = &options.cayley_sgd;
        let mut momentum = Matrix::zeros((self.n, self.p));

        for iteration in 0..stopping.max_iterations {
            if run.converged(stopping.tolerance) {
                break;
            }
            momentum = &momentum * config.momentum + &run.evaluated.gradient;
            momentum = self.project_tangent(&run.point, &momentum);
            // The low-rank Cayley transform preserves X^T X without an n-by-n
            // matrix solve; only a 2p-by-2p system is formed.
            let point = cayley_update_low_rank(&run.point, &momentum, config.step_size)?;
            let evaluated = self.evaluate(&point, oracle)?;
            momentum = self.transport(&point, &momentum);
            run.accept(TrialPoint {
                point,
                evaluated,
                step_size: config.step_size,
                objective_evaluations: 1,
            });
            log_progress(options, iteration, &run.evaluated, run.last_step_size);
        }
        Ok(run.finish(stopping.tolerance))
    }

    fn optimize_lbfgs<F>(
        &self,
        mut run: OptimizationRun,
        oracle: &F,
        stopping: StoppingCriteria,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let mut history = LbfgsHistory::new(&options.lbfgs);

        for iteration in 0..stopping.max_iterations {
            if run.converged(stopping.tolerance) {
                break;
            }
            let mut direction = history.direction(&run.evaluated.gradient);
            if !matrix_is_finite(&direction)
                || frobenius_inner(&run.evaluated.gradient, &direction) >= 0.0
            {
                history.clear();
                direction = -&run.evaluated.gradient;
            }
            let trial = self.line_search(
                &run.point,
                &run.evaluated,
                &direction,
                oracle,
                &options.line_search,
            )?;

            // Every stored pair must live in the new tangent space before it is
            // combined with the new gradient difference.
            history.transport_to(self, &trial.point);
            let transported_grad = self.transport(&trial.point, &run.evaluated.gradient);
            let s = self.transport(&trial.point, &(&direction * trial.step_size));
            let y = &trial.evaluated.gradient - &transported_grad;
            history.push_if_valid(s, y);

            run.accept(trial);
            log_progress(options, iteration, &run.evaluated, run.last_step_size);
        }
        Ok(run.finish(stopping.tolerance))
    }

    fn line_search<F>(
        &self,
        x: &Matrix,
        evaluated: &EvaluatedPoint,
        direction: &Matrix,
        oracle: &F,
        method: &LineSearchMethod,
    ) -> Result<TrialPoint, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        match method {
            LineSearchMethod::FixedStep(config) => {
                let candidate = self.retract(x, direction, config.step_size)?;
                let candidate_eval = self.evaluate(&candidate, oracle)?;
                Ok(TrialPoint {
                    point: candidate,
                    evaluated: candidate_eval,
                    step_size: config.step_size,
                    objective_evaluations: 1,
                })
            }
            LineSearchMethod::BacktrackingArmijo(config) => {
                let directional_derivative = frobenius_inner(&evaluated.gradient, direction);
                if !directional_derivative.is_finite() || directional_derivative >= 0.0 {
                    return Err("line search requires a finite descent direction".to_string());
                }
                let mut step_size = config.initial_step_size;
                let mut calls = 0;
                let mut best: Option<(Matrix, EvaluatedPoint, f64)> = None;
                for _ in 0..config.max_backtracks {
                    let candidate = self.retract(x, direction, step_size)?;
                    let candidate_eval = self.evaluate(&candidate, oracle)?;
                    calls += 1;
                    if candidate_eval.energy
                        <= evaluated.energy
                            + config.sufficient_decrease * step_size * directional_derivative
                    {
                        return Ok(TrialPoint {
                            point: candidate,
                            evaluated: candidate_eval,
                            step_size,
                            objective_evaluations: calls,
                        });
                    }
                    if best
                        .as_ref()
                        .map(|(_, best_eval, _)| candidate_eval.energy < best_eval.energy)
                        .unwrap_or(true)
                    {
                        best = Some((candidate, candidate_eval, step_size));
                    }
                    step_size *= config.contraction_factor;
                    if step_size < config.min_step_size {
                        break;
                    }
                }
                match best {
                    Some((candidate, candidate_eval, step))
                        if candidate_eval.energy <= evaluated.energy =>
                    {
                        Ok(TrialPoint {
                            point: candidate,
                            evaluated: candidate_eval,
                            step_size: step,
                            objective_evaluations: calls,
                        })
                    }
                    _ => Err("backtracking line search failed to reduce the energy".to_string()),
                }
            }
        }
    }

    fn evaluate<F>(&self, x: &Matrix, oracle: &F) -> Result<EvaluatedPoint, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let evaluation = oracle(x);
        if !evaluation.energy.is_finite() {
            return Err("objective returned a non-finite energy".to_string());
        }
        self.validate_shape(&evaluation.euclidean_gradient)?;
        if !matrix_is_finite(&evaluation.euclidean_gradient) {
            return Err("objective returned a non-finite gradient".to_string());
        }
        Ok(EvaluatedPoint {
            energy: evaluation.energy,
            gradient: self.riemannian_gradient(x, &evaluation.euclidean_gradient),
        })
    }

    fn validate_shape(&self, matrix: &Matrix) -> Result<(), String> {
        if matrix.dim() != (self.n, self.p) {
            return Err(format!(
                "matrix has shape {:?}; expected ({}, {})",
                matrix.dim(),
                self.n,
                self.p
            ));
        }
        Ok(())
    }

    fn validate_options(
        &self,
        tol: f64,
        options: &ManifoldOptimizationOptions,
    ) -> Result<(), String> {
        if !tol.is_finite() || tol <= 0.0 {
            return Err("tolerance must be finite and positive".to_string());
        }
        match options.optimizer {
            ManifoldOptimizer::RiemannianGradientDescent
            | ManifoldOptimizer::RiemannianConjugateGradient => {
                validate_line_search(&options.line_search)?;
            }
            ManifoldOptimizer::RiemannianAdam | ManifoldOptimizer::RiemannianAmsGrad => {
                validate_adam_options(&options.adam)?;
            }
            ManifoldOptimizer::CayleySgd => validate_cayley_options(&options.cayley_sgd)?,
            ManifoldOptimizer::RiemannianLbfgs => {
                validate_line_search(&options.line_search)?;
                validate_lbfgs_options(&options.lbfgs)?;
            }
        }
        Ok(())
    }
}

fn log_progress(
    options: &ManifoldOptimizationOptions,
    iteration: usize,
    evaluated: &EvaluatedPoint,
    step_size: f64,
) {
    if let Some(interval) = options.log_interval {
        if interval > 0 && iteration % interval == 0 {
            println!(
                "Iteration {}: energy = {:.10}, gradient norm = {:.2e}, step = {:.2e}",
                iteration,
                evaluated.energy,
                frobenius_norm(&evaluated.gradient),
                step_size
            );
        }
    }
}

fn validate_line_search(method: &LineSearchMethod) -> Result<(), String> {
    match method {
        LineSearchMethod::FixedStep(config) => {
            if config.step_size.is_finite() && config.step_size > 0.0 {
                Ok(())
            } else {
                Err("fixed step size must be finite and positive".to_string())
            }
        }
        LineSearchMethod::BacktrackingArmijo(config) => {
            if !config.initial_step_size.is_finite() || config.initial_step_size <= 0.0 {
                return Err("initial line-search step must be finite and positive".to_string());
            }
            if !config.contraction_factor.is_finite()
                || !(0.0..1.0).contains(&config.contraction_factor)
            {
                return Err("line-search contraction factor must be in (0, 1)".to_string());
            }
            if !config.sufficient_decrease.is_finite()
                || !(0.0..1.0).contains(&config.sufficient_decrease)
            {
                return Err("Armijo sufficient decrease must be in (0, 1)".to_string());
            }
            if !config.min_step_size.is_finite()
                || config.min_step_size <= 0.0
                || config.max_backtracks == 0
            {
                return Err("line-search limits must be positive".to_string());
            }
            Ok(())
        }
    }
}

fn validate_adam_options(options: &RiemannianAdamOptions) -> Result<(), String> {
    if !options.step_size.is_finite() || options.step_size <= 0.0 {
        return Err("Adam step size must be finite and positive".to_string());
    }
    if !(0.0..1.0).contains(&options.beta1) || !(0.0..1.0).contains(&options.beta2) {
        return Err("Adam beta values must be in [0, 1)".to_string());
    }
    if !options.epsilon.is_finite() || options.epsilon <= 0.0 {
        return Err("Adam epsilon must be finite and positive".to_string());
    }
    Ok(())
}

fn validate_cayley_options(options: &CayleySgdOptions) -> Result<(), String> {
    if !options.step_size.is_finite() || options.step_size <= 0.0 {
        return Err("Cayley SGD step size must be finite and positive".to_string());
    }
    if !options.momentum.is_finite() || !(0.0..1.0).contains(&options.momentum) {
        return Err("Cayley SGD momentum must be in [0, 1)".to_string());
    }
    Ok(())
}

fn validate_lbfgs_options(options: &RiemannianLbfgsOptions) -> Result<(), String> {
    if options.memory_size == 0 {
        return Err("L-BFGS memory size must be positive".to_string());
    }
    if !options.curvature_epsilon.is_finite() || options.curvature_epsilon < 0.0 {
        return Err("L-BFGS curvature epsilon must be finite and nonnegative".to_string());
    }
    Ok(())
}

/// Apply the standard L-BFGS two-loop recursion in the current tangent space.
fn lbfgs_direction(gradient: &Matrix, history: &VecDeque<LbfgsPair>) -> Matrix {
    if history.is_empty() {
        return -gradient;
    }
    let mut q = gradient.clone();
    let mut alphas = vec![0.0; history.len()];
    for i in (0..history.len()).rev() {
        alphas[i] = history[i].rho * frobenius_inner(&history[i].s, &q);
        q -= &(&history[i].y * alphas[i]);
    }
    let newest = history.back().expect("history is nonempty");
    let yy = frobenius_inner(&newest.y, &newest.y);
    let sy = frobenius_inner(&newest.s, &newest.y);
    let scale = if yy > 1e-30 && sy.is_finite() {
        (sy / yy).clamp(1e-12, 1e12)
    } else {
        1.0
    };
    let mut r = q * scale;
    for (i, pair) in history.iter().enumerate() {
        let beta = pair.rho * frobenius_inner(&pair.y, &r);
        r += &(&pair.s * (alphas[i] - beta));
    }
    -r
}

/// Apply a Cayley transform using a rank-`2p` Woodbury solve.
///
/// For `A = M X^T - X M^T`, the update is
/// `(I + step A/2)^-1 (I - step A/2) X`. Factoring `A = U V^T`
/// reduces the solve from `n × n` to `2p × 2p`.
fn cayley_update_low_rank(x: &Matrix, gradient_like: &Matrix, step: f64) -> Result<Matrix, String> {
    let (n, p) = x.dim();
    let width = 2 * p;
    let mut u = Matrix::zeros((n, width));
    let mut v = Matrix::zeros((n, width));
    // U = [M, X] and V = [X, -M], hence U V^T = M X^T - X M^T.
    for i in 0..n {
        for j in 0..p {
            u[[i, j]] = gradient_like[[i, j]];
            u[[i, p + j]] = x[[i, j]];
            v[[i, j]] = x[[i, j]];
            v[[i, p + j]] = -gradient_like[[i, j]];
        }
    }
    let half_step = 0.5 * step;
    let ax = matmul(&u, &matmul(&transpose(&v), x));
    let rhs = x - &(ax * half_step);
    let small_system = Matrix::eye(width) + matmul(&transpose(&v), &u) * half_step;
    let small_rhs = matmul(&transpose(&v), &rhs);
    let correction = solve_linear_system(&small_system, &small_rhs)?;
    let next = rhs - &(matmul(&u, &correction) * half_step);
    if matrix_is_finite(&next) {
        Ok(next)
    } else {
        Err("Cayley update produced non-finite values".to_string())
    }
}

/// Solve `A X = B` by Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &Matrix, b: &Matrix) -> Result<Matrix, String> {
    let n = a.nrows();
    if a.ncols() != n || b.nrows() != n {
        return Err("linear solve shape mismatch".to_string());
    }
    let rhs_cols = b.ncols();
    let mut a = a.clone();
    let mut b = b.clone();
    for pivot in 0..n {
        let mut pivot_row = pivot;
        let mut pivot_abs = a[[pivot, pivot]].abs();
        for row in (pivot + 1)..n {
            let candidate = a[[row, pivot]].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs < 1e-14 {
            return Err("Cayley small system is singular".to_string());
        }
        if pivot_row != pivot {
            for col in 0..n {
                a.swap((pivot, col), (pivot_row, col));
            }
            for col in 0..rhs_cols {
                b.swap((pivot, col), (pivot_row, col));
            }
        }
        for row in (pivot + 1)..n {
            let factor = a[[row, pivot]] / a[[pivot, pivot]];
            a[[row, pivot]] = 0.0;
            for col in (pivot + 1)..n {
                a[[row, col]] -= factor * a[[pivot, col]];
            }
            for col in 0..rhs_cols {
                b[[row, col]] -= factor * b[[pivot, col]];
            }
        }
    }
    let mut solution = Matrix::zeros((n, rhs_cols));
    for row in (0..n).rev() {
        for col in 0..rhs_cols {
            let mut value = b[[row, col]];
            for k in (row + 1)..n {
                value -= a[[row, k]] * solution[[k, col]];
            }
            solution[[row, col]] = value / a[[row, row]];
        }
    }
    Ok(solution)
}

fn frobenius_inner(a: &Matrix, b: &Matrix) -> f64 {
    assert_eq!(a.dim(), b.dim(), "Frobenius inner product shape mismatch");
    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn matrix_is_finite(matrix: &Matrix) -> bool {
    matrix.iter().all(|value| value.is_finite())
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
    use ndarray::{array, Array2};
    use std::cell::Cell;

    fn rayleigh_evaluation(a: &Matrix, x: &Matrix) -> ManifoldEvaluation {
        let ax = matmul(a, x);
        ManifoldEvaluation {
            energy: -trace(&matmul(&transpose(x), &ax)),
            euclidean_gradient: -(ax * 2.0),
        }
    }

    fn assert_orthonormal(x: &Matrix, epsilon: f64) {
        let xtx = matmul(&transpose(x), x);
        for i in 0..xtx.nrows() {
            for j in 0..xtx.ncols() {
                assert_abs_diff_eq!(
                    xtx[[i, j]],
                    if i == j { 1.0 } else { 0.0 },
                    epsilon = epsilon
                );
            }
        }
    }

    fn test_problem() -> (StiefelManifold, Matrix, Matrix) {
        let manifold = StiefelManifold::new(4, 2);
        let initial =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.2, 0.3, 1.0, 0.2, 0.4, 0.1, 0.8]).unwrap();
        let a = Array2::from_diag(&array![5.0, 4.0, 2.0, 1.0]);
        (manifold, initial, a)
    }

    #[test]
    fn dimensions_distinguish_stiefel_and_grassmann() {
        let manifold = StiefelManifold::new(53, 35);
        assert_eq!(manifold.ambient_dimension(), 1855);
        assert_eq!(manifold.intrinsic_dimension(), 1225);
        assert_eq!(manifold.grassmann_dimension(), 630);
    }

    #[test]
    fn stiefel_projection_and_tangent_projection_are_valid() {
        let manifold = StiefelManifold::new(4, 2);
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]).unwrap();
        let projected = manifold.project(&x).unwrap();
        assert_orthonormal(&projected, 1e-10);

        let tangent_source =
            Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
        let tangent = manifold.project_tangent(&projected, &tangent_source);
        let xtg = matmul(&transpose(&projected), &tangent);
        assert!(frobenius_norm(&(&xtg + &transpose(&xtg))) < 1e-10);
    }

    #[test]
    fn combined_oracle_is_evaluated_once_per_fixed_step_point() {
        let (manifold, initial, a) = test_problem();
        let calls = Cell::new(0usize);
        let options = ManifoldOptimizationOptions {
            optimizer: ManifoldOptimizer::RiemannianGradientDescent,
            line_search: LineSearchMethod::FixedStep(FixedStepLineSearch { step_size: 1e-2 }),
            ..ManifoldOptimizationOptions::default()
        };
        let result = manifold
            .optimize_with_options(
                &initial,
                |x| {
                    calls.set(calls.get() + 1);
                    rayleigh_evaluation(&a, x)
                },
                3,
                1e-16,
                &options,
            )
            .unwrap();
        assert_eq!(calls.get(), 4);
        assert_eq!(result.objective_evaluations, 4);
    }

    #[test]
    fn only_the_selected_optimizer_options_are_validated() {
        let (manifold, initial, a) = test_problem();
        let mut options = ManifoldOptimizationOptions {
            optimizer: ManifoldOptimizer::RiemannianGradientDescent,
            line_search: LineSearchMethod::FixedStep(FixedStepLineSearch { step_size: 1e-2 }),
            adam: RiemannianAdamOptions {
                step_size: f64::NAN,
                ..RiemannianAdamOptions::default()
            },
            ..ManifoldOptimizationOptions::default()
        };

        let result = manifold.optimize_with_options(
            &initial,
            |x| rayleigh_evaluation(&a, x),
            1,
            1e-8,
            &options,
        );
        assert!(result.is_ok(), "an unused option block must be ignored");

        options.optimizer = ManifoldOptimizer::RiemannianAdam;
        let error = manifold
            .optimize_with_options(&initial, |x| rayleigh_evaluation(&a, x), 1, 1e-8, &options)
            .unwrap_err();
        assert!(error.contains("Adam step size"));
    }

    #[test]
    fn line_search_optimizers_decrease_energy() {
        let (manifold, initial, a) = test_problem();
        let projected = manifold.project(&initial).unwrap();
        let initial_energy = rayleigh_evaluation(&a, &projected).energy;
        for optimizer in [
            ManifoldOptimizer::RiemannianGradientDescent,
            ManifoldOptimizer::RiemannianConjugateGradient,
            ManifoldOptimizer::RiemannianLbfgs,
        ] {
            let options = ManifoldOptimizationOptions {
                optimizer,
                ..ManifoldOptimizationOptions::default()
            };
            let result = manifold
                .optimize_with_options(
                    &initial,
                    |x| rayleigh_evaluation(&a, x),
                    100,
                    1e-8,
                    &options,
                )
                .unwrap();
            assert!(result.energy < initial_energy, "optimizer {optimizer:?}");
            assert_orthonormal(&result.point, 1e-8);
        }
    }

    #[test]
    fn adaptive_optimizers_decrease_energy_and_preserve_constraints() {
        let (manifold, initial, a) = test_problem();
        let projected = manifold.project(&initial).unwrap();
        let initial_energy = rayleigh_evaluation(&a, &projected).energy;
        for optimizer in [
            ManifoldOptimizer::RiemannianAdam,
            ManifoldOptimizer::RiemannianAmsGrad,
            ManifoldOptimizer::CayleySgd,
        ] {
            let options = ManifoldOptimizationOptions {
                optimizer,
                adam: RiemannianAdamOptions {
                    step_size: 3e-2,
                    ..RiemannianAdamOptions::default()
                },
                cayley_sgd: CayleySgdOptions {
                    step_size: 1e-2,
                    momentum: 0.5,
                },
                ..ManifoldOptimizationOptions::default()
            };
            let result = manifold
                .optimize_with_options(
                    &initial,
                    |x| rayleigh_evaluation(&a, x),
                    250,
                    1e-7,
                    &options,
                )
                .unwrap();
            assert!(result.energy < initial_energy, "optimizer {optimizer:?}");
            assert_orthonormal(&result.point, 2e-8);
        }
    }

    #[test]
    fn backwards_compatible_cg_api_still_works() {
        let (manifold, initial, a) = test_problem();
        let result = manifold
            .optimize_cg(
                &initial,
                |x| rayleigh_evaluation(&a, x).euclidean_gradient,
                |x| rayleigh_evaluation(&a, x).energy,
                100,
                1e-8,
            )
            .unwrap();
        assert!(result.converged);
    }

    #[test]
    fn gradient_only_api_rejects_an_invalid_gradient_shape() {
        let (manifold, initial, _) = test_problem();
        let error = manifold
            .optimize(&initial, |_| Matrix::zeros((1, 1)), 1, 1e-8)
            .unwrap_err();
        assert!(error.contains("expected (4, 2)"));
    }
}
