//! Optimization algorithms for matrices with orthonormal columns.
//!
//! The optimizers share one value-and-gradient oracle so expensive objectives can
//! build their state once per trial point. This matters for Hartree--Fock, where
//! both the energy and gradient use the same density and Fock matrices.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldOptimizer {
    RiemannianConjugateGradient,
    RiemannianGradientDescent,
    RiemannianAdam,
    RiemannianAmsGrad,
    CayleySgd,
    RiemannianLbfgs,
}

#[derive(Debug, Clone)]
pub struct RiemannianAdamOptions {
    pub step_size: f64,
    pub beta1: f64,
    pub beta2: f64,
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

#[derive(Debug, Clone)]
pub struct CayleySgdOptions {
    pub step_size: f64,
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

#[derive(Debug, Clone)]
pub struct RiemannianLbfgsOptions {
    pub memory_size: usize,
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

#[derive(Debug, Clone)]
pub struct ManifoldOptimizationOptions {
    pub optimizer: ManifoldOptimizer,
    pub line_search: LineSearchMethod,
    pub cg_variant: ConjugateGradientVariant,
    pub adam: RiemannianAdamOptions,
    pub cayley_sgd: CayleySgdOptions,
    pub lbfgs: RiemannianLbfgsOptions,
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
#[derive(Debug, Clone)]
pub struct ManifoldEvaluation {
    pub energy: f64,
    pub euclidean_gradient: Matrix,
}

#[derive(Debug, Clone)]
pub struct ManifoldOptimizationResult {
    pub point: Matrix,
    pub energy: f64,
    pub converged: bool,
    pub iterations: usize,
    pub gradient_norm: f64,
    pub last_step_size: f64,
    /// Number of value-and-gradient oracle calls, including the initial point.
    pub objective_evaluations: usize,
}

#[derive(Debug, Clone)]
struct EvaluatedPoint {
    energy: f64,
    gradient: Matrix,
}

#[derive(Debug, Clone)]
struct LbfgsPair {
    s: Matrix,
    y: Matrix,
    rho: f64,
}

/// Represents the Stiefel manifold St(n, p) = {X in R^(n x p) : X^T X = I_p}.
pub struct StiefelManifold {
    pub n: usize,
    pub p: usize,
}

impl StiefelManifold {
    pub fn new(n: usize, p: usize) -> Self {
        assert!(p <= n, "p must be <= n for Stiefel manifold");
        Self { n, p }
    }

    /// Ambient matrix storage dimension.
    pub fn ambient_dimension(&self) -> usize {
        self.n * self.p
    }

    /// Intrinsic dimension of St(n, p).
    pub fn intrinsic_dimension(&self) -> usize {
        self.n * self.p - self.p * (self.p + 1) / 2
    }

    /// Dimension of the Grassmann quotient represented by this matrix.
    pub fn grassmann_dimension(&self) -> usize {
        self.p * (self.n - self.p)
    }

    pub fn project(&self, x: &Matrix) -> Result<Matrix, String> {
        self.validate_shape(x)?;
        project_stiefel(x)
    }

    /// Embedded-Euclidean projection onto the Stiefel tangent space at `x`.
    pub fn project_tangent(&self, x: &Matrix, ambient: &Matrix) -> Matrix {
        let xt_ambient = matmul(&transpose(x), ambient);
        let sym = &xt_ambient + &transpose(&xt_ambient);
        ambient - &matmul(x, &(&sym * 0.5))
    }

    pub fn riemannian_gradient(&self, x: &Matrix, euclidean_grad: &Matrix) -> Matrix {
        self.project_tangent(x, euclidean_grad)
    }

    /// QR-like (modified Gram--Schmidt) retraction.
    pub fn retract(&self, x: &Matrix, tangent: &Matrix, step_size: f64) -> Result<Matrix, String> {
        self.project(&(x + &(tangent * step_size)))
    }

    /// Projection transport into the tangent space at `x_next`.
    pub fn transport(&self, x_next: &Matrix, tangent: &Matrix) -> Matrix {
        self.project_tangent(x_next, tangent)
    }

    /// Legacy fixed-step RSGD helper.
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
            let grad = self.riemannian_gradient(&x, &grad_fn(&x));
            if frobenius_norm(&grad) < tol {
                break;
            }
            x = self.retract(&x, &(-&grad), step_size)?;
        }
        Ok(x)
    }

    /// Optimize using a combined value-and-gradient oracle.
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
        match options.optimizer {
            ManifoldOptimizer::RiemannianConjugateGradient => {
                self.optimize_rcg(x, evaluated, &oracle, max_iter, tol, options)
            }
            ManifoldOptimizer::RiemannianGradientDescent => {
                self.optimize_rsgd(x, evaluated, &oracle, max_iter, tol, options)
            }
            ManifoldOptimizer::RiemannianAdam => {
                self.optimize_adam(x, evaluated, &oracle, max_iter, tol, options, false)
            }
            ManifoldOptimizer::RiemannianAmsGrad => {
                self.optimize_adam(x, evaluated, &oracle, max_iter, tol, options, true)
            }
            ManifoldOptimizer::CayleySgd => {
                self.optimize_cayley_sgd(x, evaluated, &oracle, max_iter, tol, options)
            }
            ManifoldOptimizer::RiemannianLbfgs => {
                self.optimize_lbfgs(x, evaluated, &oracle, max_iter, tol, options)
            }
        }
    }

    /// Backwards-compatible conjugate-gradient entry point.
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

    /// Backwards-compatible CG API. New expensive objectives should use
    /// [`Self::optimize_with_options`] to avoid duplicate work.
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
        mut x: Matrix,
        mut evaluated: EvaluatedPoint,
        oracle: &F,
        max_iter: usize,
        tol: f64,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let mut evaluations = 1;
        let mut iterations = 0;
        let mut last_step = 0.0;
        for iter in 0..max_iter {
            if frobenius_norm(&evaluated.gradient) < tol {
                break;
            }
            let direction = -&evaluated.gradient;
            let (next_x, next_evaluated, step, calls) =
                self.line_search(&x, &evaluated, &direction, oracle, &options.line_search)?;
            x = next_x;
            evaluated = next_evaluated;
            evaluations += calls;
            iterations = iter + 1;
            last_step = step;
            log_progress(options, iter, &evaluated, step);
        }
        Ok(finish_result(
            x,
            evaluated,
            iterations,
            tol,
            last_step,
            evaluations,
        ))
    }

    fn optimize_rcg<F>(
        &self,
        mut x: Matrix,
        mut evaluated: EvaluatedPoint,
        oracle: &F,
        max_iter: usize,
        tol: f64,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let mut direction = -&evaluated.gradient;
        let mut evaluations = 1;
        let mut iterations = 0;
        let mut last_step = 0.0;

        for iter in 0..max_iter {
            if frobenius_norm(&evaluated.gradient) < tol {
                break;
            }
            if frobenius_inner(&evaluated.gradient, &direction) >= 0.0 {
                direction = -&evaluated.gradient;
            }
            let (next_x, next_evaluated, step, calls) =
                self.line_search(&x, &evaluated, &direction, oracle, &options.line_search)?;
            let transported_grad = self.transport(&next_x, &evaluated.gradient);
            let transported_direction = self.transport(&next_x, &direction);
            let beta = cg_beta(
                options.cg_variant,
                &next_evaluated.gradient,
                &transported_grad,
                &evaluated.gradient,
            );
            let mut next_direction = -&next_evaluated.gradient + &(transported_direction * beta);
            if frobenius_inner(&next_evaluated.gradient, &next_direction) >= 0.0 {
                next_direction = -&next_evaluated.gradient;
            }

            x = next_x;
            evaluated = next_evaluated;
            direction = next_direction;
            evaluations += calls;
            iterations = iter + 1;
            last_step = step;
            log_progress(options, iter, &evaluated, step);
        }
        Ok(finish_result(
            x,
            evaluated,
            iterations,
            tol,
            last_step,
            evaluations,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize_adam<F>(
        &self,
        mut x: Matrix,
        mut evaluated: EvaluatedPoint,
        oracle: &F,
        max_iter: usize,
        tol: f64,
        options: &ManifoldOptimizationOptions,
        amsgrad: bool,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let config = &options.adam;
        let mut momentum = Matrix::zeros((self.n, self.p));
        let mut second_moment = 0.0_f64;
        let mut max_second_moment = 0.0_f64;
        let mut evaluations = 1;
        let mut iterations = 0;

        for iter in 0..max_iter {
            if frobenius_norm(&evaluated.gradient) < tol {
                break;
            }
            let t = (iter + 1) as i32;
            momentum = &momentum * config.beta1 + &evaluated.gradient * (1.0 - config.beta1);
            let norm_sq = frobenius_inner(&evaluated.gradient, &evaluated.gradient);
            second_moment = config.beta2 * second_moment + (1.0 - config.beta2) * norm_sq;
            if amsgrad {
                max_second_moment = max_second_moment.max(second_moment);
            } else {
                max_second_moment = second_moment;
            }
            let m_hat = &momentum / (1.0 - config.beta1.powi(t));
            let v_hat = max_second_moment / (1.0 - config.beta2.powi(t));
            let direction = -m_hat / (v_hat.sqrt() + config.epsilon);
            let next_x = self.retract(&x, &direction, config.step_size)?;
            let next_evaluated = self.evaluate(&next_x, oracle)?;
            momentum = self.transport(&next_x, &momentum);

            x = next_x;
            evaluated = next_evaluated;
            evaluations += 1;
            iterations = iter + 1;
            log_progress(options, iter, &evaluated, config.step_size);
        }
        Ok(finish_result(
            x,
            evaluated,
            iterations,
            tol,
            if iterations == 0 {
                0.0
            } else {
                config.step_size
            },
            evaluations,
        ))
    }

    fn optimize_cayley_sgd<F>(
        &self,
        mut x: Matrix,
        mut evaluated: EvaluatedPoint,
        oracle: &F,
        max_iter: usize,
        tol: f64,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let config = &options.cayley_sgd;
        let mut momentum = Matrix::zeros((self.n, self.p));
        let mut evaluations = 1;
        let mut iterations = 0;

        for iter in 0..max_iter {
            if frobenius_norm(&evaluated.gradient) < tol {
                break;
            }
            momentum = &momentum * config.momentum + &evaluated.gradient;
            momentum = self.project_tangent(&x, &momentum);
            let next_x = cayley_update_low_rank(&x, &momentum, config.step_size)?;
            let next_evaluated = self.evaluate(&next_x, oracle)?;
            momentum = self.transport(&next_x, &momentum);

            x = next_x;
            evaluated = next_evaluated;
            evaluations += 1;
            iterations = iter + 1;
            log_progress(options, iter, &evaluated, config.step_size);
        }
        Ok(finish_result(
            x,
            evaluated,
            iterations,
            tol,
            if iterations == 0 {
                0.0
            } else {
                config.step_size
            },
            evaluations,
        ))
    }

    fn optimize_lbfgs<F>(
        &self,
        mut x: Matrix,
        mut evaluated: EvaluatedPoint,
        oracle: &F,
        max_iter: usize,
        tol: f64,
        options: &ManifoldOptimizationOptions,
    ) -> Result<ManifoldOptimizationResult, String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        let config = &options.lbfgs;
        let mut history: Vec<LbfgsPair> = Vec::with_capacity(config.memory_size);
        let mut evaluations = 1;
        let mut iterations = 0;
        let mut last_step = 0.0;

        for iter in 0..max_iter {
            if frobenius_norm(&evaluated.gradient) < tol {
                break;
            }
            let mut direction = lbfgs_direction(&evaluated.gradient, &history);
            if !matrix_is_finite(&direction)
                || frobenius_inner(&evaluated.gradient, &direction) >= 0.0
            {
                history.clear();
                direction = -&evaluated.gradient;
            }
            let (next_x, next_evaluated, step, calls) =
                self.line_search(&x, &evaluated, &direction, oracle, &options.line_search)?;

            for pair in &mut history {
                pair.s = self.transport(&next_x, &pair.s);
                pair.y = self.transport(&next_x, &pair.y);
                let curvature = frobenius_inner(&pair.s, &pair.y);
                pair.rho = if curvature > 0.0 {
                    1.0 / curvature
                } else {
                    0.0
                };
            }
            history.retain(|pair| pair.rho.is_finite() && pair.rho > 0.0);

            let transported_grad = self.transport(&next_x, &evaluated.gradient);
            let s = self.transport(&next_x, &(&direction * step));
            let y = &next_evaluated.gradient - &transported_grad;
            let curvature = frobenius_inner(&s, &y);
            let curvature_floor = config.curvature_epsilon
                * frobenius_norm(&s).max(1e-30)
                * frobenius_norm(&y).max(1e-30);
            if curvature.is_finite() && curvature > curvature_floor {
                if history.len() == config.memory_size {
                    history.remove(0);
                }
                history.push(LbfgsPair {
                    s,
                    y,
                    rho: 1.0 / curvature,
                });
            }

            x = next_x;
            evaluated = next_evaluated;
            evaluations += calls;
            iterations = iter + 1;
            last_step = step;
            log_progress(options, iter, &evaluated, step);
        }
        Ok(finish_result(
            x,
            evaluated,
            iterations,
            tol,
            last_step,
            evaluations,
        ))
    }

    fn line_search<F>(
        &self,
        x: &Matrix,
        evaluated: &EvaluatedPoint,
        direction: &Matrix,
        oracle: &F,
        method: &LineSearchMethod,
    ) -> Result<(Matrix, EvaluatedPoint, f64, usize), String>
    where
        F: Fn(&Matrix) -> ManifoldEvaluation,
    {
        match method {
            LineSearchMethod::FixedStep(config) => {
                let candidate = self.retract(x, direction, config.step_size)?;
                let candidate_eval = self.evaluate(&candidate, oracle)?;
                Ok((candidate, candidate_eval, config.step_size, 1))
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
                    if best
                        .as_ref()
                        .map(|(_, best_eval, _)| candidate_eval.energy < best_eval.energy)
                        .unwrap_or(true)
                    {
                        best = Some((candidate.clone(), candidate_eval.clone(), step_size));
                    }
                    if candidate_eval.energy
                        <= evaluated.energy
                            + config.sufficient_decrease * step_size * directional_derivative
                    {
                        return Ok((candidate, candidate_eval, step_size, calls));
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
                        Ok((candidate, candidate_eval, step, calls))
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
        validate_line_search(&options.line_search)?;
        let adam = &options.adam;
        if !adam.step_size.is_finite() || adam.step_size <= 0.0 {
            return Err("Adam step size must be finite and positive".to_string());
        }
        if !(0.0..1.0).contains(&adam.beta1) || !(0.0..1.0).contains(&adam.beta2) {
            return Err("Adam beta values must be in [0, 1)".to_string());
        }
        if !adam.epsilon.is_finite() || adam.epsilon <= 0.0 {
            return Err("Adam epsilon must be finite and positive".to_string());
        }
        let cayley = &options.cayley_sgd;
        if !cayley.step_size.is_finite() || cayley.step_size <= 0.0 {
            return Err("Cayley SGD step size must be finite and positive".to_string());
        }
        if !cayley.momentum.is_finite() || !(0.0..1.0).contains(&cayley.momentum) {
            return Err("Cayley SGD momentum must be in [0, 1)".to_string());
        }
        if options.lbfgs.memory_size == 0 {
            return Err("L-BFGS memory size must be positive".to_string());
        }
        if !options.lbfgs.curvature_epsilon.is_finite() || options.lbfgs.curvature_epsilon < 0.0 {
            return Err("L-BFGS curvature epsilon must be finite and nonnegative".to_string());
        }
        Ok(())
    }
}

fn finish_result(
    point: Matrix,
    evaluated: EvaluatedPoint,
    iterations: usize,
    tol: f64,
    last_step_size: f64,
    objective_evaluations: usize,
) -> ManifoldOptimizationResult {
    let gradient_norm = frobenius_norm(&evaluated.gradient);
    ManifoldOptimizationResult {
        point,
        energy: evaluated.energy,
        converged: gradient_norm < tol,
        iterations,
        gradient_norm,
        last_step_size,
        objective_evaluations,
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

fn lbfgs_direction(gradient: &Matrix, history: &[LbfgsPair]) -> Matrix {
    if history.is_empty() {
        return -gradient;
    }
    let mut q = gradient.clone();
    let mut alphas = vec![0.0; history.len()];
    for i in (0..history.len()).rev() {
        alphas[i] = history[i].rho * frobenius_inner(&history[i].s, &q);
        q -= &(&history[i].y * alphas[i]);
    }
    let newest = history.last().expect("history is nonempty");
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

/// Cayley transform using a rank-2p Woodbury solve rather than an n-by-n solve.
/// This is algebraically equivalent to the dense update in the research code.
fn cayley_update_low_rank(x: &Matrix, gradient_like: &Matrix, step: f64) -> Result<Matrix, String> {
    let (n, p) = x.dim();
    let width = 2 * p;
    let mut u = Matrix::zeros((n, width));
    let mut v = Matrix::zeros((n, width));
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
}
