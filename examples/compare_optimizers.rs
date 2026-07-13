//! Compare the manifold optimizers using Fock builds as the primary work unit.

use manifold_hf::{
    CayleySgdOptions, HartreeFock, ManifoldOptimizationOptions, ManifoldOptimizer, Molecule,
    RiemannianAdamOptions,
};

fn main() -> Result<(), String> {
    let hf = HartreeFock::new(Molecule::h2o())?;
    let optimizers = [
        ManifoldOptimizer::RiemannianGradientDescent,
        ManifoldOptimizer::RiemannianConjugateGradient,
        ManifoldOptimizer::RiemannianAdam,
        ManifoldOptimizer::RiemannianAmsGrad,
        ManifoldOptimizer::CayleySgd,
        ManifoldOptimizer::RiemannianLbfgs,
    ];

    println!(
        "{:<32} {:>16} {:>10} {:>12} {:>11}",
        "optimizer", "energy", "iterations", "Fock builds", "converged"
    );
    for optimizer in optimizers {
        let options = ManifoldOptimizationOptions {
            optimizer,
            adam: RiemannianAdamOptions {
                step_size: 3e-2,
                ..RiemannianAdamOptions::default()
            },
            cayley_sgd: CayleySgdOptions {
                step_size: 5e-3,
                momentum: 0.5,
            },
            ..ManifoldOptimizationOptions::default()
        };
        let result = hf.run_scf_manifold_with_options(500, 1e-5, &options)?;
        println!(
            "{:<32?} {:>16.10} {:>10} {:>12} {:>11}",
            optimizer, result.energy, result.iterations, result.fock_builds, result.converged
        );
    }
    Ok(())
}
