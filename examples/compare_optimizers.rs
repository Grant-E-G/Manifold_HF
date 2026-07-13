//! Compare the manifold optimizers using Fock builds as the primary work unit.

use manifold_hf::{
    CayleySgdOptions, HartreeFock, ManifoldOptimizationOptions, ManifoldOptimizer, Molecule,
    RiemannianAdamOptions,
};

const MAX_ITERATIONS: usize = 500;
const GRADIENT_TOLERANCE: f64 = 1e-5;

/// Keep the comparison setup in one place so the loop below only describes the
/// experiment. Fixed-step methods need smaller problem-specific steps than the
/// default Armijo-based methods.
fn options_for(optimizer: ManifoldOptimizer) -> ManifoldOptimizationOptions {
    ManifoldOptimizationOptions {
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
    }
}

fn main() -> Result<(), String> {
    let hf = HartreeFock::new(Molecule::h2o())?;

    println!(
        "{:<32} {:>16} {:>10} {:>12} {:>11}",
        "optimizer", "energy", "iterations", "Fock builds", "converged"
    );
    for optimizer in ManifoldOptimizer::ALL {
        let options = options_for(optimizer);
        let result =
            hf.run_scf_manifold_with_options(MAX_ITERATIONS, GRADIENT_TOLERANCE, &options)?;
        println!(
            "{:<32} {:>16.10} {:>10} {:>12} {:>11}",
            optimizer, result.energy, result.iterations, result.fock_builds, result.converged
        );
    }
    Ok(())
}
