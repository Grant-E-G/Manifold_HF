use manifold_hf::{Molecule, HartreeFock};

fn main() {
    println!("=== Manifold Hartree-Fock ===");
    println!("Quantum chemistry with manifold optimization in Rust\n");

    // Create H2 molecule
    let molecule = Molecule::h2();
    println!("Molecule: H2");
    println!("Bond length: 1.4 Bohr");
    println!("Number of electrons: {}", molecule.num_electrons());
    println!("Nuclear repulsion energy: {:.6} Hartree\n", molecule.nuclear_repulsion());

    // Create Hartree-Fock calculator
    let hf = HartreeFock::new(molecule);

    println!("=== Standard SCF ===");
    match hf.run_scf(50, 1e-6) {
        Ok(result) => {
            println!("\nFinal Results:");
            println!("Converged: {}", result.converged);
            println!("Total energy: {:.10} Hartree", result.energy);
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\n=== SCF with Manifold Optimization ===");
    match hf.run_scf_manifold(100, 1e-6) {
        Ok(result) => {
            println!("\nFinal Results:");
            println!("Converged: {}", result.converged);
            println!("Total energy: {:.10} Hartree", result.energy);
        }
        Err(e) => println!("Error: {}", e),
    }
}
