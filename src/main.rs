use manifold_hf::{Molecule, HartreeFock};

fn main() {
    println!("=== Manifold Hartree-Fock ===");
    println!("Quantum chemistry with manifold optimization in Rust\n");

    // Calculate all three molecules: H2, H2O, and D2O
    calculate_molecule("H2 (Hydrogen)", Molecule::h2());
    println!("\n{}\n", "=".repeat(60));
    calculate_molecule("H2O (Water)", Molecule::h2o());
    println!("\n{}\n", "=".repeat(60));
    calculate_molecule("D2O (Heavy Water)", Molecule::d2o());
}

fn calculate_molecule(name: &str, molecule: Molecule) {
    println!("Molecule: {}", name);
    println!("Number of electrons: {}", molecule.num_electrons());
    println!("Occupied orbitals: {}", molecule.num_occupied());
    println!("Nuclear repulsion energy: {:.6} Hartree\n", molecule.nuclear_repulsion());

    // Create Hartree-Fock calculator
    let hf = HartreeFock::new(molecule);

    println!("=== Standard SCF ===");
    match hf.run_scf(50, 1e-6) {
        Ok(result) => {
            println!("Converged: {}", result.converged);
            println!("Total energy: {:.10} Hartree", result.energy);
            println!("             {:.6} eV", result.energy * 27.211);
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\n=== SCF with Manifold Optimization ===");
    match hf.run_scf_manifold(100, 1e-6) {
        Ok(result) => {
            println!("Converged: {}", result.converged);
            println!("Total energy: {:.10} Hartree", result.energy);
            println!("             {:.6} eV", result.energy * 27.211);
        }
        Err(e) => println!("Error: {}", e),
    }
}
