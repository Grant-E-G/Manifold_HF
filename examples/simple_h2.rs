//! Simple H2 molecule example
//!
//! This example demonstrates using the Hartree-Fock method
//! with manifold optimization on a simple H2 molecule.

use manifold_hf::{Molecule, HartreeFock};

fn main() {
    println!("=== Hydrogen Molecule (H2) Example ===\n");

    // Create H2 molecule at standard bond length
    let h2 = Molecule::h2();
    
    println!("System Information:");
    println!("  Molecule: H2");
    println!("  Bond length: 1.4 Bohr (0.741 Angstrom)");
    println!("  Electrons: {}", h2.num_electrons());
    println!("  Occupied orbitals: {}", h2.num_occupied());
    println!("  Nuclear repulsion: {:.6} Hartree\n", h2.nuclear_repulsion());

    // Initialize Hartree-Fock calculator
    let hf = HartreeFock::new(h2);

    // Run manifold-optimized SCF
    println!("Starting manifold-optimized Hartree-Fock calculation...\n");
    
    match hf.run_scf_manifold(100, 1e-6) {
        Ok(result) => {
            println!("\n=== Results ===");
            println!("Converged: {}", result.converged);
            println!("Total energy: {:.10} Hartree", result.energy);
            println!("              {:.10} eV", result.energy * 27.211);
            
            if result.converged {
                println!("\n✓ Calculation completed successfully!");
            }
        }
        Err(e) => {
            eprintln!("Error during calculation: {}", e);
        }
    }
}
