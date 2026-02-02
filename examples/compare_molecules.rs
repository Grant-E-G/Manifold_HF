//! Comparison of H2, H2O, and D2O molecules
//!
//! This example demonstrates Hartree-Fock calculations with manifold optimization
//! on hydrogen (H2), water (H2O), and heavy water (D2O) molecules.

use manifold_hf::{Molecule, HartreeFock};

fn main() {
    println!("=== Molecular Comparison: H2, H2O, and D2O ===\n");
    println!("Demonstrating Hartree-Fock with Manifold Optimization\n");

    // Array of molecules to calculate
    let molecules = vec![
        ("H2 (Hydrogen)", Molecule::h2()),
        ("H2O (Water)", Molecule::h2o()),
        ("D2O (Heavy Water)", Molecule::d2o()),
    ];

    let mut results = Vec::new();

    for (name, molecule) in molecules {
        println!("{}", "=".repeat(70));
        println!("Calculating: {}", name);
        println!("{}", "=".repeat(70));
        
        print_molecule_info(&molecule, name);
        
        let hf = HartreeFock::new(molecule).expect("Failed to build STO-3G basis");
        
        println!("\nRunning manifold-optimized Hartree-Fock...");
        match hf.run_scf_manifold(100, 1e-6) {
            Ok(result) => {
                println!("\n✓ Calculation completed");
                println!("  Energy: {:.10} Hartree", result.energy);
                println!("          {:.6} eV", result.energy * 27.211);
                println!("  Converged: {}", result.converged);
                
                results.push((name, result.energy));
            }
            Err(e) => {
                eprintln!("✗ Error: {}", e);
            }
        }
        println!();
    }

    // Summary comparison
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY: Energy Comparison");
    println!("{}", "=".repeat(70));
    println!("{:<20} {:>20} {:>20}", "Molecule", "Energy (Hartree)", "Energy (eV)");
    println!("{}", "-".repeat(70));
    
    for (name, energy) in results {
        println!("{:<20} {:>20.10} {:>20.6}", 
                 name, energy, energy * 27.211);
    }
    
    println!("\n{}", "=".repeat(70));
    println!("Note: D2O and H2O have identical electronic structures in the");
    println!("Born-Oppenheimer approximation (same atomic numbers). Nuclear");
    println!("mass differences only affect vibrational properties.");
    println!("{}", "=".repeat(70));
}

fn print_molecule_info(molecule: &Molecule, name: &str) {
    println!("\nMolecule Information:");
    println!("  Name: {}", name);
    println!("  Atoms: {}", molecule.atoms.len());
    println!("  Electrons: {}", molecule.num_electrons());
    println!("  Occupied orbitals: {}", molecule.num_occupied());
    println!("  Nuclear repulsion: {:.6} Hartree", molecule.nuclear_repulsion());
    println!("                     {:.6} eV", molecule.nuclear_repulsion() * 27.211);
}
