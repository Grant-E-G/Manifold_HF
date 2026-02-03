use std::fs;
use std::path::Path;

use manifold_hf::molecule::Atom;
use manifold_hf::{HartreeFock, Molecule};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Benchmarks {
    version: u32,
    molecules: Vec<BenchmarkMolecule>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkMolecule {
    name: String,
    charge: i32,
    multiplicity: u32,
    #[serde(default)]
    tags: Vec<String>,
    atoms: Vec<BenchmarkAtom>,
    reference: BenchmarkReference,
    geometry: BenchmarkGeometry,
}

#[derive(Debug, Deserialize)]
struct BenchmarkAtom {
    atomic_number: u32,
    position: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct BenchmarkReference {
    energy_total: f64,
    energy_electronic: f64,
    energy_nuclear: f64,
}

#[derive(Debug, Deserialize)]
struct BenchmarkGeometry {
    bond_lengths: Vec<BondLength>,
    bond_angles: Vec<BondAngle>,
}

#[derive(Debug, Deserialize)]
struct BondLength {
    atoms: [usize; 2],
    value: f64,
}

#[derive(Debug, Deserialize)]
struct BondAngle {
    atoms: [usize; 3],
    value: f64,
}

fn should_run(tags: &[String]) -> bool {
    let run_all = matches!(
        std::env::var("MANIFOLD_HF_BENCHMARKS").as_deref(),
        Ok("1") | Ok("full") | Ok("all") | Ok("true")
    );
    if run_all {
        return true;
    }
    tags.iter().any(|tag| tag == "quick")
}

#[test]
fn compare_benchmarks_to_reference() {
    let path = Path::new("data/benchmarks.json");
    if !path.exists() {
        eprintln!("Skipping benchmarks: data/benchmarks.json not found.");
        return;
    }

    let contents = fs::read_to_string(path).expect("Failed to read benchmarks.json");
    let data: Benchmarks = serde_json::from_str(&contents).expect("Invalid benchmarks.json");
    assert!(data.version >= 1);

    let run_all = matches!(
        std::env::var("MANIFOLD_HF_BENCHMARKS").as_deref(),
        Ok("1") | Ok("full") | Ok("all") | Ok("true")
    );
    let energy_tol = if run_all { 5e-3 } else { 1e-3 };

    for entry in data.molecules {
        if !should_run(&entry.tags) {
            continue;
        }

        assert_eq!(
            entry.multiplicity, 1,
            "Only closed-shell benchmarks supported ({})",
            entry.name
        );

        let atoms = entry
            .atoms
            .iter()
            .map(|atom| Atom::new(atom.atomic_number, atom.position))
            .collect::<Vec<_>>();
        let molecule = Molecule::new(atoms, entry.charge, entry.multiplicity);

        let hf = HartreeFock::new(molecule.clone()).expect("Failed to build HF");
        let result = hf.run_scf(100, 1e-6).expect("SCF failed");

        let nuclear = molecule.nuclear_repulsion();
        let electronic = result.energy - nuclear;

        assert!(
            (result.energy - entry.reference.energy_total).abs() <= energy_tol,
            "Total energy mismatch for {}",
            entry.name
        );
        assert!(
            (electronic - entry.reference.energy_electronic).abs() <= energy_tol,
            "Electronic energy mismatch for {}",
            entry.name
        );
        assert!(
            (nuclear - entry.reference.energy_nuclear).abs() <= 1e-6,
            "Nuclear repulsion mismatch for {}",
            entry.name
        );

        for bond in &entry.geometry.bond_lengths {
            let dist = molecule.bond_length(bond.atoms[0], bond.atoms[1]);
            assert!(
                (dist - bond.value).abs() <= 1e-6,
                "Bond length mismatch for {}",
                entry.name
            );
        }

        for angle_def in &entry.geometry.bond_angles {
            let value =
                molecule.bond_angle(angle_def.atoms[0], angle_def.atoms[1], angle_def.atoms[2]);
            assert!(
                (value - angle_def.value).abs() <= 1e-6,
                "Bond angle mismatch for {}",
                entry.name
            );
        }
    }
}
