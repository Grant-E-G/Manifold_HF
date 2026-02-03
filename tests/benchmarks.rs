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
    tags.iter().any(|tag| tag == "small")
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
    let abs_energy_tol: f64 = if run_all { 5e-3 } else { 1e-3 };
    let rel_energy_tol: f64 = 6e-2;

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
        let mut result = hf.run_scf(100, 1e-6).expect("SCF failed");
        if !result.converged {
            result = hf
                .run_scf_manifold(200, 1e-6)
                .expect("Manifold SCF failed");
        }

        let nuclear = molecule.nuclear_repulsion();
        let electronic = result.energy - nuclear;

        if std::env::var("MANIFOLD_HF_DEBUG").as_deref() == Ok("1") {
            let n = hf.basis.size();
            let mut g = manifold_hf::linalg::Matrix::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        for l in 0..n {
                            let dkl = result.density[[k, l]];
                            let ijkl = ((i * n + j) * n + k) * n + l;
                            let ikjl = ((i * n + k) * n + j) * n + l;
                            sum += dkl * (hf.eri[ijkl] - 0.5 * hf.eri[ikjl]);
                        }
                    }
                    g[[i, j]] = sum;
                }
            }
            let h_plus_f = &hf.core_hamiltonian + &(&hf.core_hamiltonian + &g);
            let energy_check = 0.5 * manifold_hf::linalg::trace(&manifold_hf::linalg::matmul(
                &result.density,
                &h_plus_f,
            )) + nuclear;
            let e_one = manifold_hf::linalg::trace(&manifold_hf::linalg::matmul(
                &result.density,
                &hf.core_hamiltonian,
            ));
            let e_two = 0.5
                * manifold_hf::linalg::trace(&manifold_hf::linalg::matmul(
                    &result.density,
                    &g,
                ));
            eprintln!(
                "Debug {}: E_total {:.6}, E_check {:.6}, E_one {:.6}, E_two {:.6}",
                entry.name, result.energy, energy_check, e_one, e_two
            );
        }

        let energy_tol = abs_energy_tol.max(rel_energy_tol * entry.reference.energy_total.abs());
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
        let nuclear_tol = 1e-5_f64.max(1e-6 * entry.reference.energy_nuclear.abs());
        assert!(
            (nuclear - entry.reference.energy_nuclear).abs() <= nuclear_tol,
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
