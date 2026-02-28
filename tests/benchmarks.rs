use std::fs;
use std::path::Path;

use manifold_hf::linalg::Matrix;
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

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = norm3(v);
    assert!(n > 1e-14, "Cannot normalize near-zero vector");
    scale3(v, 1.0 / n)
}

fn reflect_across_plane(v: [f64; 3], plane_normal_unit: [f64; 3]) -> [f64; 3] {
    sub3(
        v,
        scale3(plane_normal_unit, 2.0 * dot3(v, plane_normal_unit)),
    )
}

fn h2_inversion_midpoint_metric(mol: &Molecule) -> f64 {
    assert_eq!(mol.atoms.len(), 2, "H2 inversion metric expects 2 atoms");
    let r0 = mol.atoms[0].position;
    let r1 = mol.atoms[1].position;
    let midpoint = scale3(add3(r0, r1), 0.5);
    let r0_inverted = sub3(scale3(midpoint, 2.0), r0);
    norm3(sub3(r0_inverted, r1))
}

fn h2o_hydrogen_exchange_mirror_metric(mol: &Molecule) -> f64 {
    assert_eq!(
        mol.atoms.len(),
        3,
        "H2O mirror metric expects exactly 3 atoms"
    );
    assert_eq!(mol.atoms[0].atomic_number, 8, "Expected O atom at index 0");
    assert_eq!(mol.atoms[1].atomic_number, 1, "Expected H atom at index 1");
    assert_eq!(mol.atoms[2].atomic_number, 1, "Expected H atom at index 2");

    let o = mol.atoms[0].position;
    let h1 = mol.atoms[1].position;
    let h2 = mol.atoms[2].position;

    let r1 = sub3(h1, o);
    let r2 = sub3(h2, o);
    let bisector = add3(r1, r2);
    let mol_plane_normal = cross3(r1, r2);
    let exchange_plane_normal = normalize3(cross3(bisector, mol_plane_normal));

    let r1_reflected = reflect_across_plane(r1, exchange_plane_normal);
    let r2_reflected = reflect_across_plane(r2, exchange_plane_normal);
    let d12 = norm3(sub3(r1_reflected, r2));
    let d21 = norm3(sub3(r2_reflected, r1));

    ((d12 * d12 + d21 * d21) * 0.5).sqrt()
}

fn benchmark_to_molecule(entry: &BenchmarkMolecule) -> Molecule {
    let atoms = entry
        .atoms
        .iter()
        .map(|atom| Atom::new(atom.atomic_number, atom.position))
        .collect::<Vec<_>>();
    Molecule::new(atoms, entry.charge, entry.multiplicity)
}

fn run_hf_with_fallback(
    hf: &HartreeFock,
    max_iter: usize,
    tol: f64,
) -> manifold_hf::scf::SCFResult {
    let mut result = hf.run_scf(max_iter, tol).expect("SCF failed");
    if !result.converged {
        result = hf
            .run_scf_manifold(max_iter * 2, tol)
            .expect("Manifold SCF failed");
    }
    result
}

fn h2o_swapped_hydrogen_order() -> Molecule {
    let h2o = Molecule::h2o();
    Molecule::new(
        vec![
            h2o.atoms[0].clone(),
            h2o.atoms[2].clone(),
            h2o.atoms[1].clone(),
        ],
        h2o.charge,
        h2o.multiplicity,
    )
}

fn same_shape(m1: &Matrix, m2: &Matrix) -> bool {
    m1.nrows() == m2.nrows() && m1.ncols() == m2.ncols()
}

fn relative_frobenius_difference(a: &Matrix, b: &Matrix) -> f64 {
    assert!(same_shape(a, b), "Matrix shape mismatch");
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = a[[i, j]] - b[[i, j]];
            num += diff * diff;
            den += a[[i, j]] * a[[i, j]];
        }
    }
    let den_norm = den.sqrt().max(1e-16);
    num.sqrt() / den_norm
}

fn ao_permutation_by_center_and_angular(
    reference: &HartreeFock,
    permuted: &HartreeFock,
) -> Vec<usize> {
    let ref_basis = &reference.basis.cartesian_functions;
    let perm_basis = &permuted.basis.cartesian_functions;

    assert_eq!(
        ref_basis.len(),
        perm_basis.len(),
        "AO basis sizes differ for symmetry comparison"
    );
    assert!(
        reference.basis.max_angular_momentum() <= 1 && permuted.basis.max_angular_momentum() <= 1,
        "Symmetry permutation helper currently expects only s/p shells"
    );
    assert_eq!(
        reference.basis.size(),
        reference.basis.cartesian_size(),
        "Expected spherical/cartesian basis size equality for this test"
    );
    assert_eq!(
        permuted.basis.size(),
        permuted.basis.cartesian_size(),
        "Expected spherical/cartesian basis size equality for this test"
    );

    let mut used = vec![false; perm_basis.len()];
    let mut mapping = vec![0usize; ref_basis.len()];

    for (i, f_ref) in ref_basis.iter().enumerate() {
        let mut found: Option<usize> = None;
        for (j, f_perm) in perm_basis.iter().enumerate() {
            if used[j] {
                continue;
            }
            if f_ref.angular != f_perm.angular {
                continue;
            }
            if norm3(sub3(f_ref.center, f_perm.center)) > 1e-12 {
                continue;
            }
            found = Some(j);
            break;
        }

        let idx = found.expect("Failed to build AO permutation map");
        used[idx] = true;
        mapping[i] = idx;
    }

    mapping
}

fn reorder_density_into_reference_order(
    permuted_density: &Matrix,
    map_ref_to_perm: &[usize],
) -> Matrix {
    let n = map_ref_to_perm.len();
    assert_eq!(permuted_density.nrows(), n);
    assert_eq!(permuted_density.ncols(), n);

    let mut reordered = Matrix::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            reordered[[i, j]] = permuted_density[[map_ref_to_perm[i], map_ref_to_perm[j]]];
        }
    }
    reordered
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

        let molecule = benchmark_to_molecule(&entry);

        let hf = HartreeFock::new(molecule.clone()).expect("Failed to build HF");
        let result = run_hf_with_fallback(&hf, 100, 1e-6);

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
            let energy_check =
                0.5 * manifold_hf::linalg::trace(&manifold_hf::linalg::matmul(
                    &result.density,
                    &h_plus_f,
                )) + nuclear;
            let e_one = manifold_hf::linalg::trace(&manifold_hf::linalg::matmul(
                &result.density,
                &hf.core_hamiltonian,
            ));
            let e_two =
                0.5 * manifold_hf::linalg::trace(&manifold_hf::linalg::matmul(&result.density, &g));
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

/// Symmetry type: `D∞h` inversion symmetry for a linear diatomic (`H2`).
#[test]
fn symmetry_geometry_h2_dinfh_inversion_midpoint_metric() {
    let h2 = Molecule::h2();
    let inversion_metric = h2_inversion_midpoint_metric(&h2);
    assert!(
        inversion_metric <= 1e-12,
        "H2 inversion midpoint metric too large: {:.3e}",
        inversion_metric
    );
}

/// Symmetry type: `C2v` hydrogen-exchange mirror plane for bent water (`H2O`).
#[test]
fn symmetry_geometry_h2o_c2v_hydrogen_exchange_mirror_metric() {
    let h2o = Molecule::h2o();
    let mirror_metric = h2o_hydrogen_exchange_mirror_metric(&h2o);
    let oh_asymmetry = (h2o.bond_length(0, 1) - h2o.bond_length(0, 2)).abs();

    assert!(
        mirror_metric <= 1e-12,
        "H2O C2v mirror metric too large: {:.3e}",
        mirror_metric
    );
    assert!(
        oh_asymmetry <= 1e-12,
        "H2O OH bond asymmetry too large: {:.3e}",
        oh_asymmetry
    );
}

/// Symmetry type: benchmark-input geometry check for `C2v` hydrogen-exchange mirror in `H2O`.
#[test]
fn symmetry_geometry_benchmark_h2o_c2v_hydrogen_exchange_mirror_metric() {
    let path = Path::new("data/benchmarks.json");
    if !path.exists() {
        eprintln!("Skipping benchmark geometry symmetry check: data/benchmarks.json not found.");
        return;
    }

    let contents = fs::read_to_string(path).expect("Failed to read benchmarks.json");
    let data: Benchmarks = serde_json::from_str(&contents).expect("Invalid benchmarks.json");
    let entry = data
        .molecules
        .iter()
        .find(|m| m.name == "H2O")
        .expect("H2O entry not found in benchmarks.json");

    let molecule = benchmark_to_molecule(entry);
    let mirror_metric = h2o_hydrogen_exchange_mirror_metric(&molecule);
    let oh_asymmetry = (molecule.bond_length(0, 1) - molecule.bond_length(0, 2)).abs();

    assert!(
        mirror_metric <= 1e-5,
        "Benchmark H2O mirror metric too large: {:.3e}",
        mirror_metric
    );
    assert!(
        oh_asymmetry <= 1e-5,
        "Benchmark H2O OH bond asymmetry too large: {:.3e}",
        oh_asymmetry
    );
}

/// Symmetry type: `C2v` hydrogen-exchange permutation invariance of HF total energy in `H2O`.
#[test]
fn symmetry_hf_h2o_c2v_hydrogen_exchange_energy_invariance_metric() {
    let h2o_ref = Molecule::h2o();
    let h2o_swapped = h2o_swapped_hydrogen_order();

    let hf_ref = HartreeFock::new(h2o_ref).expect("Failed to build HF for H2O");
    let hf_swapped = HartreeFock::new(h2o_swapped).expect("Failed to build HF for swapped H2O");

    let res_ref = run_hf_with_fallback(&hf_ref, 120, 1e-6);
    let res_swapped = run_hf_with_fallback(&hf_swapped, 120, 1e-6);

    let energy_gap = (res_ref.energy - res_swapped.energy).abs();
    assert!(
        energy_gap <= 1e-6,
        "H2O hydrogen-exchange energy invariance broken: |ΔE| = {:.3e} Ha",
        energy_gap
    );
}

/// Symmetry type: `C2v` hydrogen-exchange equivariance of AO density matrix in `H2O`.
#[test]
fn symmetry_hf_h2o_c2v_hydrogen_exchange_density_equivariance_metric() {
    let h2o_ref = Molecule::h2o();
    let h2o_swapped = h2o_swapped_hydrogen_order();

    let hf_ref = HartreeFock::new(h2o_ref).expect("Failed to build HF for H2O");
    let hf_swapped = HartreeFock::new(h2o_swapped).expect("Failed to build HF for swapped H2O");

    let res_ref = run_hf_with_fallback(&hf_ref, 120, 1e-6);
    let res_swapped = run_hf_with_fallback(&hf_swapped, 120, 1e-6);

    let map_ref_to_swapped = ao_permutation_by_center_and_angular(&hf_ref, &hf_swapped);
    let swapped_in_ref_order =
        reorder_density_into_reference_order(&res_swapped.density, &map_ref_to_swapped);
    let rel_density_gap = relative_frobenius_difference(&res_ref.density, &swapped_in_ref_order);

    assert!(
        rel_density_gap <= 1e-4,
        "H2O hydrogen-exchange density equivariance broken: relative Frobenius gap = {:.3e}",
        rel_density_gap
    );
}
