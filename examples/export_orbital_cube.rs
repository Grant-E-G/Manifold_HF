//! Export molecular orbital values on a 3D grid as Gaussian cube files.
//!
//! These cube files can be visualized with standard tools (e.g. VMD) or with the
//! provided Plotly helper script in `scripts/visualize_orbitals_3d.py`.

use manifold_hf::molecule::Atom;
use manifold_hf::{HartreeFock, Molecule};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct BenchmarksFile {
    molecules: Vec<BenchmarkMolecule>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkMolecule {
    name: String,
    charge: i32,
    multiplicity: u32,
    atoms: Vec<BenchmarkAtom>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkAtom {
    atomic_number: u32,
    position: [f64; 3], // Bohr
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut molecule_name = "h2o".to_string();
    let mut benchmarks_path: Option<String> = None;
    let mut benchmarks_name: Option<String> = None;
    let mut output = "orbital.cube".to_string();
    let mut orbital_indices: Vec<usize> = Vec::new();
    let mut all_orbitals = false;
    let mut grid: usize = 40;
    let mut padding_bohr: f64 = 5.0;
    let mut auto_expand = false;
    let mut auto_expand_tol: f64 = 1e-4;
    let mut auto_expand_step: f64 = 1.0;
    let mut auto_expand_max_padding: f64 = 12.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "h2" | "h2o" | "d2o" => {
                molecule_name = args[i].to_lowercase();
            }
            "--benchmarks" => {
                if let Some(path) = args.get(i + 1) {
                    benchmarks_path = Some(path.clone());
                    i += 1;
                }
            }
            "--name" => {
                if let Some(name) = args.get(i + 1) {
                    benchmarks_name = Some(name.clone());
                    i += 1;
                }
            }
            "--out" => {
                if let Some(path) = args.get(i + 1) {
                    output = path.clone();
                    i += 1;
                }
            }
            "--orbital" | "--orbitals" => {
                if let Some(value) = args.get(i + 1) {
                    for part in value.split(',') {
                        if part.trim().is_empty() {
                            continue;
                        }
                        let idx: usize = part
                            .trim()
                            .parse()
                            .unwrap_or_else(|_| panic!("Invalid orbital index: {}", part));
                        orbital_indices.push(idx);
                    }
                    i += 1;
                }
            }
            "--all-orbitals" => {
                all_orbitals = true;
            }
            "--auto-expand" | "--auto-padding" => {
                auto_expand = true;
            }
            "--auto-expand-tol" => {
                if let Some(value) = args.get(i + 1) {
                    auto_expand_tol = value
                        .parse()
                        .unwrap_or_else(|_| panic!("Invalid --auto-expand-tol value: {}", value));
                    i += 1;
                }
            }
            "--auto-expand-step" => {
                if let Some(value) = args.get(i + 1) {
                    auto_expand_step = value
                        .parse()
                        .unwrap_or_else(|_| panic!("Invalid --auto-expand-step value: {}", value));
                    i += 1;
                }
            }
            "--auto-expand-max-padding" => {
                if let Some(value) = args.get(i + 1) {
                    auto_expand_max_padding = value.parse().unwrap_or_else(|_| {
                        panic!("Invalid --auto-expand-max-padding value: {}", value)
                    });
                    i += 1;
                }
            }
            "--grid" => {
                if let Some(value) = args.get(i + 1) {
                    grid = value
                        .parse()
                        .unwrap_or_else(|_| panic!("Invalid --grid value: {}", value));
                    i += 1;
                }
            }
            "--padding" => {
                if let Some(value) = args.get(i + 1) {
                    padding_bohr = value
                        .parse()
                        .unwrap_or_else(|_| panic!("Invalid --padding value: {}", value));
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                return;
            }
        }
        i += 1;
    }

    let molecule = if let Some(path) = benchmarks_path.as_deref() {
        let name = benchmarks_name
            .as_deref()
            .unwrap_or_else(|| panic!("--name is required when using --benchmarks"));
        load_molecule_from_benchmarks(path, name)
            .unwrap_or_else(|err| panic!("Failed to load benchmarks molecule: {}", err))
    } else {
        match molecule_name.as_str() {
            "h2" => Molecule::h2(),
            "d2o" => Molecule::d2o(),
            _ => Molecule::h2o(),
        }
    };

    let hf = HartreeFock::new(molecule.clone()).expect("Failed to build STO-3G basis");
    let mut result = hf
        .run_scf(100, 1e-6)
        .expect("SCF failed while preparing cube export");
    if !result.converged {
        result = hf
            .run_scf_manifold(200, 1e-6)
            .expect("Manifold SCF failed while preparing cube export");
    }

    let n_orb = result.coefficients.ncols();
    let n_occ = molecule.num_occupied();
    println!(
        "SCF done: n_orb={}, n_occ={} (occupied orbitals: 0..{}), converged={}",
        n_orb,
        n_occ,
        n_occ.saturating_sub(1),
        result.converged
    );
    let orbitals: Vec<usize> = if all_orbitals {
        (0..n_orb).collect()
    } else if !orbital_indices.is_empty() {
        orbital_indices
    } else {
        vec![0]
    };

    let coeffs_cart = hf.basis.to_cartesian_coefficients(&result.coefficients);

    if auto_expand {
        let mut padding = padding_bohr;
        if auto_expand_step <= 0.0 {
            panic!("--auto-expand-step must be > 0");
        }
        if auto_expand_max_padding < padding {
            panic!("--auto-expand-max-padding must be >= --padding");
        }

        loop {
            let (origin, dx, dy, dz, nx, ny, nz) = grid_from_molecule(&molecule, grid, padding);
            let worst = worst_boundary_ratio(
                &hf,
                &coeffs_cart,
                &orbitals,
                origin,
                dx,
                dy,
                dz,
                nx,
                ny,
                nz,
            );
            println!(
                "Auto-expand: padding={:.3} Bohr worst_boundary_ratio={:.3e} (tol={:.1e})",
                padding, worst, auto_expand_tol
            );
            if worst <= auto_expand_tol || padding >= auto_expand_max_padding - 1e-12 {
                break;
            }
            padding += auto_expand_step;
            if padding > auto_expand_max_padding {
                padding = auto_expand_max_padding;
            }
        }

        padding_bohr = padding;
        println!("Using padding {:.3} Bohr", padding_bohr);
    }

    let multi = orbitals.len() > 1;
    for idx in orbitals.iter().copied() {
        if idx >= n_orb {
            panic!("Orbital index {} out of range (n_orb={})", idx, n_orb);
        }
        let path = output_path_for_orbital(&output, idx, multi);
        write_orbital_cube(&path, &molecule, &hf, &result.coefficients, idx, grid, padding_bohr)
            .unwrap_or_else(|err| panic!("Failed to write cube {}: {}", path, err));
        println!("Wrote {}", path);
    }
}

fn worst_boundary_ratio(
    hf: &HartreeFock,
    coeffs_cart: &manifold_hf::Matrix,
    orbitals: &[usize],
    origin: [f64; 3],
    dx: f64,
    dy: f64,
    dz: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> f64 {
    let n_cart = hf.basis.cartesian_size();
    let mut max_abs = vec![0.0f64; orbitals.len()];
    let mut boundary_abs = vec![0.0f64; orbitals.len()];
    let mut vals = vec![0.0f64; orbitals.len()];

    for iz in 0..nz {
        let z = origin[2] + dz * iz as f64;
        let on_z = iz == 0 || iz + 1 == nz;
        for iy in 0..ny {
            let y = origin[1] + dy * iy as f64;
            let on_y = iy == 0 || iy + 1 == ny;
            for ix in 0..nx {
                let x = origin[0] + dx * ix as f64;
                let on_x = ix == 0 || ix + 1 == nx;
                let is_boundary = on_x || on_y || on_z;
                let point = [x, y, z];

                vals.as_mut_slice().fill(0.0);
                for mu in 0..n_cart {
                    let phi = hf.basis.cartesian_functions[mu].evaluate(point);
                    if phi.abs() < 1e-18 {
                        continue;
                    }
                    for (k, &orb) in orbitals.iter().enumerate() {
                        let c = coeffs_cart[[mu, orb]];
                        if c.abs() < 1e-12 {
                            continue;
                        }
                        vals[k] += c * phi;
                    }
                }

                for k in 0..orbitals.len() {
                    let a = vals[k].abs();
                    if a > max_abs[k] {
                        max_abs[k] = a;
                    }
                    if is_boundary && a > boundary_abs[k] {
                        boundary_abs[k] = a;
                    }
                }
            }
        }
    }

    let mut worst: f64 = 0.0;
    for k in 0..orbitals.len() {
        if max_abs[k] > 0.0 {
            worst = worst.max(boundary_abs[k] / max_abs[k]);
        }
    }
    worst
}

fn print_usage() {
    eprintln!(
        "Usage:\n  cargo run --example export_orbital_cube -- [h2|h2o|d2o] [--out FILE.cube] [--orbital N[,M..]] [--all-orbitals] [--grid N] [--padding BOHR] [--auto-expand]\n\nAuto-expand options:\n  --auto-expand-tol <f64>          (default 1e-4)\n  --auto-expand-step <f64>         (default 1.0)\n  --auto-expand-max-padding <f64>  (default 12.0)\n\nBenchmark geometry:\n  cargo run --example export_orbital_cube -- --benchmarks data/benchmarks.json --name H2O --orbital 0 --out h2o.cube\n"
    );
}

fn load_molecule_from_benchmarks(path: &str, name: &str) -> Result<Molecule, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let data: BenchmarksFile = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let entry = data
        .molecules
        .iter()
        .find(|m| m.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| format!("Molecule '{}' not found in {}", name, path))?;

    let atoms = entry
        .atoms
        .iter()
        .map(|a| Atom::new(a.atomic_number, a.position))
        .collect::<Vec<_>>();
    Ok(Molecule::new(atoms, entry.charge, entry.multiplicity))
}

fn output_path_for_orbital(base: &str, orbital: usize, force_suffix: bool) -> String {
    if force_suffix {
        if let Some((prefix, _)) = base.rsplit_once(".cube") {
            format!("{}_orb{}.cube", prefix, orbital)
        } else {
            format!("{}_orb{}.cube", base, orbital)
        }
    } else {
        base.to_string()
    }
}

fn write_orbital_cube(
    path: &str,
    molecule: &Molecule,
    hf: &HartreeFock,
    coefficients_sph: &manifold_hf::Matrix,
    orbital_index: usize,
    grid: usize,
    padding_bohr: f64,
) -> Result<(), String> {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let (origin, dx, dy, dz, nx, ny, nz) = grid_from_molecule(molecule, grid, padding_bohr);

    let coeffs_cart = hf.basis.to_cartesian_coefficients(coefficients_sph);
    let n_cart = hf.basis.cartesian_size();

    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut w = BufWriter::new(file);

    writeln!(w, "Manifold_HF cube file (Bohr)").map_err(|e| e.to_string())?;
    let n_occ = molecule.num_occupied();
    let tag = if orbital_index + 1 == n_occ {
        "HOMO"
    } else if orbital_index == n_occ {
        "LUMO"
    } else if orbital_index < n_occ {
        "occ"
    } else {
        "vir"
    };
    writeln!(
        w,
        "Orbital {} ({}) on {}x{}x{} grid (padding {:.3} Bohr)",
        orbital_index, tag, nx, ny, nz, padding_bohr
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        w,
        "{:5} {:12.6} {:12.6} {:12.6}",
        molecule.atoms.len(),
        origin[0],
        origin[1],
        origin[2]
    )
    .map_err(|e| e.to_string())?;
    writeln!(w, "{:5} {:12.6} {:12.6} {:12.6}", nx, dx, 0.0, 0.0).map_err(|e| e.to_string())?;
    writeln!(w, "{:5} {:12.6} {:12.6} {:12.6}", ny, 0.0, dy, 0.0).map_err(|e| e.to_string())?;
    writeln!(w, "{:5} {:12.6} {:12.6} {:12.6}", nz, 0.0, 0.0, dz).map_err(|e| e.to_string())?;

    for atom in &molecule.atoms {
        writeln!(
            w,
            "{:5} {:12.6} {:12.6} {:12.6} {:12.6}",
            atom.atomic_number,
            atom.nuclear_charge(),
            atom.position[0],
            atom.position[1],
            atom.position[2]
        )
        .map_err(|e| e.to_string())?;
    }

    let mut values_on_line = 0usize;
    for iz in 0..nz {
        let z = origin[2] + dz * iz as f64;
        for iy in 0..ny {
            let y = origin[1] + dy * iy as f64;
            for ix in 0..nx {
                let x = origin[0] + dx * ix as f64;
                let point = [x, y, z];
                let mut value = 0.0;
                for mu in 0..n_cart {
                    let c = coeffs_cart[[mu, orbital_index]];
                    if c.abs() < 1e-12 {
                        continue;
                    }
                    value += c * hf.basis.cartesian_functions[mu].evaluate(point);
                }

                write!(w, " {:13.5e}", value).map_err(|e| e.to_string())?;
                values_on_line += 1;
                if values_on_line == 6 {
                    writeln!(w).map_err(|e| e.to_string())?;
                    values_on_line = 0;
                }
            }
        }
    }
    if values_on_line != 0 {
        writeln!(w).map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn grid_from_molecule(
    molecule: &Molecule,
    grid: usize,
    padding_bohr: f64,
) -> ([f64; 3], f64, f64, f64, usize, usize, usize) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for atom in &molecule.atoms {
        for d in 0..3 {
            min[d] = min[d].min(atom.position[d]);
            max[d] = max[d].max(atom.position[d]);
        }
    }

    for d in 0..3 {
        min[d] -= padding_bohr;
        max[d] += padding_bohr;
        if (max[d] - min[d]).abs() < 1e-8 {
            min[d] -= padding_bohr;
            max[d] += padding_bohr;
        }
    }

    let nx = grid.max(10);
    let ny = grid.max(10);
    let nz = grid.max(10);
    let dx = (max[0] - min[0]) / (nx as f64 - 1.0);
    let dy = (max[1] - min[1]) / (ny as f64 - 1.0);
    let dz = (max[2] - min[2]) / (nz as f64 - 1.0);
    (min, dx, dy, dz, nx, ny, nz)
}
