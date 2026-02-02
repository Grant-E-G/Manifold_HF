//! Render a molecule diagram as SVG.

use manifold_hf::{
    DiagramOptions, HartreeFock, LengthUnit, Molecule, OrbitalSettings, ProjectionPlane,
    render_molecule_svg, render_molecule_svg_with_orbital, render_molecule_svg_with_orbitals,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut molecule_name = "h2o".to_string();
    let mut output = "molecule.svg".to_string();
    let mut orbital_index: Option<usize> = None;
    let mut auto_expand_bounds = false;
    let mut all_orbitals = false;
    let mut title_override: Option<String> = None;
    let mut description_override: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "h2" | "h2o" | "d2o" => {
                molecule_name = args[i].to_lowercase();
            }
            "--out" => {
                if let Some(path) = args.get(i + 1) {
                    output = path.clone();
                    i += 1;
                }
            }
            "--orbital" => {
                if let Some(index) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    orbital_index = Some(index);
                    i += 1;
                }
            }
            "--auto-bounds" | "--orbital-auto-bounds" => {
                auto_expand_bounds = true;
            }
            "--all-orbitals" => {
                all_orbitals = true;
            }
            "--title" => {
                if let Some(value) = args.get(i + 1) {
                    title_override = Some(value.clone());
                    i += 1;
                }
            }
            "--description" | "--desc" => {
                if let Some(value) = args.get(i + 1) {
                    description_override = Some(value.clone());
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

    let molecule = match molecule_name.as_str() {
        "h2" => Molecule::h2(),
        "d2o" => Molecule::d2o(),
        _ => Molecule::h2o(),
    };

    let mut options = DiagramOptions::default();
    options.projection = ProjectionPlane::YZ;
    options.length_unit = LengthUnit::Angstrom;
    options.annotate_bonds = true;
    options.annotate_angles = true;

    let formula = molecule_name.to_uppercase();
    let projection_label = match options.projection {
        ProjectionPlane::XY => "XY",
        ProjectionPlane::XZ => "XZ",
        ProjectionPlane::YZ => "YZ",
    };
    let unit_label = match options.length_unit {
        LengthUnit::Bohr => "Bohr",
        LengthUnit::Angstrom => "Angstrom",
    };
    let (default_title, default_description) = if all_orbitals {
        (
            format!("{} Orbitals", formula),
            format!(
                "All molecular orbitals overlaid (projection {}, units {}).",
                projection_label, unit_label
            ),
        )
    } else if let Some(index) = orbital_index {
        (
            format!("{} Orbital {}", formula, index),
            format!(
                "Molecular orbital {} (projection {}, units {}).",
                index, projection_label, unit_label
            ),
        )
    } else {
        (
            format!("{} Molecule", formula),
            format!(
                "Molecular geometry (projection {}, units {}).",
                projection_label, unit_label
            ),
        )
    };
    options.title = Some(title_override.unwrap_or(default_title));
    options.description = Some(description_override.unwrap_or(default_description));

    let svg = if all_orbitals {
        let hf = HartreeFock::new(molecule.clone());
        let result = hf
            .run_scf(100, 1e-6)
            .expect("SCF failed while preparing orbital visualization");
        let orbital = OrbitalSettings {
            auto_expand_bounds,
            ..Default::default()
        };
        let orbital_indices: Vec<usize> = (0..result.coefficients.ncols()).collect();
        render_molecule_svg_with_orbitals(
            &molecule,
            &hf.basis,
            &result.coefficients,
            &options,
            &orbital_indices,
            &orbital,
        )
        .expect("Failed to render SVG with orbitals")
    } else if let Some(index) = orbital_index {
        let hf = HartreeFock::new(molecule.clone());
        let result = hf
            .run_scf(100, 1e-6)
            .expect("SCF failed while preparing orbital visualization");
        let orbital = OrbitalSettings {
            orbital_index: index,
            auto_expand_bounds,
            ..Default::default()
        };
        render_molecule_svg_with_orbital(
            &molecule,
            &hf.basis,
            &result.coefficients,
            &options,
            &orbital,
        )
        .expect("Failed to render SVG with orbital")
    } else {
        render_molecule_svg(&molecule, &options).expect("Failed to render SVG")
    };

    std::fs::write(&output, svg).expect("Failed to write SVG output");
    println!("Wrote {}", output);
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --example render_molecule -- [h2|h2o|d2o] [--out FILE.svg] [--orbital N] [--all-orbitals] [--auto-bounds] [--title TEXT] [--description TEXT]"
    );
}
