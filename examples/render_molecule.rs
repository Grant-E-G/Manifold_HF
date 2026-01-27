//! Render a molecule diagram as SVG.

use manifold_hf::{
    DiagramOptions, HartreeFock, LengthUnit, Molecule, OrbitalSettings, ProjectionPlane,
    render_molecule_svg, render_molecule_svg_with_orbital,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut molecule_name = "h2o".to_string();
    let mut output = "molecule.svg".to_string();
    let mut orbital_index: Option<usize> = None;

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

    let svg = if let Some(index) = orbital_index {
        let hf = HartreeFock::new(molecule.clone());
        let result = hf
            .run_scf(30, 1e-6)
            .expect("SCF failed while preparing orbital visualization");
        let orbital = OrbitalSettings {
            orbital_index: index,
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
        "Usage: cargo run --example render_molecule -- [h2|h2o|d2o] [--out FILE.svg] [--orbital N]"
    );
}
