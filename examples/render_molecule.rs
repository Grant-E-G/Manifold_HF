//! Render a molecule diagram as SVG.

use manifold_hf::basis::BasisSet;
use manifold_hf::{
    render_molecule_svg, render_molecule_svg_with_density, render_molecule_svg_with_orbital,
    render_molecule_svg_with_orbitals, DensitySettings, DiagramMetric, DiagramOptions, HartreeFock,
    LengthUnit, Matrix, Molecule, OrbitalSettings, ProjectionPlane,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    BondingOrbitals,
    SingleOrbital,
    AllOrbitals,
    Density,
    GeometryOnly,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut molecule_name = "h2o".to_string();
    let mut output = "molecule.svg".to_string();
    let mut render_mode = RenderMode::BondingOrbitals;
    let mut orbital_index: usize = 0;
    let mut auto_expand_bounds = false;
    let mut debug_metrics = false;
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
                    orbital_index = index;
                    render_mode = RenderMode::SingleOrbital;
                    i += 1;
                }
            }
            "--auto-bounds" | "--orbital-auto-bounds" => {
                auto_expand_bounds = true;
            }
            "--bonding-orbitals" => {
                render_mode = RenderMode::BondingOrbitals;
            }
            "--all-orbitals" => {
                render_mode = RenderMode::AllOrbitals;
            }
            "--density" => {
                render_mode = RenderMode::Density;
            }
            "--no-density" | "--geometry-only" => {
                render_mode = RenderMode::GeometryOnly;
            }
            "--debug-metrics" => {
                debug_metrics = true;
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
    options.width = 1280;
    options.height = 840;
    options.projection = ProjectionPlane::YZ;
    options.length_unit = LengthUnit::Angstrom;
    options.annotate_bonds = true;
    options.annotate_angles = true;
    options.show_metrics_panel = true;
    options.metrics = build_visual_metrics(&molecule, molecule_name.as_str(), debug_metrics);

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
    let (default_title, default_description) = match render_mode {
        RenderMode::AllOrbitals => (
            format!("{} Orbitals", formula),
            format!(
                "All molecular orbitals overlaid (projection {}, units {}).",
                projection_label, unit_label
            ),
        ),
        RenderMode::SingleOrbital => (
            format!("{} Orbital {}", formula, orbital_index),
            format!(
                "Molecular orbital {} (projection {}, units {}).",
                orbital_index, projection_label, unit_label
            ),
        ),
        RenderMode::BondingOrbitals => (
            format!("{} Bonding Orbitals", formula),
            format!(
                "Valence occupied orbitals used for bonding/geometry overlays (projection {}, units {}).",
                projection_label, unit_label
            ),
        ),
        RenderMode::Density => (
            format!("{} Electron Density", formula),
            format!(
                "HF electron probability density on a {} plane (units {}).",
                projection_label, unit_label
            ),
        ),
        RenderMode::GeometryOnly => (
            format!("{} Molecule", formula),
            format!(
                "Molecular geometry (projection {}, units {}).",
                projection_label, unit_label
            ),
        ),
    };
    options.title = Some(title_override.unwrap_or(default_title));
    options.description = Some(description_override.unwrap_or(default_description));

    let svg = render_svg(
        &molecule,
        &mut options,
        render_mode,
        orbital_index,
        auto_expand_bounds,
    );

    std::fs::write(&output, svg).expect("Failed to write SVG output");
    println!("Wrote {}", output);
}

fn render_svg(
    molecule: &Molecule,
    options: &mut DiagramOptions,
    render_mode: RenderMode,
    orbital_index: usize,
    auto_expand_bounds: bool,
) -> String {
    match render_mode {
        RenderMode::AllOrbitals | RenderMode::SingleOrbital | RenderMode::BondingOrbitals => {
            render_orbital_visualization(
                molecule,
                options,
                render_mode,
                orbital_index,
                auto_expand_bounds,
            )
        }
        RenderMode::Density => render_density_visualization(molecule, options, auto_expand_bounds),
        RenderMode::GeometryOnly => {
            append_geometry_source_metric(&mut options.metrics, "nuclear positions");
            render_molecule_svg(molecule, options).expect("Failed to render SVG")
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --example render_molecule -- [h2|h2o|d2o] [--out FILE.svg] [--bonding-orbitals] [--orbital N] [--all-orbitals] [--density|--no-density] [--auto-bounds] [--debug-metrics] [--title TEXT] [--description TEXT]"
    );
}

fn render_orbital_visualization(
    molecule: &Molecule,
    options: &mut DiagramOptions,
    render_mode: RenderMode,
    orbital_index: usize,
    auto_expand_bounds: bool,
) -> String {
    let (hf, result) =
        solve_hf_for_visualization(molecule, "SCF failed while preparing orbital visualization");
    let orbital = OrbitalSettings {
        orbital_index: if matches!(render_mode, RenderMode::SingleOrbital) {
            orbital_index
        } else {
            0
        },
        auto_expand_bounds,
        ..Default::default()
    };
    let orbital_indices = match render_mode {
        RenderMode::AllOrbitals => (0..result.coefficients.ncols()).collect(),
        RenderMode::SingleOrbital => vec![orbital_index],
        RenderMode::BondingOrbitals => {
            default_bonding_orbital_indices(molecule, result.coefficients.ncols())
        }
        RenderMode::Density | RenderMode::GeometryOnly => unreachable!(),
    };

    append_geometry_source_metric(&mut options.metrics, "orbital probability");
    options.metrics.push(DiagramMetric::info(
        "MO set",
        format_orbital_index_list(&orbital_indices),
    ));
    append_orbital_exchange_symmetry_metrics(
        &mut options.metrics,
        molecule,
        &hf.basis,
        &result.coefficients,
        &orbital_indices,
        options.projection,
        orbital.plane_offset,
    );

    if matches!(render_mode, RenderMode::SingleOrbital) {
        render_molecule_svg_with_orbital(
            molecule,
            &hf.basis,
            &result.coefficients,
            options,
            &orbital,
        )
        .expect("Failed to render SVG with orbital")
    } else {
        render_molecule_svg_with_orbitals(
            molecule,
            &hf.basis,
            &result.coefficients,
            options,
            &orbital_indices,
            &orbital,
        )
        .expect("Failed to render SVG with orbitals")
    }
}

fn render_density_visualization(
    molecule: &Molecule,
    options: &mut DiagramOptions,
    auto_expand_bounds: bool,
) -> String {
    let (hf, result) =
        solve_hf_for_visualization(molecule, "SCF failed while preparing density visualization");
    let density = DensitySettings {
        auto_expand_bounds,
        ..Default::default()
    };
    append_geometry_source_metric(&mut options.metrics, "nuclear positions");
    render_molecule_svg_with_density(molecule, &hf.basis, &result.density, options, &density)
        .expect("Failed to render SVG with electron density")
}

fn solve_hf_for_visualization(
    molecule: &Molecule,
    scf_error: &str,
) -> (HartreeFock, manifold_hf::scf::SCFResult) {
    let hf = HartreeFock::new(molecule.clone()).expect("Failed to build STO-3G basis");
    let result = hf.run_scf(100, 1e-6).expect(scf_error);
    (hf, result)
}

fn append_geometry_source_metric(metrics: &mut Vec<DiagramMetric>, source: &str) {
    metrics.push(DiagramMetric::info("Geometry source", source));
}

fn default_bonding_orbital_indices(molecule: &Molecule, total_orbitals: usize) -> Vec<usize> {
    let occupied = molecule.num_occupied().min(total_orbitals);
    if occupied == 0 {
        return Vec::new();
    }
    let core_occupied = estimate_core_occupied_orbitals(molecule);
    let start = core_occupied.min(occupied.saturating_sub(1));
    let mut indices: Vec<usize> = (start..occupied).collect();
    if indices.is_empty() {
        indices.push(occupied - 1);
    }
    indices
}

fn estimate_core_occupied_orbitals(molecule: &Molecule) -> usize {
    molecule
        .atoms
        .iter()
        .map(|atom| usize::from(atom.atomic_number > 2))
        .sum()
}

fn format_orbital_index_list(indices: &[usize]) -> String {
    if indices.is_empty() {
        return "none".to_string();
    }
    if indices.len() == 1 {
        return format!("MO {}", indices[0]);
    }
    let joined = indices
        .iter()
        .map(|idx| idx.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("MOs {}", joined)
}

#[derive(Debug, Clone, Copy)]
struct OrbitalExchangeMetric {
    max_residual: f64,
    worst_orbital: usize,
    matched_orbital: usize,
}

fn append_orbital_exchange_symmetry_metrics(
    metrics: &mut Vec<DiagramMetric>,
    molecule: &Molecule,
    basis: &BasisSet,
    coefficients: &Matrix,
    orbital_indices: &[usize],
    projection: ProjectionPlane,
    plane_offset: f64,
) {
    if !(molecule.atoms.len() == 3 && water_like_indices(molecule).is_some()) {
        return;
    }
    if orbital_indices.len() < 2 {
        metrics.push(DiagramMetric::warning(
            "Orb exch max residual (Bohr)",
            "need >=2 orbitals",
        ));
        return;
    }
    match h2o_orbital_exchange_max_residual_metric(
        molecule,
        basis,
        coefficients,
        orbital_indices,
        projection,
        plane_offset,
    ) {
        Some(metric) => {
            metrics.push(residual_metric(
                "Orb exch max residual (Bohr)",
                metric.max_residual,
                5e-3,
                5e-2,
            ));
            metrics.push(DiagramMetric::info(
                "Orb exch worst pair",
                format!(
                    "MO {} -> MO {}",
                    metric.worst_orbital, metric.matched_orbital
                ),
            ));
        }
        None => {
            metrics.push(DiagramMetric::warning(
                "Orb exch max residual (Bohr)",
                "unable to evaluate",
            ));
        }
    }
}

fn h2o_orbital_exchange_max_residual_metric(
    molecule: &Molecule,
    basis: &BasisSet,
    coefficients: &Matrix,
    orbital_indices: &[usize],
    projection: ProjectionPlane,
    plane_offset: f64,
) -> Option<OrbitalExchangeMetric> {
    let (oxygen, h1, h2) = water_like_indices(molecule)?;
    if coefficients.nrows() != basis.size() {
        return None;
    }

    let o = molecule.atoms[oxygen].position;
    let r1 = sub3(molecule.atoms[h1].position, o);
    let r2 = sub3(molecule.atoms[h2].position, o);
    let bisector = add3(r1, r2);
    let plane_normal = cross3(r1, r2);
    let exchange_plane_normal = normalize3(cross3(bisector, plane_normal));

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for atom in &molecule.atoms {
        let p = project3(atom.position, projection);
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    if min_x == max_x {
        min_x -= 1.0;
        max_x += 1.0;
    }
    if min_y == max_y {
        min_y -= 1.0;
        max_y += 1.0;
    }
    let padding = 2.0;
    min_x -= padding;
    max_x += padding;
    min_y -= padding;
    max_y += padding;

    let coeffs_cart = basis.to_cartesian_coefficients(coefficients);
    let centroids: Vec<(usize, [f64; 2])> = orbital_indices
        .iter()
        .filter_map(|&orbital_index| {
            if orbital_index >= coefficients.ncols() {
                return None;
            }
            let centroid = orbital_probability_centroid_projected(
                basis,
                &coeffs_cart,
                orbital_index,
                projection,
                plane_offset,
                [min_x, max_x, min_y, max_y],
                84,
            )?;
            Some((orbital_index, centroid))
        })
        .collect();
    if centroids.is_empty() {
        return None;
    }

    let oxygen_p = project3(molecule.atoms[oxygen].position, projection);
    let h1_p = project3(molecule.atoms[h1].position, projection);
    let h2_p = project3(molecule.atoms[h2].position, projection);

    let mut ranked = centroids
        .iter()
        .enumerate()
        .map(|(idx, (_, c))| {
            let d_o = ((c[0] - oxygen_p[0]).powi(2) + (c[1] - oxygen_p[1]).powi(2)).sqrt();
            let d_h1 = ((c[0] - h1_p[0]).powi(2) + (c[1] - h1_p[1]).powi(2)).sqrt();
            let d_h2 = ((c[0] - h2_p[0]).powi(2) + (c[1] - h2_p[1]).powi(2)).sqrt();
            let h_score = d_o - d_h1.min(d_h2);
            (idx, h_score)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut active = ranked
        .iter()
        .take(2)
        .map(|(idx, _)| centroids[*idx])
        .collect::<Vec<_>>();
    if active.len() < 2 {
        active = centroids.clone();
    }
    if active.len() < 2 {
        return None;
    }

    let mut max_residual = 0.0;
    let mut worst_orbital = active[0].0;
    let mut matched_orbital = active[0].0;

    for (orbital_i, centroid_i) in &active {
        let point_i = unproject3(*centroid_i, projection, plane_offset);
        let reflected_i = add3(
            o,
            reflect_across_plane(sub3(point_i, o), exchange_plane_normal),
        );
        let reflected_i_2d = project3(reflected_i, projection);

        let mut best = f64::INFINITY;
        let mut best_j = *orbital_i;
        for (orbital_j, centroid_j) in &active {
            let d = ((reflected_i_2d[0] - centroid_j[0]).powi(2)
                + (reflected_i_2d[1] - centroid_j[1]).powi(2))
            .sqrt();
            if d < best {
                best = d;
                best_j = *orbital_j;
            }
        }

        if best > max_residual {
            max_residual = best;
            worst_orbital = *orbital_i;
            matched_orbital = best_j;
        }
    }

    Some(OrbitalExchangeMetric {
        max_residual,
        worst_orbital,
        matched_orbital,
    })
}

fn orbital_probability_centroid_projected(
    basis: &BasisSet,
    coeffs_cart: &Matrix,
    orbital_index: usize,
    projection: ProjectionPlane,
    plane_offset: f64,
    bounds: [f64; 4],
    grid: usize,
) -> Option<[f64; 2]> {
    let n_cart = basis.cartesian_size();
    if coeffs_cart.nrows() != n_cart || orbital_index >= coeffs_cart.ncols() {
        return None;
    }

    let grid = grid.max(24);
    let min_x = bounds[0];
    let max_x = bounds[1];
    let min_y = bounds[2];
    let max_y = bounds[3];
    let dx = (max_x - min_x) / (grid as f64 - 1.0);
    let dy = (max_y - min_y) / (grid as f64 - 1.0);

    let mut sum_w = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for j in 0..grid {
        for i in 0..grid {
            let x = min_x + dx * i as f64;
            let y = min_y + dy * j as f64;
            let point = unproject3([x, y], projection, plane_offset);
            let mut value = 0.0;
            for k in 0..n_cart {
                let coeff = coeffs_cart[[k, orbital_index]];
                if coeff.abs() < 1e-10 {
                    continue;
                }
                value += coeff * basis.cartesian_functions[k].evaluate(point);
            }
            let w = value * value;
            if w < 1e-14 {
                continue;
            }
            sum_w += w;
            sum_x += x * w;
            sum_y += y * w;
        }
    }

    if sum_w <= 1e-16 {
        return None;
    }
    Some([sum_x / sum_w, sum_y / sum_w])
}

fn project3(position: [f64; 3], plane: ProjectionPlane) -> [f64; 2] {
    match plane {
        ProjectionPlane::XY => [position[0], position[1]],
        ProjectionPlane::XZ => [position[0], position[2]],
        ProjectionPlane::YZ => [position[1], position[2]],
    }
}

fn unproject3(position: [f64; 2], plane: ProjectionPlane, offset: f64) -> [f64; 3] {
    match plane {
        ProjectionPlane::XY => [position[0], position[1], offset],
        ProjectionPlane::XZ => [position[0], offset, position[1]],
        ProjectionPlane::YZ => [offset, position[0], position[1]],
    }
}

fn build_visual_metrics(
    molecule: &Molecule,
    molecule_name: &str,
    include_hf_debug: bool,
) -> Vec<DiagramMetric> {
    let mut metrics = Vec::new();

    match molecule_name {
        "h2" => {
            metrics.push(DiagramMetric::info("Symmetry", "Dinfh linear inversion"));
            let inversion = h2_inversion_midpoint_metric(molecule);
            metrics.push(residual_metric(
                "Inversion residual",
                inversion,
                1e-12,
                1e-8,
            ));
        }
        "h2o" | "d2o" => {
            metrics.push(DiagramMetric::info(
                "Symmetry",
                "C2v bent H-exchange mirror",
            ));
            let mirror = h2o_hydrogen_exchange_mirror_metric(molecule);
            metrics.push(residual_metric(
                "Nuclear mirror residual",
                mirror,
                1e-12,
                1e-7,
            ));

            if let Some((oxygen, h1, h2)) = water_like_indices(molecule) {
                let oh_asym =
                    (molecule.bond_length(oxygen, h1) - molecule.bond_length(oxygen, h2)).abs();
                metrics.push(residual_metric("OH asymmetry", oh_asym, 1e-12, 1e-7));
            }

            if include_hf_debug {
                metrics.push(DiagramMetric::info("HF diagnostics", "H-order invariance"));
                match h2o_swapped_hydrogen_order(molecule) {
                    Some(swapped) => match (
                        HartreeFock::new(molecule.clone()),
                        HartreeFock::new(swapped),
                    ) {
                        (Ok(hf_ref), Ok(hf_swapped)) => {
                            match (
                                run_hf_with_fallback(&hf_ref, 120, 1e-6),
                                run_hf_with_fallback(&hf_swapped, 120, 1e-6),
                            ) {
                                (
                                    Ok((e_ref, d_ref, fallback_ref)),
                                    Ok((e_swap, d_swap, fallback_swap)),
                                ) => {
                                    let d_e = (e_ref - e_swap).abs();
                                    metrics.push(residual_metric("HF |delta E|", d_e, 1e-9, 1e-6));
                                    let density_gap =
                                        relative_frobenius_difference(&d_ref, &d_swap);
                                    metrics.push(residual_metric(
                                        "Density gap",
                                        density_gap,
                                        1e-8,
                                        1e-4,
                                    ));
                                    metrics.push(scf_mode_metric("SCF reference", fallback_ref));
                                    metrics.push(scf_mode_metric("SCF swapped", fallback_swap));
                                }
                                (Err(err), _) | (_, Err(err)) => {
                                    metrics.push(DiagramMetric::critical("HF diagnostic", err));
                                }
                            }
                        }
                        (Err(err), _) | (_, Err(err)) => {
                            metrics.push(DiagramMetric::critical("HF setup", err));
                        }
                    },
                    None => {
                        metrics.push(DiagramMetric::warning(
                            "HF diagnostics",
                            "not a 2H+1O water-like geometry",
                        ));
                    }
                }
            }
        }
        _ => {}
    }

    metrics
}

fn residual_metric(label: &str, value: f64, good: f64, warn: f64) -> DiagramMetric {
    let rendered = format!("{:.3e}", value);
    if value <= good {
        DiagramMetric::good(label, rendered)
    } else if value <= warn {
        DiagramMetric::warning(label, rendered)
    } else {
        DiagramMetric::critical(label, rendered)
    }
}

fn scf_mode_metric(label: &str, used_manifold_fallback: bool) -> DiagramMetric {
    if used_manifold_fallback {
        DiagramMetric::warning(label, "manifold fallback")
    } else {
        DiagramMetric::good(label, "standard SCF")
    }
}

fn run_hf_with_fallback(
    hf: &HartreeFock,
    max_iter: usize,
    tol: f64,
) -> Result<(f64, Matrix, bool), String> {
    let mut result = hf.run_scf(max_iter, tol)?;
    let mut used_manifold_fallback = false;
    if !result.converged {
        result = hf.run_scf_manifold(max_iter * 2, tol)?;
        used_manifold_fallback = true;
    }
    Ok((result.energy, result.density, used_manifold_fallback))
}

fn relative_frobenius_difference(a: &Matrix, b: &Matrix) -> f64 {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return f64::INFINITY;
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let diff = a[[i, j]] - b[[i, j]];
            num += diff * diff;
            den += a[[i, j]] * a[[i, j]];
        }
    }
    num.sqrt() / den.sqrt().max(1e-16)
}

fn h2o_swapped_hydrogen_order(molecule: &Molecule) -> Option<Molecule> {
    let (oxygen, h1, h2) = water_like_indices(molecule)?;
    let atoms = vec![
        molecule.atoms[oxygen].clone(),
        molecule.atoms[h2].clone(),
        molecule.atoms[h1].clone(),
    ];
    Some(Molecule::new(atoms, molecule.charge, molecule.multiplicity))
}

fn water_like_indices(molecule: &Molecule) -> Option<(usize, usize, usize)> {
    if molecule.atoms.len() != 3 {
        return None;
    }
    let oxygen = molecule.atoms.iter().position(|a| a.atomic_number == 8)?;
    let hydrogens: Vec<usize> = molecule
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(idx, atom)| (atom.atomic_number == 1).then_some(idx))
        .collect();
    if hydrogens.len() != 2 {
        return None;
    }
    Some((oxygen, hydrogens[0], hydrogens[1]))
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
    let n = norm3(v).max(1e-16);
    scale3(v, 1.0 / n)
}

fn reflect_across_plane(v: [f64; 3], plane_normal_unit: [f64; 3]) -> [f64; 3] {
    sub3(
        v,
        scale3(plane_normal_unit, 2.0 * dot3(v, plane_normal_unit)),
    )
}

fn h2_inversion_midpoint_metric(molecule: &Molecule) -> f64 {
    if molecule.atoms.len() != 2 {
        return f64::INFINITY;
    }
    let r0 = molecule.atoms[0].position;
    let r1 = molecule.atoms[1].position;
    let midpoint = scale3(add3(r0, r1), 0.5);
    let r0_inverted = sub3(scale3(midpoint, 2.0), r0);
    norm3(sub3(r0_inverted, r1))
}

fn h2o_hydrogen_exchange_mirror_metric(molecule: &Molecule) -> f64 {
    let (oxygen, h1, h2) = match water_like_indices(molecule) {
        Some(idxs) => idxs,
        None => return f64::INFINITY,
    };
    let o = molecule.atoms[oxygen].position;
    let r1 = sub3(molecule.atoms[h1].position, o);
    let r2 = sub3(molecule.atoms[h2].position, o);
    let bisector = add3(r1, r2);
    let plane_normal = cross3(r1, r2);
    let exchange_plane_normal = normalize3(cross3(bisector, plane_normal));
    let r1_reflected = reflect_across_plane(r1, exchange_plane_normal);
    let r2_reflected = reflect_across_plane(r2, exchange_plane_normal);
    let d12 = norm3(sub3(r1_reflected, r2));
    let d21 = norm3(sub3(r2_reflected, r1));
    ((d12 * d12 + d21 * d21) * 0.5).sqrt()
}
