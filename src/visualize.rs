//! Molecular visualization utilities
//!
//! Generates simple SVG diagrams for atomic configurations and molecular orbitals.

use crate::basis::BasisSet;
use crate::linalg::Matrix;
use crate::molecule::{Atom, Molecule};

const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_92;

#[derive(Debug, Clone, Copy)]
pub enum ProjectionPlane {
    XY,
    XZ,
    YZ,
}

#[derive(Debug, Clone, Copy)]
pub enum LengthUnit {
    Bohr,
    Angstrom,
}

#[derive(Debug, Clone)]
pub struct DiagramOptions {
    pub width: u32,
    pub height: u32,
    pub padding_px: f64,
    pub projection: ProjectionPlane,
    pub length_unit: LengthUnit,
    pub annotate_bonds: bool,
    pub annotate_angles: bool,
    pub show_atom_labels: bool,
    pub bond_scale: f64,
    pub atom_radius_px: f64,
    pub world_padding_bohr: f64,
    pub background: String,
}

impl Default for DiagramOptions {
    fn default() -> Self {
        Self {
            width: 900,
            height: 650,
            padding_px: 50.0,
            projection: ProjectionPlane::YZ,
            length_unit: LengthUnit::Angstrom,
            annotate_bonds: true,
            annotate_angles: true,
            show_atom_labels: true,
            bond_scale: 1.25,
            atom_radius_px: 14.0,
            world_padding_bohr: 0.6,
            background: "#ffffff".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrbitalSettings {
    pub orbital_index: usize,
    pub grid: usize,
    pub cutoff_fraction: f64,
    pub alpha: f64,
    pub plane_offset: f64,
}

impl Default for OrbitalSettings {
    fn default() -> Self {
        Self {
            orbital_index: 0,
            grid: 90,
            cutoff_fraction: 0.18,
            alpha: 0.6,
            plane_offset: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct Bond {
    i: usize,
    j: usize,
    length_bohr: f64,
}

#[derive(Debug, Clone, Copy)]
struct Vec2 {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy)]
struct Bounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

#[derive(Debug, Clone, Copy)]
struct Transform {
    scale: f64,
    min_x: f64,
    max_y: f64,
    padding_px: f64,
}

impl Transform {
    fn to_screen(&self, p: Vec2) -> Vec2 {
        Vec2 {
            x: self.padding_px + (p.x - self.min_x) * self.scale,
            y: self.padding_px + (self.max_y - p.y) * self.scale,
        }
    }

    fn length_to_screen(&self, len: f64) -> f64 {
        len * self.scale
    }
}

pub fn render_molecule_svg(molecule: &Molecule, options: &DiagramOptions) -> Result<String, String> {
    render_molecule_svg_internal(molecule, options, None, None)
}

pub fn render_molecule_svg_with_orbital(
    molecule: &Molecule,
    basis: &BasisSet,
    coefficients: &Matrix,
    options: &DiagramOptions,
    orbital: &OrbitalSettings,
) -> Result<String, String> {
    render_molecule_svg_internal(
        molecule,
        options,
        Some((basis, coefficients)),
        Some(orbital),
    )
}

fn render_molecule_svg_internal(
    molecule: &Molecule,
    options: &DiagramOptions,
    orbital_basis: Option<(&BasisSet, &Matrix)>,
    orbital_settings: Option<&OrbitalSettings>,
) -> Result<String, String> {
    let projected = project_atoms(molecule, options.projection);
    let bounds = compute_bounds(&projected, options.world_padding_bohr);

    let transform = compute_transform(&bounds, options);
    let bonds = detect_bonds(molecule, options.bond_scale);

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns='http://www.w3.org/2000/svg' width='{}' height='{}' viewBox='0 0 {} {}'>",
        options.width,
        options.height,
        options.width,
        options.height
    ));
    svg.push_str(&format!(
        "<rect width='100%' height='100%' fill='{}'/>",
        options.background
    ));

    if let (Some((basis, coefficients)), Some(settings)) = (orbital_basis, orbital_settings) {
        let field = compute_orbital_field(options, &bounds, basis, coefficients, settings)?;
        svg.push_str("<g opacity='1'>");
        svg.push_str(&field);
        svg.push_str("</g>");
    }

    svg.push_str("<g stroke='#4a4a4a' stroke-width='2' fill='none'>");
    for bond in &bonds {
        let a = transform.to_screen(projected[bond.i]);
        let b = transform.to_screen(projected[bond.j]);
        svg.push_str(&format!(
            "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}'/>",
            a.x, a.y, b.x, b.y
        ));
    }
    svg.push_str("</g>");

    if options.annotate_angles {
        svg.push_str("<g stroke='#555555' stroke-width='1.5' fill='none'>");
        let angle_annotations = angle_annotations(&projected, &bonds, &transform);
        svg.push_str(&angle_annotations.paths);
        svg.push_str("</g>");
        svg.push_str("<g fill='#2b2b2b' font-size='13' font-family='DejaVu Sans, Arial, sans-serif'>");
        svg.push_str(&angle_annotations.labels);
        svg.push_str("</g>");
    }

    if options.annotate_bonds {
        svg.push_str("<g fill='#2b2b2b' font-size='13' font-family='DejaVu Sans, Arial, sans-serif'>");
        for bond in &bonds {
            let a = transform.to_screen(projected[bond.i]);
            let b = transform.to_screen(projected[bond.j]);
            let mid = Vec2 {
                x: (a.x + b.x) * 0.5,
                y: (a.y + b.y) * 0.5,
            };
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let len = (dx * dx + dy * dy).sqrt().max(1.0);
            let nx = -dy / len;
            let ny = dx / len;
            let label_offset = 14.0;
            let label_pos = Vec2 {
                x: mid.x + nx * label_offset,
                y: mid.y + ny * label_offset,
            };
            let label = format_length(bond.length_bohr, options.length_unit);
            svg.push_str(&format!(
                "<text x='{:.2}' y='{:.2}' text-anchor='middle'>{}</text>",
                label_pos.x, label_pos.y, label
            ));
        }
        svg.push_str("</g>");
    }

    svg.push_str("<g stroke='#222222' stroke-width='1.5'>");
    for (idx, atom) in molecule.atoms.iter().enumerate() {
        let center = transform.to_screen(projected[idx]);
        let radius = atom_radius_px(atom, options);
        let fill = atom_color(atom);
        svg.push_str(&format!(
            "<circle cx='{:.2}' cy='{:.2}' r='{:.2}' fill='{}'/>",
            center.x, center.y, radius, fill
        ));
    }
    svg.push_str("</g>");

    if options.show_atom_labels {
        svg.push_str("<g fill='#111111' font-size='14' font-family='DejaVu Sans, Arial, sans-serif'>");
        for (idx, atom) in molecule.atoms.iter().enumerate() {
            let center = transform.to_screen(projected[idx]);
            let label = element_symbol(atom.atomic_number);
            svg.push_str(&format!(
                "<text x='{:.2}' y='{:.2}' text-anchor='middle' dominant-baseline='middle'>{}</text>",
                center.x,
                center.y,
                label
            ));
        }
        svg.push_str("</g>");
    }

    svg.push_str("</svg>");

    Ok(svg)
}

fn project_atoms(molecule: &Molecule, plane: ProjectionPlane) -> Vec<Vec2> {
    molecule
        .atoms
        .iter()
        .map(|atom| project_point(atom.position, plane))
        .collect()
}

fn project_point(position: [f64; 3], plane: ProjectionPlane) -> Vec2 {
    match plane {
        ProjectionPlane::XY => Vec2 {
            x: position[0],
            y: position[1],
        },
        ProjectionPlane::XZ => Vec2 {
            x: position[0],
            y: position[2],
        },
        ProjectionPlane::YZ => Vec2 {
            x: position[1],
            y: position[2],
        },
    }
}

fn compute_bounds(points: &[Vec2], padding: f64) -> Bounds {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    if min_x == max_x {
        min_x -= 1.0;
        max_x += 1.0;
    }
    if min_y == max_y {
        min_y -= 1.0;
        max_y += 1.0;
    }

    Bounds {
        min_x: min_x - padding,
        max_x: max_x + padding,
        min_y: min_y - padding,
        max_y: max_y + padding,
    }
}

fn compute_transform(bounds: &Bounds, options: &DiagramOptions) -> Transform {
    let width = options.width as f64;
    let height = options.height as f64;
    let range_x = (bounds.max_x - bounds.min_x).max(1.0);
    let range_y = (bounds.max_y - bounds.min_y).max(1.0);
    let scale_x = (width - 2.0 * options.padding_px) / range_x;
    let scale_y = (height - 2.0 * options.padding_px) / range_y;

    Transform {
        scale: scale_x.min(scale_y),
        min_x: bounds.min_x,
        max_y: bounds.max_y,
        padding_px: options.padding_px,
    }
}

fn detect_bonds(molecule: &Molecule, scale: f64) -> Vec<Bond> {
    let mut bonds = Vec::new();
    let atoms = &molecule.atoms;

    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let distance = distance_bohr(&atoms[i], &atoms[j]);
            let threshold = bond_threshold_bohr(atoms[i].atomic_number, atoms[j].atomic_number, scale);
            if distance <= threshold {
                bonds.push(Bond {
                    i,
                    j,
                    length_bohr: distance,
                });
            }
        }
    }

    bonds
}

fn distance_bohr(a: &Atom, b: &Atom) -> f64 {
    let dx = a.position[0] - b.position[0];
    let dy = a.position[1] - b.position[1];
    let dz = a.position[2] - b.position[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn bond_threshold_bohr(z1: u32, z2: u32, scale: f64) -> f64 {
    let r1 = covalent_radius_angstrom(z1).unwrap_or(0.75);
    let r2 = covalent_radius_angstrom(z2).unwrap_or(0.75);
    let threshold_angstrom = (r1 + r2) * scale;
    threshold_angstrom / BOHR_TO_ANGSTROM
}

fn covalent_radius_angstrom(z: u32) -> Option<f64> {
    match z {
        1 => Some(0.31),
        2 => Some(0.28),
        5 => Some(0.85),
        6 => Some(0.76),
        7 => Some(0.71),
        8 => Some(0.66),
        9 => Some(0.57),
        10 => Some(0.58),
        _ => None,
    }
}

fn element_symbol(z: u32) -> String {
    match z {
        1 => "H".to_string(),
        2 => "He".to_string(),
        5 => "B".to_string(),
        6 => "C".to_string(),
        7 => "N".to_string(),
        8 => "O".to_string(),
        9 => "F".to_string(),
        10 => "Ne".to_string(),
        _ => format!("Z{}", z),
    }
}

fn atom_color(atom: &Atom) -> String {
    match atom.atomic_number {
        1 => "#f8f8f8".to_string(),
        6 => "#2c2c2c".to_string(),
        7 => "#5dade2".to_string(),
        8 => "#e74c3c".to_string(),
        9 => "#58d68d".to_string(),
        _ => "#b0b0b0".to_string(),
    }
}

fn atom_radius_px(atom: &Atom, options: &DiagramOptions) -> f64 {
    match atom.atomic_number {
        1 => options.atom_radius_px * 0.75,
        8 => options.atom_radius_px * 1.15,
        _ => options.atom_radius_px,
    }
}

fn format_length(length_bohr: f64, unit: LengthUnit) -> String {
    match unit {
        LengthUnit::Bohr => format!("{:.2} Bohr", length_bohr),
        LengthUnit::Angstrom => {
            let angstrom = length_bohr * BOHR_TO_ANGSTROM;
            format!("{:.2} Angstrom", angstrom)
        }
    }
}

struct AngleAnnotations {
    paths: String,
    labels: String,
}

fn angle_annotations(projected: &[Vec2], bonds: &[Bond], transform: &Transform) -> AngleAnnotations {
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); projected.len()];
    for bond in bonds {
        neighbors[bond.i].push(bond.j);
        neighbors[bond.j].push(bond.i);
    }

    let mut paths = String::new();
    let mut labels = String::new();

    for (center_idx, neigh) in neighbors.iter().enumerate() {
        if neigh.len() < 2 {
            continue;
        }
        for i in 0..neigh.len() {
            for j in (i + 1)..neigh.len() {
                let a_idx = neigh[i];
                let b_idx = neigh[j];
                let center = projected[center_idx];
                let a = projected[a_idx];
                let b = projected[b_idx];
                let va = Vec2 {
                    x: a.x - center.x,
                    y: a.y - center.y,
                };
                let vb = Vec2 {
                    x: b.x - center.x,
                    y: b.y - center.y,
                };
                let angle = angle_between(va, vb);
                if !angle.is_finite() {
                    continue;
                }
                let angle_deg = angle.to_degrees();
                let label = format!("{:.1} deg", angle_deg);

                let radius_world = 0.35 * (vec2_len(va).min(vec2_len(vb))).max(0.6);
                let arc = arc_path(center, va, vb, radius_world, transform);
                paths.push_str(&arc);

                let bisector = normalize(Vec2 {
                    x: va.x + vb.x,
                    y: va.y + vb.y,
                });
                let label_world = if bisector.x.is_finite() && bisector.y.is_finite() {
                    Vec2 {
                        x: center.x + bisector.x * radius_world * 1.2,
                        y: center.y + bisector.y * radius_world * 1.2,
                    }
                } else {
                    center
                };
                let label_screen = transform.to_screen(label_world);
                labels.push_str(&format!(
                    "<text x='{:.2}' y='{:.2}' text-anchor='middle'>{}</text>",
                    label_screen.x, label_screen.y, label
                ));
            }
        }
    }

    AngleAnnotations { paths, labels }
}

fn arc_path(center: Vec2, a_vec: Vec2, b_vec: Vec2, radius: f64, transform: &Transform) -> String {
    let a_unit = normalize(a_vec);
    let b_unit = normalize(b_vec);
    if !a_unit.x.is_finite() || !b_unit.x.is_finite() {
        return String::new();
    }
    let start = Vec2 {
        x: center.x + a_unit.x * radius,
        y: center.y + a_unit.y * radius,
    };
    let end = Vec2 {
        x: center.x + b_unit.x * radius,
        y: center.y + b_unit.y * radius,
    };
    let start_screen = transform.to_screen(start);
    let end_screen = transform.to_screen(end);
    let radius_px = transform.length_to_screen(radius);
    let sweep = if cross(a_unit, b_unit) >= 0.0 { 1 } else { 0 };

    format!(
        "<path d='M {:.2} {:.2} A {:.2} {:.2} 0 0 {} {:.2} {:.2}'/>",
        start_screen.x,
        start_screen.y,
        radius_px,
        radius_px,
        sweep,
        end_screen.x,
        end_screen.y
    )
}

fn angle_between(a: Vec2, b: Vec2) -> f64 {
    let denom = vec2_len(a) * vec2_len(b);
    if denom == 0.0 {
        return f64::NAN;
    }
    let mut cos = (a.x * b.x + a.y * b.y) / denom;
    if cos > 1.0 {
        cos = 1.0;
    }
    if cos < -1.0 {
        cos = -1.0;
    }
    cos.acos()
}

fn vec2_len(v: Vec2) -> f64 {
    (v.x * v.x + v.y * v.y).sqrt()
}

fn normalize(v: Vec2) -> Vec2 {
    let len = vec2_len(v);
    if len == 0.0 {
        return Vec2 { x: f64::NAN, y: f64::NAN };
    }
    Vec2 {
        x: v.x / len,
        y: v.y / len,
    }
}

fn cross(a: Vec2, b: Vec2) -> f64 {
    a.x * b.y - a.y * b.x
}

fn compute_orbital_field(
    options: &DiagramOptions,
    bounds: &Bounds,
    basis: &BasisSet,
    coefficients: &Matrix,
    settings: &OrbitalSettings,
) -> Result<String, String> {
    let n_basis = basis.size();
    if settings.orbital_index >= coefficients.ncols() {
        return Err(format!(
            "orbital index {} out of range ({} columns)",
            settings.orbital_index,
            coefficients.ncols()
        ));
    }
    if n_basis != coefficients.nrows() {
        return Err("coefficients matrix size does not match basis".to_string());
    }

    let grid = settings.grid.max(10);
    let dx = (bounds.max_x - bounds.min_x) / (grid as f64 - 1.0);
    let dy = (bounds.max_y - bounds.min_y) / (grid as f64 - 1.0);

    let mut values = Vec::with_capacity(grid * grid);
    let mut max_abs: f64 = 0.0;

    for j in 0..grid {
        for i in 0..grid {
            let x = bounds.min_x + dx * i as f64;
            let y = bounds.min_y + dy * j as f64;
            let point = expand_point(options.projection, x, y, settings.plane_offset);
            let mut value = 0.0;
            for k in 0..n_basis {
                let coeff = coefficients[[k, settings.orbital_index]];
                if coeff.abs() < 1e-10 {
                    continue;
                }
                value += coeff * basis.functions[k].evaluate(point);
            }
            max_abs = max_abs.max(value.abs());
            values.push(value);
        }
    }

    if max_abs == 0.0 {
        return Ok(String::new());
    }

    let transform = compute_transform(bounds, options);
    let cell_w = transform.length_to_screen(dx);
    let cell_h = transform.length_to_screen(dy);
    let cutoff = settings.cutoff_fraction.clamp(0.0, 1.0) * max_abs;

    let mut svg = String::new();

    for (idx, value) in values.iter().enumerate() {
        if value.abs() < cutoff {
            continue;
        }
        let i = idx % grid;
        let j = idx / grid;
        let x = bounds.min_x + dx * i as f64;
        let y = bounds.min_y + dy * j as f64;
        let pos = transform.to_screen(Vec2 { x, y });
        let alpha = (value.abs() / max_abs) * settings.alpha;
        let color = if *value >= 0.0 { "#3b6fb6" } else { "#d1495b" };
        svg.push_str(&format!(
            "<rect x='{:.2}' y='{:.2}' width='{:.2}' height='{:.2}' fill='{}' fill-opacity='{:.3}'/>",
            pos.x - cell_w * 0.5,
            pos.y - cell_h * 0.5,
            cell_w,
            cell_h,
            color,
            alpha.min(0.9)
        ));
    }

    Ok(svg)
}

fn expand_point(plane: ProjectionPlane, x: f64, y: f64, offset: f64) -> [f64; 3] {
    match plane {
        ProjectionPlane::XY => [x, y, offset],
        ProjectionPlane::XZ => [x, offset, y],
        ProjectionPlane::YZ => [offset, x, y],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bond_detection_h2() {
        let mol = Molecule::h2();
        let bonds = detect_bonds(&mol, 1.25);
        assert_eq!(bonds.len(), 1);
        assert!(bonds[0].length_bohr > 0.0);
    }

    #[test]
    fn test_angle_water_projection() {
        let mol = Molecule::h2o();
        let projected = project_atoms(&mol, ProjectionPlane::YZ);
        let bonds = detect_bonds(&mol, 1.25);
        let options = DiagramOptions::default();
        let transform = compute_transform(&compute_bounds(&projected, 0.5), &options);
        let annotations = angle_annotations(&projected, &bonds, &transform);
        assert!(!annotations.labels.is_empty());
    }
}
