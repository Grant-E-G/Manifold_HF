//! Molecular visualization utilities
//!
//! Generates simple SVG diagrams for atomic configurations and molecular orbitals.

use crate::basis::BasisSet;
use crate::linalg::Matrix;
use crate::molecule::{Atom, Molecule};
use std::collections::BTreeMap;

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

#[derive(Debug, Clone, Copy)]
pub enum MetricTone {
    Info,
    Good,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct DiagramMetric {
    pub label: String,
    pub value: String,
    pub tone: MetricTone,
}

impl DiagramMetric {
    pub fn info(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            tone: MetricTone::Info,
        }
    }

    pub fn good(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            tone: MetricTone::Good,
        }
    }

    pub fn warning(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            tone: MetricTone::Warning,
        }
    }

    pub fn critical(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            tone: MetricTone::Critical,
        }
    }
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
    pub title: Option<String>,
    pub description: Option<String>,
    pub metrics: Vec<DiagramMetric>,
    pub show_metrics_panel: bool,
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
            world_padding_bohr: 2.0,
            background: "#ffffff".to_string(),
            title: None,
            description: None,
            metrics: Vec::new(),
            show_metrics_panel: true,
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
    pub auto_expand_bounds: bool,
}

impl Default for OrbitalSettings {
    fn default() -> Self {
        Self {
            orbital_index: 0,
            grid: 90,
            cutoff_fraction: 0.18,
            alpha: 0.6,
            plane_offset: 0.0,
            auto_expand_bounds: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DensitySettings {
    pub grid: usize,
    pub cutoff_fraction: f64,
    pub alpha: f64,
    pub plane_offset: f64,
    pub auto_expand_bounds: bool,
    pub color: String,
}

impl Default for DensitySettings {
    fn default() -> Self {
        Self {
            grid: 90,
            cutoff_fraction: 0.03,
            alpha: 0.85,
            plane_offset: 0.0,
            auto_expand_bounds: false,
            color: "#1f4d8b".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
enum OrbitalColorMode {
    Phase,
    Solid(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrbitalRenderStyle {
    Filled,
    Contours,
}

#[derive(Debug, Clone)]
struct OrbitalLayer {
    settings: OrbitalSettings,
    color_mode: OrbitalColorMode,
    render_style: OrbitalRenderStyle,
}

struct OrbitalRenderInfo {
    svg: String,
    centroid: Option<Vec2>,
    legend_color: Option<String>,
    label: String,
}

#[derive(Debug, Clone)]
struct Bond {
    i: usize,
    j: usize,
    length_bohr: f64,
}

#[derive(Debug, Clone)]
struct OrbitalDerivedGeometry {
    points: Vec<Vec2>,
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

const METRICS_PANEL_WIDTH_PX: f64 = 310.0;
const METRICS_PANEL_GUTTER_PX: f64 = 22.0;

fn reserved_right_sidebar_px(options: &DiagramOptions) -> f64 {
    if options.show_metrics_panel {
        METRICS_PANEL_WIDTH_PX + METRICS_PANEL_GUTTER_PX
    } else {
        0.0
    }
}

pub fn render_molecule_svg(
    molecule: &Molecule,
    options: &DiagramOptions,
) -> Result<String, String> {
    render_molecule_svg_internal(molecule, options, None, None, None)
}

pub fn render_molecule_svg_with_orbital(
    molecule: &Molecule,
    basis: &BasisSet,
    coefficients: &Matrix,
    options: &DiagramOptions,
    orbital: &OrbitalSettings,
) -> Result<String, String> {
    let layer = OrbitalLayer {
        settings: orbital.clone(),
        color_mode: OrbitalColorMode::Phase,
        render_style: OrbitalRenderStyle::Filled,
    };
    render_molecule_svg_internal(
        molecule,
        options,
        Some((basis, coefficients)),
        Some(vec![layer]),
        None,
    )
}

pub fn render_molecule_svg_with_orbitals(
    molecule: &Molecule,
    basis: &BasisSet,
    coefficients: &Matrix,
    options: &DiagramOptions,
    orbital_indices: &[usize],
    orbital: &OrbitalSettings,
) -> Result<String, String> {
    let palette = default_orbital_palette(orbital_indices.len().max(1));
    let layers = orbital_indices
        .iter()
        .enumerate()
        .map(|(idx, &orbital_index)| OrbitalLayer {
            settings: OrbitalSettings {
                orbital_index,
                ..orbital.clone()
            },
            color_mode: OrbitalColorMode::Solid(palette[idx].clone()),
            render_style: OrbitalRenderStyle::Contours,
        })
        .collect();

    render_molecule_svg_internal(
        molecule,
        options,
        Some((basis, coefficients)),
        Some(layers),
        None,
    )
}

pub fn render_molecule_svg_with_density(
    molecule: &Molecule,
    basis: &BasisSet,
    density: &Matrix,
    options: &DiagramOptions,
    density_settings: &DensitySettings,
) -> Result<String, String> {
    render_molecule_svg_internal(
        molecule,
        options,
        None,
        None,
        Some((basis, density, density_settings)),
    )
}

fn render_molecule_svg_internal(
    molecule: &Molecule,
    options: &DiagramOptions,
    orbital_basis: Option<(&BasisSet, &Matrix)>,
    orbital_layers: Option<Vec<OrbitalLayer>>,
    density_overlay: Option<(&BasisSet, &Matrix, &DensitySettings)>,
) -> Result<String, String> {
    let projected = project_atoms(molecule, options.projection);
    let mut bounds = compute_bounds(&projected, options.world_padding_bohr);
    let layers_ref = orbital_layers.as_deref();
    if let (Some((basis, coefficients)), Some(layers)) = (orbital_basis, layers_ref) {
        if layers.iter().any(|layer| layer.settings.auto_expand_bounds) {
            bounds = auto_expand_bounds_layers(
                options.projection,
                &bounds,
                basis,
                coefficients,
                layers,
            )?;
        }
    }
    if let Some((basis, density, density_settings)) = density_overlay {
        if density_settings.auto_expand_bounds {
            bounds = auto_expand_bounds_density(
                options.projection,
                &bounds,
                basis,
                density,
                density_settings,
            )?;
        }
    }
    let transform = compute_transform(&bounds, options, reserved_right_sidebar_px(options));
    let bonds = detect_bonds(molecule, options.bond_scale);

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns='http://www.w3.org/2000/svg' width='{}' height='{}' viewBox='0 0 {} {}'>",
        options.width, options.height, options.width, options.height
    ));
    svg.push_str(&format!(
        "<rect width='100%' height='100%' fill='{}'/>",
        options.background
    ));

    if let Some(title) = options.title.as_ref() {
        svg.push_str(&format!("<title>{}</title>", escape_xml(title)));
    }
    if let Some(description) = options.description.as_ref() {
        svg.push_str(&format!("<desc>{}</desc>", escape_xml(description)));
    }

    if options.title.is_some() || options.description.is_some() {
        let x = options.padding_px;
        let mut y = options.padding_px * 0.6;
        if let Some(title) = options.title.as_ref() {
            svg.push_str("<g fill='#111111' font-size='18' font-family='DejaVu Sans, Arial, sans-serif' font-weight='600'>");
            svg.push_str(&format!(
                "<text x='{:.2}' y='{:.2}' text-anchor='start' dominant-baseline='hanging'>{}</text>",
                x,
                y,
                escape_xml(title)
            ));
            svg.push_str("</g>");
            y += 20.0;
        }
        if let Some(description) = options.description.as_ref() {
            svg.push_str(
                "<g fill='#333333' font-size='12' font-family='DejaVu Sans, Arial, sans-serif'>",
            );
            svg.push_str(&format!(
                "<text x='{:.2}' y='{:.2}' text-anchor='start' dominant-baseline='hanging'>{}</text>",
                x,
                y,
                escape_xml(description)
            ));
            svg.push_str("</g>");
        }
    }

    if let Some((basis, density, density_settings)) = density_overlay {
        let density_svg = render_density_layer_svg(
            options.projection,
            &bounds,
            &transform,
            basis,
            density,
            density_settings,
        )?;
        svg.push_str("<g opacity='1'>");
        svg.push_str(&density_svg);
        svg.push_str("</g>");
    }

    let mut orbital_renders: Vec<OrbitalRenderInfo> = Vec::new();
    if let (Some((basis, coefficients)), Some(layers)) = (orbital_basis, layers_ref) {
        let mut field = String::new();
        for layer in layers {
            let render = render_orbital_layer_svg(
                options.projection,
                &bounds,
                &transform,
                basis,
                coefficients,
                &layer.settings,
                &layer.color_mode,
                layer.render_style,
            )?;
            field.push_str(&render.svg);
            orbital_renders.push(render);
        }
        svg.push_str("<g opacity='1'>");
        svg.push_str(&field);
        svg.push_str("</g>");
    }

    let mut geometry_points = projected.clone();
    let mut bond_lengths_for_labels: Vec<f64> = bonds.iter().map(|bond| bond.length_bohr).collect();
    let mut orbital_geometry_guides = String::new();
    if let (Some((basis, coefficients)), Some(layers)) = (orbital_basis, layers_ref) {
        if let Ok(derived) = derive_geometry_from_orbital_probability(
            options.projection,
            &bounds,
            basis,
            coefficients,
            layers,
            &projected,
        ) {
            if derived.points.len() == projected.len() {
                bond_lengths_for_labels = bonds
                    .iter()
                    .map(|bond| vec2_distance(derived.points[bond.i], derived.points[bond.j]))
                    .collect();
                orbital_geometry_guides =
                    render_orbital_geometry_guides(&projected, &derived.points, &transform);
                geometry_points = derived.points;
            }
        }
    }

    svg.push_str("<g stroke='#4a4a4a' stroke-width='2' fill='none'>");
    for bond in &bonds {
        let a = transform.to_screen(geometry_points[bond.i]);
        let b = transform.to_screen(geometry_points[bond.j]);
        svg.push_str(&format!(
            "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}'/>",
            a.x, a.y, b.x, b.y
        ));
    }
    svg.push_str("</g>");

    if options.annotate_angles {
        svg.push_str("<g stroke='#555555' stroke-width='1.5' fill='none'>");
        let angle_annotations = angle_annotations(&geometry_points, &bonds, &transform);
        svg.push_str(&angle_annotations.paths);
        svg.push_str("</g>");
        svg.push_str(
            "<g fill='#2b2b2b' font-size='13' font-family='DejaVu Sans, Arial, sans-serif'>",
        );
        svg.push_str(&angle_annotations.labels);
        svg.push_str("</g>");
    }

    if options.annotate_bonds {
        svg.push_str(
            "<g fill='#2b2b2b' font-size='13' font-family='DejaVu Sans, Arial, sans-serif'>",
        );
        for (bond_idx, bond) in bonds.iter().enumerate() {
            let a = transform.to_screen(geometry_points[bond.i]);
            let b = transform.to_screen(geometry_points[bond.j]);
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
            let label = format_length(bond_lengths_for_labels[bond_idx], options.length_unit);
            svg.push_str(&format!(
                "<text x='{:.2}' y='{:.2}' text-anchor='middle'>{}</text>",
                label_pos.x, label_pos.y, label
            ));
        }
        svg.push_str("</g>");
    }

    if !orbital_geometry_guides.is_empty() {
        svg.push_str(&orbital_geometry_guides);
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
        svg.push_str(
            "<g fill='#111111' font-size='14' font-family='DejaVu Sans, Arial, sans-serif'>",
        );
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

    if !orbital_renders.is_empty() {
        svg.push_str(&render_orbital_centroids(&orbital_renders, &transform));
        svg.push_str(&render_orbital_legend(options, &orbital_renders));
        if layers_ref
            .map(|layers| {
                layers
                    .iter()
                    .any(|layer| layer.render_style == OrbitalRenderStyle::Contours)
            })
            .unwrap_or(false)
        {
            svg.push_str(&render_contour_scale_note(options));
        }
    }

    if options.show_metrics_panel {
        svg.push_str(&render_metrics_panel(
            molecule,
            options,
            &bonds,
            layers_ref.map_or(0, |layers| layers.len()),
        ));
    }

    svg.push_str(&scale_bar_svg(options, &bounds, &transform));

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

fn compute_transform(
    bounds: &Bounds,
    options: &DiagramOptions,
    reserved_right_px: f64,
) -> Transform {
    let width = options.width as f64;
    let height = options.height as f64;
    let range_x = (bounds.max_x - bounds.min_x).max(1.0);
    let range_y = (bounds.max_y - bounds.min_y).max(1.0);
    let drawable_width = (width - 2.0 * options.padding_px - reserved_right_px).max(120.0);
    let scale_x = drawable_width / range_x;
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
            let threshold =
                bond_threshold_bohr(atoms[i].atomic_number, atoms[j].atomic_number, scale);
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

fn projection_label(plane: ProjectionPlane) -> &'static str {
    match plane {
        ProjectionPlane::XY => "XY",
        ProjectionPlane::XZ => "XZ",
        ProjectionPlane::YZ => "YZ",
    }
}

fn length_unit_label(unit: LengthUnit) -> &'static str {
    match unit {
        LengthUnit::Bohr => "Bohr",
        LengthUnit::Angstrom => "Angstrom",
    }
}

fn molecular_formula(molecule: &Molecule) -> String {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for atom in &molecule.atoms {
        *counts
            .entry(element_symbol(atom.atomic_number))
            .or_insert(0) += 1;
    }

    let mut symbols: Vec<String> = counts.keys().cloned().collect();
    symbols.sort_by(|a, b| {
        let rank = |symbol: &str| match symbol {
            "C" => 0,
            "H" => 1,
            _ => 2,
        };
        rank(a).cmp(&rank(b)).then_with(|| a.cmp(b))
    });

    let mut formula = String::new();
    for symbol in symbols {
        let count = counts[&symbol];
        formula.push_str(&symbol);
        if count > 1 {
            formula.push_str(&count.to_string());
        }
    }
    formula
}

fn metric_tone_color(tone: MetricTone) -> &'static str {
    match tone {
        MetricTone::Info => "#3f3f46",
        MetricTone::Good => "#15803d",
        MetricTone::Warning => "#b45309",
        MetricTone::Critical => "#b91c1c",
    }
}

fn render_metric_rows(
    svg: &mut String,
    rows: &[DiagramMetric],
    x0: f64,
    panel_width: f64,
    mut y_start: f64,
) -> f64 {
    let label_x = x0 + 18.0;
    let value_x = x0 + panel_width - 12.0;
    let row_h = 18.0;

    for row in rows {
        let dot_y = y_start + 7.5;
        svg.push_str(&format!(
            "<circle cx='{:.2}' cy='{:.2}' r='3.0' fill='{}'/>",
            x0 + 8.0,
            dot_y,
            metric_tone_color(row.tone)
        ));
        svg.push_str(&format!(
            "<text x='{:.2}' y='{:.2}' fill='#111827' font-size='11.5' font-family='DejaVu Sans, Arial, sans-serif'>{}</text>",
            label_x,
            y_start + 11.0,
            escape_xml(&row.label)
        ));
        svg.push_str(&format!(
            "<text x='{:.2}' y='{:.2}' text-anchor='end' fill='{}' font-size='11.5' font-family='DejaVu Sans, Arial, sans-serif' font-weight='600'>{}</text>",
            value_x,
            y_start + 11.0,
            metric_tone_color(row.tone),
            escape_xml(&row.value)
        ));
        y_start += row_h;
    }

    y_start
}

fn render_metrics_panel(
    molecule: &Molecule,
    options: &DiagramOptions,
    bonds: &[Bond],
    displayed_orbitals: usize,
) -> String {
    let summary_rows = vec![
        DiagramMetric::info("Formula", molecular_formula(molecule)),
        DiagramMetric::info("Atoms", molecule.atoms.len().to_string()),
        DiagramMetric::info("Electrons", molecule.num_electrons().to_string()),
        DiagramMetric::info("Occupied MOs", molecule.num_occupied().to_string()),
        DiagramMetric::info("Detected bonds", bonds.len().to_string()),
        DiagramMetric::info("Projection", projection_label(options.projection)),
        DiagramMetric::info("Units", length_unit_label(options.length_unit)),
        DiagramMetric::info(
            "Displayed orbitals",
            if displayed_orbitals == 0 {
                "geometry only".to_string()
            } else {
                displayed_orbitals.to_string()
            },
        ),
    ];

    let debug_rows = &options.metrics;
    let row_h = 18.0;
    let padding = 10.0;
    let header_h = 28.0;
    let section_gap = if debug_rows.is_empty() { 0.0 } else { 14.0 };
    let debug_title_h = if debug_rows.is_empty() { 0.0 } else { 14.0 };
    let panel_width = METRICS_PANEL_WIDTH_PX;
    let panel_height = header_h
        + padding * 2.0
        + row_h * summary_rows.len() as f64
        + section_gap
        + debug_title_h
        + row_h * debug_rows.len() as f64;

    let x0 = options.width as f64 - options.padding_px - panel_width;
    let y0 = (options.height as f64 - options.padding_px * 0.75 - panel_height).max(8.0);

    let mut svg = String::new();
    svg.push_str("<g>");
    svg.push_str(&format!(
        "<rect x='{:.2}' y='{:.2}' width='{:.2}' height='{:.2}' rx='10' ry='10' fill='#fffdf8' fill-opacity='0.93' stroke='#cbd5e1'/>",
        x0, y0, panel_width, panel_height
    ));
    svg.push_str(&format!(
        "<rect x='{:.2}' y='{:.2}' width='{:.2}' height='{:.2}' rx='10' ry='10' fill='#0f172a'/>",
        x0, y0, panel_width, header_h
    ));
    svg.push_str(&format!(
        "<text x='{:.2}' y='{:.2}' fill='#f8fafc' font-size='12' font-family='DejaVu Sans, Arial, sans-serif' font-weight='700'>Figure Metrics</text>",
        x0 + 12.0,
        y0 + 18.0
    ));

    let mut y_cursor = y0 + header_h + padding;
    y_cursor = render_metric_rows(
        &mut svg,
        &summary_rows,
        x0 + padding,
        panel_width - 2.0 * padding,
        y_cursor,
    );

    if !debug_rows.is_empty() {
        y_cursor += 4.0;
        svg.push_str(&format!(
            "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}' stroke='#d4d4d8' stroke-width='1'/>",
            x0 + padding,
            y_cursor,
            x0 + panel_width - padding,
            y_cursor
        ));
        y_cursor += 12.0;
        svg.push_str(&format!(
            "<text x='{:.2}' y='{:.2}' fill='#334155' font-size='11' font-family='DejaVu Sans, Arial, sans-serif' font-weight='700'>Debug Diagnostics</text>",
            x0 + padding,
            y_cursor
        ));
        y_cursor += 4.0;
        render_metric_rows(
            &mut svg,
            debug_rows,
            x0 + padding,
            panel_width - 2.0 * padding,
            y_cursor,
        );
    }

    svg.push_str("</g>");
    svg
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

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn default_orbital_palette(count: usize) -> Vec<String> {
    let n = count.max(1);
    let mut colors = Vec::with_capacity(n);
    for i in 0..n {
        let hue = (i as f64) / (n as f64);
        let (r, g, b) = hsl_to_rgb(hue, 0.62, 0.45);
        colors.push(format!("#{:02x}{:02x}{:02x}", r, g, b));
    }
    colors
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = (h * 6.0) % 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());

    let (r1, g1, b1) = if (0.0..1.0).contains(&h_prime) {
        (c, x, 0.0)
    } else if (1.0..2.0).contains(&h_prime) {
        (x, c, 0.0)
    } else if (2.0..3.0).contains(&h_prime) {
        (0.0, c, x)
    } else if (3.0..4.0).contains(&h_prime) {
        (0.0, x, c)
    } else if (4.0..5.0).contains(&h_prime) {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    let m = l - c * 0.5;
    let to_u8 = |v: f64| ((v + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    (to_u8(r1), to_u8(g1), to_u8(b1))
}

fn orbital_centroid(field: &OrbitalField, bounds: &Bounds) -> Option<Vec2> {
    let mut weight_sum = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;

    for (idx, value) in field.values.iter().enumerate() {
        let weight = value * value;
        if weight == 0.0 {
            continue;
        }
        let i = idx % field.grid;
        let j = idx / field.grid;
        let x = bounds.min_x + field.dx * i as f64;
        let y = bounds.min_y + field.dy * j as f64;
        weight_sum += weight;
        sum_x += x * weight;
        sum_y += y * weight;
    }

    if weight_sum <= 0.0 {
        return None;
    }

    Some(Vec2 {
        x: sum_x / weight_sum,
        y: sum_y / weight_sum,
    })
}

fn render_orbital_centroids(infos: &[OrbitalRenderInfo], transform: &Transform) -> String {
    let mut svg = String::new();
    svg.push_str("<g stroke-width='2'>");
    for info in infos {
        let centroid = match info.centroid {
            Some(c) => c,
            None => continue,
        };
        let color = info.legend_color.as_deref().unwrap_or("#111111");
        let screen = transform.to_screen(centroid);
        svg.push_str(&format!(
            "<circle cx='{:.2}' cy='{:.2}' r='4.2' fill='#ffffff' stroke='{}'/>",
            screen.x, screen.y, color
        ));
        svg.push_str(&format!(
            "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}' stroke='{}'/>",
            screen.x - 6.0,
            screen.y,
            screen.x + 6.0,
            screen.y,
            color
        ));
        svg.push_str(&format!(
            "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}' stroke='{}'/>",
            screen.x,
            screen.y - 6.0,
            screen.x,
            screen.y + 6.0,
            color
        ));
        svg.push_str(&format!(
            "<text x='{:.2}' y='{:.2}' fill='{}' font-size='11' font-family='DejaVu Sans, Arial, sans-serif'>{}</text>",
            screen.x + 8.0,
            screen.y - 8.0,
            color,
            escape_xml(&info.label)
        ));
    }
    svg.push_str("</g>");
    svg
}

fn render_orbital_legend(options: &DiagramOptions, infos: &[OrbitalRenderInfo]) -> String {
    let entries: Vec<&OrbitalRenderInfo> = infos
        .iter()
        .filter(|info| info.legend_color.is_some())
        .collect();
    if entries.len() < 2 {
        return String::new();
    }

    let swatch = 10.0;
    let line_height = 16.0;
    let padding = 8.0;
    let width = 120.0;
    let height = padding * 2.0 + line_height * entries.len() as f64;
    let x0 = options.width as f64 - options.padding_px - width;
    let y0 = options.padding_px * 0.6;

    let mut svg = String::new();
    svg.push_str("<g>");
    svg.push_str(&format!(
        "<rect x='{:.2}' y='{:.2}' width='{:.2}' height='{:.2}' fill='#ffffff' fill-opacity='0.85' stroke='#cccccc'/>",
        x0, y0, width, height
    ));
    for (idx, info) in entries.iter().enumerate() {
        let color = info.legend_color.as_deref().unwrap_or("#111111");
        let y = y0 + padding + line_height * idx as f64 + 10.0;
        svg.push_str(&format!(
            "<rect x='{:.2}' y='{:.2}' width='{:.2}' height='{:.2}' fill='{}'/>",
            x0 + padding,
            y - swatch * 0.75,
            swatch,
            swatch,
            color
        ));
        svg.push_str(&format!(
            "<text x='{:.2}' y='{:.2}' fill='#111111' font-size='12' font-family='DejaVu Sans, Arial, sans-serif'>{}</text>",
            x0 + padding + swatch + 6.0,
            y,
            escape_xml(&info.label)
        ));
    }
    svg.push_str("</g>");
    svg
}

fn render_contour_scale_note(options: &DiagramOptions) -> String {
    let x0 = options.width as f64 - options.padding_px - 208.0;
    let y0 = options.padding_px * 0.6 + 88.0;
    let mut svg = String::new();
    svg.push_str("<g>");
    svg.push_str(&format!(
        "<rect x='{:.2}' y='{:.2}' width='208.00' height='26.00' fill='#ffffff' fill-opacity='0.82' stroke='#d4d4d8'/>",
        x0, y0
    ));
    svg.push_str(&format!(
        "<text x='{:.2}' y='{:.2}' fill='#334155' font-size='11.5' font-family='DejaVu Sans, Arial, sans-serif'>Contours: 85/65/45/30% of |psi| max</text>",
        x0 + 8.0,
        y0 + 17.0
    ));
    svg.push_str("</g>");
    svg
}

fn scale_bar_svg(options: &DiagramOptions, bounds: &Bounds, transform: &Transform) -> String {
    let width_bohr = (bounds.max_x - bounds.min_x).max(1.0);
    let width_units = match options.length_unit {
        LengthUnit::Bohr => width_bohr,
        LengthUnit::Angstrom => width_bohr * BOHR_TO_ANGSTROM,
    };
    let target_units = width_units * 0.25;
    let nice_units = nice_length(target_units);
    if !nice_units.is_finite() || nice_units <= 0.0 {
        return String::new();
    }
    let length_bohr = match options.length_unit {
        LengthUnit::Bohr => nice_units,
        LengthUnit::Angstrom => nice_units / BOHR_TO_ANGSTROM,
    };
    let length_px = transform.length_to_screen(length_bohr);
    if length_px <= 0.0 {
        return String::new();
    }

    let x0 = options.padding_px;
    let y0 = options.height as f64 - options.padding_px * 0.6;
    let tick = 6.0;
    let label = format_length(length_bohr, options.length_unit);

    let mut svg = String::new();
    svg.push_str("<g stroke='#222222' stroke-width='2'>");
    svg.push_str(&format!(
        "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}'/>",
        x0,
        y0,
        x0 + length_px,
        y0
    ));
    svg.push_str(&format!(
        "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}'/>",
        x0,
        y0 - tick * 0.5,
        x0,
        y0 + tick * 0.5
    ));
    svg.push_str(&format!(
        "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}'/>",
        x0 + length_px,
        y0 - tick * 0.5,
        x0 + length_px,
        y0 + tick * 0.5
    ));
    svg.push_str("</g>");
    svg.push_str("<g fill='#222222' font-size='12' font-family='DejaVu Sans, Arial, sans-serif'>");
    svg.push_str(&format!(
        "<text x='{:.2}' y='{:.2}' text-anchor='middle'>{}</text>",
        x0 + length_px * 0.5,
        y0 + 14.0,
        label
    ));
    svg.push_str("</g>");

    svg
}

fn nice_length(target: f64) -> f64 {
    if !target.is_finite() || target <= 0.0 {
        return 0.0;
    }
    let exp = 10_f64.powf(target.log10().floor());
    let frac = target / exp;
    let nice_frac = if frac <= 0.2 {
        0.2
    } else if frac <= 0.5 {
        0.5
    } else if frac <= 1.0 {
        1.0
    } else if frac <= 2.0 {
        2.0
    } else if frac <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice_frac * exp
}

struct AngleAnnotations {
    paths: String,
    labels: String,
}

fn angle_annotations(
    projected: &[Vec2],
    bonds: &[Bond],
    transform: &Transform,
) -> AngleAnnotations {
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
        start_screen.x, start_screen.y, radius_px, radius_px, sweep, end_screen.x, end_screen.y
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

fn vec2_distance(a: Vec2, b: Vec2) -> f64 {
    vec2_len(Vec2 {
        x: a.x - b.x,
        y: a.y - b.y,
    })
}

fn normalize(v: Vec2) -> Vec2 {
    let len = vec2_len(v);
    if len == 0.0 {
        return Vec2 {
            x: f64::NAN,
            y: f64::NAN,
        };
    }
    Vec2 {
        x: v.x / len,
        y: v.y / len,
    }
}

fn cross(a: Vec2, b: Vec2) -> f64 {
    a.x * b.y - a.y * b.x
}

fn derive_geometry_from_orbital_probability(
    projection: ProjectionPlane,
    bounds: &Bounds,
    basis: &BasisSet,
    coefficients: &Matrix,
    layers: &[OrbitalLayer],
    projected_atoms: &[Vec2],
) -> Result<OrbitalDerivedGeometry, String> {
    if projected_atoms.is_empty() || layers.is_empty() {
        return Err("missing atoms or orbital layers".to_string());
    }
    let n_basis = basis.size();
    if n_basis != coefficients.nrows() {
        return Err("coefficients matrix size does not match basis".to_string());
    }
    if layers
        .iter()
        .any(|layer| layer.settings.orbital_index >= coefficients.ncols())
    {
        return Err("orbital layer index out of range".to_string());
    }

    let grid = layers
        .iter()
        .map(|layer| layer.settings.grid)
        .max()
        .unwrap_or(90)
        .max(48);
    let dx = (bounds.max_x - bounds.min_x) / (grid as f64 - 1.0);
    let dy = (bounds.max_y - bounds.min_y) / (grid as f64 - 1.0);

    let coeffs_cart = basis.to_cartesian_coefficients(coefficients);
    let n_cart = basis.cartesian_size();
    let layer_specs: Vec<(usize, f64)> = layers
        .iter()
        .map(|layer| (layer.settings.orbital_index, layer.settings.plane_offset))
        .collect();

    let mut probabilities = Vec::with_capacity(grid * grid);
    let mut max_probability: f64 = 0.0;
    for j in 0..grid {
        for i in 0..grid {
            let x = bounds.min_x + dx * i as f64;
            let y = bounds.min_y + dy * j as f64;
            let mut probability = 0.0;
            for (orbital_index, plane_offset) in &layer_specs {
                let point = expand_point(projection, x, y, *plane_offset);
                let mut value = 0.0;
                for k in 0..n_cart {
                    let coeff = coeffs_cart[[k, *orbital_index]];
                    if coeff.abs() < 1e-10 {
                        continue;
                    }
                    value += coeff * basis.cartesian_functions[k].evaluate(point);
                }
                probability += value * value;
            }
            max_probability = max_probability.max(probability);
            probabilities.push(probability);
        }
    }
    if max_probability <= 0.0 {
        return Err("orbital probability field is zero".to_string());
    }

    let threshold = 0.05 * max_probability;
    let mut sum_x = vec![0.0; projected_atoms.len()];
    let mut sum_y = vec![0.0; projected_atoms.len()];
    let mut sum_w = vec![0.0; projected_atoms.len()];
    for (idx, probability) in probabilities.iter().enumerate() {
        if *probability < threshold {
            continue;
        }
        let i = idx % grid;
        let j = idx / grid;
        let x = bounds.min_x + dx * i as f64;
        let y = bounds.min_y + dy * j as f64;

        let mut nearest = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (atom_idx, atom_pos) in projected_atoms.iter().enumerate() {
            let d2 = (x - atom_pos.x).powi(2) + (y - atom_pos.y).powi(2);
            if d2 < best_d2 {
                best_d2 = d2;
                nearest = atom_idx;
            }
        }

        sum_x[nearest] += probability * x;
        sum_y[nearest] += probability * y;
        sum_w[nearest] += probability;
    }

    let mut points = projected_atoms.to_vec();
    // Keep derived centers near nuclei so overlays remain interpretable.
    let max_shift_bohr = 1.25;
    for atom_idx in 0..points.len() {
        if sum_w[atom_idx] <= 0.0 {
            continue;
        }
        let mut derived = Vec2 {
            x: sum_x[atom_idx] / sum_w[atom_idx],
            y: sum_y[atom_idx] / sum_w[atom_idx],
        };
        let shift = vec2_distance(points[atom_idx], derived);
        if shift > max_shift_bohr {
            let t = max_shift_bohr / shift;
            derived = Vec2 {
                x: points[atom_idx].x + (derived.x - points[atom_idx].x) * t,
                y: points[atom_idx].y + (derived.y - points[atom_idx].y) * t,
            };
        }
        points[atom_idx] = derived;
    }

    Ok(OrbitalDerivedGeometry { points })
}

fn render_orbital_geometry_guides(
    nuclear_points: &[Vec2],
    derived_points: &[Vec2],
    transform: &Transform,
) -> String {
    if nuclear_points.len() != derived_points.len() {
        return String::new();
    }
    let mut svg = String::new();
    svg.push_str("<g id='orbital-geometry-guides' stroke='#0f766e' fill='none' stroke-width='1.2' opacity='0.92'>");
    for i in 0..nuclear_points.len() {
        let shift = vec2_distance(nuclear_points[i], derived_points[i]);
        if shift <= 1e-4 {
            continue;
        }
        let a = transform.to_screen(nuclear_points[i]);
        let b = transform.to_screen(derived_points[i]);
        svg.push_str(&format!(
            "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}' stroke-dasharray='3 2'/>",
            a.x, a.y, b.x, b.y
        ));
        svg.push_str(&format!(
            "<circle cx='{:.2}' cy='{:.2}' r='3.2' fill='#ffffff'/>",
            b.x, b.y
        ));
    }
    svg.push_str("</g>");
    svg
}

struct OrbitalField {
    values: Vec<f64>,
    max_abs: f64,
    edge_max_abs: f64,
    grid: usize,
    dx: f64,
    dy: f64,
    plane_offset: f64,
}

struct DensityField {
    values: Vec<f64>,
    max_value: f64,
    edge_max_value: f64,
    grid: usize,
    dx: f64,
    dy: f64,
}

fn compute_orbital_values_at_offset(
    projection: ProjectionPlane,
    bounds: &Bounds,
    basis: &BasisSet,
    coefficients: &Matrix,
    orbital_index: usize,
    grid: usize,
    plane_offset: f64,
) -> Result<OrbitalField, String> {
    let dx = (bounds.max_x - bounds.min_x) / (grid as f64 - 1.0);
    let dy = (bounds.max_y - bounds.min_y) / (grid as f64 - 1.0);
    let coeffs_cart = basis.to_cartesian_coefficients(coefficients);
    let n_cart = basis.cartesian_size();

    let mut values = Vec::with_capacity(grid * grid);
    let mut max_abs: f64 = 0.0;
    let mut edge_max_abs: f64 = 0.0;

    for j in 0..grid {
        for i in 0..grid {
            let x = bounds.min_x + dx * i as f64;
            let y = bounds.min_y + dy * j as f64;
            let point = expand_point(projection, x, y, plane_offset);
            let mut value = 0.0;
            for k in 0..n_cart {
                let coeff = coeffs_cart[[k, orbital_index]];
                if coeff.abs() < 1e-10 {
                    continue;
                }
                value += coeff * basis.cartesian_functions[k].evaluate(point);
            }
            max_abs = max_abs.max(value.abs());
            if i == 0 || j == 0 || i + 1 == grid || j + 1 == grid {
                edge_max_abs = edge_max_abs.max(value.abs());
            }
            values.push(value);
        }
    }

    Ok(OrbitalField {
        values,
        max_abs,
        edge_max_abs,
        grid,
        dx,
        dy,
        plane_offset,
    })
}

fn compute_orbital_values(
    projection: ProjectionPlane,
    bounds: &Bounds,
    basis: &BasisSet,
    coefficients: &Matrix,
    settings: &OrbitalSettings,
) -> Result<OrbitalField, String> {
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
    let mut field = compute_orbital_values_at_offset(
        projection,
        bounds,
        basis,
        coefficients,
        settings.orbital_index,
        grid,
        settings.plane_offset,
    )?;

    // Some orbitals are exactly zero on symmetric slice planes (e.g. a nodal
    // plane through the molecule). Nudge the slice for visualization so the
    // user still sees the requested MO.
    if field.max_abs < 1e-12 && settings.plane_offset.abs() < 1e-12 {
        for offset in [0.12, -0.12, 0.24, -0.24] {
            let candidate = compute_orbital_values_at_offset(
                projection,
                bounds,
                basis,
                coefficients,
                settings.orbital_index,
                grid,
                offset,
            )?;
            if candidate.max_abs > field.max_abs {
                field = candidate;
            }
            if field.max_abs >= 1e-12 {
                break;
            }
        }
    }

    Ok(field)
}

fn render_orbital_layer_svg(
    projection: ProjectionPlane,
    bounds: &Bounds,
    transform: &Transform,
    basis: &BasisSet,
    coefficients: &Matrix,
    settings: &OrbitalSettings,
    color_mode: &OrbitalColorMode,
    render_style: OrbitalRenderStyle,
) -> Result<OrbitalRenderInfo, String> {
    let field = compute_orbital_values(projection, bounds, basis, coefficients, settings)?;
    if field.max_abs == 0.0 {
        let legend_color = match color_mode {
            OrbitalColorMode::Solid(color) => Some(color.clone()),
            OrbitalColorMode::Phase => None,
        };
        return Ok(OrbitalRenderInfo {
            svg: String::new(),
            centroid: None,
            legend_color,
            label: format!("MO {}", settings.orbital_index),
        });
    }

    let mut svg = String::new();
    match render_style {
        OrbitalRenderStyle::Filled => {
            let cell_w = transform.length_to_screen(field.dx);
            let cell_h = transform.length_to_screen(field.dy);
            let cutoff = settings.cutoff_fraction.clamp(0.0, 1.0) * field.max_abs;

            for (idx, value) in field.values.iter().enumerate() {
                if value.abs() < cutoff {
                    continue;
                }
                let i = idx % field.grid;
                let j = idx / field.grid;
                let x = bounds.min_x + field.dx * i as f64;
                let y = bounds.min_y + field.dy * j as f64;
                let pos = transform.to_screen(Vec2 { x, y });
                let alpha = (value.abs() / field.max_abs) * settings.alpha;
                let color = match color_mode {
                    OrbitalColorMode::Phase => {
                        if *value >= 0.0 {
                            "#3b6fb6"
                        } else {
                            "#d1495b"
                        }
                    }
                    OrbitalColorMode::Solid(color) => color.as_str(),
                };
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
        }
        OrbitalRenderStyle::Contours => {
            let base_color = match color_mode {
                OrbitalColorMode::Phase => "#3b6fb6",
                OrbitalColorMode::Solid(color) => color.as_str(),
            };
            svg.push_str(&render_orbital_contours_svg(
                &field,
                bounds,
                transform,
                base_color,
                settings.alpha.clamp(0.1, 1.0),
            ));
        }
    }

    let legend_color = match color_mode {
        OrbitalColorMode::Solid(color) => Some(color.clone()),
        OrbitalColorMode::Phase => None,
    };
    let label = if (field.plane_offset - settings.plane_offset).abs() > 1e-9 {
        format!(
            "MO {} (offset {:.2})",
            settings.orbital_index, field.plane_offset
        )
    } else {
        format!("MO {}", settings.orbital_index)
    };

    Ok(OrbitalRenderInfo {
        svg,
        centroid: orbital_centroid(&field, bounds),
        legend_color,
        label,
    })
}

fn render_orbital_contours_svg(
    field: &OrbitalField,
    bounds: &Bounds,
    transform: &Transform,
    color: &str,
    alpha: f64,
) -> String {
    let levels = [0.85, 0.65, 0.45, 0.30];
    let dash = [None, Some("7 3"), Some("5 3"), Some("3 3")];

    let mut svg = String::new();
    for (idx, level_fraction) in levels.iter().enumerate() {
        let level = field.max_abs * level_fraction;
        let segments = contour_segments(field, bounds, level);
        if segments.is_empty() {
            continue;
        }

        let stroke_width = (1.8 - idx as f64 * 0.25).max(0.9);
        let stroke_opacity = (alpha * (0.95 - idx as f64 * 0.16)).clamp(0.2, 0.95);
        svg.push_str(&format!(
            "<g stroke='{}' stroke-width='{:.2}' stroke-opacity='{:.3}' fill='none'",
            color, stroke_width, stroke_opacity
        ));
        if let Some(pattern) = dash[idx] {
            svg.push_str(&format!(" stroke-dasharray='{}'", pattern));
        }
        svg.push('>');
        for (a, b) in segments {
            let sa = transform.to_screen(a);
            let sb = transform.to_screen(b);
            svg.push_str(&format!(
                "<line x1='{:.2}' y1='{:.2}' x2='{:.2}' y2='{:.2}'/>",
                sa.x, sa.y, sb.x, sb.y
            ));
        }
        svg.push_str("</g>");
    }
    svg
}

fn contour_segments(field: &OrbitalField, bounds: &Bounds, level: f64) -> Vec<(Vec2, Vec2)> {
    let mut segments = Vec::new();
    if field.grid < 2 || level <= 0.0 {
        return segments;
    }

    for j in 0..(field.grid - 1) {
        for i in 0..(field.grid - 1) {
            let p0 = Vec2 {
                x: bounds.min_x + field.dx * i as f64,
                y: bounds.min_y + field.dy * j as f64,
            };
            let p1 = Vec2 {
                x: bounds.min_x + field.dx * (i + 1) as f64,
                y: bounds.min_y + field.dy * j as f64,
            };
            let p2 = Vec2 {
                x: bounds.min_x + field.dx * (i + 1) as f64,
                y: bounds.min_y + field.dy * (j + 1) as f64,
            };
            let p3 = Vec2 {
                x: bounds.min_x + field.dx * i as f64,
                y: bounds.min_y + field.dy * (j + 1) as f64,
            };

            let v0 = field.values[j * field.grid + i].abs();
            let v1 = field.values[j * field.grid + (i + 1)].abs();
            let v2 = field.values[(j + 1) * field.grid + (i + 1)].abs();
            let v3 = field.values[(j + 1) * field.grid + i].abs();

            let mut points = Vec::with_capacity(4);
            add_contour_intersection(&mut points, v0, p0, v1, p1, level);
            add_contour_intersection(&mut points, v1, p1, v2, p2, level);
            add_contour_intersection(&mut points, v2, p2, v3, p3, level);
            add_contour_intersection(&mut points, v3, p3, v0, p0, level);

            match points.len() {
                2 => segments.push((points[0], points[1])),
                4 => {
                    let center = (v0 + v1 + v2 + v3) * 0.25;
                    if center >= level {
                        segments.push((points[0], points[3]));
                        segments.push((points[1], points[2]));
                    } else {
                        segments.push((points[0], points[1]));
                        segments.push((points[2], points[3]));
                    }
                }
                _ => {}
            }
        }
    }

    segments
}

fn add_contour_intersection(out: &mut Vec<Vec2>, va: f64, pa: Vec2, vb: f64, pb: Vec2, level: f64) {
    let da = va - level;
    let db = vb - level;
    let crosses = (da >= 0.0 && db < 0.0) || (da < 0.0 && db >= 0.0);
    if !crosses {
        return;
    }

    let t = if (vb - va).abs() < 1e-14 {
        0.5
    } else {
        ((level - va) / (vb - va)).clamp(0.0, 1.0)
    };
    out.push(Vec2 {
        x: pa.x + (pb.x - pa.x) * t,
        y: pa.y + (pb.y - pa.y) * t,
    });
}

fn compute_density_values(
    projection: ProjectionPlane,
    bounds: &Bounds,
    basis: &BasisSet,
    density: &Matrix,
    settings: &DensitySettings,
) -> Result<DensityField, String> {
    let n_basis = basis.size();
    if density.nrows() != n_basis || density.ncols() != n_basis {
        return Err("density matrix size does not match basis".to_string());
    }

    let grid = settings.grid.max(10);
    let dx = (bounds.max_x - bounds.min_x) / (grid as f64 - 1.0);
    let dy = (bounds.max_y - bounds.min_y) / (grid as f64 - 1.0);

    let density_cart = basis.to_cartesian_density(density);
    let n_cart = basis.cartesian_size();

    let mut values = Vec::with_capacity(grid * grid);
    let mut max_value: f64 = 0.0;
    let mut edge_max_value: f64 = 0.0;
    let mut phi = vec![0.0; n_cart];

    for j in 0..grid {
        for i in 0..grid {
            let x = bounds.min_x + dx * i as f64;
            let y = bounds.min_y + dy * j as f64;
            let point = expand_point(projection, x, y, settings.plane_offset);

            for (k, basis_fn) in basis.cartesian_functions.iter().enumerate() {
                phi[k] = basis_fn.evaluate(point);
            }

            let mut rho = 0.0;
            for mu in 0..n_cart {
                let phi_mu = phi[mu];
                if phi_mu.abs() < 1e-12 {
                    continue;
                }
                for nu in 0..n_cart {
                    let phi_nu = phi[nu];
                    if phi_nu.abs() < 1e-12 {
                        continue;
                    }
                    rho += density_cart[[mu, nu]] * phi_mu * phi_nu;
                }
            }
            let rho = rho.max(0.0);
            max_value = max_value.max(rho);
            if i == 0 || j == 0 || i + 1 == grid || j + 1 == grid {
                edge_max_value = edge_max_value.max(rho);
            }
            values.push(rho);
        }
    }

    Ok(DensityField {
        values,
        max_value,
        edge_max_value,
        grid,
        dx,
        dy,
    })
}

fn render_density_layer_svg(
    projection: ProjectionPlane,
    bounds: &Bounds,
    transform: &Transform,
    basis: &BasisSet,
    density: &Matrix,
    settings: &DensitySettings,
) -> Result<String, String> {
    let field = compute_density_values(projection, bounds, basis, density, settings)?;
    if field.max_value <= 0.0 {
        return Ok(String::new());
    }

    let cell_w = transform.length_to_screen(field.dx);
    let cell_h = transform.length_to_screen(field.dy);
    let cutoff = settings.cutoff_fraction.clamp(0.0, 1.0) * field.max_value;
    let base_color = settings.color.as_str();

    let mut svg = String::new();
    for (idx, value) in field.values.iter().enumerate() {
        if *value < cutoff {
            continue;
        }
        let i = idx % field.grid;
        let j = idx / field.grid;
        let x = bounds.min_x + field.dx * i as f64;
        let y = bounds.min_y + field.dy * j as f64;
        let pos = transform.to_screen(Vec2 { x, y });
        let prob = (value / field.max_value).clamp(0.0, 1.0);
        // Darker means higher probability: keep hue fixed and vary opacity.
        let alpha = settings.alpha * prob;
        svg.push_str(&format!(
            "<rect x='{:.2}' y='{:.2}' width='{:.2}' height='{:.2}' fill='{}' fill-opacity='{:.3}'/>",
            pos.x - cell_w * 0.5,
            pos.y - cell_h * 0.5,
            cell_w,
            cell_h,
            base_color,
            alpha.min(0.96)
        ));
    }

    Ok(svg)
}

fn auto_expand_bounds_layers(
    projection: ProjectionPlane,
    start: &Bounds,
    basis: &BasisSet,
    coefficients: &Matrix,
    layers: &[OrbitalLayer],
) -> Result<Bounds, String> {
    let mut bounds = *start;
    let max_iters = 6;
    for _ in 0..max_iters {
        let mut needs_expand = false;
        for layer in layers {
            let cutoff_fraction = layer.settings.cutoff_fraction.clamp(0.0, 1.0);
            if cutoff_fraction <= 0.0 {
                continue;
            }
            let field =
                compute_orbital_values(projection, &bounds, basis, coefficients, &layer.settings)?;
            if field.max_abs == 0.0 {
                continue;
            }
            let cutoff = cutoff_fraction * field.max_abs;
            if field.edge_max_abs >= cutoff {
                needs_expand = true;
                break;
            }
        }
        if !needs_expand {
            break;
        }

        let range_x = (bounds.max_x - bounds.min_x).max(1.0);
        let range_y = (bounds.max_y - bounds.min_y).max(1.0);
        let expand_x = range_x * 0.25 + 0.4;
        let expand_y = range_y * 0.25 + 0.4;
        bounds = Bounds {
            min_x: bounds.min_x - expand_x,
            max_x: bounds.max_x + expand_x,
            min_y: bounds.min_y - expand_y,
            max_y: bounds.max_y + expand_y,
        };
    }

    Ok(bounds)
}

fn auto_expand_bounds_density(
    projection: ProjectionPlane,
    start: &Bounds,
    basis: &BasisSet,
    density: &Matrix,
    settings: &DensitySettings,
) -> Result<Bounds, String> {
    let mut bounds = *start;
    let max_iters = 6;
    for _ in 0..max_iters {
        let cutoff_fraction = settings.cutoff_fraction.clamp(0.0, 1.0);
        if cutoff_fraction <= 0.0 {
            break;
        }
        let field = compute_density_values(projection, &bounds, basis, density, settings)?;
        if field.max_value <= 0.0 {
            break;
        }
        let cutoff = cutoff_fraction * field.max_value;
        if field.edge_max_value < cutoff {
            break;
        }

        let range_x = (bounds.max_x - bounds.min_x).max(1.0);
        let range_y = (bounds.max_y - bounds.min_y).max(1.0);
        let expand_x = range_x * 0.25 + 0.4;
        let expand_y = range_y * 0.25 + 0.4;
        bounds = Bounds {
            min_x: bounds.min_x - expand_x,
            max_x: bounds.max_x + expand_x,
            min_y: bounds.min_y - expand_y,
            max_y: bounds.max_y + expand_y,
        };
    }
    Ok(bounds)
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
    use crate::HartreeFock;

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
        let transform = compute_transform(
            &compute_bounds(&projected, 0.5),
            &options,
            reserved_right_sidebar_px(&options),
        );
        let annotations = angle_annotations(&projected, &bonds, &transform);
        assert!(!annotations.labels.is_empty());
    }

    #[test]
    fn test_metrics_panel_renders_debug_rows() {
        let mol = Molecule::h2o();
        let mut options = DiagramOptions::default();
        options.metrics = vec![
            DiagramMetric::good("Mirror residual", "1.2e-12"),
            DiagramMetric::warning("Density gap", "2.1e-04"),
        ];
        let svg = render_molecule_svg(&mol, &options).expect("SVG render failed");
        assert!(svg.contains("Figure Metrics"));
        assert!(svg.contains("Debug Diagnostics"));
        assert!(svg.contains("Mirror residual"));
        assert!(svg.contains("Density gap"));
    }

    #[test]
    fn test_density_overlay_renders_probability_cells() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol.clone()).expect("HF setup failed");
        let scf = hf.run_scf(100, 1e-6).expect("SCF failed");
        let options = DiagramOptions::default();
        let density_settings = DensitySettings::default();
        let svg = render_molecule_svg_with_density(
            &mol,
            &hf.basis,
            &scf.density,
            &options,
            &density_settings,
        )
        .expect("density render failed");
        assert!(svg.contains("fill-opacity"));
        assert!(svg.contains(&density_settings.color));
    }

    #[test]
    fn test_orbital_overlay_includes_orbital_geometry_guides() {
        let mol = Molecule::h2();
        let hf = HartreeFock::new(mol.clone()).expect("HF setup failed");
        let scf = hf.run_scf(100, 1e-6).expect("SCF failed");
        let options = DiagramOptions::default();
        let orbital = OrbitalSettings::default();
        let svg = render_molecule_svg_with_orbital(
            &mol,
            &hf.basis,
            &scf.coefficients,
            &options,
            &orbital,
        )
        .expect("orbital render failed");
        assert!(svg.contains("id='orbital-geometry-guides'"));
    }
}
