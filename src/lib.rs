//! # Manifold Hartree-Fock
//!
//! This library implements the Hartree-Fock method for quantum chemistry calculations
//! using manifold optimization techniques. The code follows a functional programming style.
//!
//! ## Modules
//!
//! - `linalg`: Linear algebra utilities in functional style
//! - `molecule`: Molecular system representation
//! - `basis`: Basis set functions
//! - `scf`: Self-Consistent Field (Hartree-Fock) implementation
//! - `manifold`: Manifold optimization on Stiefel manifold

pub mod linalg;
pub mod molecule;
pub mod basis;
pub mod scf;
pub mod manifold;
pub mod visualize;

/// Re-export commonly used types
pub use linalg::{Matrix, Vector};
pub use molecule::Molecule;
pub use scf::HartreeFock;
pub use manifold::StiefelManifold;
pub use visualize::{
    DiagramOptions, LengthUnit, OrbitalSettings, ProjectionPlane, render_molecule_svg,
    render_molecule_svg_with_orbital,
};
