//! Basis set functions for quantum chemistry
//!
//! This module provides Gaussian basis functions in a functional style.

use crate::linalg::{Matrix, zeros};
use crate::molecule::Molecule;

/// Represents a primitive Gaussian function
#[derive(Debug, Clone)]
pub struct GaussianPrimitive {
    /// Exponent (alpha)
    pub exponent: f64,
    /// Contraction coefficient
    pub coefficient: f64,
}

impl GaussianPrimitive {
    pub fn new(exponent: f64, coefficient: f64) -> Self {
        Self { exponent, coefficient }
    }
}

/// Represents a contracted Gaussian basis function (CGBF)
#[derive(Debug, Clone)]
pub struct BasisFunction {
    /// Center position [x, y, z]
    pub center: [f64; 3],
    /// Primitive Gaussians
    pub primitives: Vec<GaussianPrimitive>,
}

impl BasisFunction {
    pub fn new(center: [f64; 3], primitives: Vec<GaussianPrimitive>) -> Self {
        Self {
            center,
            primitives,
        }
    }

    /// Evaluates the basis function at a given point
    pub fn evaluate(&self, point: [f64; 3]) -> f64 {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;

        self.primitives.iter()
            .map(|p| p.coefficient * (-p.exponent * r2).exp())
            .sum()
    }
}

/// A minimal basis set (STO-3G like)
pub struct BasisSet {
    pub functions: Vec<BasisFunction>,
}

impl BasisSet {
    /// Creates a minimal basis set for a molecule
    pub fn minimal(molecule: &Molecule) -> Self {
        let functions = molecule.atoms.iter()
            .flat_map(|atom| Self::minimal_atom_basis(atom.atomic_number, atom.position))
            .collect();

        Self { functions }
    }

    /// Creates minimal basis for a single atom (STO-3G style)
    fn minimal_atom_basis(atomic_number: u32, position: [f64; 3]) -> Vec<BasisFunction> {
        match atomic_number {
            1 => {
                // Hydrogen: 1s orbital (STO-3G)
                vec![BasisFunction::new(
                    position,
                    vec![
                        GaussianPrimitive::new(3.425250914, 0.154328967),
                        GaussianPrimitive::new(0.623913730, 0.535328142),
                        GaussianPrimitive::new(0.168855404, 0.444634542),
                    ],
                )]
            }
            6..=10 => {
                // C-Ne: 1s, 2s, 2px, 2py, 2pz orbitals (5 basis functions)
                // NOTE: Simplified implementation - p orbitals use same angular=[0,0,0]
                // In a complete implementation, these would have angular=[1,0,0], [0,1,0], [0,0,1]
                // and require proper p-type integral evaluation
                vec![
                    // 1s orbital (core)
                    BasisFunction::new(
                        position,
                        vec![
                            GaussianPrimitive::new(10.0, 0.4),
                            GaussianPrimitive::new(2.0, 0.6),
                        ],
                    ),
                    // 2s orbital (valence)
                    BasisFunction::new(
                        position,
                        vec![
                            GaussianPrimitive::new(1.5, 0.5),
                            GaussianPrimitive::new(0.4, 0.5),
                        ],
                    ),
                    // 2p orbitals (3 functions for x, y, z)
                    BasisFunction::new(
                        position,
                        vec![
                            GaussianPrimitive::new(1.0, 0.5),
                            GaussianPrimitive::new(0.3, 0.5),
                        ],
                    ),
                    BasisFunction::new(
                        position,
                        vec![
                            GaussianPrimitive::new(1.0, 0.5),
                            GaussianPrimitive::new(0.3, 0.5),
                        ],
                    ),
                    BasisFunction::new(
                        position,
                        vec![
                            GaussianPrimitive::new(1.0, 0.5),
                            GaussianPrimitive::new(0.3, 0.5),
                        ],
                    ),
                ]
            }
            2..=5 => {
                // He-B: 1s and 2s orbitals (simplified)
                vec![
                    BasisFunction::new(
                        position,
                        vec![
                            GaussianPrimitive::new(5.0, 0.5),
                            GaussianPrimitive::new(1.0, 0.5),
                        ],
                    ),
                    BasisFunction::new(
                        position,
                        vec![
                            GaussianPrimitive::new(1.0, 0.5),
                            GaussianPrimitive::new(0.2, 0.5),
                        ],
                    ),
                ]
            }
            _ => vec![],
        }
    }

    /// Number of basis functions
    pub fn size(&self) -> usize {
        self.functions.len()
    }

    /// Computes overlap matrix S
    pub fn overlap_matrix(&self) -> Matrix {
        let n = self.size();
        let mut s = zeros(n, n);

        for i in 0..n {
            for j in 0..=i {
                let sij = self.overlap(&self.functions[i], &self.functions[j]);
                s[[i, j]] = sij;
                s[[j, i]] = sij;
            }
        }

        s
    }

    /// Computes overlap between two basis functions
    fn overlap(&self, a: &BasisFunction, b: &BasisFunction) -> f64 {
        // Simplified overlap for s-type Gaussians
        let dx = a.center[0] - b.center[0];
        let dy = a.center[1] - b.center[1];
        let dz = a.center[2] - b.center[2];
        let rab2 = dx * dx + dy * dy + dz * dz;

        let mut total = 0.0;
        for pa in &a.primitives {
            for pb in &b.primitives {
                let gamma = pa.exponent + pb.exponent;
                let prefactor = (std::f64::consts::PI / gamma).powf(1.5);
                let exponential = (-pa.exponent * pb.exponent / gamma * rab2).exp();
                total += pa.coefficient * pb.coefficient * prefactor * exponential;
            }
        }

        total
    }

    /// Computes one-electron (kinetic + nuclear attraction) matrix
    pub fn core_hamiltonian(&self, molecule: &Molecule) -> Matrix {
        let n = self.size();
        let mut h = zeros(n, n);

        for i in 0..n {
            for j in 0..=i {
                let hij = self.kinetic(&self.functions[i], &self.functions[j])
                    + self.nuclear_attraction(&self.functions[i], &self.functions[j], molecule);
                h[[i, j]] = hij;
                h[[j, i]] = hij;
            }
        }

        h
    }

    /// Computes kinetic energy integral (simplified)
    fn kinetic(&self, a: &BasisFunction, b: &BasisFunction) -> f64 {
        // Very simplified - in practice needs proper integral evaluation
        let sab = self.overlap(a, b);
        
        // Approximate kinetic energy
        let mut t = 0.0;
        for pa in &a.primitives {
            for pb in &b.primitives {
                t += pa.coefficient * pb.coefficient * 
                     pa.exponent * pb.exponent / (pa.exponent + pb.exponent) * sab;
            }
        }
        
        t * 0.5
    }

    /// Computes nuclear attraction integral (simplified)
    fn nuclear_attraction(&self, a: &BasisFunction, b: &BasisFunction, molecule: &Molecule) -> f64 {
        // Very simplified - in practice needs proper integral evaluation
        let sab = self.overlap(a, b);
        
        let mut v = 0.0;
        for atom in &molecule.atoms {
            let dx_a = a.center[0] - atom.position[0];
            let dy_a = a.center[1] - atom.position[1];
            let dz_a = a.center[2] - atom.position[2];
            let ra = (dx_a * dx_a + dy_a * dy_a + dz_a * dz_a).sqrt().max(0.1);

            v -= atom.nuclear_charge() * sab / ra;
        }

        v
    }

    /// Computes two-electron repulsion integrals (very simplified)
    pub fn two_electron_integrals(&self) -> Vec<f64> {
        // Approximate ERI scale factor
        const APPROX_ERI_VALUE: f64 = 0.1;
        
        let n = self.size();
        let size = n * n * n * n;
        
        // Simplified - just use approximate values
        // NOTE: Real implementation would compute (ij|kl) integrals properly
        vec![APPROX_ERI_VALUE; size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_h2() {
        let mol = Molecule::h2();
        let basis = BasisSet::minimal(&mol);
        assert_eq!(basis.size(), 2); // Two hydrogen atoms = 2 basis functions
    }

    #[test]
    fn test_overlap_matrix() {
        let mol = Molecule::h2();
        let basis = BasisSet::minimal(&mol);
        let s = basis.overlap_matrix();
        
        // Diagonal elements should be approximately 1 for normalized functions
        assert!(s[[0, 0]] > 0.5);
        assert!(s[[1, 1]] > 0.5);
    }
}
