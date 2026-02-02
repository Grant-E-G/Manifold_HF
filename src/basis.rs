//! Basis set functions for quantum chemistry
//!
//! This module provides Gaussian basis functions in a functional style.

use crate::basis_data::STO3G_JSON;
use crate::linalg::{Matrix, zeros};
use crate::molecule::Molecule;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

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
    /// Angular momentum exponents [l, m, n]
    pub angular: [u32; 3],
}

impl BasisFunction {
    pub fn new(center: [f64; 3], primitives: Vec<GaussianPrimitive>, angular: [u32; 3]) -> Self {
        Self {
            center,
            primitives,
            angular,
        }
    }

    /// Evaluates the basis function at a given point
    pub fn evaluate(&self, point: [f64; 3]) -> f64 {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let r2 = dx * dx + dy * dy + dz * dz;

        let ang = dx.powi(self.angular[0] as i32)
            * dy.powi(self.angular[1] as i32)
            * dz.powi(self.angular[2] as i32);

        self.primitives
            .iter()
            .map(|p| p.coefficient * (-p.exponent * r2).exp())
            .sum::<f64>()
            * ang
    }
}

/// STO-3G basis set
pub struct BasisSet {
    pub functions: Vec<BasisFunction>,
}

#[derive(Debug, Deserialize)]
struct BseBasis {
    elements: HashMap<String, BseElement>,
}

#[derive(Debug, Deserialize)]
struct BseElement {
    electron_shells: Vec<BseShell>,
}

#[derive(Debug, Deserialize)]
struct BseShell {
    angular_momentum: Vec<u32>,
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

/// STO-3G data is sourced from the Basis Set Exchange (see data/sto-3g.SOURCE.txt).
fn sto3g_data() -> Result<&'static BseBasis, String> {
    static STO3G_CACHE: OnceLock<Result<BseBasis, String>> = OnceLock::new();
    match STO3G_CACHE.get_or_init(|| serde_json::from_str(STO3G_JSON).map_err(|e| e.to_string()))
    {
        Ok(data) => Ok(data),
        Err(err) => Err(err.clone()),
    }
}

fn parse_f64_list(values: &[String]) -> Result<Vec<f64>, String> {
    values
        .iter()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("{} ({})", value, err))
        })
        .collect()
}

fn cartesian_exponents(l: u32) -> Vec<[u32; 3]> {
    let mut exps = Vec::new();
    for lx in (0..=l).rev() {
        for ly in (0..=l - lx).rev() {
            let lz = l - lx - ly;
            exps.push([lx, ly, lz]);
        }
    }
    exps
}

fn double_factorial(mut n: i32) -> f64 {
    if n <= 0 {
        return 1.0;
    }
    let mut value = 1.0;
    while n > 1 {
        value *= n as f64;
        n -= 2;
    }
    value
}

fn primitive_normalization(alpha: f64, angular: [u32; 3]) -> f64 {
    let l = angular[0] as i32;
    let m = angular[1] as i32;
    let n = angular[2] as i32;
    let lmn = (l + m + n) as f64;
    let prefactor = (2.0 * alpha / std::f64::consts::PI).powf(0.75);
    let ang_factor = (4.0 * alpha).powf(0.5 * lmn);
    let denom = (double_factorial(2 * l - 1)
        * double_factorial(2 * m - 1)
        * double_factorial(2 * n - 1))
        .sqrt();
    prefactor * ang_factor / denom
}

fn primitive_overlap_same_center(alpha: f64, beta: f64, angular: [u32; 3]) -> f64 {
    let l = angular[0] as i32;
    let m = angular[1] as i32;
    let n = angular[2] as i32;
    let gamma = alpha + beta;
    let lsum = l + m + n;
    let df = double_factorial(2 * l - 1)
        * double_factorial(2 * m - 1)
        * double_factorial(2 * n - 1);
    let denom = 2.0_f64.powi(lsum) * gamma.powf(lsum as f64 + 1.5);
    std::f64::consts::PI.powf(1.5) * df / denom
}

fn build_primitives(
    exponents: &[f64],
    coefficients: &[f64],
    angular: [u32; 3],
) -> Vec<GaussianPrimitive> {
    let mut primitives: Vec<GaussianPrimitive> = exponents
        .iter()
        .zip(coefficients.iter())
        .map(|(&exp, &coeff)| {
            let norm = primitive_normalization(exp, angular);
            GaussianPrimitive::new(exp, coeff * norm)
        })
        .collect();

    let mut overlap_sum = 0.0;
    for i in 0..primitives.len() {
        for j in 0..primitives.len() {
            let s = primitive_overlap_same_center(
                primitives[i].exponent,
                primitives[j].exponent,
                angular,
            );
            overlap_sum += primitives[i].coefficient * primitives[j].coefficient * s;
        }
    }

    if overlap_sum > 0.0 {
        let norm = 1.0 / overlap_sum.sqrt();
        for prim in &mut primitives {
            prim.coefficient *= norm;
        }
    }

    primitives
}

impl BasisSet {
    /// Creates a minimal basis set for a molecule
    pub fn minimal(molecule: &Molecule) -> Result<Self, String> {
        Self::sto3g(molecule)
    }

    /// Builds STO-3G basis functions using data from the Basis Set Exchange.
    pub fn sto3g(molecule: &Molecule) -> Result<Self, String> {
        let data = sto3g_data()?;
        let mut functions = Vec::new();

        for atom in &molecule.atoms {
            let key = atom.atomic_number.to_string();
            let element = data.elements.get(&key).ok_or_else(|| {
                format!("STO-3G data missing element Z={}", atom.atomic_number)
            })?;

            for shell in &element.electron_shells {
                if shell.angular_momentum.len() != shell.coefficients.len() {
                    return Err(format!(
                        "STO-3G shell mismatch for Z={}: angular_momentum {:?} coefficients {}",
                        atom.atomic_number,
                        shell.angular_momentum,
                        shell.coefficients.len()
                    ));
                }

                let exponents = parse_f64_list(&shell.exponents).map_err(|err| {
                    format!("STO-3G exponent parse error for Z={}: {}", atom.atomic_number, err)
                })?;

                for (idx, &l) in shell.angular_momentum.iter().enumerate() {
                    let coeffs = parse_f64_list(&shell.coefficients[idx]).map_err(|err| {
                        format!(
                            "STO-3G coefficient parse error for Z={}, l={}: {}",
                            atom.atomic_number, l, err
                        )
                    })?;
                    if coeffs.len() != exponents.len() {
                        return Err(format!(
                            "STO-3G shell length mismatch for Z={}, l={}: {} exponents vs {} coeffs",
                            atom.atomic_number,
                            l,
                            exponents.len(),
                            coeffs.len()
                        ));
                    }

                    for angular in cartesian_exponents(l) {
                        let primitives = build_primitives(&exponents, &coeffs, angular);
                        functions.push(BasisFunction::new(atom.position, primitives, angular));
                    }
                }
            }
        }

        Ok(Self { functions })
    }

    /// Maximum angular momentum (l) in the basis.
    pub fn max_angular_momentum(&self) -> u32 {
        self.functions
            .iter()
            .map(|f| f.angular[0] + f.angular[1] + f.angular[2])
            .max()
            .unwrap_or(0)
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
        let basis = BasisSet::minimal(&mol).unwrap();
        assert_eq!(basis.size(), 2); // Two hydrogen atoms = 2 basis functions
    }

    #[test]
    fn test_overlap_matrix() {
        let mol = Molecule::h2();
        let basis = BasisSet::minimal(&mol).unwrap();
        let s = basis.overlap_matrix();
        
        // Diagonal elements should be approximately 1 for normalized functions
        assert!(s[[0, 0]] > 0.5);
        assert!(s[[1, 1]] > 0.5);
    }
}
