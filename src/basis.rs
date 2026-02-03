//! Basis set functions for quantum chemistry
//!
//! This module provides Gaussian basis functions in a functional style.

use crate::basis_data::STO3G_JSON;
use crate::linalg::{Matrix, matmul, transpose, zeros};
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
    pub cartesian_functions: Vec<BasisFunction>,
    cart_to_sph: Matrix,
    sph_to_cart: Matrix,
    spherical_size: usize,
}

#[derive(Debug, Clone)]
struct Shell {
    l: u32,
    start: usize,
    n_cart: usize,
    n_sph: usize,
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

fn cartesian_count(l: u32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

fn cartesian_to_spherical_matrix(l: u32) -> Result<Matrix, String> {
    let n_cart = cartesian_count(l);
    let n_sph = if l <= 1 { n_cart } else { (2 * l + 1) as usize };
    let mut mat = Matrix::zeros((n_sph, n_cart));

    match l {
        0 => {
            mat[[0, 0]] = 1.0;
        }
        1 => {
            // Cartesian order: x, y, z
            for i in 0..3 {
                mat[[i, i]] = 1.0;
            }
        }
        2 => {
            // Cartesian order from cartesian_exponents: xx, xy, xz, yy, yz, zz
            let inv_sqrt6 = 1.0 / 6.0_f64.sqrt();
            let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
            // d_{z^2}
            mat[[0, 0]] = -inv_sqrt6; // xx
            mat[[0, 3]] = -inv_sqrt6; // yy
            mat[[0, 5]] = 2.0 * inv_sqrt6; // zz
            // d_{x^2-y^2}
            mat[[1, 0]] = inv_sqrt2; // xx
            mat[[1, 3]] = -inv_sqrt2; // yy
            // d_{xy}
            mat[[2, 1]] = 1.0; // xy
            // d_{xz}
            mat[[3, 2]] = 1.0; // xz
            // d_{yz}
            mat[[4, 4]] = 1.0; // yz
        }
        _ => {
            return Err(format!(
                "spherical transform for l={} is not implemented",
                l
            ));
        }
    }

    Ok(mat)
}

fn build_spherical_transform(
    shells: &[Shell],
    n_cart_total: usize,
) -> Result<(Matrix, Matrix, usize), String> {
    let n_sph_total: usize = shells.iter().map(|s| s.n_sph as usize).sum();
    let mut cart_to_sph = Matrix::zeros((n_sph_total, n_cart_total));
    let mut sph_offset = 0usize;
    for shell in shells {
        let transform = cartesian_to_spherical_matrix(shell.l)?;
        for r in 0..shell.n_sph {
            for c in 0..shell.n_cart {
                cart_to_sph[[sph_offset + r, shell.start + c]] = transform[[r, c]];
            }
        }
        sph_offset += shell.n_sph;
    }

    let sph_to_cart = transpose(&cart_to_sph);
    Ok((cart_to_sph, sph_to_cart, n_sph_total))
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

fn hermite_coeffs(
    l1: u32,
    l2: u32,
    pa: f64,
    pb: f64,
    gamma: f64,
    mu: f64,
    ab: f64,
) -> Vec<f64> {
    let l1 = l1 as usize;
    let l2 = l2 as usize;
    let max_t = l1 + l2;
    let mut e = vec![vec![vec![0.0; max_t + 2]; l2 + 1]; l1 + 1];
    e[0][0][0] = (-mu * ab * ab).exp();

    for i in 0..l1 {
        for t in 0..=max_t {
            let term = e[i][0][t];
            let mut value = pa * term;
            if t > 0 {
                value += e[i][0][t - 1] / (2.0 * gamma);
            }
            value += (t as f64 + 1.0) * e[i][0][t + 1];
            e[i + 1][0][t] = value;
        }
    }

    for j in 0..l2 {
        for i in 0..=l1 {
            for t in 0..=max_t {
                let term = e[i][j][t];
                let mut value = pb * term;
                if t > 0 {
                    value += e[i][j][t - 1] / (2.0 * gamma);
                }
                value += (t as f64 + 1.0) * e[i][j][t + 1];
                e[i][j + 1][t] = value;
            }
        }
    }

    e[l1][l2][0..=max_t].to_vec()
}

fn overlap_1d(
    l1: u32,
    l2: u32,
    pa: f64,
    pb: f64,
    gamma: f64,
    mu: f64,
    ab: f64,
) -> f64 {
    let coeffs = hermite_coeffs(l1, l2, pa, pb, gamma, mu, ab);
    coeffs[0] * (std::f64::consts::PI / gamma).sqrt()
}

fn kinetic_1d(
    l1: u32,
    l2: u32,
    beta: f64,
    pa: f64,
    pb: f64,
    gamma: f64,
    mu: f64,
    ab: f64,
) -> f64 {
    let s = overlap_1d(l1, l2, pa, pb, gamma, mu, ab);
    let s_plus2 = overlap_1d(l1, l2 + 2, pa, pb, gamma, mu, ab);
    let s_minus2 = if l2 >= 2 {
        overlap_1d(l1, l2 - 2, pa, pb, gamma, mu, ab)
    } else {
        0.0
    };
    let l2f = l2 as f64;
    -0.5
        * (l2f * (l2f - 1.0) * s_minus2
            - 2.0 * beta * (2.0 * l2f + 1.0) * s
            + 4.0 * beta * beta * s_plus2)
}

fn overlap_primitive(
    a: &BasisFunction,
    b: &BasisFunction,
    alpha: f64,
    beta: f64,
) -> f64 {
    let gamma = alpha + beta;
    let mu = alpha * beta / gamma;
    let p = [
        (alpha * a.center[0] + beta * b.center[0]) / gamma,
        (alpha * a.center[1] + beta * b.center[1]) / gamma,
        (alpha * a.center[2] + beta * b.center[2]) / gamma,
    ];
    let pa = [p[0] - a.center[0], p[1] - a.center[1], p[2] - a.center[2]];
    let pb = [p[0] - b.center[0], p[1] - b.center[1], p[2] - b.center[2]];
    let ab = [
        a.center[0] - b.center[0],
        a.center[1] - b.center[1],
        a.center[2] - b.center[2],
    ];

    let sx = overlap_1d(a.angular[0], b.angular[0], pa[0], pb[0], gamma, mu, ab[0]);
    let sy = overlap_1d(a.angular[1], b.angular[1], pa[1], pb[1], gamma, mu, ab[1]);
    let sz = overlap_1d(a.angular[2], b.angular[2], pa[2], pb[2], gamma, mu, ab[2]);
    sx * sy * sz
}

fn kinetic_primitive(
    a: &BasisFunction,
    b: &BasisFunction,
    alpha: f64,
    beta: f64,
) -> f64 {
    let gamma = alpha + beta;
    let mu = alpha * beta / gamma;
    let p = [
        (alpha * a.center[0] + beta * b.center[0]) / gamma,
        (alpha * a.center[1] + beta * b.center[1]) / gamma,
        (alpha * a.center[2] + beta * b.center[2]) / gamma,
    ];
    let pa = [p[0] - a.center[0], p[1] - a.center[1], p[2] - a.center[2]];
    let pb = [p[0] - b.center[0], p[1] - b.center[1], p[2] - b.center[2]];
    let ab = [
        a.center[0] - b.center[0],
        a.center[1] - b.center[1],
        a.center[2] - b.center[2],
    ];

    let sx = overlap_1d(a.angular[0], b.angular[0], pa[0], pb[0], gamma, mu, ab[0]);
    let sy = overlap_1d(a.angular[1], b.angular[1], pa[1], pb[1], gamma, mu, ab[1]);
    let sz = overlap_1d(a.angular[2], b.angular[2], pa[2], pb[2], gamma, mu, ab[2]);

    let tx = kinetic_1d(a.angular[0], b.angular[0], beta, pa[0], pb[0], gamma, mu, ab[0]);
    let ty = kinetic_1d(a.angular[1], b.angular[1], beta, pa[1], pb[1], gamma, mu, ab[1]);
    let tz = kinetic_1d(a.angular[2], b.angular[2], beta, pa[2], pb[2], gamma, mu, ab[2]);

    tx * sy * sz + sx * ty * sz + sx * sy * tz
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let poly = (((a5 * t + a4) * t + a3) * t + a2) * t + a1;
    let y = 1.0 - poly * t * (-x * x).exp();
    sign * y
}

fn boys(n: usize, t: f64) -> f64 {
    if t < 1e-8 {
        let mut sum = 1.0 / (2.0 * n as f64 + 1.0);
        let mut term = 1.0;
        for k in 1..6 {
            term *= -t / k as f64;
            sum += term / (2.0 * n as f64 + 2.0 * k as f64 + 1.0);
        }
        return sum;
    }

    let sqrt_t = t.sqrt();
    let f0 = 0.5 * (std::f64::consts::PI).sqrt() * erf_approx(sqrt_t) / sqrt_t;
    if n == 0 {
        return f0;
    }
    let exp_t = (-t).exp();
    let mut f = f0;
    for m in 0..n {
        f = ((2.0 * m as f64 + 1.0) * f - exp_t) / (2.0 * t);
    }
    f
}

fn hermite_coulomb(
    t_max: usize,
    u_max: usize,
    v_max: usize,
    n_max: usize,
    gamma: f64,
    pc: [f64; 3],
) -> Vec<Vec<Vec<Vec<f64>>>> {
    let mut r = vec![vec![vec![vec![0.0; n_max + 1]; v_max + 1]; u_max + 1]; t_max + 1];
    let t = gamma * (pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]);
    for n in 0..=n_max {
        r[0][0][0][n] = (-2.0 * gamma).powi(n as i32) * boys(n, t);
    }

    if n_max == 0 {
        return r;
    }

    for tx in 1..=t_max {
        for n in (0..=n_max - 1).rev() {
            let term1 = if tx >= 2 {
                (tx - 1) as f64 * r[tx - 2][0][0][n + 1]
            } else {
                0.0
            };
            r[tx][0][0][n] = term1 + pc[0] * r[tx - 1][0][0][n + 1];
        }
    }

    for uy in 1..=u_max {
        for tx in 0..=t_max {
            for n in (0..=n_max - 1).rev() {
                let term1 = if uy >= 2 {
                    (uy - 1) as f64 * r[tx][uy - 2][0][n + 1]
                } else {
                    0.0
                };
                r[tx][uy][0][n] = term1 + pc[1] * r[tx][uy - 1][0][n + 1];
            }
        }
    }

    for vz in 1..=v_max {
        for tx in 0..=t_max {
            for uy in 0..=u_max {
                for n in (0..=n_max - 1).rev() {
                    let term1 = if vz >= 2 {
                        (vz - 1) as f64 * r[tx][uy][vz - 2][n + 1]
                    } else {
                        0.0
                    };
                    r[tx][uy][vz][n] = term1 + pc[2] * r[tx][uy][vz - 1][n + 1];
                }
            }
        }
    }

    r
}

fn nuclear_attraction_primitive(
    a: &BasisFunction,
    b: &BasisFunction,
    alpha: f64,
    beta: f64,
    center: [f64; 3],
) -> f64 {
    let gamma = alpha + beta;
    let mu = alpha * beta / gamma;
    let p = [
        (alpha * a.center[0] + beta * b.center[0]) / gamma,
        (alpha * a.center[1] + beta * b.center[1]) / gamma,
        (alpha * a.center[2] + beta * b.center[2]) / gamma,
    ];
    let pa = [p[0] - a.center[0], p[1] - a.center[1], p[2] - a.center[2]];
    let pb = [p[0] - b.center[0], p[1] - b.center[1], p[2] - b.center[2]];
    let ab = [
        a.center[0] - b.center[0],
        a.center[1] - b.center[1],
        a.center[2] - b.center[2],
    ];
    let pc = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];

    let ex = hermite_coeffs(a.angular[0], b.angular[0], pa[0], pb[0], gamma, mu, ab[0]);
    let ey = hermite_coeffs(a.angular[1], b.angular[1], pa[1], pb[1], gamma, mu, ab[1]);
    let ez = hermite_coeffs(a.angular[2], b.angular[2], pa[2], pb[2], gamma, mu, ab[2]);

    let t_max = ex.len() - 1;
    let u_max = ey.len() - 1;
    let v_max = ez.len() - 1;
    let n_max = t_max + u_max + v_max;
    let r = hermite_coulomb(t_max, u_max, v_max, n_max, gamma, pc);

    let mut sum = 0.0;
    for t in 0..=t_max {
        for u in 0..=u_max {
            for v in 0..=v_max {
                sum += ex[t] * ey[u] * ez[v] * r[t][u][v][0];
            }
        }
    }

    -2.0 * std::f64::consts::PI / gamma * sum
}

fn electron_repulsion_primitive(
    a: &BasisFunction,
    b: &BasisFunction,
    c: &BasisFunction,
    d: &BasisFunction,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
) -> f64 {
    let p = alpha + beta;
    let q = gamma + delta;
    let mu = alpha * beta / p;
    let nu = gamma * delta / q;
    let p_center = [
        (alpha * a.center[0] + beta * b.center[0]) / p,
        (alpha * a.center[1] + beta * b.center[1]) / p,
        (alpha * a.center[2] + beta * b.center[2]) / p,
    ];
    let q_center = [
        (gamma * c.center[0] + delta * d.center[0]) / q,
        (gamma * c.center[1] + delta * d.center[1]) / q,
        (gamma * c.center[2] + delta * d.center[2]) / q,
    ];

    let pa = [p_center[0] - a.center[0], p_center[1] - a.center[1], p_center[2] - a.center[2]];
    let pb = [p_center[0] - b.center[0], p_center[1] - b.center[1], p_center[2] - b.center[2]];
    let qc = [q_center[0] - c.center[0], q_center[1] - c.center[1], q_center[2] - c.center[2]];
    let qd = [q_center[0] - d.center[0], q_center[1] - d.center[1], q_center[2] - d.center[2]];

    let ab = [a.center[0] - b.center[0], a.center[1] - b.center[1], a.center[2] - b.center[2]];
    let cd = [c.center[0] - d.center[0], c.center[1] - d.center[1], c.center[2] - d.center[2]];
    let pq = [p_center[0] - q_center[0], p_center[1] - q_center[1], p_center[2] - q_center[2]];

    let ex_ab = hermite_coeffs(a.angular[0], b.angular[0], pa[0], pb[0], p, mu, ab[0]);
    let ey_ab = hermite_coeffs(a.angular[1], b.angular[1], pa[1], pb[1], p, mu, ab[1]);
    let ez_ab = hermite_coeffs(a.angular[2], b.angular[2], pa[2], pb[2], p, mu, ab[2]);

    let ex_cd = hermite_coeffs(c.angular[0], d.angular[0], qc[0], qd[0], q, nu, cd[0]);
    let ey_cd = hermite_coeffs(c.angular[1], d.angular[1], qc[1], qd[1], q, nu, cd[1]);
    let ez_cd = hermite_coeffs(c.angular[2], d.angular[2], qc[2], qd[2], q, nu, cd[2]);

    let t_max = ex_ab.len() - 1 + ex_cd.len() - 1;
    let u_max = ey_ab.len() - 1 + ey_cd.len() - 1;
    let v_max = ez_ab.len() - 1 + ez_cd.len() - 1;
    let n_max = t_max + u_max + v_max;

    let rho = p * q / (p + q);
    let r = hermite_coulomb(t_max, u_max, v_max, n_max, rho, pq);

    let mut sum = 0.0;
    for t in 0..ex_ab.len() {
        for u in 0..ey_ab.len() {
            for v in 0..ez_ab.len() {
                let eab = ex_ab[t] * ey_ab[u] * ez_ab[v];
                if eab == 0.0 {
                    continue;
                }
                for tp in 0..ex_cd.len() {
                    for up in 0..ey_cd.len() {
                        for vp in 0..ez_cd.len() {
                            let ecd = ex_cd[tp] * ey_cd[up] * ez_cd[vp];
                            if ecd == 0.0 {
                                continue;
                            }
                            sum += eab * ecd * r[t + tp][u + up][v + vp][0];
                        }
                    }
                }
            }
        }
    }

    let prefactor = 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt());
    prefactor * sum
}

impl BasisSet {
    /// Creates a minimal basis set for a molecule
    pub fn minimal(molecule: &Molecule) -> Result<Self, String> {
        Self::sto3g(molecule)
    }

    /// Builds STO-3G basis functions using data from the Basis Set Exchange.
    pub fn sto3g(molecule: &Molecule) -> Result<Self, String> {
        let data = sto3g_data()?;
        let mut cartesian_functions = Vec::new();
        let mut shells = Vec::new();

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

                    let start = cartesian_functions.len();
                    for angular in cartesian_exponents(l) {
                        let primitives = build_primitives(&exponents, &coeffs, angular);
                        cartesian_functions.push(BasisFunction::new(atom.position, primitives, angular));
                    }
                    let n_cart = cartesian_count(l);
                    let n_sph = if l <= 1 { n_cart } else { (2 * l + 1) as usize };
                    shells.push(Shell {
                        l,
                        start,
                        n_cart,
                        n_sph,
                    });
                }
            }
        }

        let (cart_to_sph, sph_to_cart, spherical_size) =
            build_spherical_transform(&shells, cartesian_functions.len())?;

        Ok(Self {
            cartesian_functions,
            cart_to_sph,
            sph_to_cart,
            spherical_size,
        })
    }

    /// Maximum angular momentum (l) in the basis.
    pub fn max_angular_momentum(&self) -> u32 {
        self.cartesian_functions
            .iter()
            .map(|f| f.angular[0] + f.angular[1] + f.angular[2])
            .max()
            .unwrap_or(0)
    }

    /// Number of basis functions
    pub fn size(&self) -> usize {
        self.spherical_size
    }

    pub fn cartesian_size(&self) -> usize {
        self.cartesian_functions.len()
    }

    pub fn to_cartesian_coefficients(&self, coeffs: &Matrix) -> Matrix {
        matmul(&self.sph_to_cart, coeffs)
    }

    /// Computes overlap matrix S
    pub fn overlap_matrix(&self) -> Matrix {
        let n_cart = self.cartesian_size();
        let mut s_cart = zeros(n_cart, n_cart);

        for i in 0..n_cart {
            for j in 0..=i {
                let sij = self.overlap(&self.cartesian_functions[i], &self.cartesian_functions[j]);
                s_cart[[i, j]] = sij;
                s_cart[[j, i]] = sij;
            }
        }

        let s_sph = matmul(&self.cart_to_sph, &matmul(&s_cart, &self.sph_to_cart));
        s_sph
    }

    /// Computes overlap between two basis functions
    fn overlap(&self, a: &BasisFunction, b: &BasisFunction) -> f64 {
        let mut total = 0.0;
        for pa in &a.primitives {
            for pb in &b.primitives {
                let s = overlap_primitive(a, b, pa.exponent, pb.exponent);
                total += pa.coefficient * pb.coefficient * s;
            }
        }

        total
    }

    /// Computes one-electron (kinetic + nuclear attraction) matrix
    pub fn core_hamiltonian(&self, molecule: &Molecule) -> Matrix {
        let n_cart = self.cartesian_size();
        let mut h_cart = zeros(n_cart, n_cart);

        for i in 0..n_cart {
            for j in 0..=i {
                let hij = self.kinetic(&self.cartesian_functions[i], &self.cartesian_functions[j])
                    + self.nuclear_attraction(&self.cartesian_functions[i], &self.cartesian_functions[j], molecule);
                h_cart[[i, j]] = hij;
                h_cart[[j, i]] = hij;
            }
        }

        matmul(&self.cart_to_sph, &matmul(&h_cart, &self.sph_to_cart))
    }

    /// Computes kinetic energy integral (simplified)
    fn kinetic(&self, a: &BasisFunction, b: &BasisFunction) -> f64 {
        let mut total = 0.0;
        for pa in &a.primitives {
            for pb in &b.primitives {
                let t = kinetic_primitive(a, b, pa.exponent, pb.exponent);
                total += pa.coefficient * pb.coefficient * t;
            }
        }
        total
    }

    /// Computes nuclear attraction integral (simplified)
    fn nuclear_attraction(&self, a: &BasisFunction, b: &BasisFunction, molecule: &Molecule) -> f64 {
        let mut v = 0.0;
        for atom in &molecule.atoms {
            let mut term = 0.0;
            for pa in &a.primitives {
                for pb in &b.primitives {
                    let value = nuclear_attraction_primitive(
                        a,
                        b,
                        pa.exponent,
                        pb.exponent,
                        atom.position,
                    );
                    term += pa.coefficient * pb.coefficient * value;
                }
            }
            v += atom.nuclear_charge() * term;
        }

        v
    }

    /// Computes two-electron repulsion integrals (ij|kl)
    pub fn two_electron_integrals(&self) -> Vec<f64> {
        let n_cart = self.cartesian_size();
        let size_cart = n_cart * n_cart * n_cart * n_cart;
        let mut eri_cart = vec![0.0; size_cart];

        for i in 0..n_cart {
            for j in 0..n_cart {
                for k in 0..n_cart {
                    for l in 0..n_cart {
                        let mut value = 0.0;
                        let a = &self.cartesian_functions[i];
                        let b = &self.cartesian_functions[j];
                        let c = &self.cartesian_functions[k];
                        let d = &self.cartesian_functions[l];
                        for pa in &a.primitives {
                            for pb in &b.primitives {
                                for pc in &c.primitives {
                                    for pd in &d.primitives {
                                        let integral = electron_repulsion_primitive(
                                            a,
                                            b,
                                            c,
                                            d,
                                            pa.exponent,
                                            pb.exponent,
                                            pc.exponent,
                                            pd.exponent,
                                        );
                                        value += pa.coefficient
                                            * pb.coefficient
                                            * pc.coefficient
                                            * pd.coefficient
                                            * integral;
                                    }
                                }
                            }
                        }
                        let idx = ((i * n_cart + j) * n_cart + k) * n_cart + l;
                        eri_cart[idx] = value;
                    }
                }
            }
        }

        self.transform_eri_cartesian_to_spherical(&eri_cart)
    }

    fn transform_eri_cartesian_to_spherical(&self, eri_cart: &[f64]) -> Vec<f64> {
        let n_cart = self.cartesian_size();
        let n_sph = self.size();
        let idx4 = |a: usize, b: usize, c: usize, d: usize, n: usize| -> usize {
            ((a * n + b) * n + c) * n + d
        };

        let mut t = vec![0.0; n_sph * n_cart * n_cart * n_cart];
        for a in 0..n_sph {
            for i in 0..n_cart {
                let c_ai = self.cart_to_sph[[a, i]];
                if c_ai == 0.0 {
                    continue;
                }
                for j in 0..n_cart {
                    for k in 0..n_cart {
                        for l in 0..n_cart {
                            let idx_cart = idx4(i, j, k, l, n_cart);
                            let idx_t = ((a * n_cart + j) * n_cart + k) * n_cart + l;
                            t[idx_t] += c_ai * eri_cart[idx_cart];
                        }
                    }
                }
            }
        }

        let mut u = vec![0.0; n_sph * n_sph * n_cart * n_cart];
        for a in 0..n_sph {
            for b in 0..n_sph {
                for j in 0..n_cart {
                    let c_bj = self.cart_to_sph[[b, j]];
                    if c_bj == 0.0 {
                        continue;
                    }
                    for k in 0..n_cart {
                        for l in 0..n_cart {
                            let idx_t = ((a * n_cart + j) * n_cart + k) * n_cart + l;
                            let idx_u = ((a * n_sph + b) * n_cart + k) * n_cart + l;
                            u[idx_u] += c_bj * t[idx_t];
                        }
                    }
                }
            }
        }

        let mut v = vec![0.0; n_sph * n_sph * n_sph * n_cart];
        for a in 0..n_sph {
            for b in 0..n_sph {
                for c in 0..n_sph {
                    for k in 0..n_cart {
                        let c_ck = self.cart_to_sph[[c, k]];
                        if c_ck == 0.0 {
                            continue;
                        }
                        for l in 0..n_cart {
                            let idx_u = ((a * n_sph + b) * n_cart + k) * n_cart + l;
                            let idx_v = ((a * n_sph + b) * n_sph + c) * n_cart + l;
                            v[idx_v] += c_ck * u[idx_u];
                        }
                    }
                }
            }
        }

        let mut eri_sph = vec![0.0; n_sph * n_sph * n_sph * n_sph];
        for a in 0..n_sph {
            for b in 0..n_sph {
                for c in 0..n_sph {
                    for d in 0..n_sph {
                        for l in 0..n_cart {
                            let c_dl = self.cart_to_sph[[d, l]];
                            if c_dl == 0.0 {
                                continue;
                            }
                            let idx_v = ((a * n_sph + b) * n_sph + c) * n_cart + l;
                            let idx_sph = idx4(a, b, c, d, n_sph);
                            eri_sph[idx_sph] += c_dl * v[idx_v];
                        }
                    }
                }
            }
        }

        eri_sph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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

    #[test]
    fn test_d_transform_orthonormal() {
        let m = cartesian_to_spherical_matrix(2).unwrap();
        let mtm = matmul(&m, &transpose(&m));
        for i in 0..5 {
            assert_abs_diff_eq!(mtm[[i, i]], 1.0, epsilon = 1e-10);
            for j in 0..5 {
                if i != j {
                    assert_abs_diff_eq!(mtm[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }
}
