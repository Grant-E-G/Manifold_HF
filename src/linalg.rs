//! Linear algebra utilities in functional style
//!
//! This module provides pure functional wrappers for matrix and vector manipulations.
//! All operations are implemented in pure Rust without external BLAS/LAPACK dependencies.

use ndarray::{Array1, Array2};

/// Type alias for matrix (2D array)
pub type Matrix = Array2<f64>;

/// Type alias for vector (1D array)
pub type Vector = Array1<f64>;

/// Creates an identity matrix of given size
///
/// # Examples
/// ```
/// use manifold_hf::linalg::identity;
/// let i = identity(3);
/// assert_eq!(i.shape(), &[3, 3]);
/// ```
pub fn identity(n: usize) -> Matrix {
    Array2::eye(n)
}

/// Creates a zero matrix of given dimensions
pub fn zeros(rows: usize, cols: usize) -> Matrix {
    Array2::zeros((rows, cols))
}

/// Matrix multiplication (functional style)
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    a.dot(b)
}

/// Matrix transpose (returns new matrix)
pub fn transpose(m: &Matrix) -> Matrix {
    m.t().to_owned()
}

/// Computes eigenvalues and eigenvectors using Jacobi algorithm
///
/// This is a pure Rust implementation for symmetric matrices
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column vectors
pub fn eig(m: &Matrix) -> Result<(Vector, Matrix), String> {
    let n = m.nrows();
    if n != m.ncols() {
        return Err("Matrix must be square".to_string());
    }

    // Use Jacobi algorithm for eigenvalue decomposition
    jacobi_eigenvalue(m)
}

/// Jacobi eigenvalue algorithm for symmetric matrices
fn jacobi_eigenvalue(a: &Matrix) -> Result<(Vector, Matrix), String> {
    let n = a.nrows();
    let mut a_work = a.clone();
    let mut v = identity(n);
    
    let max_iter = 100;
    let tolerance = 1e-10;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let (p, q, max_val) = find_max_offdiag(&a_work);
        
        if max_val.abs() < tolerance {
            break;
        }

        // Compute rotation angle
        let theta = if (a_work[[p, p]] - a_work[[q, q]]).abs() < 1e-10 {
            std::f64::consts::PI / 4.0
        } else {
            0.5 * (2.0 * a_work[[p, q]] / (a_work[[p, p]] - a_work[[q, q]])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation
        apply_jacobi_rotation(&mut a_work, &mut v, p, q, c, s);
    }

    // Extract eigenvalues from diagonal
    let eigenvalues = Vector::from_vec((0..n).map(|i| a_work[[i, i]]).collect());
    
    Ok((eigenvalues, v))
}

/// Finds the largest off-diagonal element
fn find_max_offdiag(a: &Matrix) -> (usize, usize, f64) {
    let n = a.nrows();
    let mut max_val = 0.0;
    let mut max_p = 0;
    let mut max_q = 1;

    for i in 0..n {
        for j in (i + 1)..n {
            let val = a[[i, j]].abs();
            if val > max_val {
                max_val = val;
                max_p = i;
                max_q = j;
            }
        }
    }

    (max_p, max_q, max_val)
}

/// Applies Jacobi rotation to matrix and eigenvector matrix
fn apply_jacobi_rotation(a: &mut Matrix, v: &mut Matrix, p: usize, q: usize, c: f64, s: f64) {
    let n = a.nrows();

    // Rotate rows and columns of A
    for i in 0..n {
        if i != p && i != q {
            let aip = a[[i, p]];
            let aiq = a[[i, q]];
            a[[i, p]] = c * aip - s * aiq;
            a[[p, i]] = a[[i, p]];
            a[[i, q]] = s * aip + c * aiq;
            a[[q, i]] = a[[i, q]];
        }
    }

    let app = a[[p, p]];
    let aqq = a[[q, q]];
    let apq = a[[p, q]];

    a[[p, p]] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    a[[q, q]] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    a[[p, q]] = 0.0;
    a[[q, p]] = 0.0;

    // Update eigenvectors
    for i in 0..n {
        let vip = v[[i, p]];
        let viq = v[[i, q]];
        v[[i, p]] = c * vip - s * viq;
        v[[i, q]] = s * vip + c * viq;
    }
}

/// Computes the trace of a matrix
pub fn trace(m: &Matrix) -> f64 {
    m.diag().sum()
}

/// Frobenius norm of a matrix
pub fn frobenius_norm(m: &Matrix) -> f64 {
    m.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Orthogonalizes columns of a matrix using Gram-Schmidt
pub fn orthogonalize(m: &Matrix) -> Result<Matrix, String> {
    gram_schmidt(m)
}

/// Gram-Schmidt orthogonalization
fn gram_schmidt(a: &Matrix) -> Result<Matrix, String> {
    let (n, k) = a.dim();
    let mut q = zeros(n, k);

    for j in 0..k {
        let mut v = a.column(j).to_owned();
        
        // Subtract projections onto previous vectors
        for i in 0..j {
            let qi = q.column(i);
            let proj = v.dot(&qi);
            v = v - &(&qi.to_owned() * proj);
        }

        // Normalize
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err("Linear dependent vectors".to_string());
        }

        let v_normalized = &v / norm;
        for i in 0..n {
            q[[i, j]] = v_normalized[i];
        }
    }

    Ok(q)
}

/// Projects a matrix onto the Stiefel manifold (orthogonal columns)
///
/// This ensures the columns are orthonormal
pub fn project_stiefel(m: &Matrix) -> Result<Matrix, String> {
    orthogonalize(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_identity() {
        let i = identity(3);
        assert_eq!(i[[0, 0]], 1.0);
        assert_eq!(i[[1, 1]], 1.0);
        assert_eq!(i[[0, 1]], 0.0);
    }

    #[test]
    fn test_matmul() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = matmul(&a, &b);
        assert_abs_diff_eq!(c[[0, 0]], 19.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace() {
        let m = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_abs_diff_eq!(trace(&m), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gram_schmidt() {
        let m = Array2::from_shape_vec((3, 2), vec![
            1.0, 1.0,
            0.0, 1.0,
            0.0, 0.0,
        ]).unwrap();
        
        let q = gram_schmidt(&m).unwrap();
        
        // Check orthonormality
        let qtq = matmul(&transpose(&q), &q);
        assert_abs_diff_eq!(qtq[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qtq[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qtq[[0, 1]], 0.0, epsilon = 1e-10);
    }
}
