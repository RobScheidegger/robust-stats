use ndarray::Array2;
use ndarray_linalg::{Eigh, Norm, UPLO};

use crate::matrix::FastMatrix;

pub fn robust_mean_pgd(
    x: &FastMatrix<f32>,
    epsilon: f32,
    output: &mut FastMatrix<f32>,
    num_iterations: Option<usize>,
) {
    let n = x.n;
    let d = x.d;
    let epsilon_n = (epsilon * n as f32).round() as usize;
    let step_size = 1.0 / (n as f32);

    let x = x.to_array_view();
    let xt = x.t().to_owned();
    let mut w = Array2::<f32>::ones((1, n)) / (n as f32);
    let mut dw = Array2::<f32>::zeros((n, n));
    let mut xw_outer = Array2::<f32>::zeros((d, d));

    for _ in 0..num_iterations.unwrap_or(10) {
        let xw = &xt * &w;
        for i in 0..n {
            dw[[i, i]] = w[[0, i]];
        }
        for i in 0..d {
            for j in 0..d {
                xw_outer[[i, j]] = xw[[0, i]] * xw[[0, j]];
            }
        }

        // check outer product here
        let sigma_w = &xt.dot(&dw).dot(&x) - &xw_outer;
        let (eigs, vecs) = sigma_w.eigh(UPLO::Lower).unwrap();
        // max eigenvalue
        let mut max_eig_idx = 0;
        let mut max_eig = eigs[0];
        for i in 1..d {
            let eig_norm = eigs[i];
            if eig_norm > max_eig {
                max_eig = eig_norm;
                max_eig_idx = i;
            }
        }
        let u = vecs.column(max_eig_idx).to_owned();

        let xu = x.dot(&u);
        let uxw = &u.dot(&xw)[[0]];
        let nabla_f_u = &xu * &xu - 2.0 * uxw * &xu;
        w = &w - step_size * &nabla_f_u / nabla_f_u.norm();

        project_onto_capped_simplex_simple(w.as_slice_mut().unwrap(), 1.0 / (n - epsilon_n) as f32);
    }

    // fill output using weight as weighted average for the samples in X
    let output_slice = output.get_slice_mut();
    for i in 0..d {
        output_slice[i] = 0.0;
        for j in 0..n {
            output_slice[i] += w[[0, j]] * x[[j, i]];
        }
    }
}

fn project_onto_capped_simplex_simple(w: &mut [f32], cap: f32) {
    let mut t_l = w.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() - 1.0;
    let mut t_r = w
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
        .to_owned();
    for _ in 0..50 {
        let t = (t_l + t_r) / 2.0;
        let mut sum = 0.0;
        for i in 0..w.len() {
            sum += (w[i] - t).max(0.0).min(cap);
        }
        if sum < 1.0 {
            t_r = t;
        } else {
            t_l = t;
        }
    }
    for i in 0..w.len() {
        w[i] = (w[i] - t_l).max(0.0).min(cap);
    }
}
