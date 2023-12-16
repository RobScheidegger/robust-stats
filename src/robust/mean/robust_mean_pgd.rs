// // function [mu] = robust_mean_pgd(X, eps)
// // % N = number of samples, d = dimension.

// // w = ones(N, 1) / N;
// // for itr = 1:nItr

// //     Xw = X' * w;
// //     Sigma_w_fun = @(v) X' * (w .* (X * v)) - Xw * Xw' * v;
// //     [u, lambda1] = eigs(Sigma_w_fun, d, 1);

// //     % Compute the gradient of spectral norm (assuming unique top eigenvalue)
// //     Xu = X * u;
// //     nabla_f_w = Xu .* Xu - 2 * (w' * Xu) * Xu;
// //     old_w = w;
// //     w = w - stepSz * nabla_f_w / norm(nabla_f_w);
// //     % Projecting w onto the feasible region
// //     w = project_onto_capped_simplex_simple(w, 1 / (N - epsN));

// //     % Use adaptive step size.
// //     %   If objective function decreases, take larger steps.
// //     %   If objective function increases, take smaller steps.
// //     Sigma_w_fun = @(v) X' * (w .* (X * v)) - Xw * Xw' * v;
// //     [~, new_lambda1] = eigs(Sigma_w_fun, d, 1);
// //     if (new_lambda1 < lambda1)
// //         stepSz = stepSz * 2;
// //     else
// //         stepSz = stepSz / 4;
// //         w = old_w;
// //     end
// // end
// // mu = X' * w;
// // end

use crate::matrix::FastMatrix;

pub fn robust_mean_pgd(x: &FastMatrix<f32>, epsilon: f32, output: &mut FastMatrix<f32>) {
    return;
    // let n = x.n;
    // let d = x.d;

    // let epsilon_n = (epsilon * n as f32).round() as usize;
    // let step_size = 1.0 / (n as f32);

    // const NUM_ITERATIONS: usize = 10;

    // let mut w = Array2::<f32>::ones((n, 1)) / (n as f32);
    // let X = x.to_array_view();
    // let X_t = X.t().to_owned();
    // let Xw = &X_t * &w;
    // let mut Dw = Array2::<f32>::zeros((n, n));
    // for i in 0..n {
    //     Dw[[i, i]] = w[[i, 0]];
    // }
    // let Xw_t = Xw.t().to_owned();
    // let E = &X_t * Dw * &X - (&Xw) * Xw_t;

    // let (u, lambda) = E.eig().unwrap();

    // let mut best_eigenvalue_index = 0;
    // let mut best_eigenvalue_distance = lambda[0] - 1.abs() - 1;

    // for i in 1..d {
    //     if lambda[i] > best_eigenvalue {
    //         best_eigenvalue_index = i;
    //         best_eigenvalue = lambda[i];
    //     }
    // }
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
