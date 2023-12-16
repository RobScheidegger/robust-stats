use crate::math::utils::{argsort, erfc, median_in_place};
use crate::matrix::FastMatrix;

use ndarray::{Array2, Axis};
use ndarray_linalg::*;

pub fn robust_mean_filter(x: &FastMatrix<f32>, epsilon: f32, output: &mut FastMatrix<f32>) {
    let n = x.n;
    let d = x.d;

    const CHER: f32 = 2.5;
    const TAU: f32 = 0.1;

    let threshold = epsilon * (1.0 / epsilon).ln() as f32;

    if n == 1 {
        let output_slice = output.get_slice_mut();
        let row_slice = x.get_row_slice(0);
        for di in 0..d {
            output_slice[di] = row_slice[di];
        }
        return;
    }

    let empirical_mean = x
        .to_array_view()
        .mean_axis(Axis(0))
        .expect("Mean should succeed");

    let centered_data = (&x.to_array_view()
        - &empirical_mean
            .broadcast((n, d))
            .expect("Broadcast should succeed"))
        / (n as f32).sqrt();

    let centered_data_transpose = centered_data.t().to_owned();
    let (u, s, _) = centered_data_transpose
        .svd(true, false)
        .expect("SVD failed to converge");

    let lambda = s[0].pow(2.0);
    let v = u.expect("V should be populated").column(0).to_owned();

    if lambda < 1.0 + 3.0 * threshold {
        let output_slice = output.get_slice_mut();
        for di in 0..d {
            output_slice[di] = empirical_mean[di];
        }
        return;
    }

    let delta = 2.0 * epsilon;
    let sample_view = x.to_array_view();
    let mut projected_data_raw = sample_view.dot(&v);

    let med = median_in_place(&mut projected_data_raw);
    let projected_data = projected_data_raw.map(|x| (x - med).abs());

    let sorted_indices = argsort(&projected_data.as_slice().expect("Should be a slice"));

    let mut idx = 0;
    for i in 0..n {
        let index = sorted_indices[i];
        let t = projected_data[index] - delta;

        if (n - i) as f32
            > (0.5 * CHER * (n as f32) * erfc(t / (2.0).sqrt())
                + epsilon / (d as f32) * ((d as f32) * epsilon / TAU).ln())
        {
            break;
        }
        idx += 1;
    }

    if idx == 0 || idx >= n - 1 {
        let output_slice = output.get_slice_mut();
        for di in 0..d {
            output_slice[di] = empirical_mean[di];
        }
        return;
    }

    let mut x_prime = Array2::<f32>::zeros((idx, d));
    for i in 0..idx {
        let index = sorted_indices[i];
        let row = sample_view.row(index);
        x_prime.row_mut(i).assign(&row);
    }

    return robust_mean_filter(
        &FastMatrix::from_ptr(x_prime.as_mut_ptr(), idx, d),
        epsilon,
        output,
    );
}
