use crate::matrix::FastMatrix;

use ndarray_linalg::*;
use ndarray_stats::CorrelationExt;
use ordered_float::NotNan;
use priority_queue::PriorityQueue;

pub fn robust_mean_heuristic(x: &FastMatrix<f32>, epsilon: f32, output: &mut FastMatrix<f32>) {
    let n = x.n;
    let d = x.d;

    let x_view = x.to_array_view();

    let cov_x = x_view.cov(1.0).unwrap();

    let (mut v1, _) = cov_x.eig().unwrap();
    let v = FastMatrix::from_ptr(v1.as_mut_ptr(), d, 1);
    let v_slice = v.get_row_slice(0);

    // Compute and store the regular mean of the data
    crate::mean::nd_mean(x, output);
    let mut mean_x = output.get_row_slice_mut(0);

    let mut pq: PriorityQueue<usize, NotNan<f32>> = PriorityQueue::new();

    // Remove the epsilon * n fraction of points with the largest projection onto the first eigenvector
    for i in 0..n {
        let row = x.get_row_slice(i);
        let mut projection: f32 = 0.0;
        for j in 0..d {
            projection += row[j] * v_slice[j].re();
        }
        pq.push(i, NotNan::new(projection).unwrap());
    }

    // TODO: Remove old values to keep the PQ size small
    let mut num_to_remove = (epsilon * n as f32) as usize;
    let float_n = n as f32;
    // First, scale all of the mean values up again by a factor of n
    for j in 0..d {
        mean_x[j] *= float_n;
    }

    while num_to_remove > 0 {
        let (i, _) = pq.pop().unwrap();
        let row = x.get_row_slice(i);
        for j in 0..d {
            mean_x[j] -= row[j];
        }
        num_to_remove -= 1;
    }

    // Scale all of the value back down by a factor of n - epsilon * n
    let scale = 1.0 / (float_n - epsilon * float_n);
    for j in 0..d {
        mean_x[j] *= scale;
    }

    // Output is then defined!
}
