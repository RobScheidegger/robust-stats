use crate::matrix::FastMatrix;

pub fn nd_mean(x: &FastMatrix<f32>, output: &mut FastMatrix<f32>) {
    let n = x.n;
    let d = x.d;
    let output_slice: &mut [f32] = output.get_row_slice_mut(0);

    for ni in 0..n {
        let row = x.get_row_slice(ni);
        for di in 0..d {
            output_slice[di] += row[di];
        }
    }

    for di in 0..d {
        output_slice[di] /= n as f32;
    }
}
