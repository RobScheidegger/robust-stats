use ndarray::Array1;
use special::Error;

/// Computes the median of a mutable array in-place by sorting the input array.
pub fn median_in_place(x: &mut Array1<f32>) -> f32 {
    let slice = x.as_slice_mut().unwrap();
    slice.sort_by(|a, b| a.partial_cmp(b).expect("Comparison should work"));
    let n = slice.len();

    if n % 2 == 0 {
        let median_index = n / 2;
        return slice[median_index];
    } else {
        let median_index = (n - 1) / 2;
        return slice[median_index];
    }
}

/// Computes the complement of the `erf` function.
pub fn erfc(x: f32) -> f32 {
    1.0 - x.error()
}

/// Computes the indices that would sort the given input float array.
pub fn argsort(data: &[f32]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| a.partial_cmp(b).expect("Comparison should work"));
    indices
}
