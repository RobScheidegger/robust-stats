use crate::matrix::FastMatrix;
use numpy::{ndarray::ArrayD, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn robust_stats<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "mean")]
    fn mean<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f32>) -> &'py PyArrayDyn<f32> {
        // Make a d dimensional output array/vector
        let n = x.shape()[0];
        let d = x.shape()[1];
        let mut result_array = ArrayD::<f32>::zeros(vec![d]);

        let input_matrix = FastMatrix::from_ptr(x.as_raw_array_mut().as_mut_ptr(), n, d);
        let mut output_matrix = FastMatrix::from_ptr(result_array.as_mut_ptr(), 1, d);

        let output_slice: &mut [f32] = output_matrix.get_row_slice_mut(0);

        for ni in 0..n {
            let row = input_matrix.get_row_slice(ni);
            for di in 0..d {
                output_slice[di] += row[di];
            }
        }

        let n: f32 = n as f32;

        for di in 0..d {
            output_slice[di] /= n as f32;
        }

        return result_array.into_pyarray(py);
    }

    // // wrapper of `mult`
    // #[pyfn(m)]
    // #[pyo3(name = "robust_mean")]
    // fn mult_py<'py>(a: f64, x: &'py PyArrayDyn<f32>, epsilon: f32) -> &'py PyArrayDyn<f32> {
    //     let x = unsafe { x.as_array_mut() };
    //     x *= a as f32;
    // }

    Ok(())
}
