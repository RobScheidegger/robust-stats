extern crate rblas as blas;

use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn basic_mean_benchmark<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "mean_numpy")]
    fn mean_numpy<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f32>) -> &'py PyArrayDyn<f32> {
        let dims = x.shape();
        if dims.len() != 2 {
            panic!("Expected a 2D array!");
        }

        return x
            .as_array()
            .mean_axis(numpy::ndarray::Axis(0))
            .expect("Mean should not fail.")
            .into_pyarray(py);
    }

    #[target_feature(enable = "avx2,avx,sse2")]
    #[pyfn(m)]
    #[pyo3(name = "mean_native")]
    unsafe fn mean_native<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
    ) -> &'py PyArrayDyn<f32> {
        let dims = x.shape();
        if dims.len() != 2 {
            panic!("Expected a 2D array!");
        }

        let x = x.as_array();
        let n = dims[0];
        let d = dims[1];

        // Allocate a new array to store the result
        let mut result = ArrayD::<f32>::zeros(vec![d]);
        let input_ptr = x.as_ptr();
        let input_stride = x.strides()[0] as usize;
        let result_view = result.as_mut_ptr();
        for ni in 0..n {
            for di in 0..d {
                let input_location = unsafe { input_ptr.offset((ni * input_stride + di) as isize) };
                let output_location = unsafe { result_view.offset((di) as isize) };
                unsafe { *output_location += *input_location };
            }
        }

        for di in 0..d {
            let output_location = unsafe { result_view.offset((di) as isize) };
            unsafe { *output_location /= n as f32 };
        }

        return result.into_pyarray(py);
    }
    Ok(())
}
