extern crate rblas as blas;

use cblas::sasum;
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

    #[pyfn(m)]
    #[pyo3(name = "mean_native")]
    fn mean_native<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f32>) -> &'py PyArrayDyn<f32> {
        let dims = x.shape();
        if dims.len() != 2 {
            panic!("Expected a 2D array!");
        }

        let x = x.as_array();
        let n = dims[0];
        let d = dims[1];

        // Allocate a new array to store the result
        let mut result = ArrayD::<f32>::zeros(vec![dims[1]]);
        let input_ptr = x.as_ptr();
        let input_stride = x.strides()[0] as usize;
        let result_view = result.as_mut_ptr();
        for j in 0..d {
            let mut sum: f32 = 0.0;
            let output_location = unsafe { result_view.offset((j) as isize) };
            for i in 0..n {
                sum += unsafe { *input_ptr.offset((i * input_stride + j) as isize) };
            }
            unsafe {
                *output_location = sum / (n as f32);
            }
        }

        return result.into_pyarray(py);
    }

    #[pyfn(m)]
    #[pyo3(name = "mean_native_transpose")]
    fn mean_native_transpose<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
    ) -> &'py PyArrayDyn<f32> {
        // Assume that the input is transposed, so it is a 2D array with shape (d, n)

        let dims = x.shape();
        if dims.len() != 2 {
            panic!("Expected a 2D array!");
        }

        let x = x.as_array();
        let d = dims[0];
        let n = dims[1];

        // Allocate a new array to store the result
        let mut result = ArrayD::<f32>::zeros(vec![dims[1]]);
        let input_ptr = x.as_ptr();

        let input_stride = x.strides()[0] as usize;
        println!("input_strides: {:?}, n = {}, d = {}", x.strides(), n, d);
        let result_view = result.as_mut_ptr();
        for j in 0..d {
            let mut sum: f32 = 0.0;
            let output_location = unsafe { result_view.offset((j) as isize) };
            let input_pointer = unsafe { input_ptr.offset((j * input_stride) as isize) };
            for i in 0..n {
                sum += unsafe { *input_pointer.offset((i) as isize) };
            }
            unsafe {
                *output_location = sum / (n as f32);
            }
        }

        return result.into_pyarray(py);
    }

    #[pyfn(m)]
    #[pyo3(name = "mean_native_transpose_unroll")]
    fn mean_native_transpose_unroll<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
    ) -> &'py PyArrayDyn<f32> {
        // Assume that the input is transposed, so it is a 2D array with shape (d, n)

        let dims = x.shape();
        if dims.len() != 2 {
            panic!("Expected a 2D array!");
        }

        let x = x.as_array();
        let d = dims[0];
        let n = dims[1];

        // Allocate a new array to store the result
        let mut result = ArrayD::<f32>::zeros(vec![dims[1]]);
        let input_ptr = x.as_ptr();
        let input_stride = x.strides()[0] as usize;
        let result_view = result.as_mut_ptr();
        for j in 0..d {
            let mut sum: f32 = 0.0;
            let output_location = unsafe { result_view.offset((j) as isize) };
            let input_pointer = unsafe { input_ptr.offset((j * input_stride) as isize) };
            let mut ptr = input_pointer;
            for _ in 0..(n / 32) {
                let slice = unsafe { std::slice::from_raw_parts(ptr, 32) };
                sum += slice.iter().sum::<f32>();
                ptr = unsafe { ptr.offset(32) };
            }
            unsafe {
                *output_location = sum / (n as f32);
            }
        }

        return result.into_pyarray(py);
    }

    #[pyfn(m)]
    #[pyo3(name = "mean_native_transpose_blas")]
    fn mean_native_transpose_blas<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
    ) -> &'py PyArrayDyn<f32> {
        // Assume that the input is transposed, so it is a 2D array with shape (d, n)

        let dims = x.shape();
        if dims.len() != 2 {
            panic!("Expected a 2D array!");
        }

        let x = x.as_array();
        let d = dims[0];
        let n = dims[1];

        let mut result = ArrayD::<f32>::zeros(vec![dims[1]]);
        let input_ptr = x.as_ptr();
        let input_stride = x.strides()[0] as usize;
        let result_view = result.as_mut_ptr();
        for j in 0..d {
            let output_location = unsafe { result_view.offset((j) as isize) };
            let input_pointer = unsafe { input_ptr.offset((j * input_stride) as isize) };
            let data_vec: &[f32] = unsafe { std::slice::from_raw_parts(input_pointer, n) };

            unsafe {
                *output_location = sasum(n as i32, data_vec, 1);
            }
        }

        return result.into_pyarray(py);
    }

    #[pyfn(m)]
    #[pyo3(name = "mean_native_blas")]
    fn mean_native_blas<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
    ) -> &'py PyArrayDyn<f32> {
        // Assume that the input is transposed, so it is a 2D array with shape (d, n)

        let dims = x.shape();
        if dims.len() != 2 {
            panic!("Expected a 2D array!");
        }

        let x = x.as_array();
        let n = dims[0];
        let d = dims[1];

        // Allocate a new array to store the result
        let mut result = ArrayD::<f32>::zeros(vec![dims[1]]);
        let input_ptr = x.as_ptr();
        let input_stride = x.strides()[0] as usize;
        let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, n * input_stride) };
        let result_view = result.as_mut_ptr();
        for j in 0..d {
            let output_location = unsafe { result_view.offset((j) as isize) };

            unsafe {
                *output_location = sasum(n as i32, input_slice, input_stride as i32);
            }
        }

        return result.into_pyarray(py);
    }

    Ok(())
}
