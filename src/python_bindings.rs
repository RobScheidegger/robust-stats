use crate::mean::nd_mean;
use crate::{matrix::FastMatrix, robust};
use numpy::{ndarray::ArrayD, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn robust_stats<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "mean")]
    fn mean<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f32>) -> &'py PyArrayDyn<f32> {
        let n = x.shape()[0];
        let d = x.shape()[1];
        let mut result_array = ArrayD::<f32>::zeros(vec![d]);

        let input_matrix = FastMatrix::from_ptr(x.as_raw_array_mut().as_mut_ptr(), n, d);
        let mut output_matrix = FastMatrix::from_ptr(result_array.as_mut_ptr(), 1, d);

        nd_mean(&input_matrix, &mut output_matrix);

        return result_array.into_pyarray(py);
    }
    #[pyfn(m)]
    #[pyo3(name = "robust_mean_heuristic")]
    fn robust_mean_heuristic<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
        epsilon: f32,
    ) -> &'py PyArrayDyn<f32> {
        let n = x.shape()[0];
        let d = x.shape()[1];
        let mut result_array = ArrayD::<f32>::zeros(vec![d]);

        let input_matrix = FastMatrix::from_ptr(x.as_raw_array_mut().as_mut_ptr(), n, d);
        let mut output_matrix = FastMatrix::from_ptr(result_array.as_mut_ptr(), 1, d);

        robust::mean::robust_mean_heuristic::robust_mean_heuristic(
            &input_matrix,
            epsilon,
            &mut output_matrix,
        );

        return result_array.into_pyarray(py);
    }

    #[pyfn(m)]
    #[pyo3(name = "robust_mean_filter")]
    fn robust_mean_filter<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
        epsilon: f32,
    ) -> &'py PyArrayDyn<f32> {
        let n = x.shape()[0];
        let d = x.shape()[1];
        let mut result_array = ArrayD::<f32>::zeros(vec![d]);

        let input_matrix = FastMatrix::from_ptr(x.as_raw_array_mut().as_mut_ptr(), n, d);
        let mut output_matrix = FastMatrix::from_ptr(result_array.as_mut_ptr(), 1, d);

        robust::mean::robust_mean_filter::robust_mean_filter(
            &input_matrix,
            epsilon,
            &mut output_matrix,
        );

        return result_array.into_pyarray(py);
    }

    #[pyfn(m)]
    #[pyo3(name = "robust_mean_pgd")]
    fn robust_mean_pgd<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'py, f32>,
        epsilon: f32,
    ) -> &'py PyArrayDyn<f32> {
        let n = x.shape()[0];
        let d = x.shape()[1];
        let mut result_array = ArrayD::<f32>::zeros(vec![d]);

        let input_matrix = FastMatrix::from_ptr(x.as_raw_array_mut().as_mut_ptr(), n, d);
        let mut output_matrix = FastMatrix::from_ptr(result_array.as_mut_ptr(), 1, d);

        robust::mean::robust_mean_pgd::robust_mean_pgd(&input_matrix, epsilon, &mut output_matrix);

        return result_array.into_pyarray(py);
    }

    Ok(())
}
