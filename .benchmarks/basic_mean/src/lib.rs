use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::IntoPy;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn basic_mean_benchmark<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "mean_numpy")]
    fn mean_numpy<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, f32>) -> Option<f32> {
        let dims = x.shape();
        if dims.len() != 2 {
            panic!("Expected a 2D array!");
        }

        let n = dims[0] as f32;

        let a = x.as_array();
        //.mean_axis(numpy::ndarray::Axis(0));
        //.expect("Mean should not fail.");
        //.into_pyarray(py);
        return Some(0.0);
    }

    Ok(())
}
