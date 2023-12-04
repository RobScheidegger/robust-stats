use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn robust_mean<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    Ok(())
}
