mod number_backend;
mod string_backend;
use number_backend::distance_function;
use pyo3::prelude::*;
use string_backend::ip_to_float;

#[pymodule]
fn sk_transformers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_function, m)?)?;
    m.add_function(wrap_pyfunction!(ip_to_float, m)?)?;
    Ok(())
}
