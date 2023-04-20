use geoutils::Location;
use numpy::borrow::PyReadonlyArray1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn distance_function<'a>(
    latitude_1: PyReadonlyArray1<'a, f64>,
    longitude_1: PyReadonlyArray1<'a, f64>,
    latitude_2: PyReadonlyArray1<'a, f64>,
    longitude_2: PyReadonlyArray1<'a, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let latitude_1 = latitude_1.as_array();
    let longitude_1 = longitude_1.as_array();
    let latitude_2 = latitude_2.as_array();
    let longitude_2 = longitude_2.as_array();

    let distances = latitude_1
        .iter()
        .zip(longitude_1.iter())
        .zip(latitude_2.iter().zip(longitude_2.iter()))
        .map(|((lat1, lon1), (lat2, lon2))| {
            let location_1 = Location::new(*lat1, *lon1);
            let location_2 = Location::new(*lat2, *lon2);
            location_1
                .distance_to(&location_2)
                .map(|distance| distance.meters() / 1000.0)
                .map_err(|err| PyErr::new::<PyValueError, _>(err))
        })
        .collect::<Result<Vec<f64>, PyErr>>()?;

    let result = ndarray::Array1::from(distances);

    Python::with_gil(|py| Ok(result.into_pyarray(py).to_owned()))
}

#[pymodule]
fn sk_transformers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_function, m)?)?;
    Ok(())
}
