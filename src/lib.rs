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
                .map_err(PyErr::new::<PyValueError, _>)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_distance_function() {
        pyo3::prepare_freethreaded_python();

        let result = Python::with_gil(|py: Python| {
            let latitudes_1 = vec![
                52.380001,
                50.033333,
                48.353802,
                48.353802,
                51.289501,
                53.63040161,
            ];
            let longitudes_1 = vec![13.5225, 8.570556, 11.7861, 11.7861, 6.76678, 9.988229752];
            let latitudes_2 = vec![
                50.033333,
                52.380001,
                48.353802,
                51.289501,
                53.63040161,
                48.353802,
            ];
            let longitudes_2 = vec![8.570556, 13.5225, 11.7861, 6.76678, 9.988229752, 11.7861];

            let latitudes_1 = Array1::from(latitudes_1);
            let longitudes_1 = Array1::from(longitudes_1);
            let latitudes_2 = Array1::from(latitudes_2);
            let longitudes_2 = Array1::from(longitudes_2);

            let latitudes_1_py = latitudes_1.into_pyarray(py).readonly().to_owned();
            let longitudes_1_py = longitudes_1.into_pyarray(py).readonly().to_owned();
            let latitudes_2_py = latitudes_2.into_pyarray(py).readonly().to_owned();
            let longitudes_2_py = longitudes_2.into_pyarray(py).readonly().to_owned();

            distance_function(
                latitudes_1_py,
                longitudes_1_py,
                latitudes_2_py,
                longitudes_2_py,
            )
        })
        .unwrap();

        let expected_distances: Vec<f64> = vec![
            433.338219, 433.338219, 0.000000, 486.704807, 340.222735, 600.376258,
        ];
        let computed_distances: Vec<f64> =
            Python::with_gil(|py| unsafe { result.as_ref(py).as_array().to_vec() });

        for (expected, computed) in expected_distances.iter().zip(computed_distances.iter()) {
            assert!((expected - computed).abs() < 1e-6, "Mismatch in distances");
        }
    }
}
