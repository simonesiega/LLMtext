use pyo3::prelude::*;

#[pyfunction]
fn hello_rust() -> PyResult<String> {
    println!("Hello from Rust!");
    Ok("Hello from Python!".to_string())
}

#[pymodule]
fn subtk(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    Ok(())
}