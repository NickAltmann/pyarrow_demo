use std::sync::Arc;

use pyo3::prelude::*;
use numpy::PyReadonlyArrayDyn;
use arrow::{
    array::{Array, ArrayData, make_array, AsArray, UInt64Array},
    pyarrow::PyArrowType,
};


pub fn index_left<T>(list_input: &[T], value: &T, left_count: Option<usize>) -> usize
where for<'a> &'a T: PartialOrd + PartialEq
{
    let lc = left_count.unwrap_or(0_usize);
    let n = list_input.len();
    match n {
        1 => panic!("`index_left` designed for intervals. Cannot index sequence of length 1."),
        2 => lc,
        _ => {
            let split = (n - 1_usize) / 2_usize;  // this will take the floor of result
            if n == 3 && value == &list_input[split] {
                lc
            } else if value <= &list_input[split] {
                index_left(&list_input[..=split], value, Some(lc))
            } else {
                index_left(&list_input[split..], value, Some(lc + split))
            }
        }
    }
}

#[pyfunction]
pub fn index_left_from_rust (list_input: Vec<f64>, value: f64) -> usize {
    index_left(&list_input[..], &value, Some(0))
}


#[pyfunction]
fn index_left_arrow(array: PyArrowType<ArrayData>, value: f64) -> usize {
    let array = array.0; // Extract from PyArrowType wrapper
    let array: Arc<dyn Array> = make_array(array); // Convert ArrayData to ArrayRef
    //let vw = array.as_any().downcast_ref::<Float64Array>().unwrap().values();
    let vw = array.as_primitive::<arrow::datatypes::Float64Type>().values();
    
    index_left(vw, &value, Some(0))
}


#[pyfunction]
fn index_left_arrow_vec(source: PyArrowType<ArrayData>, targets: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let source_array = source.0; // Extract from PyArrowType wrapper
    let source_array: Arc<dyn Array> = make_array(source_array); // Convert ArrayData to ArrayRef
    let source_vw = source_array.as_primitive::<arrow::datatypes::Float64Type>().values();

    let target_array = targets.0;
    let target_array: Arc<dyn Array> = make_array(target_array);

    let target_view = target_array.as_primitive::<arrow::datatypes::Int32Type>().values();
    let results_array: UInt64Array = target_view.iter().map(|x| { let y = *x as f64; index_left(source_vw, &y, Some(0)) as u64 } ).collect();
    //let target_view = target_array.as_primitive::<arrow::datatypes::Float64Type>().values();
    //let results_array: UInt64Array = target_view.iter().map(|x| index_left(source_vw, x, Some(0)) as u64 ).collect();

    Ok(PyArrowType(results_array.into_data()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyarrow_demo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(index_left_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(index_left_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(index_left_arrow_vec, m)?)?;

    #[pyfn(m)]
    #[pyo3(name = "index_left_np")]
    fn index_left_np(
        x: PyReadonlyArrayDyn<f64>,
        v: f64
    ) -> usize {
        let (x_vec, _o) = x.as_array().as_standard_layout().into_owned().into_raw_vec_and_offset();       
        index_left(&x_vec, &v, Some(0))
    }    

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
     fn test_index_left() {
        let v = vec![1, 2, 3, 4, 6, 8, 10];
        let s = 7;
        let r = index_left(&v, &s, None);
        assert_eq!(r, 4);
     }
}