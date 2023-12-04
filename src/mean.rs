// unsafe fn mean<'py>(x: PyReadonlyArrayDyn<'py, f32>) -> &'py PyArrayDyn<f32> {
//     let dims = x.shape();
//     if dims.len() != 2 {
//         panic!("Expected a 2D array!");
//     }

//     let x = x.as_array();
//     let n = dims[0];
//     let d = dims[1];

//     // Allocate a new array to store the result
//     let mut result = ArrayD::<f32>::zeros(vec![d]);
//     let input_ptr = x.as_ptr();
//     let input_stride = x.strides()[0] as usize;
//     let result_view = result.as_mut_ptr();
//     for ni in 0..n {
//         for di in 0..d {
//             let input_location = unsafe { input_ptr.offset((ni * input_stride + di) as isize) };
//             let output_location = unsafe { result_view.offset((di) as isize) };
//             unsafe { *output_location += *input_location };
//         }
//     }

//     for di in 0..d {
//         let output_location = unsafe { result_view.offset((di) as isize) };
//         unsafe { *output_location /= n as f32 };
//     }

//     return result.into_pyarray(py);
// }

use crate::matrix::FastMatrix;

pub fn nd_mean(x: &FastMatrix<f32>, output: &mut FastMatrix<f32>) {
    let n = x.n;
    let d = x.d;
    let output_slice: &mut [f32] = output.get_row_slice_mut(0);

    for ni in 0..n {
        let row = x.get_row_slice(ni);
        for di in 0..d {
            output_slice[di] += row[di];
        }
    }

    let n: f32 = n as f32;

    for di in 0..d {
        output_slice[di] /= n as f32;
    }
}
