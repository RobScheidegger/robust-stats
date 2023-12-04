use nalgebra::DMatrix;

/// A simple, performant matrix implementation for use in the robust_stats crate.
/// Uses an unsafe C-style syntax to be performant, without sacrificing (too much) safety.
pub struct FastMatrix<T> {
    pub data: *mut T,
    rows: usize,
    cols: usize,
}

impl<T> FastMatrix<T> {
    pub fn from_ptr(ptr: *mut T, rows: usize, cols: usize) -> Self {
        FastMatrix {
            data: ptr,
            rows,
            cols,
        }
    }

    pub fn get_row_slice(&self, row: usize) -> &[T] {
        let start = row * self.cols;
        return unsafe { std::slice::from_raw_parts(self.data.offset(start as isize), self.cols) };
    }

    pub fn get_row_slice_mut(&mut self, row: usize) -> &mut [T] {
        let start = row * self.cols;
        return unsafe {
            std::slice::from_raw_parts_mut(self.data.offset(start as isize), self.cols)
        };
    }

    pub fn get_slice(&self) -> &[T] {
        return unsafe { std::slice::from_raw_parts(self.data, self.cols * self.rows) };
    }
}
