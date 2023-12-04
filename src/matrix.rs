use numpy::ndarray::ArrayView2;

/// A simple, performant matrix implementation for use in the robust_stats crate.
/// Uses an unsafe C-style syntax to be performant, without sacrificing (too much) safety.
pub struct FastMatrix<T> {
    pub data: *mut T,
    pub n: usize,
    pub d: usize,
}

impl<T> FastMatrix<T> {
    pub fn from_ptr(ptr: *mut T, rows: usize, cols: usize) -> Self {
        FastMatrix {
            data: ptr,
            n: rows,
            d: cols,
        }
    }

    pub fn get_row_slice(&self, row: usize) -> &[T] {
        let start = row * self.d;
        return unsafe { std::slice::from_raw_parts(self.data.offset(start as isize), self.d) };
    }

    pub fn get_row_slice_mut(&mut self, row: usize) -> &mut [T] {
        let start = row * self.d;
        return unsafe { std::slice::from_raw_parts_mut(self.data.offset(start as isize), self.d) };
    }

    pub fn get_slice(&self) -> &[T] {
        return unsafe { std::slice::from_raw_parts(self.data, self.d * self.n) };
    }

    pub fn get_slice_mut(&mut self) -> &mut [T] {
        return unsafe { std::slice::from_raw_parts_mut(self.data, self.d * self.n) };
    }

    pub fn to_array_view(&self) -> ArrayView2<T> {
        return unsafe { ArrayView2::from_shape_ptr((self.n, self.d), self.data) };
    }
}
