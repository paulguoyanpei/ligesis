#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<i64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<i64>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0; rows * cols],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[i64] {
        &self.data
    }

    pub fn into_data(self) -> Vec<i64> {
        self.data
    }

    pub fn get(&self, r: usize, c: usize) -> i64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: i64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn row(&self, r: usize) -> &[i64] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn add_row_vector(&self, bias: &[i64]) -> Matrix {
        assert_eq!(self.cols, bias.len());
        let mut out = self.clone();
        for row in out.data.chunks_mut(self.cols) {
            for (x, b) in row.iter_mut().zip(bias) {
                *x += *b;
            }
        }
        out
    }

    pub fn add_matrix(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        Matrix::new(
            self.rows,
            self.cols,
            self.data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }

    pub fn matmul(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.cols, rhs.rows);
        let mut out = Matrix::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut acc = 0i128;
                for k in 0..self.cols {
                    acc += self.get(i, k) as i128 * rhs.get(k, j) as i128;
                }
                out.set(i, j, checked_i128_to_i64(acc));
            }
        }
        out
    }

    pub fn matmul_transposed_rhs(&self, rhs: &Matrix) -> Matrix {
        assert_eq!(self.cols, rhs.cols);
        let mut out = Matrix::zeros(self.rows, rhs.rows);
        for i in 0..self.rows {
            for j in 0..rhs.rows {
                let mut acc = 0i128;
                for k in 0..self.cols {
                    acc += self.get(i, k) as i128 * rhs.get(j, k) as i128;
                }
                out.set(i, j, checked_i128_to_i64(acc));
            }
        }
        out
    }

    pub fn floor_div_const(&self, divisor: i64) -> Matrix {
        Matrix::new(
            self.rows,
            self.cols,
            self.data
                .iter()
                .map(|&x| floor_div_i128(x as i128, divisor as i128))
                .collect(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor3 {
    d0: usize,
    d1: usize,
    d2: usize,
    data: Vec<i64>,
}

impl Tensor3 {
    pub fn new(d0: usize, d1: usize, d2: usize, data: Vec<i64>) -> Self {
        assert_eq!(d0 * d1 * d2, data.len());
        Self { d0, d1, d2, data }
    }

    pub fn zeros(d0: usize, d1: usize, d2: usize) -> Self {
        Self {
            d0,
            d1,
            d2,
            data: vec![0; d0 * d1 * d2],
        }
    }

    pub fn d0(&self) -> usize {
        self.d0
    }

    pub fn d1(&self) -> usize {
        self.d1
    }

    pub fn d2(&self) -> usize {
        self.d2
    }

    pub fn data(&self) -> &[i64] {
        &self.data
    }

    pub fn get(&self, a: usize, b: usize, c: usize) -> i64 {
        self.data[(a * self.d1 + b) * self.d2 + c]
    }

    pub fn set(&mut self, a: usize, b: usize, c: usize, v: i64) {
        self.data[(a * self.d1 + b) * self.d2 + c] = v;
    }

    pub fn head_matrix(&self, h: usize) -> Matrix {
        let mut out = Vec::with_capacity(self.d1 * self.d2);
        for i in 0..self.d1 {
            for j in 0..self.d2 {
                out.push(self.get(h, i, j));
            }
        }
        Matrix::new(self.d1, self.d2, out)
    }
}

pub fn checked_i128_to_i64(x: i128) -> i64 {
    assert!(
        x >= i64::MIN as i128 && x <= i64::MAX as i128,
        "i64 overflow: {x}"
    );
    x as i64
}

pub fn floor_div_i128(a: i128, b: i128) -> i64 {
    assert!(b > 0);
    let mut q = a / b;
    let r = a % b;
    if r < 0 {
        q -= 1;
    }
    checked_i128_to_i64(q)
}

pub fn euclidean_remainder(a: i128, b: i128) -> i64 {
    assert!(b > 0);
    let q = floor_div_i128(a, b) as i128;
    checked_i128_to_i64(a - q * b)
}

pub fn round_sqrt(n: i128) -> i64 {
    assert!(n >= 0);
    let root = (n as f64).sqrt();
    checked_i128_to_i64(root.round() as i128)
}
