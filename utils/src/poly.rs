use p3_field::Field;

#[derive(Debug, Clone)]
pub struct MlPoly<F: Field>(pub Vec<F>);

impl<F: Field> MlPoly<F> {
    pub fn eval(self, point: &[F]) -> F {
        let mut scratch = self.0;
        let mut cur_len = scratch.len() >> 1;
        assert_eq!(1 << point.len(), scratch.len());
        for r in point.iter() {
            for i in 0..cur_len {
                scratch[i] = scratch[i * 2] + (scratch[i * 2 + 1] - scratch[i * 2]) * (*r);
            }
            cur_len >>= 1;
        }
        scratch[0]
    }

    pub fn split(&self, n: usize) -> Vec<MlPoly<F>> {
        assert_eq!(n & (n - 1), 0);
        let len = self.0.len();
        let num_chunks = len / n;
        let mut polies = (0..num_chunks).map(|_| vec![]).collect::<Vec<_>>();
        for i in (0..len).step_by(num_chunks) {
            for j in 0..num_chunks {
                polies[j].push(self.0[i + j]);
            }
        }
        polies.into_iter().map(|x| MlPoly(x)).collect()
    }

    pub fn fold(&mut self, point: &[F]) {
        let mut cur_len = self.0.len();
        for r in point.iter() {
            cur_len >>= 1;
            for i in 0..cur_len {
                self.0[i] = self.0[i * 2] + (self.0[i * 2 + 1] - self.0[i * 2]) * (*r);
            }
        }
        self.0.truncate(cur_len);
    }

    pub fn new_eq(point: &Vec<F>) -> Self {
        let mut evals = vec![F::ONE];
        for &i in point.iter().rev() {
            evals = evals
                .iter()
                .flat_map(|&x| [(F::ONE - i) * x, i * x])
                .collect()
        }
        MlPoly(evals)
    }

    pub fn eval_eq(point1: &Vec<F>, point2: &Vec<F>) -> F {
        let mut res = F::ONE;
        for (&i, &j) in point1.iter().zip(point2.iter()) {
            res *= i * j + (F::ONE - i) * (F::ONE - j);
        }
        res
    }
}
