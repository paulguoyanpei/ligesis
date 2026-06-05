use p3_field::Field;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

pub const RANDOM_ORACLE_LEN: usize = 100_000;

pub struct RandomOracle<F: Field> {
    fields: Vec<F>,
    ints: Vec<usize>,
    fields_idx: usize,
    ints_idx: usize,
}

impl<F: Field> RandomOracle<F> {
    pub fn new<R: Rng + ?Sized>(rng: &mut R) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self {
            fields: (0..RANDOM_ORACLE_LEN).map(|_| rng.random()).collect(),
            ints: (0..RANDOM_ORACLE_LEN)
                .map(|_| rng.random::<u64>() as usize)
                .collect(),
            fields_idx: 0,
            ints_idx: 0,
        }
    }

    pub fn next_field(&mut self) -> F {
        let res = self.fields[self.fields_idx];
        self.fields_idx += 1;
        res
    }

    pub fn next_n_fields(&mut self, n: usize) -> Vec<F> {
        let res = self.fields[self.fields_idx..(self.fields_idx + n)].to_vec();
        self.fields_idx += n;
        res
    }

    pub fn next_int(&mut self) -> usize {
        let res = self.ints[self.ints_idx];
        self.ints_idx += 1;
        res
    }

    pub fn next_n_ints(&mut self, n: usize) -> Vec<usize> {
        let res = self.ints[self.ints_idx..(self.ints_idx + n)].to_vec();
        self.ints_idx += n;
        res
    }

    pub fn restart(&mut self) {
        self.fields_idx = 0;
        self.ints_idx = 0;
    }
}
