//! This module is to generate some random sequence to assess the performance.
//! Usually, it would not be used in the real-applications.
use rand::seq::SliceRandom;
#[derive(Debug, Clone, Copy)]
pub struct Profile {
    pub sub: f64,
    pub del: f64,
    pub ins: f64,
}
impl Profile {
    pub fn sum(&self) -> f64 {
        self.sub + self.del + self.ins
    }
    pub fn norm(&self) -> Self {
        let sum = self.sum();
        Self {
            sub: self.sub / sum,
            del: self.del / sum,
            ins: self.ins / sum,
        }
    }
    pub fn mul(&self, x: f64) -> Self {
        Self {
            sub: self.sub * x,
            ins: self.ins * x,
            del: self.del * x,
        }
    }
}

pub const PROFILE: Profile = Profile {
    sub: 0.04,
    del: 0.04,
    ins: 0.07,
};

pub const CCS_PROFILE: Profile = Profile {
    sub: 0.002,
    del: 0.004,
    ins: 0.004,
};

#[derive(Debug, Clone, Copy)]
enum Op {
    Match,
    MisMatch,
    Del,
    In,
}
impl Op {
    fn weight(self, p: &Profile) -> f64 {
        match self {
            Op::Match => 1. - p.sub - p.del - p.ins,
            Op::MisMatch => p.sub,
            Op::Del => p.del,
            Op::In => p.ins,
        }
    }
}
const OPERATIONS: [Op; 4] = [Op::Match, Op::MisMatch, Op::Del, Op::In];
pub fn introduce_randomness<T: rand::Rng>(seq: &[u8], rng: &mut T, p: &Profile) -> Vec<u8> {
    let mut res = vec![];
    let mut remainings: Vec<_> = seq.iter().copied().rev().collect();
    while !remainings.is_empty() {
        match *OPERATIONS.choose_weighted(rng, |e| e.weight(p)).unwrap() {
            Op::Match => res.push(remainings.pop().unwrap()),
            Op::MisMatch => res.push(choose_base(rng, remainings.pop().unwrap())),
            Op::In => res.push(random_base(rng)),
            Op::Del => {
                remainings.pop().unwrap();
            }
        }
    }
    res
}
pub fn introduce_errors<T: rand::Rng>(
    seq: &[u8],
    rng: &mut T,
    sub: usize,
    del: usize,
    ins: usize,
) -> Vec<u8> {
    // Alignment operations.
    let mut operations = vec![
        vec![Op::Match; seq.len() - sub - del],
        vec![Op::MisMatch; sub],
        vec![Op::Del; del],
        vec![Op::In; ins],
    ]
    .concat();
    operations.shuffle(rng);
    let mut res = vec![];
    let mut remainings: Vec<_> = seq.iter().copied().rev().collect();
    for op in operations {
        match op {
            Op::Match => res.push(remainings.pop().unwrap()),
            Op::MisMatch => res.push(choose_base(rng, remainings.pop().unwrap())),
            Op::In => res.push(random_base(rng)),
            Op::Del => {
                remainings.pop().unwrap();
            }
        }
    }
    res
}
pub fn generate_seq<T: rand::Rng>(rng: &mut T, len: usize) -> Vec<u8> {
    let bases = b"ACTG";
    (0..len)
        .filter_map(|_| bases.choose(rng))
        .copied()
        .collect()
}

fn choose_base<T: rand::Rng>(rng: &mut T, base: u8) -> u8 {
    let bases: Vec<u8> = b"ATCG".iter().filter(|&&e| e != base).copied().collect();
    *bases.choose_weighted(rng, |_| 1. / 3.).unwrap()
}
fn random_base<T: rand::Rng>(rng: &mut T) -> u8 {
    *b"ATGC".choose_weighted(rng, |_| 1. / 4.).unwrap()
}
