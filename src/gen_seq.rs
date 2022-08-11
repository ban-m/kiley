//! This module is to generate some random sequence to assess the performance.
//! Usually, it would not be used in the real-applications.
use crate::op::Op;
use rand::seq::SliceRandom;

/// Generate sequence.
pub trait Generate {
    fn gen<R: rand::Rng>(&self, seq: &[u8], rng: &mut R) -> Vec<u8>;
}

#[derive(Debug, Clone, Copy)]
pub struct Profile {
    pub sub: f64,
    pub del: f64,
    pub ins: f64,
}
impl Generate for Profile {
    fn gen<R: rand::Rng>(&self, seq: &[u8], rng: &mut R) -> Vec<u8> {
        introduce_randomness(seq, rng, self)
    }
}

impl Profile {
    pub fn new(sub: f64, del: f64, ins: f64) -> Self {
        Self { sub, del, ins }
    }
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
    sub: 0.01,
    del: 0.01,
    ins: 0.01,
};

pub const CCS_PROFILE: Profile = Profile {
    sub: 0.002,
    del: 0.004,
    ins: 0.004,
};

impl Op {
    fn weight(self, p: &Profile) -> f64 {
        match self {
            Op::Match => 1. - p.sub - p.del - p.ins,
            Op::Mismatch => p.sub,
            Op::Del => p.del,
            Op::Ins => p.ins,
        }
    }
}
const OPERATIONS: [Op; 4] = [Op::Match, Op::Mismatch, Op::Del, Op::Ins];
pub fn introduce_randomness<T: rand::Rng>(seq: &[u8], rng: &mut T, p: &Profile) -> Vec<u8> {
    let mut res = vec![];
    let mut remainings: Vec<_> = seq.iter().copied().rev().collect();
    while !remainings.is_empty() {
        match *OPERATIONS.choose_weighted(rng, |e| e.weight(p)).unwrap() {
            Op::Match => res.push(remainings.pop().unwrap()),
            Op::Mismatch => res.push(choose_base(rng, remainings.pop().unwrap())),
            Op::Ins => res.push(random_base(rng)),
            Op::Del => {
                remainings.pop().unwrap();
            }
        }
    }
    res
}

#[derive(Debug, Clone)]
pub struct ProfileWithContext {
    subst: f64,
    mat_mat: f64,
    mat_del: f64,
    mat_ins: f64,
    ins_mat: f64,
    ins_ins: f64,
    ins_del: f64,
    del_mat: f64,
    del_ins: f64,
    del_del: f64,
}

impl std::default::Default for ProfileWithContext {
    fn default() -> Self {
        Self {
            subst: 0.02,
            mat_mat: 0.90,
            mat_del: 0.05,
            mat_ins: 0.05,
            ins_mat: 0.5,
            ins_ins: 0.46,
            ins_del: 0.04,
            del_mat: 0.46,
            del_ins: 0.04,
            del_del: 0.5,
        }
    }
}

pub fn introduce_randomness_with_context<R: rand::Rng>(
    seq: &[u8],
    rng: &mut R,
    p: &ProfileWithContext,
) -> Vec<u8> {
    // 0->Mat,1->Ins,2->Del
    let mut state = 0;
    let mut idx = 0;
    let states = [0, 1, 2];
    let mut res = vec![];
    while idx < seq.len() {
        let weights = match state {
            0 => [p.mat_mat, p.mat_ins, p.mat_del],
            1 => [p.ins_mat, p.ins_ins, p.ins_del],
            _ => [p.del_mat, p.del_ins, p.del_del],
        };
        state = *states.choose_weighted(rng, |&s| weights[s]).unwrap();
        match state {
            0 if rng.gen_bool(p.subst) => res.push(choose_base(rng, seq[idx])),
            0 => res.push(seq[idx]),
            1 => res.push(random_base(rng)),
            _ => {}
        }
        idx += match state {
            0 | 1 => 1,
            _ => 0,
        };
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
        vec![Op::Mismatch; sub],
        vec![Op::Del; del],
        vec![Op::Ins; ins],
    ]
    .concat();
    operations.shuffle(rng);
    let mut res = vec![];
    let mut remainings: Vec<_> = seq.iter().copied().rev().collect();
    for op in operations {
        match op {
            Op::Match => res.push(remainings.pop().unwrap()),
            Op::Mismatch => res.push(choose_base(rng, remainings.pop().unwrap())),
            Op::Ins => res.push(random_base(rng)),
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
