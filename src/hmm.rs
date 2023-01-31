//! An tiny implementation of pair hidden Markov models.
use crate::op::Op;
use rand::Rng;
pub mod banded;
pub mod full;
pub mod guided;
use crate::logsumexp;
use crate::EP;
use serde::{Deserialize, Serialize};

const fn base_table() -> [usize; 256] {
    let mut slots = [0; 256];
    slots[b'A' as usize] = 0;
    slots[b'C' as usize] = 1;
    slots[b'G' as usize] = 2;
    slots[b'T' as usize] = 3;
    slots[b'a' as usize] = 0;
    slots[b'c' as usize] = 1;
    slots[b'g' as usize] = 2;
    slots[b't' as usize] = 3;
    slots
}
const BASE_TABLE: [usize; 256] = base_table();

/// HMM. As a rule of thumb, we do not take logarithm of each field; Scaling would be
/// better both in performance and computational stability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairHiddenMarkovModel {
    /// Prob from mat.
    /// Pr{Mat->Mat},
    pub mat_mat: f64,
    /// Pr{Mat->Ins}
    pub mat_ins: f64,
    /// Pr{Mat->Del}
    pub mat_del: f64,
    /// Pr{Ins->Mat}
    pub ins_mat: f64,
    /// Pr{Ins->Ins}
    pub ins_ins: f64,
    /// Pr{Ins->Del}
    pub ins_del: f64,
    /// Pr{Del->Mat}.
    pub del_mat: f64,
    /// Pr{Del -> Ins},
    pub del_ins: f64,
    /// Pr{Del->Del}
    pub del_del: f64,
    /// 4 * ref_base + query_base = Pr{Query|Ref}
    pub mat_emit: [f64; 16],
    /// 4 * prev_base + query_base = Pr{Query|Previous Query}.
    /// The last four slot is the initial emittion, 1/4.
    pub ins_emit: [f64; 20],
}

impl crate::gen_seq::Generate for PairHiddenMarkovModel {
    fn gen<R: rand::Rng>(&self, seq: &[u8], rng: &mut R) -> Vec<u8> {
        use rand::seq::SliceRandom;
        let states = [Op::Match, Op::Del, Op::Ins];
        let mut current = Op::Match;
        let mut gen = vec![];
        let mut seq = seq.iter().peekable();
        while seq.peek().is_some() {
            current = *states
                .choose_weighted(rng, |to| self.weight(current, *to))
                .unwrap();
            match current {
                Op::Match => {
                    let base = BASE_TABLE[*seq.next().unwrap() as usize] as usize;
                    let distr = &self.mat_emit[4 * base..4 * base + 4];
                    let pos = *[0, 1, 2, 3].choose_weighted(rng, |&i| distr[i]).unwrap();
                    gen.push(b"ACGT"[pos]);
                }
                Op::Ins => {
                    assert!(seq.next().is_some());
                    let prev = gen
                        .last()
                        .map(|b| BASE_TABLE[*b as usize] as usize)
                        .unwrap_or(4);
                    let distr = &self.ins_emit[4 * prev..4 * prev + 4];
                    let pos = *[0, 1, 2, 3].choose_weighted(rng, |&i| distr[i]).unwrap();
                    gen.push(b"ACGT"[pos]);
                }
                Op::Del => assert!(seq.next().is_some()),
                _ => panic!(),
            }
        }
        gen
    }
}

impl std::fmt::Display for PairHiddenMarkovModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "MAT:{:.3}\t{:.3}\t{:.3}",
            self.mat_mat, self.mat_ins, self.mat_del
        )?;
        writeln!(
            f,
            "INS:{:.3}\t{:.3}\t{:.3}",
            self.ins_mat, self.ins_ins, self.ins_del
        )?;
        writeln!(
            f,
            "DEL:{:.3}\t{:.3}\t{:.3}",
            self.del_mat, self.del_ins, self.del_del
        )?;
        for obs in self.mat_emit.chunks_exact(4) {
            let [a, c, g, t] = [obs[0], obs[1], obs[2], obs[3]];
            writeln!(f, "Obs:{a:.3}\t{c:.3}\t{g:.3}\t{t:.3}")?;
        }
        for (i, obs) in self.ins_emit.chunks_exact(4).enumerate() {
            let [a, c, g, t] = [obs[0], obs[1], obs[2], obs[3]];
            write!(f, "Ins:{a:.3}\t{c:.3}\t{g:.3}\t{t:.3}")?;
            if i < 4 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    Match,
    Del,
    Ins,
}

impl std::default::Default for PairHiddenMarkovModel {
    fn default() -> Self {
        let mat = (0.96, 0.02, 0.02);
        let ins = (0.85, 0.10, 0.05);
        let del = (0.85, 0.10, 0.05);
        let mat_emits = vec![
            vec![0.97, 0.01, 0.01, 0.01],
            vec![0.01, 0.97, 0.01, 0.01],
            vec![0.01, 0.01, 0.97, 0.01],
            vec![0.01, 0.01, 0.01, 0.97],
        ]
        .concat();
        let ins_emits = vec![vec![0.25; 4]; 5].concat();
        Self::new(mat, ins, del, &mat_emits, &ins_emits)
    }
}

impl PairHiddenMarkovModel {
    fn weight(&self, from: Op, to: Op) -> f64 {
        match (from, to) {
            (Op::Match, Op::Match) => self.mat_mat,
            (Op::Match, Op::Ins) => self.mat_ins,
            (Op::Match, Op::Del) => self.mat_del,
            (Op::Ins, Op::Match) => self.ins_mat,
            (Op::Ins, Op::Ins) => self.ins_ins,
            (Op::Ins, Op::Del) => self.ins_del,
            (Op::Del, Op::Match) => self.del_mat,
            (Op::Del, Op::Ins) => self.del_ins,
            (Op::Del, Op::Del) => self.del_del,
            _ => panic!(),
        }
    }
    pub fn new(
        (mat_mat, mat_ins, mat_del): (f64, f64, f64),
        (ins_mat, ins_ins, ins_del): (f64, f64, f64),
        (del_mat, del_ins, del_del): (f64, f64, f64),
        mat_emit: &[f64],
        ins_emit: &[f64],
    ) -> Self {
        assert_eq!(mat_emit.len(), 16);
        assert_eq!(ins_emit.len(), 20);
        assert!(0f64 <= mat_mat && 0f64 <= mat_ins && 0f64 <= mat_del);
        assert!(0f64 <= ins_mat && 0f64 <= ins_ins && 0f64 <= ins_del);
        assert!(0f64 <= del_mat && 0f64 <= del_ins && 0f64 <= del_del);
        assert!(mat_emit.iter().all(|&x| 0f64 <= x));
        let mat = mat_mat + mat_ins + mat_del;
        let ins = ins_mat + ins_ins + ins_del;
        let del = del_mat + del_ins + del_del;
        let mut mat_emit_norm = [0f64; 16];
        for (from, to) in mat_emit
            .chunks_exact(4)
            .zip(mat_emit_norm.chunks_exact_mut(4))
        {
            let sum: f64 = from.iter().sum();
            to.iter_mut().zip(from).for_each(|(x, y)| *x = *y / sum);
        }
        let mut ins_emit_norm = [0f64; 20];
        for (from, to) in ins_emit
            .chunks_exact(4)
            .zip(ins_emit_norm.chunks_exact_mut(4))
        {
            let sum: f64 = from.iter().sum();
            to.iter_mut().zip(from).for_each(|(x, y)| *x = *y / sum);
        }
        Self {
            mat_mat: mat_mat / mat,
            mat_ins: mat_ins / mat,
            mat_del: mat_del / mat,
            ins_mat: ins_mat / ins,
            ins_ins: ins_ins / ins,
            ins_del: ins_del / ins,
            del_mat: del_mat / del,
            del_ins: del_ins / del,
            del_del: del_del / del,
            mat_emit: mat_emit_norm,
            ins_emit: ins_emit_norm,
        }
    }
    fn uniform(unif: f64) -> Self {
        Self {
            mat_mat: unif,
            mat_ins: unif,
            mat_del: unif,
            ins_mat: unif,
            ins_ins: unif,
            ins_del: unif,
            del_mat: unif,
            del_ins: unif,
            del_del: unif,
            mat_emit: [unif; 16],
            ins_emit: [unif; 20],
        }
    }
    fn zeros() -> Self {
        Self {
            mat_mat: 0f64,
            mat_ins: 0f64,
            mat_del: 0f64,
            ins_mat: 0f64,
            ins_ins: 0f64,
            ins_del: 0f64,
            del_mat: 0f64,
            del_ins: 0f64,
            del_del: 0f64,
            mat_emit: [0f64; 16],
            ins_emit: [0f64; 20],
        }
    }
    fn obs(&self, r: u8, q: u8) -> f64 {
        let index = (BASE_TABLE[r as usize] << 2) | BASE_TABLE[q as usize];
        self.mat_emit[index as usize]
    }
    fn del(&self, _r: u8) -> f64 {
        1f64
    }
    fn ins(&self, q: u8, prev: Option<u8>) -> f64 {
        let prev = prev.unwrap_or(4);
        let index = (BASE_TABLE[prev as usize] << 2) | BASE_TABLE[q as usize];
        self.ins_emit[index as usize]
        //        0.25f64
    }

    fn to_mat(&self, (mat, ins, del): (f64, f64, f64)) -> f64 {
        mat * self.mat_mat + ins * self.ins_mat + del * self.del_mat
    }
    fn to_ins(&self, (mat, ins, del): (f64, f64, f64)) -> f64 {
        mat * self.mat_ins + ins * self.ins_ins + del * self.del_ins
    }
    fn to_del(&self, (mat, ins, del): (f64, f64, f64)) -> f64 {
        mat * self.mat_del + ins * self.ins_del + del * self.del_del
    }
    pub fn eval(&self, rs: &[u8], qs: &[u8], ops: &[Op]) -> f64 {
        use Op::*;
        let mut lk = 1f64;
        let mut current = Match;
        let (mut qpos, mut rpos) = (0, 0);
        let ops = ops.iter().map(|&op| match op {
            Mismatch => Match,
            x => x,
        });
        for op in ops {
            lk *= match (current, op) {
                (Match, Match) => self.mat_mat,
                (Match, Ins) => self.mat_ins,
                (Match, Del) => self.mat_del,
                (Ins, Match) => self.ins_mat,
                (Ins, Ins) => self.ins_ins,
                (Ins, Del) => self.ins_del,
                (Del, Match) => self.del_mat,
                (Del, Ins) => self.del_ins,
                (Del, Del) => self.del_del,
                _ => panic!(),
            };
            match op {
                Match => {
                    lk *= self.obs(qs[qpos], rs[rpos]);
                    qpos += 1;
                    rpos += 1;
                }
                Ins => {
                    let prev = (qpos != 0).then(|| qs[qpos - 1]);
                    lk *= self.ins(qs[qpos], prev);
                    qpos += 1;
                }
                Del => {
                    lk *= self.del(rs[rpos]);
                    rpos += 1;
                }
                _ => panic!(),
            };
            current = op;
        }
        lk.ln()
    }
    pub fn eval_ln(&self, rs: &[u8], qs: &[u8], ops: &[Op]) -> f64 {
        use Op::*;
        let mut lk = 0f64;
        let mut current = Match;
        let (mut qpos, mut rpos) = (0, 0);
        let ops = ops.iter().map(|&op| match op {
            Mismatch => Match,
            x => x,
        });
        for op in ops {
            lk += match (current, op) {
                (Match, Match) => self.mat_mat.ln(),
                (Match, Ins) => self.mat_ins.ln(),
                (Match, Del) => self.mat_del.ln(),
                (Ins, Match) => self.ins_mat.ln(),
                (Ins, Ins) => self.ins_ins.ln(),
                (Ins, Del) => self.ins_del.ln(),
                (Del, Match) => self.del_mat.ln(),
                (Del, Ins) => self.del_ins.ln(),
                (Del, Del) => self.del_del.ln(),
                _ => panic!(),
            };
            match op {
                Match => {
                    lk += self.obs(qs[qpos], rs[rpos]).ln();
                    qpos += 1;
                    rpos += 1;
                }
                Ins => {
                    let prev = (qpos != 0).then(|| qs[qpos - 1]);
                    lk += self.ins(qs[qpos], prev).ln();
                    qpos += 1;
                }
                Del => {
                    lk += self.del(rs[rpos]).ln();
                    rpos += 1;
                }
                _ => panic!(),
            };
            current = op;
        }
        lk
    }
}

// /// A pair hidden Markov model.
// /// To access the output prbabilities,
// /// call `.prob_of()` instead of direct membership access.
// /// In the membership description below, `X` means some arbitrary state except end state.
// /// As a principle of thumb, statistics function `*` and `*_banded`
// /// (`likelihood` and `likelihood_banded`, for example) would return the same type of values.
// /// If you want to more fine resolution outpt, please consult more specific function such as `forward` or `forward_banded`.
// #[derive(Debug, Clone)]
// pub struct PairHiddenMarkovModel {
//     /// Pr{X->Mat}
//     pub mat_ext: f64,
//     pub mat_from_ins: f64,
//     pub mat_from_del: f64,
//     /// Pr{Mat->Ins}
//     pub ins_open: f64,
//     /// Pr{Mat->Del}
//     pub del_open: f64,
//     /// Pr{Ins->Del}
//     pub del_from_ins: f64,
//     /// Pr{Del->Ins}.
//     pub ins_from_del: f64,
//     /// Pr{Del->Del}
//     pub del_ext: f64,
//     /// Pr{Ins->Ins}
//     pub ins_ext: f64,
//     // Pr{(-,base)|Del}. Bases are A,C,G,T, '-' and NULL. The last two "bases" are defined just for convinience.
//     del_emit: [f64; 6],
//     // Pr{(base,-)|Ins}
//     ins_emit: [f64; 6],
//     // Pr{(base,base)|Mat}
//     mat_emit: [f64; 64],
// }

/// Shorthand for PairHiddenMarkovModel.
#[allow(clippy::upper_case_acronyms)]
pub type PHMM = PairHiddenMarkovModel;

// #[derive(Debug, Clone, Copy)]
// pub enum State {
//     Mat,
//     Del,
//     Ins,
// }

impl std::convert::From<State> for usize {
    fn from(state: State) -> usize {
        match state {
            State::Match => 0,
            State::Del => 1,
            State::Ins => 2,
        }
    }
}

impl std::convert::From<State> for Op {
    fn from(state: State) -> Op {
        match state {
            State::Match => Op::Match,
            State::Del => Op::Del,
            State::Ins => Op::Ins,
        }
    }
}

// impl std::default::Default for PHMM {
//     fn default() -> Self {
//         let match_prob = 0.9;
//         let gap_ext_prob = 0.08;
//         let gap_output = [(4f64).recip(); 4];
//         let match_output = [
//             [0.9, 0.1 / 3., 0.1 / 3., 0.1 / 3.],
//             [0.1 / 3., 0.9, 0.1 / 3., 0.1 / 3.],
//             [0.1 / 3., 0.1 / 3., 0.9, 0.1 / 3.],
//             [0.1 / 3., 0.1 / 3., 0.1 / 3., 0.9],
//         ];
//         let quit_prob = 0.001;
//         PHMM::as_reversible(
//             match_prob,
//             gap_ext_prob,
//             &gap_output,
//             &match_output,
//             quit_prob,
//         )
//     }
// }

impl PHMM {
    // /// construct a new pair reversible HMM.
    // /// In reversible, I mean that the HMM is synmetric with respect to switching the reference and the query,
    // /// and w.r.t reversing the reference sequence and the query sequence.
    // /// In other words, it has the same probability to transit from deletion/insertion <-> matches,
    // /// same emittion probability on the deletion/insertion states, and so on.
    // /// # Example
    // /// ```rust
    // /// use kiley::hmm::PHMM;
    // /// let match_prob = 0.8;
    // /// let gap_ext_prob = 0.1;
    // /// let gap_output = [(4f64).recip(); 4];
    // /// let match_output = [
    // /// [0.7, 0.1, 0.1, 0.1],
    // /// [0.1, 0.7, 0.1, 0.1],
    // /// [0.1, 0.1, 0.7, 0.1],
    // /// [0.1, 0.1, 0.1, 0.7],
    // /// ];
    // /// let quit_prob = 0.001;
    // /// PHMM::as_reversible(
    // ///   match_prob,
    // ///   gap_ext_prob,
    // ///   &gap_output,
    // ///   &match_output,
    // ///   quit_prob,
    // /// );
    // /// ```
    // #[allow(clippy::wrong_self_convention)]
    // pub fn as_reversible(
    //     mat: f64,
    //     gap_ext: f64,
    //     gap_output: &[f64; 4],
    //     mat_output: &[[f64; 4]],
    //     _quit_prob: f64,
    // ) -> Self {
    //     assert!(mat.is_sign_positive());
    //     assert!(gap_ext.is_sign_positive());
    //     assert!(mat + gap_ext <= 1f64);
    //     let gap_open = (1f64 - mat) / 2f64;
    //     let gap_switch = 1f64 - gap_ext - mat;
    //     // let alive_prob = 1f64 - quit_prob;
    //     let mut gap_emit = [0f64; 6];
    //     gap_emit[..4].clone_from_slice(&gap_output[..4]);
    //     // Maybe we should have the matching function to compute this matrix...
    //     let mat_emit = {
    //         let mut slots = [0f64; 8 * 8];
    //         for i in 0..4 {
    //             for j in 0..4 {
    //                 slots[(i << 3) | j] = mat_output[i][j];
    //             }
    //         }
    //         slots
    //     };
    //     Self {
    //         mat_ext: mat,
    //         mat_from_del: mat,
    //         mat_from_ins: mat,
    //         ins_open: gap_open,
    //         del_open: gap_open,
    //         ins_from_del: gap_switch,
    //         del_from_ins: gap_switch,
    //         del_ext: gap_ext,
    //         ins_ext: gap_ext,
    //         del_emit: gap_emit,
    //         ins_emit: gap_emit,
    //         mat_emit,
    //     }
    // }
    fn log(x: &f64) -> f64 {
        assert!(!x.is_sign_negative());
        if x.abs() > 0.00000001 {
            x.ln()
        } else {
            EP
        }
    }
    fn logsumexp(x: f64, y: f64, z: f64) -> f64 {
        let max = x.max(y).max(z);
        max + ((x - max).exp() + (y - max).exp() + (z - max).exp()).ln()
    }
}

/// A dynamic programming table. It is a serialized 2-d array.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DPTable {
    mat_dp: Vec<f64>,
    ins_dp: Vec<f64>,
    del_dp: Vec<f64>,
    column: usize,
    row: usize,
}

impl DPTable {
    fn new(row: usize, column: usize) -> Self {
        Self {
            mat_dp: vec![0f64; column * row],
            ins_dp: vec![0f64; column * row],
            del_dp: vec![0f64; column * row],
            column,
            row,
        }
    }
    fn get_mut(&mut self, i: usize, j: usize, state: State) -> &mut f64 {
        match state {
            State::Match => self.mat_dp.get_mut(i * self.column + j).unwrap(),
            State::Del => self.del_dp.get_mut(i * self.column + j).unwrap(),
            State::Ins => self.ins_dp.get_mut(i * self.column + j).unwrap(),
        }
    }
    fn get(&self, i: usize, j: usize, state: State) -> f64 {
        match state {
            State::Match => *self.mat_dp.get(i * self.column + j).unwrap(),
            State::Del => *self.del_dp.get(i * self.column + j).unwrap(),
            State::Ins => *self.ins_dp.get(i * self.column + j).unwrap(),
        }
    }
    fn get_total_lk(&self, i: usize, j: usize) -> f64 {
        PHMM::logsumexp(
            self.get(i, j, State::Match),
            self.get(i, j, State::Ins),
            self.get(i, j, State::Del),
        )
    }
    pub fn lks_in_row(&self) -> Vec<f64> {
        self.mat_dp
            .chunks_exact(self.column)
            .zip(self.del_dp.chunks_exact(self.column))
            .zip(self.ins_dp.chunks_exact(self.column))
            .map(|((xs, ys), zs)| PHMM::logsumexp(logsumexp(xs), logsumexp(ys), logsumexp(zs)))
            .collect()
    }
    pub fn lks_in_row_by_state(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mat: Vec<_> = self
            .mat_dp
            .chunks_exact(self.column)
            .map(logsumexp)
            .collect();
        let ins: Vec<_> = self
            .ins_dp
            .chunks_exact(self.column)
            .map(logsumexp)
            .collect();
        let del: Vec<_> = self
            .del_dp
            .chunks_exact(self.column)
            .map(logsumexp)
            .collect();
        (mat, del, ins)
    }
}

/// A summary of comparison of two sequence.
#[derive(Debug, Clone)]
pub struct LikelihoodSummary {
    /// Match probability for each sequence `xs`. to get corresponding base, access mat_base below.
    pub match_prob: Vec<f64>,
    /// Matched base for match state of the i-th base of `xs`
    pub match_bases: Vec<[u8; 4]>,
    pub insertion_prob: Vec<f64>,
    pub insertion_bases: Vec<[u8; 4]>,
    pub deletion_prob: Vec<f64>,
    /// Likelihood of alignment between xs and ys.
    pub total_likelihood: f64,
    // pub likelihood_trajectry: Vec<f64>,
}

impl LikelihoodSummary {
    pub fn add(&mut self, other: &Self) {
        self.match_prob
            .iter_mut()
            .zip(other.match_prob.iter())
            .for_each(|(x, y)| *x += y);
        self.insertion_prob
            .iter_mut()
            .zip(other.insertion_prob.iter())
            .for_each(|(x, y)| *x += y);
        self.deletion_prob
            .iter_mut()
            .zip(other.deletion_prob.iter())
            .for_each(|(x, y)| *x += y);
        self.match_bases
            .iter_mut()
            .zip(other.match_bases.iter())
            .for_each(|(x, y)| x.iter_mut().zip(y.iter()).for_each(|(x, y)| *x += y));
        self.insertion_bases
            .iter_mut()
            .zip(other.insertion_bases.iter())
            .for_each(|(x, y)| x.iter_mut().zip(y.iter()).for_each(|(x, y)| *x += y));
        self.total_likelihood += other.total_likelihood;
    }
    pub fn div_probs(&mut self, r: f64) {
        self.match_prob.iter_mut().for_each(|x| *x /= r);
        self.deletion_prob.iter_mut().for_each(|x| *x /= r);
        self.insertion_prob.iter_mut().for_each(|x| *x /= r);
    }
    /// Flipping a suspicious erroneous position.
    pub fn correct_flip<R: Rng>(&self, rng: &mut R) -> Vec<u8> {
        let mut template: Vec<_> = self
            .match_bases
            .iter()
            .map(|x| Self::choose_max_base(x))
            .collect();
        let chunks = Self::chunk_by_homopolymer(&template);
        if rng.gen_bool(0.5) {
            // const THRESHOLD: f64 = 0.1;
            let (position, _) = chunks
                .iter()
                .map(|&(start, end)| {
                    let sum = self.deletion_prob[start..end].iter().sum::<f64>();
                    (start, sum)
                })
                .max_by(|x, y| (x.1).partial_cmp(&y.1).unwrap())
                .unwrap();
            template.remove(position);
        } else {
            let (start, end, _) = chunks
                .iter()
                .map(|&(start, end)| {
                    let sum = self.insertion_prob[start..end].iter().sum::<f64>();
                    (start, end, sum)
                })
                .max_by(|x, y| (x.1).partial_cmp(&y.1).unwrap())
                .unwrap();
            let (position, base) = {
                let position = (start..end)
                    .max_by(|&x, &y| {
                        self.insertion_prob[x]
                            .partial_cmp(&self.insertion_prob[y])
                            .unwrap()
                    })
                    .unwrap();
                let base = self.insertion_bases[position]
                    .iter()
                    .enumerate()
                    .max_by_key(|x| x.1)
                    .map(|(idx, _)| b"ACGT"[idx])
                    .unwrap();
                (position, base)
            };
            template.insert(position, base);
        };
        template
    }
    pub fn correct(&self, _template: &[u8]) -> Vec<u8> {
        // First, correct all substitution errors.
        let template: Vec<_> = self
            .match_bases
            .iter()
            .map(|x| Self::choose_max_base(x))
            .collect();
        // First, lets chunk the template by homopolymer run.
        let intervals = Self::chunk_by_homopolymer(&template);
        let mut polished_seq = vec![];
        for (start, end) in intervals {
            // Check if there is some insertion.
            // If there is, then, we do not remove any base and only care about insertions.
            // Note that the last putative insertion would be allowed.
            let some_ins = (start..end - 1).any(|i| 0.75 < self.insertion_prob[i]);
            if some_ins {
                let ziped = template.iter().zip(self.insertion_bases.iter());
                for (&mat, ins) in ziped.take(end).skip(start) {
                    polished_seq.push(mat);
                    polished_seq.push(Self::choose_max_base(ins));
                }
            } else {
                let del_length =
                    (self.deletion_prob[start..end].iter().sum::<f64>() / 0.75).floor() as usize;
                let take_len = (end - start).saturating_sub(del_length);
                polished_seq.extend(std::iter::repeat(template[start]).take(take_len));
                if 0.75 < self.insertion_prob[end - 1] {
                    polished_seq.push(Self::choose_max_base(&self.insertion_bases[end - 1]));
                }
            }
        }
        polished_seq
    }
    fn choose_max_base(xs: &[u8]) -> u8 {
        let (max_base, _) = xs.iter().enumerate().max_by_key(|x| x.1).unwrap();
        b"ACGT"[max_base]
    }
    fn chunk_by_homopolymer(template: &[u8]) -> Vec<(usize, usize)> {
        let mut intervals = vec![];
        let mut start = 0;
        while start < template.len() {
            let base = template[start];
            let mut end = start;
            while end < template.len() && template[end] == base {
                end += 1;
            }
            intervals.push((start, end));
            start = end;
        }
        intervals
    }
}

/// Guided version of the pair HMM on the forward and reverse strands.
#[derive(Debug, Clone)]
pub struct PairHiddenMarkovModelOnStrands {
    forward: PairHiddenMarkovModel,
    reverse: PairHiddenMarkovModel,
}

impl PairHiddenMarkovModelOnStrands {
    pub fn forward(&self) -> &PairHiddenMarkovModel {
        &self.forward
    }
    pub fn reverse(&self) -> &PairHiddenMarkovModel {
        &self.reverse
    }
    pub fn new(forward: PairHiddenMarkovModel, reverse: PairHiddenMarkovModel) -> Self {
        Self { forward, reverse }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingDataPack<'a, T, O>
where
    T: std::borrow::Borrow<[u8]> + Sync + Send,
    O: std::borrow::Borrow<[Op]> + Sync + Send,
{
    consensus: &'a [u8],
    directions: &'a [bool],
    sequences: &'a [T],
    operations: &'a [O],
}

impl<'a, T, O> TrainingDataPack<'a, T, O>
where
    T: std::borrow::Borrow<[u8]> + Sync + Send,
    O: std::borrow::Borrow<[Op]> + Sync + Send,
{
    pub fn new(
        consensus: &'a [u8],
        directions: &'a [bool],
        sequences: &'a [T],
        operations: &'a [O],
    ) -> Self {
        Self {
            consensus,
            directions,
            sequences,
            operations,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[test]
    fn align() {
        let phmm = PHMM::default();
        let (_table, ops, lk) = phmm.align(b"ACCG", b"ACCG");
        eprintln!("{:?}\t{:.3}", ops, lk);
        assert_eq!(ops, vec![Op::Match; 4]);
        let (_table, ops, lk) = phmm.align(b"ACCG", b"");
        eprintln!("{:?}\t{:.3}", ops, lk);
        assert_eq!(ops, vec![Op::Del; 4]);
        let (_table, ops, lk) = phmm.align(b"", b"ACCG");
        assert_eq!(ops, vec![Op::Ins; 4]);
        eprintln!("{:?}\t{:.3}", ops, lk);
        let (_table, ops, lk) = phmm.align(b"ATGCCGCACAGTCGAT", b"ATCCGC");
        eprintln!("{:?}\t{:.3}", ops, lk);
        use Op::*;
        let answer = vec![vec![Match; 2], vec![Del], vec![Match; 4], vec![Del; 9]].concat();
        assert_eq!(ops, answer);
    }
    // #[test]
    // fn forward() {
    //     let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
    //     let template = gen_seq::generate_seq(&mut rng, 300);
    //     let profile = gen_seq::PROFILE;
    //     let hmm = PHMM::default();
    //     for i in 0..10 {
    //         let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
    //         let (_, lkb) = hmm.likelihood_banded(&template, &seq, 100).unwrap();
    //         let (_, lk) = hmm.likelihood(&template, &seq);
    //         assert!((lkb - lk).abs() < 10., "{},{},{}", i, lkb, lk);
    //     }
    // }
    // #[test]
    // fn forward_short() {
    //     let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
    //     let template = gen_seq::generate_seq(&mut rng, 10);
    //     let hmm = PHMM::default();
    //     for i in 0..10 {
    //         let seq = gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
    //         let (_, lkb) = hmm.likelihood_banded(&template, &seq, 5).unwrap();
    //         let (_, lk) = hmm.likelihood(&template, &seq);
    //         if (lkb - lk).abs() > 0.1 {
    //             eprintln!("{}", String::from_utf8_lossy(&template));
    //             eprintln!("{}", String::from_utf8_lossy(&seq));
    //         }
    //         assert!((lkb - lk).abs() < 1f64, "{},{},{}", i, lkb, lk);
    //     }
    // }
    // #[test]
    // fn forward_banded_test() {
    //     let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
    //     let template = gen_seq::generate_seq(&mut rng, 30);
    //     let hmm = PHMM::default();
    //     let radius = 4;
    //     for _ in 0..10 {
    //         let seq = gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
    //         let (fwd, centers) = hmm.forward_banded(&template, &seq, radius);
    //         let k = (template.len() + seq.len()) as isize;
    //         let u_in_dp = (template.len() + radius) as isize - centers[k as usize];
    //         assert!(fwd.get_check(k, u_in_dp, State::Match).is_some());
    //         let table = hmm.forward(&template, &seq);
    //         let lk_banded = PHMM::logsumexp(
    //             fwd.get(k, u_in_dp, State::Match),
    //             fwd.get(k, u_in_dp, State::Del),
    //             fwd.get(k, u_in_dp, State::Ins),
    //         );
    //         let lk = table.get_total_lk(template.len(), seq.len());
    //         assert!((lk - lk_banded).abs() < 0.001, "{},{}", lk, lk_banded);
    //         let state = State::Del;
    //         for i in 0..template.len() + 1 {
    //             for j in 0..seq.len() + 1 {
    //                 let x = table.get(i, j, state);
    //                 if EP < x {
    //                     print!("{:.1}\t", x);
    //                 } else {
    //                     print!("{:.1}\t", 1f64);
    //                 }
    //             }
    //             println!();
    //         }
    //         println!();
    //         let mut dump = vec![vec![EP; seq.len() + 1]; template.len() + 1];
    //         // for k in 0..template.len() + seq.len() + 1 {
    //         for (k, center) in centers
    //             .iter()
    //             .enumerate()
    //             .take(template.len() + seq.len() + 1)
    //         {
    //             // let center = centers[k];
    //             for (pos, &lk) in fwd.get_row(k as isize, state).iter().enumerate() {
    //                 let u = pos as isize + center - radius as isize;
    //                 let (i, j) = (u, k as isize - u);
    //                 if (0..template.len() as isize + 1).contains(&i)
    //                     && (0..seq.len() as isize + 1).contains(&j)
    //                 {
    //                     dump[i as usize][j as usize] = lk;
    //                 }
    //             }
    //         }
    //         for line in dump {
    //             for x in line {
    //                 if EP < x {
    //                     print!("{:.1}\t", x);
    //                 } else {
    //                     print!("{:.1}\t", 1f64);
    //                 }
    //             }
    //             println!();
    //         }
    //         println!();
    //         for (k, center) in centers
    //             .iter()
    //             .enumerate()
    //             .take(template.len() + seq.len() + 1)
    //         {
    //             // for k in 0..template.len() + seq.len() + 1 {
    //             //     let center = centers[k];
    //             let k = k as isize;
    //             for (u, ((&mat, &del), &ins)) in fwd
    //                 .get_row(k, State::Match)
    //                 .iter()
    //                 .zip(fwd.get_row(k, State::Del).iter())
    //                 .zip(fwd.get_row(k, State::Ins).iter())
    //                 .enumerate()
    //                 .take(2 * radius - 2)
    //                 .skip(2)
    //             {
    //                 let u = u as isize + center - radius as isize;
    //                 let i = u;
    //                 let j = k as isize - u;
    //                 if 0 <= u && u <= template.len() as isize && 0 <= j && j <= seq.len() as isize {
    //                     let (i, j) = (i as usize, j as usize);
    //                     assert!((table.get(i, j, State::Match) - mat).abs() < 2.);
    //                     let del_exact = table.get(i, j, State::Del);
    //                     if 2f64 < (del_exact - del).abs() {
    //                         println!("{},{}", i, j);
    //                     }
    //                     assert!((del_exact - del).abs() < 2., "E{},B{}", del_exact, del);
    //                     let ins_exact = table.get(i, j, State::Ins);
    //                     assert!((ins_exact - ins).abs() < 2., "{},{}", ins_exact, ins);
    //                 }
    //             }
    //         }
    //     }
    // }
    // #[test]
    // fn backward_banded_test() {
    //     let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
    //     let template = gen_seq::generate_seq(&mut rng, 30);
    //     let hmm = PHMM::default();
    //     let radius = 5;
    //     for _ in 0..10 {
    //         let seq = gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
    //         let (_, centers) = hmm.forward_banded(&template, &seq, radius);
    //         let table = hmm.backward(&template, &seq);
    //         let bwd = hmm.backward_banded(&template, &seq, radius, &centers);
    //         println!();
    //         let state = State::Del;
    //         for i in 0..template.len() + 1 {
    //             for j in 0..seq.len() + 1 {
    //                 print!("{:.1}\t", table.get(i, j, state));
    //             }
    //             println!();
    //         }
    //         println!();

    //         println!();
    //         let mut dump = vec![vec![EP; seq.len() + 1]; template.len() + 1];
    //         for (k, center) in centers
    //             .iter()
    //             .enumerate()
    //             .take(template.len() + seq.len() + 1)
    //         {
    //             // for k in 0..template.len() + seq.len() + 1 {
    //             //     let center = centers[k];
    //             let k = k as isize;
    //             for (pos, &lk) in bwd.get_row(k, state).iter().enumerate() {
    //                 let u = pos as isize + center - radius as isize;
    //                 let (i, j) = (u, k as isize - u);
    //                 if (0..template.len() as isize + 1).contains(&i)
    //                     && (0..seq.len() as isize + 1).contains(&j)
    //                 {
    //                     dump[i as usize][j as usize] = lk;
    //                 }
    //             }
    //         }
    //         for line in dump {
    //             for x in line {
    //                 if EP < x {
    //                     print!("{:.1}\t", x);
    //                 } else {
    //                     print!("{:.1}\t", 1f64);
    //                 }
    //             }
    //             println!();
    //         }
    //         for (k, center) in centers
    //             .iter()
    //             .enumerate()
    //             .take(template.len() + seq.len() + 1)
    //         {
    //             // for k in 0..template.len() + seq.len() + 1 {
    //             //     let center = centers[k];
    //             let k = k as isize;
    //             for (u, ((&mat, &del), &ins)) in bwd
    //                 .get_row(k, State::Match)
    //                 .iter()
    //                 .zip(bwd.get_row(k, State::Del).iter())
    //                 .zip(bwd.get_row(k, State::Ins).iter())
    //                 .enumerate()
    //                 .take(2 * radius - 1)
    //                 .skip(2)
    //             {
    //                 let u = u as isize + center - radius as isize;
    //                 let (i, j) = (u, k as isize - u);
    //                 if 0 <= u && u <= template.len() as isize && 0 <= j && j <= seq.len() as isize {
    //                     let (i, j) = (i as usize, j as usize);
    //                     let mat_exact = table.get(i, j, State::Match);
    //                     assert!(
    //                         (mat_exact - mat).abs() < 2.,
    //                         "{},{},{},{}",
    //                         mat_exact,
    //                         mat,
    //                         i,
    //                         j
    //                     );
    //                     let diff = (table.get(i, j, State::Del) - del).abs() < 2f64;
    //                     assert!(diff, "{},{},{}", diff, i, j);
    //                     let diff = (table.get(i, j, State::Ins) - ins).abs() < 2.;
    //                     assert!(diff, "{},{}", i, j);
    //                 }
    //             }
    //         }
    //     }
    // }
    // #[test]
    // fn forward_backward_test() {
    //     let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
    //     let template = gen_seq::generate_seq(&mut rng, 100);
    //     let hmm = PHMM::default();
    //     let radius = 10;
    //     let profile = gen_seq::Profile {
    //         sub: 0.01,
    //         del: 0.01,
    //         ins: 0.01,
    //     };
    //     for _ in 0..100 {
    //         let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
    //         let profile_exact = hmm.get_profile(&template, &seq);
    //         let profile_banded = hmm.get_profile_banded(&template, &seq, radius).unwrap();
    //         assert!((profile_banded.total_likelihood - profile_exact.total_likelihood).abs() < 0.1);
    //         for (x, y) in profile_banded
    //             .match_prob
    //             .iter()
    //             .zip(profile_exact.match_prob.iter())
    //         {
    //             assert!((x - y).abs() < 0.1f64);
    //         }
    //         for (x, y) in profile_banded
    //             .deletion_prob
    //             .iter()
    //             .zip(profile_exact.deletion_prob.iter())
    //         {
    //             assert!((x - y).abs() < 0.1f64);
    //         }
    //         for (x, y) in profile_banded
    //             .insertion_prob
    //             .iter()
    //             .zip(profile_exact.insertion_prob.iter())
    //         {
    //             assert!((x - y).abs() < 0.1f64);
    //         }
    //         for (x, y) in profile_banded
    //             .match_bases
    //             .iter()
    //             .zip(profile_exact.match_bases.iter())
    //         {
    //             let diff = x
    //                 .iter()
    //                 .zip(y.iter())
    //                 .map(|(x, y)| (x != y) as u8)
    //                 .sum::<u8>();
    //             assert_eq!(diff, 0, "{:?},{:?}", x, y);
    //         }
    //     }
    // }
}
