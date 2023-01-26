// TODO: This file is too large. Split into several files.

use crate::bialignment::guided::bootstrap_ops;
use crate::bialignment::guided::re_fill_fill_range;
use crate::dptable::DPTable;
use crate::op::Op;
use serde::{Deserialize, Serialize};
use std::f64::MIN_POSITIVE;

#[derive(Debug, Clone)]
/// Configurations
pub struct HMMConfig {
    pub radius: usize,
    pub take_num: usize,
    pub ignore_edge: usize,
}

impl HMMConfig {
    pub fn new(radius: usize, take_num: usize, ignore_edge: usize) -> Self {
        Self {
            radius,
            take_num,
            ignore_edge,
        }
    }
}

fn sum((x, y, z): (f64, f64, f64)) -> f64 {
    x + y + z
}

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

pub const COPY_SIZE: usize = 3;
pub const DEL_SIZE: usize = 3;
pub const NUM_ROW: usize = 8 + COPY_SIZE + DEL_SIZE;
// After introducing mutation, we would take INACTIVE_TIME bases just as-is.
pub const INACTIVE_TIME: usize = 5;

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
                    let base = BASE_TABLE[*seq.next().unwrap() as usize];
                    let distr = &self.mat_emit[4 * base..4 * base + 4];
                    let pos = *[0, 1, 2, 3].choose_weighted(rng, |&i| distr[i]).unwrap();
                    gen.push(b"ACGT"[pos]);
                }
                Op::Ins => {
                    assert!(seq.next().is_some());
                    let prev = gen.last().map(|b| BASE_TABLE[*b as usize]).unwrap_or(4);
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
enum State {
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
    pub fn align(&self, rs: &[u8], qs: &[u8], radius: usize) -> (f64, Vec<Op>) {
        let mut ops = bootstrap_ops(rs.len(), qs.len());
        let mut memory = Memory::with_capacity(rs.len(), radius);
        let lk = self.update_aln_path(&mut memory, rs, qs, &mut ops);
        assert!(lk <= 0.0, "{},{:?}", lk, self);
        (lk, ops)
    }
    pub fn update_aln_path(
        &self,
        memory: &mut Memory,
        rs: &[u8],
        qs: &[u8],
        ops: &mut Vec<Op>,
    ) -> f64 {
        memory.set_fill_ranges(rs.len(), qs.len(), ops);
        memory.initialize();
        self.fill_viterbi(memory, rs, qs);
        let mut qpos = qs.len();
        let mut rpos = rs.len();
        assert_eq!(rpos, rs.len());
        let (mut state, score) = {
            let (mat, ins, del) = memory.pre.get(qpos, rpos);
            if ins <= mat && del <= mat {
                (State::Match, mat)
            } else if ins <= del && mat <= del {
                (State::Del, del)
            } else {
                if !(del <= ins && mat <= ins) {
                    // Fallback.
                    let rad = memory.default_radius;
                    let (_, fallback) =
                        crate::bialignment::guided::edit_dist_guided(rs, qs, ops, rad);
                    *ops = fallback;
                    let (qx, ax, rx) = crate::recover(rs, qs, ops);
                    for ((qx, ax), rx) in qx.chunks(200).zip(ax.chunks(200)).zip(rx.chunks(200)) {
                        trace!("{}", String::from_utf8_lossy(qx));
                        trace!("{}", String::from_utf8_lossy(ax));
                        trace!("{}\n", String::from_utf8_lossy(rx));
                    }
                    return -100f64 * (qs.len() + rs.len()) as f64;
                }
                (State::Ins, ins)
            }
        };
        ops.clear();
        // Trackback
        while 0 < qpos && 0 < rpos {
            let (r, q) = (rs[rpos - 1], qs[qpos - 1]);
            match state {
                State::Match => {
                    let (mat, ins, del) = memory.pre.get(qpos - 1, rpos - 1);
                    let mat_mat = mat * self.mat_mat * self.obs(r, q);
                    let del_mat = del * self.del_mat * self.obs(r, q);
                    let ins_mat = ins * self.ins_mat * self.obs(r, q);
                    state = if del_mat <= mat_mat && ins_mat <= mat_mat {
                        State::Match
                    } else if mat_mat <= ins_mat && del_mat <= ins_mat {
                        State::Ins
                    } else {
                        assert!(mat_mat <= del_mat && ins_mat <= del_mat);
                        State::Del
                    };
                    qpos -= 1;
                    rpos -= 1;
                    match q == r {
                        true => ops.push(Op::Match),
                        false => ops.push(Op::Mismatch),
                    }
                }
                State::Ins => {
                    let ins_prob = self.ins(q, (1 < qpos).then(|| qs[qpos - 2]));
                    let (mat, ins, del) = memory.pre.get(qpos - 1, rpos);
                    let mat_ins = mat * self.mat_ins * ins_prob;
                    let ins_ins = ins * self.ins_ins * ins_prob;
                    let del_ins = del * self.del_ins * ins_prob;
                    state = if ins_ins <= mat_ins && del_ins <= mat_ins {
                        State::Match
                    } else if mat_ins <= ins_ins && del_ins <= ins_ins {
                        State::Ins
                    } else {
                        assert!(mat_ins <= del_ins && ins_ins <= del_ins);
                        State::Del
                    };
                    qpos -= 1;
                    ops.push(Op::Ins);
                }
                State::Del => {
                    // No scaling factor needed.
                    let (mat, ins, del) = memory.pre.get(qpos, rpos - 1);
                    let mat_del = mat * self.mat_del * self.del(r);
                    let ins_del = ins * self.ins_del * self.del(r);
                    let del_del = del * self.del_del * self.del(r);
                    state = if ins_del <= mat_del && del_del <= mat_del {
                        State::Match
                    } else if mat_del <= ins_del && del_del <= ins_del {
                        State::Ins
                    } else {
                        assert!(mat_del <= del_del && ins_del <= del_del);
                        State::Del
                    };
                    rpos -= 1;
                    ops.push(Op::Del);
                }
            }
        }
        ops.extend(std::iter::repeat(Op::Del).take(rpos));
        ops.extend(std::iter::repeat(Op::Ins).take(qpos));
        ops.reverse();
        score.ln() + memory.pre_scl.iter().map(|x| x.ln()).sum::<f64>()
    }
    // Viterbi algorithm.
    // Maybe we need log-version....
    fn fill_viterbi(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        assert!(memory.pre_scl.is_empty());
        // 1. Initialize.
        {
            memory.pre.set(0, 0, (1f64, 0f64, 0f64));
            for j in 1..rs.len() + 1 {
                let (prev_m, _, prev_d) = memory.pre.get(0, j - 1);
                let del_cum =
                    ((prev_m * self.mat_del) + (prev_d * self.del_del)) * self.del(rs[j - 1]);
                memory.pre.set(0, j, (0f64, 0f64, del_cum));
            }
            memory.pre_scl.push(1f64);
        }
        // 2. Recur.
        for (i, &(start, end)) in memory.fill_ranges.iter().enumerate().skip(1) {
            let q = qs[i - 1];
            let ins_prob = self.ins(q, (1 < i).then(|| qs[i - 2]));
            // Ins & Matches first.
            // Scaling parameter.
            let mut sum = 0f64;
            if start == 0 {
                let (mat, ins, _) = memory.pre.get(i - 1, 0);
                let mat_ins = mat * self.mat_ins * ins_prob;
                let ins_ins = ins * self.ins_ins * ins_prob;
                memory.pre.set(i, 0, (0f64, mat_ins.max(ins_ins), 0f64));
                sum += mat_ins.max(ins_ins);
            }
            for j in start.max(1)..end {
                let r = rs[j - 1];
                let (mat, ins, del) = memory.pre.get(i - 1, j - 1);
                let now_mat = (mat * self.mat_mat)
                    .max(del * self.del_mat)
                    .max(ins * self.ins_mat)
                    * self.obs(r, q);
                let (mat, ins, del) = memory.pre.get(i - 1, j);
                let now_ins = (mat * self.mat_ins)
                    .max(ins * self.ins_ins)
                    .max(del * self.del_ins)
                    * ins_prob;
                memory.pre.set(i, j, (now_mat, now_ins, 0f64));
                sum += now_mat + now_ins;
            }
            memory.pre_scl.push(sum);
            if start == 0 {
                let slot = memory.pre.get_mut(i, 0).unwrap();
                slot.0 /= sum;
                slot.1 /= sum;
            }
            // Scaling and deletion.
            for j in start.max(1)..end {
                let r = rs[j - 1];
                let (mat, ins, del) = memory.pre.get(i, j - 1);
                let del = (mat * self.mat_del)
                    .max(ins * self.ins_del)
                    .max(del * self.del_del);
                if let Some((m, i, d)) = memory.pre.get_mut(i, j) {
                    *m /= sum;
                    *i /= sum;
                    *d = del * self.del(r);
                }
            }
        }
        assert_eq!(memory.pre_scl.len(), qs.len() + 1);
    }
    fn fill_pre_dp(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        assert!(memory.pre_scl.is_empty());
        // 1. Initialization.
        {
            memory.pre.set(0, 0, (1f64, 0f64, 0f64));
            for j in 1..rs.len() + 1 {
                let del_cum = self.to_del(memory.pre.get(0, j - 1)) * self.del(rs[j - 1]);
                memory.pre.set(0, j, (0f64, 0f64, del_cum));
            }
            memory.pre_scl.push(1f64);
        }
        // 2. Recur
        for (i, &(start, end)) in memory.fill_ranges.iter().enumerate().skip(1) {
            let q = qs[i - 1];
            let ins_prob = self.ins(q, (1 < i).then(|| qs[i - 2]));
            // Scaling
            let mut sum = 0f64;
            if start == 0 {
                let ins = self.to_ins(memory.pre.get(i - 1, 0)) * ins_prob;
                memory.pre.set(i, 0, (0f64, ins, 0f64));
                sum += ins;
            }
            // Match/Insertion first.
            for j in start.max(1)..end {
                let r = rs[j - 1];
                let mat = self.to_mat(memory.pre.get(i - 1, j - 1)) * self.obs(r, q);
                let ins = self.to_ins(memory.pre.get(i - 1, j)) * ins_prob;
                memory.pre.set(i, j, (mat, ins, 0f64));
                sum += mat + ins;
            }
            memory.pre_scl.push(sum);
            if start == 0 {
                let (m, i, _) = memory.pre.get_mut(i, 0).unwrap();
                *m /= sum;
                *i /= sum;
            }
            // deletion last.
            for j in start.max(1)..end {
                let r = rs[j - 1];
                let del = self.to_del(memory.pre.get(i, j - 1)) * self.del(r);
                if let Some((m, i, d)) = memory.pre.get_mut(i, j) {
                    *m /= sum;
                    *i /= sum;
                    *d = del;
                }
            }
        }
        assert_eq!(memory.pre_scl.len(), qs.len() + 1);
    }
    fn fill_post_dp(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        assert!(memory.post_scl.is_empty());
        // 1. Initialization
        {
            memory.post.set(qs.len(), rs.len(), (1f64, 1f64, 1f64));
            let mut acc = 1f64;
            for j in (0..rs.len()).rev() {
                acc *= self.del(rs[j]);
                let elm = (self.mat_del * acc, self.ins_del * acc, self.del_del * acc);
                memory.post.set(qs.len(), j, elm);
                acc *= self.del_del;
            }
            memory.post_scl.push(1f64);
        }
        // 2. Recur
        for (i, &(start, end)) in memory.fill_ranges.iter().enumerate().rev().skip(1) {
            let q = qs[i];
            let ins_prob = self.ins(q, (0 < i).then(|| qs[i - 1]));
            let mut sum = 0f64;
            if end == rs.len() + 1 {
                let ins = ins_prob * memory.post.get(i + 1, rs.len()).1;
                let elm = (self.mat_ins * ins, self.ins_ins * ins, self.del_ins * ins);
                sum += elm.0 + elm.1 + elm.2;
                memory.post.set(i, rs.len(), elm);
            }
            // Match/Ins first.
            for j in (start..end.min(rs.len())).rev() {
                let r = rs[j];
                let af_mat = self.obs(r, q) * memory.post.get(i + 1, j + 1).0;
                let af_ins = ins_prob * memory.post.get(i + 1, j).1;
                let mat = self.mat_mat * af_mat + self.mat_ins * af_ins;
                let ins = self.ins_mat * af_mat + self.ins_ins * af_ins;
                let del = self.del_mat * af_mat + self.del_ins * af_ins;
                sum += mat + ins + del;
                memory.post.set(i, j, (mat, ins, del));
            }
            // Scaling.
            for j in start..end {
                if let Some((m, i, d)) = memory.post.get_mut(i, j) {
                    *m /= sum;
                    *i /= sum;
                    *d /= sum;
                }
            }
            memory.post_scl.push(sum);
            // Deletion.
            for j in (start..end.min(rs.len())).rev() {
                let r = rs[j];
                let del = self.del(r) * memory.post.get(i, j + 1).2;
                if let Some((m, i, d)) = memory.post.get_mut(i, j) {
                    *m += self.mat_del * del;
                    *i += self.ins_del * del;
                    *d += self.del_del * del;
                }
            }
        }
        memory.post_scl.reverse();
        assert_eq!(memory.post_scl.len(), qs.len() + 1);
    }
    fn fill_mod_table(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        assert_eq!(memory.fill_ranges.len(), qs.len() + 1);
        let total_len = NUM_ROW * (rs.len() + 1);
        memory.mod_table.truncate(total_len);
        memory.mod_table.iter_mut().for_each(|x| *x = 0f64);
        if memory.mod_table.len() < total_len {
            let len = total_len - memory.mod_table.len();
            memory.mod_table.extend(std::iter::repeat(0f64).take(len));
        }
        // let X be the maximum of sum_{t=0}^{i} ln(Cf[t]) sum_{t=i}^{|Q|} ln(Cb[t])
        // return (X/|Q|).exp()
        let max: f64 = {
            let current: f64 = memory.post_scl.iter().map(|x| x.ln()).sum();
            memory
                .pre_scl
                .iter()
                .zip(memory.post_scl.iter())
                .scan(current, |current, (f, b)| {
                    let val = *current + f.ln();
                    *current = val - b.ln();
                    Some(val)
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap()
        };
        let scaling = (max / memory.post_scl.len() as f64).exp();
        // i-> prod_{t=0}^{i} scale[t]/scaling * prod_{t=i}^{N+1} scale[t]/scaling
        let scale_stay: Vec<_> = {
            let ln_scale = max / memory.post_scl.len() as f64;
            let current: f64 = memory.post_scl.iter().map(|x| x.ln() - ln_scale).sum();
            memory
                .pre_scl
                .iter()
                .zip(memory.post_scl.iter())
                .scan(current - ln_scale, |current, (f, b)| {
                    let val = *current + f.ln();
                    *current = val - b.ln();
                    Some(val)
                })
                .map(|lk| lk.exp())
                .collect()
        };
        // i-> prod_{t=0}^{i} scale[t]/scaling * prod_{t=i+1}^{N+1} scale[t]/scaling / scaling.
        // The last /scaling factor is needed .
        let scale_proc: Vec<_> = {
            let ln_scale = max / memory.post_scl.len() as f64;
            let current: f64 = memory
                .post_scl
                .iter()
                .skip(1)
                .map(|x| x.ln() - ln_scale)
                .sum();
            memory
                .pre_scl
                .iter()
                .zip(memory.post_scl.iter().skip(1))
                .scan(current - 2f64 * ln_scale, |current, (f, b)| {
                    let val = *current + f.ln();
                    *current = val - b.ln();
                    Some(val)
                })
                .map(|x| x.exp())
                .collect()
        };
        assert!(
            scale_proc.iter().all(|x| x.is_finite()),
            "{:?}\t{}\t{}",
            scale_proc,
            scaling,
            memory.post_scl[0],
        );
        assert!(scale_stay.iter().all(|x| x.is_finite()), "{:?}", scale_stay);
        assert!(scale_proc.iter().all(|x| !x.is_nan()), "{:?}", scale_proc);
        assert!(scale_stay.iter().all(|x| !x.is_nan()), "{:?}", scale_stay);
        assert_eq!(memory.post_scl.len(), qs.len() + 1);
        assert_eq!(memory.pre_scl.len(), qs.len() + 1);
        assert_eq!(memory.pre_scl.len() + 1, qs.len() + 2);
        let mut slots = [0f64; 8 + COPY_SIZE + DEL_SIZE];
        for ((i, &(start, end)), &q) in memory.fill_ranges.iter().enumerate().zip(qs.iter()) {
            for j in start..end {
                slots.iter_mut().for_each(|x| *x = 0f64);
                let (to_mat, to_del) = {
                    let prev = memory.pre.get(i, j);
                    (self.to_mat(prev), self.to_del(prev))
                };
                let post_no_del = memory.post.get(i, j).2 * scale_stay[i];
                let post_del_del = memory.post.get(i, j + 1).2 * scale_stay[i];
                let post_mat_mat = memory.post.get(i + 1, j + 1).0 * scale_proc[i];
                let post_ins_mat = memory.post.get(i + 1, j).0 * scale_proc[i];
                b"ACGT".iter().zip(slots.iter_mut()).for_each(|(&b, y)| {
                    let mat = self.obs(b, q) * post_mat_mat;
                    let del = self.del(b) * post_del_del;
                    *y = to_mat * mat + to_del * del;
                });
                b"ACGT"
                    .iter()
                    .zip(slots.iter_mut().skip(4))
                    .for_each(|(&b, s)| {
                        let mat = self.obs(b, q) * post_ins_mat;
                        let del = self.del(b) * post_no_del;
                        *s = to_mat * mat + to_del * del;
                    });
                // Copying the j..j+c bases ...
                (0..COPY_SIZE)
                    .filter(|len| j + len < rs.len())
                    .zip(slots.iter_mut().skip(8))
                    .for_each(|(len, y)| {
                        let pos = j + len + 1;
                        let r = rs[j];
                        let mat = self.obs(r, q) * post_mat_mat;
                        let del = self.del(r) * post_del_del;
                        let (to_mat, to_del) = {
                            let prev = memory.pre.get(i, pos);
                            (self.to_mat(prev), self.to_del(prev))
                        };
                        *y = to_mat * mat + to_del * del;
                    });
                // deleting the j..j+d bases..
                (0..DEL_SIZE)
                    .filter(|d| j + d + 1 < rs.len())
                    .zip(slots.iter_mut().skip(8 + COPY_SIZE))
                    .for_each(|(len, y)| {
                        let post = j + len + 1;
                        let r = rs[post];
                        let post_mat_mat = memory.post.get(i + 1, post + 1).0 * scale_proc[i];
                        let mat = self.obs(r, q) * post_mat_mat;
                        let post_del_del = memory.post.get(i, post + 1).2 * scale_stay[i];
                        let del = self.del(r) * post_del_del;
                        *y = to_mat * mat + to_del * del;
                    });
                let row_start = NUM_ROW * j;
                memory
                    .mod_table
                    .iter_mut()
                    .skip(row_start)
                    .zip(slots.iter())
                    .for_each(|(x, y)| *x += y);
            }
        }
        if let Some((start, end)) = memory.fill_ranges.last().copied() {
            slots.iter_mut().for_each(|x| *x = 0.0);
            let i = memory.fill_ranges.len() - 1;
            for j in start..end {
                let to_del = { self.to_del(memory.pre.get(i, j)) };
                let post_del_del = memory.post.get(i, j + 1).2 * scale_stay[i];
                let post_no_del = memory.post.get(i, j).2 * scale_stay[i];
                slots.iter_mut().for_each(|x| *x = 0f64);
                // Change the j-th base into ...
                b"ACGT".iter().zip(slots.iter_mut()).for_each(|(&b, y)| {
                    *y = to_del * post_del_del * self.del(b);
                });
                // Insertion before the j-th base ...
                b"ACGT"
                    .iter()
                    .zip(slots.iter_mut().skip(4))
                    .for_each(|(&b, y)| {
                        *y = to_del * self.del(b) * post_no_del;
                    });
                // Copying the j..j+c bases....
                (0..COPY_SIZE)
                    .filter(|len| j + len < rs.len())
                    .zip(slots.iter_mut().skip(8))
                    .for_each(|(len, y)| {
                        let r = rs[j];
                        let to_del = self.to_del(memory.pre.get(i, j + len + 1));
                        *y = to_del * self.del(r) * post_del_del;
                    });
                // Deleting the j..j+d bases
                (0..DEL_SIZE)
                    .filter(|len| j + len < rs.len())
                    .zip(slots.iter_mut().skip(8 + COPY_SIZE))
                    .for_each(|(len, y)| {
                        let post = j + len + 1;
                        *y = match rs.get(post) {
                            Some(&r) => {
                                let post_del_del = memory.post.get(i, post + 1).2 * scale_stay[i];
                                to_del * self.del(r) * post_del_del
                            }
                            None => scale_stay[i] * sum(memory.pre.get(i, j)) * memory.post_scl[i],
                        };
                    });
                let row_start = NUM_ROW * j;
                memory
                    .mod_table
                    .iter_mut()
                    .skip(row_start)
                    .zip(slots.iter())
                    .for_each(|(x, y)| *x += y);
            }
        }
        memory.pre_scl.truncate(qs.len() + 1);
        memory.post_scl.truncate(qs.len() + 1);
        let len = (qs.len() + 2) as f64;
        memory
            .mod_table
            .iter_mut()
            .for_each(|x| *x = x.max(MIN_POSITIVE).ln() + scaling.ln() * len);
    }
    fn update(&self, memory: &mut Memory, rs: &[u8], qs: &[u8], ops: &[Op]) -> Option<f64> {
        memory.set_fill_ranges(rs.len(), qs.len(), ops);
        memory.initialize();
        self.fill_pre_dp(memory, rs, qs);
        self.fill_post_dp(memory, rs, qs);
        let lk = memory.post.get(0, 0).0.ln() + memory.post_scl.iter().map(|x| x.ln()).sum::<f64>();
        let lk2 = {
            let (mat, ins, del) = memory.pre.get(qs.len(), rs.len());
            (mat + del + ins).ln() + memory.pre_scl.iter().map(|x| x.ln()).sum::<f64>()
        };
        if 0.0001 < (lk - lk2).abs() || lk.is_nan() || lk2.is_nan() {
            trace!("{},{}", lk, lk2);
            trace!("MODEL\t{}", self);
            let ops: String = ops.iter().map(|x| format!("{}", x)).collect();
            trace!("OPS\t{}", ops);
            trace!("REF\t{}\t{}", String::from_utf8_lossy(rs), rs.len());
            trace!("QRY\t{}\t{}", String::from_utf8_lossy(qs), qs.len());
            None
        } else {
            self.fill_mod_table(memory, rs, qs);
            Some(lk)
        }
    }
    fn lk(&self, memory: &mut Memory, rs: &[u8], qs: &[u8], ops: &[Op]) -> f64 {
        memory.set_fill_ranges(rs.len(), qs.len(), ops);
        memory.initialize();
        self.fill_post_dp(memory, rs, qs);
        memory.post.get(0, 0).0.ln() + memory.post_scl.iter().map(|x| x.ln()).sum::<f64>()
    }
    pub fn likelihood(&self, rs: &[u8], qs: &[u8], radius: usize) -> f64 {
        let ops = bootstrap_ops(rs.len(), qs.len());
        self.likelihood_guided(rs, qs, &ops, radius)
    }
    pub fn likelihood_guided(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: usize) -> f64 {
        let mut memory = Memory::with_capacity(rs.len(), radius);
        memory.fill_ranges.clear();
        memory
            .fill_ranges
            .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
        re_fill_fill_range(qs.len(), rs.len(), ops, radius, &mut memory.fill_ranges);
        memory.initialize();
        self.fill_pre_dp(&mut memory, rs, qs);
        let (mat, ins, del) = memory.pre.get(qs.len(), rs.len());
        assert!(!(mat + ins + del).is_nan(), "{},{},{}", mat, ins, del);
        (mat + del + ins).ln() + memory.pre_scl.iter().map(|x| x.ln()).sum::<f64>()
    }
    pub fn likelihood_guided_post(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: usize) -> f64 {
        let mut memory = Memory::with_capacity(rs.len(), radius);
        memory.fill_ranges.clear();
        memory
            .fill_ranges
            .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
        re_fill_fill_range(qs.len(), rs.len(), ops, radius, &mut memory.fill_ranges);
        memory.initialize();
        self.fill_post_dp(&mut memory, rs, qs);
        memory.post.get(0, 0).0.ln() + memory.post_scl.iter().map(|x| x.ln()).sum::<f64>()
    }
    pub fn modification_table(
        &self,
        rs: &[u8],
        qs: &[u8],
        radius: usize,
        ops: &[Op],
    ) -> Option<(Vec<f64>, f64)> {
        let mut memory = Memory::with_capacity(rs.len(), radius);
        let lk = self.update(&mut memory, rs, qs, ops)?;
        Some((memory.mod_table, lk))
    }

    /// With bootstrap operations and taking numbers.
    /// The returned operations would be consistent with the returned sequence.
    /// Only the *first* `take_num` seuqneces would be used to polish,
    /// but the operations of the rest are also updated accordingly.
    /// This is (slightly) faster than the usual/full polishing...
    pub fn polish_until_converge_with_take<T, O>(
        &self,
        draft: &[u8],
        xs: &[T],
        ops: &mut [O],
        radius: usize,
        take_num: usize,
    ) -> Vec<u8>
    where
        T: std::borrow::Borrow<[u8]>,
        O: std::borrow::BorrowMut<Vec<Op>>,
    {
        let config = HMMConfig::new(radius, take_num, 0);
        self.polish_until_converge_with_conf(draft, xs, ops, &config)
    }
    /// With bootstrap operations and taking numbers.
    /// The returned operations would be consistent with the returned sequence.
    /// Only the *first* `take_num` seuqneces would be used to polish,
    /// but the operations of the rest are also updated accordingly.
    /// This is (slightly) faster than the usual/full polishing...
    pub fn polish_until_converge_with_conf<T, O>(
        &self,
        draft: &[u8],
        xs: &[T],
        ops: &mut [O],
        config: &HMMConfig,
    ) -> Vec<u8>
    where
        T: std::borrow::Borrow<[u8]>,
        O: std::borrow::BorrowMut<Vec<Op>>,
    {
        let &HMMConfig {
            radius,
            take_num,
            ignore_edge,
        } = config;
        let take_num = take_num.min(xs.len());
        assert!(!xs.is_empty());
        let mut template = draft.to_vec();
        let len = (template.len() / 2).max(3);
        let mut modif_table = Vec::new();
        let mut memory = Memory::with_capacity(template.len(), radius);
        let mut current_max = None;
        'outer: for t in 0..100 {
            let inactive = INACTIVE_TIME + (t * INACTIVE_TIME) % len;
            modif_table.clear();
            let mut current_lk = 0f64;
            let seq_stream = ops.iter_mut().zip(xs.iter());
            for (i, (ops, seq)) in seq_stream.enumerate() {
                let ops = ops.borrow_mut();
                let _ = self.update_aln_path(&mut memory, &template, seq.borrow(), ops);
                if take_num <= i {
                    continue;
                }
                if let Some(lk_of_read) = self.update(&mut memory, &template, seq.borrow(), ops) {
                    current_lk += lk_of_read;
                    if modif_table.is_empty() {
                        modif_table.extend_from_slice(&memory.mod_table)
                    } else {
                        modif_table
                            .iter_mut()
                            .zip(memory.mod_table.iter())
                            .for_each(|(x, y)| *x += y);
                    }
                }
            }
            assert_eq!(modif_table.len(), NUM_ROW * (template.len() + 1));
            // Ignore edge regions
            for (i, lk) in modif_table.iter_mut().enumerate() {
                let pos = i / NUM_ROW;
                if pos < ignore_edge || template.len() + 1 - ignore_edge < pos {
                    *lk = -1000000000000000000000000000000f64;
                }
            }
            let changed_pos = polish_guided(&mut template, &modif_table, current_lk, inactive);
            let edit_path = changed_pos.iter().map(|&(pos, op)| {
                if op < 4 {
                    (pos, crate::op::Edit::Subst)
                } else if op < 8 {
                    (pos, crate::op::Edit::Insertion)
                } else if op < 8 + COPY_SIZE {
                    (pos, crate::op::Edit::Copy(op - 8 + 1))
                } else {
                    (pos, crate::op::Edit::Deletion(op - 8 - COPY_SIZE + 1))
                }
            });
            for (ops, seq) in ops.iter_mut().zip(xs.iter()) {
                let ops = ops.borrow_mut();
                let seq = seq.borrow();
                let (qlen, rlen) = (seq.len(), template.len());
                crate::op::fix_alignment_path(ops, edit_path.clone(), qlen, rlen);
            }
            memory.update_radius(&changed_pos, template.len());
            if changed_pos.is_empty() {
                if matches!(current_max,Some(lk) if current_lk < lk + 0.1) {
                    break 'outer;
                }
                // trace!("NOWLK\t{}", current_lk);
                current_max = Some(current_lk);
                let is_updated = ops
                    .iter_mut()
                    .zip(xs.iter())
                    .take(take_num)
                    .enumerate()
                    .map(|(_, (ops, seq))| {
                        let (ops, seq) = (ops.borrow_mut(), seq.borrow());
                        let lk = self.lk(&mut memory, &template, seq, ops);
                        let edop = crate::edlib_global(&template, seq);
                        let lk2 = self.lk(&mut memory, &template, seq, &edop);
                        if lk + 0.1 < lk2 {
                            // trace!("{t}\t{i}\t{:.3}\t{:.3}", lk, lk2);
                            *ops = edop;
                            true
                        } else {
                            false
                        }
                    })
                    .fold(false, |is_updated, b| is_updated | b);
                if !is_updated {
                    break 'outer;
                }
            }
        }
        template
    }
    /// With bootstrap operations. The returned operations would be
    /// consistent with the returned sequence.
    pub fn polish_until_converge_with<T: std::borrow::Borrow<[u8]>>(
        &self,
        draft: &[u8],
        xs: &[T],
        ops: &mut [Vec<Op>],
        radius: usize,
    ) -> Vec<u8> {
        let config = HMMConfig::new(radius, xs.len(), 0);
        self.polish_until_converge_with_conf(draft, xs, ops, &config)
    }
    pub fn polish_until_converge<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        xs: &[T],
        radius: usize,
    ) -> Vec<u8> {
        let mut ops: Vec<_> = xs
            .iter()
            .map(|seq| bootstrap_ops(template.len(), seq.borrow().len()))
            .collect();
        let config = HMMConfig::new(radius, xs.len(), 0);
        self.polish_until_converge_with_conf(template, xs, &mut ops, &config)
    }
    pub fn fit_by_alignment<T: std::borrow::Borrow<[u8]>, O: std::borrow::Borrow<[Op]>>(
        &mut self,
        template: &[u8],
        xss: &[T],
        radius: usize,
    ) {
        let ops: Vec<_> = xss
            .iter()
            .map(|xs| self.align(template, xs.borrow(), radius).1)
            .collect();
        self.fit_by_alignment_with(template, xss, &ops);
    }
    pub fn fit_by_alignment_with<T: std::borrow::Borrow<[u8]>, O: std::borrow::Borrow<[Op]>>(
        &mut self,
        template: &[u8],
        xss: &[T],
        ops: &[O],
    ) {
        // From -> To. Mat, Ins, Del = 0, 1, 2
        fn op_to_state(op: Op) -> usize {
            match op {
                Op::Mismatch => 0,
                Op::Match => 0,
                Op::Ins => 1,
                Op::Del => 2,
            }
        }
        let mut transitions = [[1f64; 3]; 3];
        let mut mat_emit = [1f64; 16];
        let mut ins_emit = [1f64; 20];
        for (ops, xs) in ops.iter().zip(xss.iter()) {
            let ops = ops.borrow();
            let xs = xs.borrow();
            if ops.len() < 2 {
                continue;
            }
            let mut state = op_to_state(ops[0]);
            let (mut rpos, mut qpos) = (0, 0);
            let rbase = BASE_TABLE[template[rpos] as usize] << 2;
            let qbase = BASE_TABLE[xs[qpos] as usize];
            match state {
                0 => {
                    mat_emit[rbase | qbase] += 1f64;
                    rpos += 1;
                    qpos += 1;
                }
                1 => {
                    ins_emit[rbase | qbase] += 1f64;
                    qpos += 1;
                }
                _ => {
                    rpos += 1;
                }
            }
            for op in ops.iter().skip(1) {
                let next = op_to_state(*op);
                transitions[state][next] += 1f64;
                state = next;
                match state {
                    0 => {
                        let rbase = BASE_TABLE[template[rpos] as usize] << 2;
                        let qbase = BASE_TABLE[xs[qpos] as usize];
                        mat_emit[rbase | qbase] += 1f64;
                        rpos += 1;
                        qpos += 1;
                    }
                    1 => {
                        let rbase = BASE_TABLE[template[rpos] as usize] << 2;
                        let qbase = BASE_TABLE[xs[qpos] as usize];
                        ins_emit[rbase | qbase] += 1f64;
                        qpos += 1;
                    }
                    _ => {
                        rpos += 1;
                    }
                }
            }
        }
        for &base in template.iter() {
            ins_emit[16 + BASE_TABLE[base as usize]] += 1.0;
        }
        for state in ins_emit.chunks_exact_mut(4) {
            let sum: f64 = state.iter().sum();
            state.iter_mut().for_each(|x| *x /= sum);
        }
        for state in mat_emit.chunks_exact_mut(4) {
            let sum: f64 = state.iter().sum();
            state.iter_mut().for_each(|x| *x /= sum);
        }
        for from_state in transitions.iter_mut() {
            let sum: f64 = from_state.iter().sum();
            from_state.iter_mut().for_each(|x| *x /= sum);
        }
        self.ins_emit = ins_emit;
        self.mat_emit = mat_emit;
        self.mat_mat = transitions[0][0];
        self.mat_ins = transitions[0][1];
        self.mat_del = transitions[0][2];
        self.ins_mat = transitions[1][0];
        self.ins_ins = transitions[1][1];
        self.ins_del = transitions[1][2];
        self.del_mat = transitions[2][0];
        self.del_ins = transitions[2][1];
        self.del_del = transitions[2][2];
    }
    pub fn fit_naive_with<T: std::borrow::Borrow<[u8]>, O: std::borrow::Borrow<[Op]>>(
        &mut self,
        template: &[u8],
        xss: &[T],
        ops: &[O],
        radius: usize,
    ) {
        assert_eq!(xss.len(), ops.len());
        let rs = template;
        let mut memory = Memory::with_capacity(template.len(), radius);
        let mut next = PairHiddenMarkovModel::uniform(0.0005);
        for (ops, seq) in ops.iter().zip(xss.iter()) {
            let qs = seq.borrow();
            let ops = ops.borrow();
            memory.fill_ranges.clear();
            memory
                .fill_ranges
                .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
            re_fill_fill_range(qs.len(), rs.len(), ops, radius, &mut memory.fill_ranges);
            memory.initialize();
            self.fill_pre_dp(&mut memory, rs, qs);
            self.fill_post_dp(&mut memory, rs, qs);
            let lk =
                memory.post.get(0, 0).0.ln() + memory.post_scl.iter().map(|x| x.ln()).sum::<f64>();
            let lk2 = {
                let (mat, ins, del) = memory.pre.get(qs.len(), rs.len());
                (mat + del + ins).ln() + memory.pre_scl.iter().map(|x| x.ln()).sum::<f64>()
            };
            if 0.0001 < (lk - lk2).abs() {
                continue;
            }
            self.register_naive(&memory, rs, qs, &mut next);
        }
        next.normalize();
        *self = next;
    }
    pub fn fit_naive_with_par<T, O>(&mut self, template: &[u8], xss: &[T], ops: &[O], radius: usize)
    where
        T: std::borrow::Borrow<[u8]> + Sync + Send,
        O: std::borrow::Borrow<[Op]> + Sync + Send,
    {
        assert_eq!(xss.len(), ops.len());
        let rs = template;
        use rayon::prelude::*;
        let mut next = Self::uniform(0.0005);
        let folded = ops
            .par_iter()
            .zip(xss.par_iter())
            .filter_map(|(ops, seq)| {
                let qs = seq.borrow();
                let ops = ops.borrow();
                let mut memory = Memory::with_capacity(template.len(), radius);
                memory.fill_ranges.clear();
                memory
                    .fill_ranges
                    .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
                re_fill_fill_range(qs.len(), rs.len(), ops, radius, &mut memory.fill_ranges);
                memory.initialize();
                self.fill_pre_dp(&mut memory, rs, qs);
                self.fill_post_dp(&mut memory, rs, qs);
                let lk = memory.post.get(0, 0).0.ln()
                    + memory.post_scl.iter().map(|x| x.ln()).sum::<f64>();
                let lk2 = {
                    let (mat, ins, del) = memory.pre.get(qs.len(), rs.len());
                    (mat + del + ins).ln() + memory.pre_scl.iter().map(|x| x.ln()).sum::<f64>()
                };
                if 0.0001 < (lk - lk2).abs() {
                    None
                } else {
                    Some((memory, qs))
                }
            })
            .fold(PairHiddenMarkovModel::zeros, |mut next, (memory, qs)| {
                self.register_naive(&memory, rs, qs, &mut next);
                next
            })
            .reduce(PairHiddenMarkovModel::zeros, |mut x, y| {
                x.merge(&y);
                x
            });
        next.merge(&folded);
        // next.regularize(0.1);
        next.normalize();
        *self = next;
    }
    pub fn fit_naive<T: std::borrow::Borrow<[u8]>>(
        &mut self,
        template: &[u8],
        xss: &[T],
        radius: usize,
    ) {
        let ops: Vec<_> = xss
            .iter()
            .map(|seq| bootstrap_ops(template.len(), seq.borrow().len()))
            .collect();
        self.fit_naive_with(template, xss, &ops, radius);
    }
    fn register_naive(&self, mem: &Memory, rs: &[u8], qs: &[u8], next: &mut Self) {
        // Scaling factors
        // i => sum_{t=0}^{i} ln C_F[t]
        let forward: Vec<_> = mem
            .pre_scl
            .iter()
            .scan(0f64, |acc, scl| {
                *acc += scl.ln();
                Some(*acc)
            })
            .collect();
        // i => sum_{t=i}^{T} ln C_B[t]
        let backward: Vec<_> = {
            let mut temp: Vec<_> = mem
                .post_scl
                .iter()
                .rev()
                .scan(0f64, |acc, scl| {
                    *acc += scl.ln();
                    Some(*acc)
                })
                .collect();
            temp.reverse();
            temp
        };
        if forward.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return;
        }
        if backward.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return;
        }
        // Obs probs (Mat)
        let mut mat_probs: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 4]; 4];
        for ((i, &(start, end)), &q) in mem.fill_ranges.iter().enumerate().skip(1).zip(qs.iter()) {
            let scale = forward[i] + backward[i];
            for j in start.max(1)..end {
                let r = rs[j - 1];
                let (before, _, _) = mem.pre.get(i, j);
                let (after, _, _) = mem.post.get(i, j);
                let lk = (before * after).max(std::f64::MIN_POSITIVE).ln() + scale;
                mat_probs[BASE_TABLE[r as usize]][BASE_TABLE[q as usize]].push(lk);
            }
        }
        for (r, lks) in mat_probs.iter().enumerate() {
            let lks: Vec<_> = lks.iter().map(|x| logsumexp(x)).collect();
            let total = logsumexp(&lks);
            let sum: f64 = lks.iter().map(|x| (x - total).exp()).sum();
            if 0.0001 < (sum - 1f64).abs() {
                continue;
            }
            for (q, lk) in lks.iter().enumerate() {
                next.mat_emit[r << 2 | q] += (lk - total).exp();
            }
        }
        // Obs probs (Ins)
        let mut ins_probs = vec![vec![Vec::new(); 4]; 4];
        for ((i, &(start, end)), &q) in mem.fill_ranges.iter().enumerate().skip(1).zip(qs.iter()) {
            let prev = match 1 < i {
                true => qs[i - 2],
                false => continue,
            };
            let scale = forward[i] + backward[i];
            for j in start.max(1)..end {
                let (_, before, _) = mem.pre.get(i, j);
                let (_, after, _) = mem.post.get(i, j - 1);
                let lk = (before * after).max(std::f64::MIN_POSITIVE).ln() + scale;
                ins_probs[BASE_TABLE[prev as usize]][BASE_TABLE[q as usize]].push(lk);
            }
        }
        for (prev, lks) in ins_probs.iter().enumerate() {
            let lks: Vec<_> = lks.iter().map(|x| logsumexp(x)).collect();
            let total = logsumexp(&lks);
            let sum: f64 = lks.iter().map(|x| (x - total).exp()).sum();
            if 0.0001 < (sum - 1f64).abs() {
                continue;
            }
            for (q, lk) in lks.iter().enumerate() {
                next.ins_emit[prev << 2 | q] += (lk - total).exp();
            }
        }
        // Transition probs
        let mut mat_to = (vec![], vec![], vec![]);
        let mut ins_to = (vec![], vec![], vec![]);
        let mut del_to = (vec![], vec![], vec![]);
        for ((i, &(start, end)), &q) in mem.fill_ranges.iter().enumerate().zip(qs.iter()) {
            let ins_prob = self.ins(q, (0 < i).then(|| qs[i - 1]));
            let stay = forward[i] + backward[i];
            let proceed = forward[i] + backward[i + 1];
            for (j, &r) in rs.iter().enumerate().take(end).skip(start) {
                let (from_mat, from_ins, from_del) = mem.pre.get(i, j);
                let after_mat = mem.post.get(i + 1, j + 1).0;
                let after_ins = mem.post.get(i + 1, j).1;
                let after_del = mem.post.get(i, j + 1).2;
                let mat_to_mat = from_mat * self.mat_mat * self.obs(r, q) * after_mat;
                let mat_to_ins = from_mat * self.mat_ins * ins_prob * after_ins;
                let mat_to_del = from_mat * self.mat_del * self.del(r) * after_del;
                mat_to.0.push(mat_to_mat.max(MIN_POSITIVE).ln() + proceed);
                mat_to.1.push(mat_to_ins.max(MIN_POSITIVE).ln() + proceed);
                mat_to.2.push(mat_to_del.max(MIN_POSITIVE).ln() + stay);
                let ins_to_mat = from_ins * self.ins_mat * self.obs(r, q) * after_mat;
                let ins_to_ins = from_ins * self.ins_ins * ins_prob * after_ins;
                let ins_to_del = from_ins * self.ins_del * self.del(r) * after_del;
                ins_to.0.push(ins_to_mat.max(MIN_POSITIVE).ln() + proceed);
                ins_to.1.push(ins_to_ins.max(MIN_POSITIVE).ln() + proceed);
                ins_to.2.push(ins_to_del.max(MIN_POSITIVE).ln() + stay);
                let del_to_mat = from_del * self.del_mat * self.obs(r, q) * after_mat;
                let del_to_ins = from_del * self.del_ins * ins_prob * after_ins;
                let del_to_del = from_del * self.del_del * self.del(r) * after_del;
                del_to.0.push(del_to_mat.max(MIN_POSITIVE).ln() + proceed);
                del_to.1.push(del_to_ins.max(MIN_POSITIVE).ln() + proceed);
                del_to.2.push(del_to_del.max(MIN_POSITIVE).ln() + stay);
            }
        }
        let mat_to_mat = logsumexp(&mat_to.0);
        let mat_to_ins = logsumexp(&mat_to.1);
        let mat_to_del = logsumexp(&mat_to.2);
        let from_mat = logsumexp(&[mat_to_mat, mat_to_ins, mat_to_del]);
        next.mat_mat += (mat_to_mat - from_mat).exp();
        next.mat_ins += (mat_to_ins - from_mat).exp();
        next.mat_del += (mat_to_del - from_mat).exp();
        let ins_to_mat = logsumexp(&ins_to.0);
        let ins_to_ins = logsumexp(&ins_to.1);
        let ins_to_del = logsumexp(&ins_to.2);
        let from_ins = logsumexp(&[ins_to_mat, ins_to_ins, ins_to_del]);
        next.ins_mat += (ins_to_mat - from_ins).exp();
        next.ins_ins += (ins_to_ins - from_ins).exp();
        next.ins_del += (ins_to_del - from_ins).exp();
        let del_to_mat = logsumexp(&del_to.0);
        let del_to_ins = logsumexp(&del_to.1);
        let del_to_del = logsumexp(&del_to.2);
        let from_del = logsumexp(&[del_to_mat, del_to_del, del_to_ins]);
        next.del_mat += (del_to_mat - from_del).exp();
        next.del_ins += (del_to_ins - from_del).exp();
        next.del_del += (del_to_del - from_del).exp();
    }
    fn merge(&mut self, other: &Self) {
        self.mat_mat += other.mat_mat;
        self.mat_ins += other.mat_ins;
        self.mat_del += other.mat_del;
        self.ins_mat += other.ins_mat;
        self.ins_del += other.ins_del;
        self.ins_ins += other.ins_ins;
        self.del_mat += other.del_mat;
        self.del_del += other.del_del;
        self.del_ins += other.del_ins;
        self.mat_emit
            .iter_mut()
            .zip(other.mat_emit.iter())
            .for_each(|(x, y)| *x += y);
        self.ins_emit
            .iter_mut()
            .zip(other.ins_emit.iter())
            .for_each(|(x, y)| *x += y);
    }

    fn normalize(&mut self) {
        let mat_sum = self.mat_mat + self.mat_ins + self.mat_del;
        self.mat_mat /= mat_sum;
        self.mat_ins /= mat_sum;
        self.mat_del /= mat_sum;
        let ins_sum = self.ins_mat + self.ins_del + self.ins_ins;
        self.ins_mat /= ins_sum;
        self.ins_del /= ins_sum;
        self.ins_ins /= ins_sum;
        let del_sum = self.del_mat + self.del_del + self.del_ins;
        self.del_mat /= del_sum;
        self.del_del /= del_sum;
        self.del_ins /= del_sum;
        for obss in self.mat_emit.chunks_exact_mut(4) {
            let sum: f64 = obss.iter().sum();
            obss.iter_mut().for_each(|x| *x /= sum);
        }
        for inss in self.ins_emit.chunks_exact_mut(4) {
            let sum: f64 = inss.iter().sum();
            inss.iter_mut().for_each(|x| *x /= sum);
        }
    }
}

fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        -10000000000000000000000000000000f64
    } else {
        let max = *xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
    }
}

pub struct Memory {
    pre: DPTable<(f64, f64, f64)>,
    post: DPTable<(f64, f64, f64)>,
    // Scaling factor.
    pre_scl: Vec<f64>,
    post_scl: Vec<f64>,
    // Radius parameters.
    default_radius: usize,
    radius: Vec<usize>,
    fill_ranges: Vec<(usize, usize)>,
    mod_table: Vec<f64>,
}

impl Memory {
    const MIN_RADIUS: usize = 4;
    pub fn mod_table(&self) -> &[f64] {
        &self.mod_table
    }
    pub fn with_capacity(rlen: usize, radius: usize) -> Self {
        let fill_ranges = Vec::with_capacity(radius * rlen * 3);
        let mod_table = Vec::with_capacity(radius * rlen * 3);
        Self {
            fill_ranges,
            mod_table,
            pre: DPTable::with_capacity(rlen, radius, (0f64, 0f64, 0f64)),
            post: DPTable::with_capacity(rlen, radius, (0f64, 0f64, 0f64)),
            pre_scl: Vec::with_capacity(5 * rlen / 2),
            post_scl: Vec::with_capacity(5 * rlen / 2),
            default_radius: radius,
            radius: Vec::with_capacity(5 * rlen / 2),
        }
    }
    fn initialize(&mut self) {
        self.pre.initialize((0.0, 0.0, 0.0), &self.fill_ranges);
        self.post.initialize((0.0, 0.0, 0.0), &self.fill_ranges);
        self.pre_scl.clear();
        self.post_scl.clear();
    }
    fn set_fill_ranges(&mut self, rlen: usize, qlen: usize, ops: &[Op]) {
        self.fill_ranges.clear();
        self.fill_ranges
            .extend(std::iter::repeat((rlen + 1, 0)).take(qlen + 1));
        let fill_len = (rlen + 1).saturating_sub(self.radius.len());
        self.radius
            .extend(std::iter::repeat(self.default_radius).take(fill_len));
        re_fill_fill_range(qlen, rlen, ops, self.default_radius, &mut self.fill_ranges);
    }
    fn update_radius(&mut self, updated_position: &[(usize, usize)], len: usize) {
        let orig_len = self.radius.len();
        let mut prev_pos = 0;
        const SCALE: usize = 2;
        for &(position, op) in updated_position.iter() {
            for pos in prev_pos..position {
                self.radius
                    .push((self.radius[pos] / SCALE).max(Self::MIN_RADIUS));
            }
            prev_pos = if op < 4 {
                // Mism
                self.radius.push(self.default_radius);
                position + 1
            } else if op < 8 {
                // Insertion
                self.radius.push(self.default_radius);
                self.radius.push(self.default_radius);
                position + 1
            } else if op < 8 + COPY_SIZE {
                // Copying.
                let len = op - 8 + 2;
                self.radius
                    .extend(std::iter::repeat(self.default_radius).take(len));
                position + 1
            } else {
                // Deletion
                position + op - 8 - COPY_SIZE + 1
            };
        }
        // Finally, prev_pos -> prev_len, exact matches.
        for pos in prev_pos..orig_len {
            self.radius
                .push((self.radius[pos] / SCALE).max(Self::MIN_RADIUS));
        }
        let mut idx = 0;
        self.radius.retain(|_| {
            idx += 1;
            orig_len < idx
        });
        assert_eq!(self.radius.len(), len + 1);
    }
}

// Minimum required improvement on the likelihood.
// In a desirable case, it is exactly zero, but as a matter of fact,
// the likelihood is sometimes wobble between very small values,
// so this "min-requirement" is nessesarry.
const MIN_UP: f64 = 0.00001;

// TODO:Maybe we should choose a position to be changed by window.
fn polish_guided(
    template: &mut Vec<u8>,
    modif_table: &[f64],
    current_lk: f64,
    inactive: usize,
) -> Vec<(usize, usize)> {
    let orig_len = template.len();
    let mut changed_positions = vec![];
    let mut modif_table = modif_table.chunks_exact(NUM_ROW);
    let mut pos = 0;
    while let Some(row) = modif_table.next() {
        if row.iter().any(|x| x.is_nan()) {
            panic!("{:?}", row);
        }
        let (op, &lk) = row
            .iter()
            .enumerate()
            .max_by(|x, y| (x.1).partial_cmp(y.1).unwrap())
            .unwrap();
        if current_lk + MIN_UP < lk && pos < orig_len {
            changed_positions.push((pos, op));
            if op < 4 {
                // Mutateion.
                template.push(b"ACGT"[op]);
                pos += 1;
            } else if op < 8 {
                // Insertion before this base.
                template.push(b"ACGT"[op - 4]);
                template.push(template[pos]);
                pos += 1;
            } else if op < 8 + COPY_SIZE {
                // copy from here to here + (op - 8) + 1 base
                let len = (op - 8) + 1;
                for i in pos..pos + len {
                    template.push(template[i]);
                }
                template.push(template[pos]);
                pos += 1;
            } else if op < NUM_ROW {
                // Delete from here to here + 1 + (op - 8 - COPY_SIZE).
                let del_size = op - 8 - COPY_SIZE + 1;
                pos += del_size;
                (0..del_size - 1).filter_map(|_| modif_table.next()).count();
            } else {
                unreachable!()
            }
            for i in pos..(pos + inactive).min(orig_len) {
                template.push(template[i]);
            }
            pos = (pos + inactive).min(orig_len);
            (0..inactive).filter_map(|_| modif_table.next()).count();
        } else if pos < orig_len {
            template.push(template[pos]);
            pos += 1;
        } else if current_lk + MIN_UP < lk && (4..8).contains(&op) {
            changed_positions.push((pos, op));
            // Here, we need to consider the last insertion...
            template.push(b"ACGT"[op - 4]);
        }
    }
    assert_eq!(pos, orig_len);
    let mut idx = 0;
    template.retain(|_| {
        idx += 1;
        orig_len < idx
    });
    changed_positions
}

/// Guided version of the pair HMM on the forward and reverse strands.
#[derive(Debug, Clone)]
pub struct PairHiddenMarkovModelOnStrands {
    forward: PairHiddenMarkovModel,
    reverse: PairHiddenMarkovModel,
}

impl PairHiddenMarkovModelOnStrands {
    pub fn new(forward: PairHiddenMarkovModel, reverse: PairHiddenMarkovModel) -> Self {
        Self { forward, reverse }
    }
    /// With bootstrap operations and taking numbers.
    /// The returned operations would be consistent with the returned sequence.
    /// Only the *first* `take_num` seuqneces would be used to polish,
    /// but the operations of the rest are also updated accordingly.
    /// This is (slightly) faster than the usual/full polishing...
    /// strands: `true` if forward.
    pub fn polish_until_converge_with_conf<T, O>(
        &self,
        draft: &[u8],
        xs: &[T],
        ops: &mut [O],
        strands: &[bool],
        config: &HMMConfig,
    ) -> Vec<u8>
    where
        T: std::borrow::Borrow<[u8]>,
        O: std::borrow::BorrowMut<Vec<Op>>,
    {
        let &HMMConfig {
            radius,
            take_num,
            ignore_edge,
        } = config;
        let take_num = take_num.min(xs.len());
        assert!(!xs.is_empty());
        let mut template = draft.to_vec();
        let len = (template.len() / 2).max(3);
        let mut modif_table = Vec::new();
        let mut memory = Memory::with_capacity(template.len(), radius);
        let mut current_max = None;
        'outer: for t in 0..100 {
            let inactive = INACTIVE_TIME + (t * INACTIVE_TIME) % len;
            modif_table.clear();
            let mut current_lk = 0f64;
            let seq_stream = ops.iter_mut().zip(xs.iter()).zip(strands.iter());
            for (i, ((ops, seq), strand)) in seq_stream.enumerate() {
                let ops = ops.borrow_mut();
                let model = match strand {
                    true => &self.forward,
                    false => &self.reverse,
                };
                model.update_aln_path(&mut memory, &template, seq.borrow(), ops);
                if take_num <= i {
                    continue;
                }
                if let Some(lk_of_read) = model.update(&mut memory, &template, seq.borrow(), ops) {
                    current_lk += lk_of_read;
                    if modif_table.is_empty() {
                        modif_table.extend_from_slice(&memory.mod_table)
                    } else {
                        modif_table
                            .iter_mut()
                            .zip(memory.mod_table.iter())
                            .for_each(|(x, y)| *x += y);
                    }
                }
            }
            assert_eq!(modif_table.len(), NUM_ROW * (template.len() + 1));
            // Ignore edge regions
            for (i, lk) in modif_table.iter_mut().enumerate() {
                let pos = i / NUM_ROW;
                if pos < ignore_edge || template.len() + 1 - ignore_edge < pos {
                    *lk = -1000000000000000000000000000000f64;
                }
            }
            let changed_pos = polish_guided(&mut template, &modif_table, current_lk, inactive);
            let edit_path = changed_pos.iter().map(|&(pos, op)| {
                if op < 4 {
                    (pos, crate::op::Edit::Subst)
                } else if op < 8 {
                    (pos, crate::op::Edit::Insertion)
                } else if op < 8 + COPY_SIZE {
                    (pos, crate::op::Edit::Copy(op - 8 + 1))
                } else {
                    (pos, crate::op::Edit::Deletion(op - 8 - COPY_SIZE + 1))
                }
            });
            for (ops, seq) in ops.iter_mut().zip(xs.iter()) {
                let ops = ops.borrow_mut();
                let seq = seq.borrow();
                let (qlen, rlen) = (seq.len(), template.len());
                crate::op::fix_alignment_path(ops, edit_path.clone(), qlen, rlen);
            }
            memory.update_radius(&changed_pos, template.len());
            if changed_pos.is_empty() {
                if matches!(current_max,Some(lk) if current_lk < lk + 0.1) {
                    break 'outer;
                }
                // trace!("NOWLK\t{}", current_lk);
                current_max = Some(current_lk);
                let is_updated = ops
                    .iter_mut()
                    .zip(xs.iter())
                    .zip(strands.iter())
                    .take(take_num)
                    .enumerate()
                    .map(|(_, ((ops, seq), strand))| {
                        let model = match strand {
                            true => &self.forward,
                            false => &self.reverse,
                        };
                        let (ops, seq) = (ops.borrow_mut(), seq.borrow());
                        let lk = model.lk(&mut memory, &template, seq, ops);
                        let edop = crate::edlib_global(&template, seq);
                        let lk2 = model.lk(&mut memory, &template, seq, &edop);
                        if lk + 0.1 < lk2 {
                            // trace!("{t}\t{i}\t{:.3}\t{:.3}", lk, lk2);
                            *ops = edop;
                            true
                        } else {
                            false
                        }
                    })
                    .fold(false, |is_updated, b| is_updated | b);
                if !is_updated {
                    break 'outer;
                }
            }
        }
        template
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::gen_seq;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;

    #[test]
    fn align() {
        let model = PairHiddenMarkovModel::default();
        let (lk, ops) = model.align(b"ACCG", b"ACCG", 5);
        eprintln!("{:?}\t{:.3}", ops, lk);
        assert_eq!(ops, vec![Op::Match; 4]);
        let (lk, ops) = model.align(b"ACCG", b"", 2);
        eprintln!("{:?}\t{:.3}", ops, lk);
        assert_eq!(ops, vec![Op::Del; 4]);
        let (lk, ops) = model.align(b"", b"ACCG", 2);
        assert_eq!(ops, vec![Op::Ins; 4]);
        eprintln!("{:?}\t{:.3}", ops, lk);
        let (lk, ops) = model.align(b"ATGCCGCACAGTCGAT", b"ATCCGC", 5);
        eprintln!("{:?}\t{:.3}", ops, lk);
        use Op::*;
        let answer = vec![vec![Match; 2], vec![Del], vec![Match; 4], vec![Del; 9]].concat();
        assert_eq!(ops, answer);
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 300);
        let profile = gen_seq::PROFILE;
        let hmm = PairHiddenMarkovModel::default();
        let radius = 50;
        let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
        hmm.align(&template, &seq, radius);
    }
    #[test]
    fn likelihood() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 300);
        let profile = gen_seq::PROFILE;
        let hmm = PairHiddenMarkovModel::default();
        let radius = 50;
        let same_lk = hmm.likelihood(&template, &template, radius);
        let (lk, _) = hmm.align(&template, &template, radius);
        let expect = (template.len() as f64 - 1f64) * hmm.mat_mat.ln()
            + (template.len() as f64) * hmm.mat_emit[0].ln();
        assert!(lk < same_lk);
        assert!(expect < same_lk, "{},{}", expect, same_lk);
        for i in 0..10 {
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let lk = hmm.likelihood(&template, &seq, radius);
            assert!(lk < same_lk, "{},{},{}", i, same_lk, lk);
            let seq2 = gen_seq::introduce_randomness(&seq, &mut rng, &profile);
            let lk2 = hmm.likelihood(&template, &seq2, radius);
            assert!(lk2 < lk, "{},{},{}", i, lk2, lk);
        }
    }
    #[test]
    fn likelihood2() {
        let seq1 = b"AC";
        let seq2 = b"CT";
        use Op::*;
        // There are no del->ins transition. why?
        let ops = vec![
            vec![Match; 2],
            vec![Ins, Del, Match],
            vec![Del, Ins, Match],
            vec![Ins, Match, Del],
            vec![Match, Ins, Del],
            vec![Del, Match, Ins],
            vec![Match, Del, Ins],
            vec![Del, Del, Ins, Ins],
            vec![Del, Ins, Del, Ins],
            vec![Del, Ins, Ins, Del],
            vec![Ins, Del, Del, Ins],
            vec![Ins, Del, Ins, Del],
            vec![Ins, Ins, Del, Del],
        ];
        let hmm = PairHiddenMarkovModel::default();
        let lks: Vec<_> = ops.iter().map(|ops| hmm.eval(seq1, seq2, ops)).collect();
        let lk_e = logsumexp(&lks);
        eprintln!("{:?},{}", lks, lk_e);
        let radius = 5;
        let lk = hmm.likelihood(seq1, seq2, radius);
        assert!((lk_e - lk).abs() < 0.001, "{},{}", lk_e, lk);
    }
    #[test]
    fn modification_table() {
        for seed in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198 + seed);
            let template = gen_seq::generate_seq(&mut rng, 70);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let radius = 50;
            let query = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let (modif_table, ops) = {
                let mut memory = Memory::with_capacity(template.len(), radius);
                let ops = bootstrap_ops(template.len(), query.len());
                let lk = hmm.update(&mut memory, &template, &query, &ops).unwrap();
                println!("LK\t{}", lk);
                (memory.mod_table, ops)
            };
            let mut memory = Memory::with_capacity(template.len(), radius);
            let mut mod_version = template.clone();
            println!("{}", seed);
            // Mutation error
            for (j, modif_table) in modif_table
                .chunks_exact(NUM_ROW)
                .take(template.len())
                .enumerate()
            {
                println!("{}", j);
                let orig = mod_version[j];
                for (&base, lk_m) in b"ACGT".iter().zip(modif_table) {
                    mod_version[j] = base;
                    let lk = hmm.update(&mut memory, &mod_version, &query, &ops).unwrap();
                    assert!((lk - lk_m).abs() < 0.0001, "{},{},mod", lk, lk_m);
                    println!("M\t{}\t{}", j, (lk - lk_m).abs());
                    mod_version[j] = orig;
                }
                // Insertion error
                for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                    mod_version.insert(j, base);
                    let lk = hmm.update(&mut memory, &mod_version, &query, &ops).unwrap();
                    assert!((lk - lk_m).abs() < 0.0001, "{},{}", lk, lk_m);
                    println!("I\t{}\t{}", j, (lk - lk_m).abs());
                    mod_version.remove(j);
                }
                // Copying mod
                for len in (0..COPY_SIZE).filter(|c| j + c < template.len()) {
                    let lk_m = modif_table[8 + len];
                    let mod_version: Vec<_> = template[..j + len + 1]
                        .iter()
                        .chain(template[j..].iter())
                        .copied()
                        .collect();
                    let lk = hmm.update(&mut memory, &mod_version, &query, &ops).unwrap();
                    println!("C\t{}\t{}\t{}", j, len, (lk - lk_m).abs());
                    assert!((lk - lk_m).abs() < 0.0001, "{},{}", lk, lk_m);
                }
                // Deletion error
                for len in (0..DEL_SIZE).filter(|d| j + d < template.len()) {
                    let lk_m = modif_table[8 + COPY_SIZE + len];
                    let mod_version: Vec<_> = template[..j]
                        .iter()
                        .chain(template[j + len + 1..].iter())
                        .copied()
                        .collect();
                    let lk = hmm.update(&mut memory, &mod_version, &query, &ops).unwrap();
                    println!("D\t{}\t{}\t{}", j, len, lk - lk_m);
                    assert!(
                        (lk - lk_m).abs() < 0.01,
                        "{},{},{}",
                        lk,
                        lk_m,
                        String::from_utf8_lossy(&template)
                    );
                }
            }
            let modif_table = modif_table
                .chunks_exact(NUM_ROW)
                .nth(template.len())
                .unwrap();
            for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                mod_version.push(base);
                let lk = hmm.update(&mut memory, &mod_version, &query, &ops).unwrap();
                assert!((lk - lk_m).abs() < 1.0001);
                mod_version.pop();
            }
        }
    }
    #[test]
    fn modification_table2() {
        use crate::gen_seq;
        let coverage = 10;
        let radius = 20;
        let prof = gen_seq::ProfileWithContext::default();
        let seed = 36;
        let len = 500;
        let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
        let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
        let seqs: Vec<_> = (0..coverage)
            .map(|_| gen_seq::introduce_randomness_with_context(&template, &mut rng, &prof))
            .collect();
        let hmm = super::PairHiddenMarkovModel::default();
        let template = crate::ternary_consensus_by_chunk(&seqs, radius);
        let query = &seqs[0];
        let (modif_table, ops) = {
            let mut memory = Memory::with_capacity(template.len(), radius);
            let ops = bootstrap_ops(template.len(), query.len());
            let lk = hmm.update(&mut memory, &template, query, &ops).unwrap();
            println!("LK\t{}", lk);
            (memory.mod_table, ops)
        };
        let mut memory = Memory::with_capacity(template.len(), radius);
        let mut mod_version = template.clone();
        println!("{}", seed);
        // Mutation error
        for (j, modif_table) in modif_table
            .chunks_exact(NUM_ROW)
            .take(template.len())
            .enumerate()
        {
            println!("{}", j);
            let orig = mod_version[j];
            for (&base, lk_m) in b"ACGT".iter().zip(modif_table) {
                mod_version[j] = base;
                let lk = hmm.update(&mut memory, &mod_version, query, &ops).unwrap();
                assert!((lk - lk_m).abs() < 0.0001, "{},{},mod", lk, lk_m);
                println!("M\t{}\t{}", j, (lk - lk_m).abs());
                mod_version[j] = orig;
            }
            // Insertion error
            for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                mod_version.insert(j, base);
                let lk = hmm.update(&mut memory, &mod_version, query, &ops).unwrap();
                assert!((lk - lk_m).abs() < 0.0001, "{},{}", lk, lk_m);
                println!("I\t{}\t{}", j, (lk - lk_m).abs());
                mod_version.remove(j);
            }
            // Copying mod
            for len in (0..COPY_SIZE).filter(|c| j + c < template.len()) {
                let lk_m = modif_table[8 + len];
                let mod_version: Vec<_> = template[..j + len + 1]
                    .iter()
                    .chain(template[j..].iter())
                    .copied()
                    .collect();
                let lk = hmm.update(&mut memory, &mod_version, query, &ops).unwrap();
                println!("C\t{}\t{}\t{}", j, len, (lk - lk_m).abs());
                assert!((lk - lk_m).abs() < 0.0001, "{},{}", lk, lk_m);
            }
            // Deletion error
            for len in (0..DEL_SIZE).filter(|d| j + d < template.len()) {
                let lk_m = modif_table[8 + COPY_SIZE + len];
                let mod_version: Vec<_> = template[..j]
                    .iter()
                    .chain(template[j + len + 1..].iter())
                    .copied()
                    .collect();
                let lk = hmm.update(&mut memory, &mod_version, query, &ops).unwrap();
                println!("D\t{}\t{}\t{}", j, len, lk - lk_m);
                assert!(
                    (lk - lk_m).abs() < 0.01,
                    "{},{},{}",
                    lk,
                    lk_m,
                    String::from_utf8_lossy(&template)
                );
            }
        }
        let modif_table = modif_table
            .chunks_exact(NUM_ROW)
            .nth(template.len())
            .unwrap();
        for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
            mod_version.push(base);
            let lk = hmm.update(&mut memory, &mod_version, query, &ops).unwrap();
            assert!((lk - lk_m).abs() < 1.0001);
            mod_version.pop();
        }
    }
    #[test]
    fn polish_test() {
        let hmm = PairHiddenMarkovModel::default();
        let radius = 50;
        for seed in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198 + seed);
            let template = gen_seq::generate_seq(&mut rng, 70);
            let diff = gen_seq::introduce_errors(&template, &mut rng, 2, 2, 2);
            let polished = hmm.polish_until_converge(&diff, &[template.as_slice()], radius);
            if polished != template {
                println!("{}", String::from_utf8_lossy(&polished));
                println!("{}", String::from_utf8_lossy(&template));
                let prev = crate::bialignment::edit_dist(&template, &diff);
                let now = crate::bialignment::edit_dist(&template, &polished);
                println!("{},{}", prev, now);
                panic!()
            }
        }
        for seed in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198 + seed);
            let template = gen_seq::generate_seq(&mut rng, 700);
            let diff = gen_seq::introduce_errors(&template, &mut rng, 2, 2, 2);
            let profile = gen_seq::PROFILE;
            let seqs: Vec<_> = (0..20)
                .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &profile))
                .collect();
            let polished = hmm.polish_until_converge(&diff, &seqs, radius);
            if polished != template {
                println!("{}", String::from_utf8_lossy(&polished));
                println!("{}", String::from_utf8_lossy(&template));
                let prev = crate::bialignment::edit_dist(&template, &diff);
                let now = crate::bialignment::edit_dist(&template, &polished);
                println!("{},{}", prev, now);
                panic!()
            }
        }
    }
    #[test]
    fn update_aln_path_test() {
        let radius = 10;
        let hmm = PairHiddenMarkovModel::default();
        for seed in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198 + seed);
            let template = gen_seq::generate_seq(&mut rng, 70);
            let mut memory = Memory::with_capacity(template.len(), radius);
            let diff = gen_seq::introduce_randomness(&template, &mut rng, &gen_seq::PROFILE);
            let mut ops = bootstrap_ops(template.len(), diff.len());
            hmm.update_aln_path(&mut memory, &template, &diff, &mut ops);
        }
    }
    #[test]
    fn polish_long_insertion() {
        let hmm = PairHiddenMarkovModel::default();
        let radius = 50;
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let head = gen_seq::generate_seq(&mut rng, 100);
        let repeat = gen_seq::generate_seq(&mut rng, 200);
        let tail = gen_seq::generate_seq(&mut rng, 100);
        let template = vec![head.clone(), repeat.clone(), tail.clone()].concat();
        let answer = vec![head, repeat.clone(), repeat, tail].concat();
        let profile = gen_seq::PROFILE;
        let seqs: Vec<_> = (0..20)
            .map(|_| gen_seq::introduce_randomness(&answer, &mut rng, &profile))
            .collect();
        let polished = hmm.polish_until_converge(&template, &seqs, radius);
        if polished != answer {
            println!("{}", String::from_utf8_lossy(&polished));
            println!("{}", String::from_utf8_lossy(&answer));
            let prev = crate::bialignment::edit_dist(&template, &answer);
            let now = crate::bialignment::edit_dist(&polished, &answer);
            println!("{}->{}", prev, now);
            panic!()
        }
    }
}
