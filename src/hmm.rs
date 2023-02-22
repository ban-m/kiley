//! An tiny implementation of pair hidden Markov models.
use crate::op::Op;
pub mod full;
pub mod guided;
pub mod guided_antidiagonal;
use crate::EP;
use serde::{Deserialize, Serialize};

pub const COPY_SIZE: usize = 3;
pub const DEL_SIZE: usize = 3;
pub const COPY_DEL_MAX: usize = 3;
pub const NUM_ROW: usize = 8 + COPY_SIZE + DEL_SIZE;

fn usize_to_edit_op(op: usize) -> crate::op::Edit {
    use crate::op::Edit;
    assert!(op < NUM_ROW);
    if op < 4 {
        Edit::Subst
    } else if op < 8 {
        Edit::Insertion
    } else if op < 8 + COPY_SIZE {
        Edit::Copy(op - 8 + 1)
    } else {
        Edit::Deletion(op - 8 - COPY_SIZE + 1)
    }
}

#[derive(Debug, Clone)]
/// Configurations
pub struct HMMPolishConfig {
    pub radius: usize,
    pub take_num: usize,
    pub ignore_edge: usize,
}

impl HMMPolishConfig {
    pub fn new(radius: usize, take_num: usize, ignore_edge: usize) -> Self {
        Self {
            radius,
            take_num,
            ignore_edge,
        }
    }
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
    /// Create a new pair-hidden Markov model.
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

    fn obs(&self, r: u8, q: u8) -> f64 {
        let index = (BASE_TABLE[r as usize] << 2) | BASE_TABLE[q as usize];
        self.mat_emit[index as usize]
    }
    const fn del(&self, _r: u8) -> f64 {
        1f64
    }
    fn ins(&self, q: u8, prev: Option<u8>) -> f64 {
        let prev = prev.unwrap_or(4);
        let index = (BASE_TABLE[prev as usize] << 2) | BASE_TABLE[q as usize];
        self.ins_emit[index as usize]
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
    /// Return the log-likelihood of the alignment (`ops`).
    /// It is calculated in a normal, i.e., non-log, space.
    /// Thus, if the `ops` is long, the returned value might be NaN.
    /// In that case, use `Self::eval_ln()` instead.
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
    /// Return the log-likelihood of the alignment (`ops`).
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

/// Shorthand for PairHiddenMarkovModel.
#[allow(clippy::upper_case_acronyms)]
pub type PHMM = PairHiddenMarkovModel;

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

impl PHMM {
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

/// A dynamic programming table. It is a serialized 2-d array. i -> query, j -> Reference!
#[derive(Debug, Clone)]
pub(crate) struct DPTable {
    mat_dp: Vec<f64>,
    ins_dp: Vec<f64>,
    del_dp: Vec<f64>,
    column: usize,
    #[allow(dead_code)]
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
}

// Minimum required improvement on the likelihood.
// In a desirable case, it is exactly zero, but as a matter of fact,
// the likelihood is sometimes wobble between very small values,
// so this "min-requirement" is nessesarry.
const MIN_UP: f64 = 0.00001;

pub(crate) fn polish_by_modification_table(
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
#[derive(Debug, Clone, Default)]
pub struct PairHiddenMarkovModelOnStrands {
    forward: PairHiddenMarkovModel,
    reverse: PairHiddenMarkovModel,
}

impl PairHiddenMarkovModelOnStrands {
    /// Return the model on the forward strand.
    pub fn forward(&self) -> &PairHiddenMarkovModel {
        &self.forward
    }
    /// Return the model on the reverse strand.
    pub fn reverse(&self) -> &PairHiddenMarkovModel {
        &self.reverse
    }
    /// Create a new instance.
    pub fn new(forward: PairHiddenMarkovModel, reverse: PairHiddenMarkovModel) -> Self {
        Self { forward, reverse }
    }
}

impl std::fmt::Display for PairHiddenMarkovModelOnStrands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FORWARD\n{}", self.forward)?;
        write!(f, "REVERSE\n{}", self.reverse)
    }
}

/// Dataset used to train the parameters of a [`PairHiddenMarkovModelOnStrands`]
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
    /// Create a new datasets. They consist of the template (`consensus`), `directions` of the aligned `sequences` with `operations`.
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
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    #[test]
    fn align() {
        let phmm = PHMM::default();
        let (lk, ops) = phmm.align(b"ACCG", b"ACCG");
        eprintln!("{:?}\t{:.3}", ops, lk);
        assert_eq!(ops, vec![Op::Match; 4]);
        let (lk, ops) = phmm.align(b"ACCG", b"");
        eprintln!("{:?}\t{:.3}", ops, lk);
        assert_eq!(ops, vec![Op::Del; 4]);
        let (lk, ops) = phmm.align(b"", b"ACCG");
        assert_eq!(ops, vec![Op::Ins; 4]);
        eprintln!("{:?}\t{:.3}", ops, lk);
        let (lk, ops) = phmm.align(b"ATGCCGCACAGTCGAT", b"ATCCGC");
        eprintln!("{:?}\t{:.3}", ops, lk);
        use Op::*;
        let answer = vec![vec![Match; 2], vec![Del], vec![Match; 4], vec![Del; 9]].concat();
        assert_eq!(ops, answer);
    }
    #[test]
    fn hmm_full_guided() {
        let len = 100;
        let hmm = PairHiddenMarkovModel::default();
        let profile = crate::gen_seq::Profile::new(0.01, 0.01, 0.01);
        for seed in 0..100 {
            let mut rng: Xoshiro256PlusPlus = SeedableRng::seed_from_u64(seed);
            let template = crate::gen_seq::generate_seq(&mut rng, len);
            let query = crate::gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let (lk_full, path_full) = hmm.align(&template, &query);
            let (lk_guided, path_guided) = hmm.align(&template, &query);
            assert_eq!(path_full, path_guided);
            assert!((lk_full - lk_guided).abs() < 0.001);
            let likelihood_full = hmm.likelihood(&template, &query);
            let likelihood_guided = hmm.likelihood_guided(&template, &query, &path_full, 20);
            assert!((likelihood_full - likelihood_guided).abs() < 0.001);
        }
    }
}
