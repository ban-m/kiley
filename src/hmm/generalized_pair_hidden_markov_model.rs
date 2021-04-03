//! This module defines generalized pair hidden Markov models.
//! In contrast to the usual pair hidden Markov model, where each state
//! should output either a match, a deletion, or an insertion,
//! a GpHMM can output any of them.
//! However, the usual algorithm such as Viterbi, Forward, Backward, and Forward-Backward algorithm could be run on a GpHMM in the
//! same time with respect to the O-notation. Specifically, all of them run in O(|X||Y||S|) time, where X and Y are the reference and the query, and S is the states.
//! Note: I do not know the Baum-Whelch algorithm for trailing parameters for a pair HMM. Maybe there is. If you know how to train parameters in pair HMM, please send me a link to some papers/codes for that purpose.

/// A generalized pair hidden Markov model.
/// Usually, it is more stressful to use short-hand notation `GPHMM`.
/// This model generates alignment, not a pair of sequence. This distinction is very vague, but
/// one should be careful when using pair-HMM.
/// For example,
/// ```
/// AAA-AGT-T
/// AA-CCGTGT
/// ```
/// is an alignment between a pair of sequence `AAAAGTT` and `AACCGTGT`. Of course, there are other alignment for these two sequence.
/// So, there is possibly many alignment for a pair of sequence.
/// Currently, this function is for DNA alignment with no anbiguous bases such as N. In other words,
/// The input sequences should be strings on the alphabet b"ATGCacgt", and there is no distinction between lower-case letter and upper-case letter.
/// Note that ,for each exact algorithm, there is `banded` version approximate the result.
/// If the computational time matters, please try these method, even though it can reach sub-optimal solutions.
#[derive(Debug, Clone)]
pub struct GeneralizedPairHiddenMarkovModel {
    // Number of states.
    states: usize,
    // Transition between each states.
    // This is transposed matrix of transition matrix. So,
    // by accessing to * self.states + from, we can get the transition probability from `from` to `to`.
    transition_matrix: Vec<f64>,
    // obseration on a state for a alignment operation.
    // By accessing [states << 6 | x << 3 | y], we can get the obvervasion probability Pr{(x,y)|x}
    observation_matrix: Vec<f64>,
    // Initial distribution. Should be normalized to 1.
    initial_distribution: Vec<f64>,
}

pub type GPHMM = GeneralizedPairHiddenMarkovModel;

// We define additional structure for easy implementation.
// This is offset around a DP table for each state.
const OFFSET: usize = 3;
// This is the DP table. We have one sheet for each state.
#[derive(Debug, Clone)]
struct DPTable {
    column: usize,
    row: usize,
    states: usize,
    data: Vec<f64>,
}

impl DPTable {
    // Create new (row x column x state) DP table.
    // If you access invalid index such as -1 or [column][row][states], it would return default value..
    fn new(row: usize, column: usize, states: usize, default: f64) -> Self {
        let len = (row + 2 * OFFSET) * (column + 2 * OFFSET) * states;
        Self {
            column,
            row,
            states,
            data: vec![default; len],
        }
    }
    // Maybe I can move these implementations to std::slice::SliceIndex.
    fn get(&self, i: isize, j: isize, s: isize) -> Option<&f64> {
        let index = self.get_index(i, j, s);
        if 0 <= index {
            self.data.get(index as usize)
        } else {
            None
        }
    }
    fn get_index(&self, i: isize, j: isize, s: isize) -> isize {
        let column_size = (self.states * (self.column + 2 * OFFSET)) as isize;
        let column_pos = (j + OFFSET as isize) * self.states as isize;
        column_size * (i + OFFSET as isize) + column_pos + s
    }
    fn get_mut(&mut self, i: isize, j: isize, s: isize) -> Option<&mut f64> {
        let index = self.get_index(i, j, s);
        if 0 <= index {
            self.data.get_mut(index as usize)
        } else {
            None
        }
    }
    // Sum dp[i][j][s] over j and s.
    fn total(&self, i: usize) -> f64 {
        let start = self.get_index(i as isize, 0, 0) as usize;
        let length = self.column * self.states;
        let range = start..start + length;
        self.data[range].iter().sum::<f64>()
    }
    // Divide dp[i] by `by`
    fn div(&mut self, i: usize, by: f64) {
        let start = self.get_index(i as isize, 0, 0) as usize;
        let length = self.column * self.states;
        let range = start..start + length;
        self.data[range].iter_mut().for_each(|x| *x /= by);
    }
}

impl std::ops::Index<(isize, isize, isize)> for DPTable {
    type Output = f64;
    fn index(&self, (i, j, s): (isize, isize, isize)) -> &Self::Output {
        self.get(i, j, s).unwrap()
    }
}

impl std::ops::IndexMut<(isize, isize, isize)> for DPTable {
    fn index_mut(&mut self, (i, j, s): (isize, isize, isize)) -> &mut Self::Output {
        self.get_mut(i, j, s).unwrap()
    }
}

use super::Op;
use crate::padseq;
use crate::padseq::PadSeq;
use padseq::GAP;
const EP: f64 = -1000000000000000000000f64;
fn log(x: &f64) -> f64 {
    assert!(!x.is_sign_negative());
    if f64::EPSILON < x.abs() {
        x.ln()
    } else {
        EP
    }
}

impl std::default::Default for GPHMM {
    fn default() -> Self {
        // Return very simple single state hidden Markov model.
        let mat = 0.2;
        let mism = 0.01;
        let match_prob = [
            mat, mism, mism, mism, mism, mat, mism, mism, mism, mism, mat, mism, mism, mism, mism,
            mat,
        ];
        let gap = 0.01;
        let del_prob = [gap; 4];
        let ins_prob = [gap; 4];
        GPHMM::new(
            1,
            &[vec![1f64]],
            &[match_prob],
            &[del_prob],
            &[ins_prob],
            &[1f64],
        )
    }
}

impl GPHMM {
    /// Create a new generalized pair hidden Markov model.
    /// There is a few restriction on the input arguments.
    /// 1. transition matrix should be states x states matrix,
    /// all rowsum to be 1. In other words, transition_matrix[i][j] = Pr(i -> j).
    /// 2. Match_prob, del_prob, ins_prob, and initial_distribution should be the length of `states`.
    /// 3. The sum of initial_distribution should be 1.
    /// 4. The sum of match_prob[x][y] + del_prob[x] + ins_prob[x] should be 1.
    /// Each 16-length array of match probabilities is:
    /// [(A,A), (A,C), (A,G), (A,T), (C,A), ... ,(T,T)]
    /// and Each array of del_prob is [(A,-), ...,(T,-)],
    /// and each array of ins_prob is [(-,A), ... ,(-,A)]
    pub fn new(
        states: usize,
        transition_matrix: &[Vec<f64>],
        match_prob: &[[f64; 16]],
        del_prob: &[[f64; 4]],
        ins_prob: &[[f64; 4]],
        initial_distribution: &[f64],
    ) -> Self {
        let mut observation_matrix = vec![0f64; (states << 6) | (8 * 8)];
        for (state, ((mat, del), ins)) in match_prob
            .iter()
            .zip(del_prob.iter())
            .zip(ins_prob.iter())
            .enumerate()
        {
            for x in 0..4 {
                observation_matrix[(state << 6) | (x << 3) | GAP as usize] = del[x];
                observation_matrix[(state << 6) | ((GAP as usize) << 3) | x] = ins[x];
                for y in 0..4 {
                    observation_matrix[(state << 6) | (x << 3) | y] = mat[x << 2 | y];
                }
            }
        }
        let mut transposed_transition = vec![0f64; states * states];
        for from in 0..states {
            for to in 0..states {
                // This is transposed!
                transposed_transition[to * states + from] = transition_matrix[from][to];
            }
        }
        Self {
            states,
            transition_matrix: transposed_transition,
            observation_matrix,
            initial_distribution: initial_distribution.to_vec(),
        }
    }
    /// Return a simple single state pair HMM for computing conditional likelihood, LK(y|x).
    /// (match_prob + del_prob) should be smaller than 1f64.
    pub fn new_conditional_single_state(match_prob: f64, del_prob: f64) -> Self {
        let states = 1;
        let mismatch = (1f64 - match_prob - del_prob) / 3f64;
        let transition_matrix = vec![1f64];
        let match_prob = [
            match_prob, mismatch, mismatch, mismatch, mismatch, match_prob, mismatch, mismatch,
            mismatch, mismatch, match_prob, mismatch, mismatch, mismatch, mismatch, match_prob,
        ];
        let del_prob = [del_prob; 4];
        let ins_prob = [4f64.recip(); 4];
        Self::new(
            states,
            &[transition_matrix],
            &[match_prob],
            &[del_prob],
            &[ins_prob],
            &[1f64],
        )
    }
    /// Return a three states pair HMM for computing conditional likelihood, LK(y|x)
    pub fn new_conditional_three_state(
        match_prob: f64,
        gap_open: f64,
        gap_ext: f64,
        match_emit: f64,
    ) -> Self {
        let states = 3;
        let transition_matrix = [
            vec![match_prob, gap_open, gap_open],
            vec![1f64 - gap_ext - gap_open, gap_ext, gap_open],
            vec![1f64 - gap_ext - gap_open, gap_ext, gap_open],
        ];
        let (mat, mism) = (match_emit, (1f64 - match_emit) / 3f64);
        let match_emit = [
            mat, mism, mism, mism, mism, mat, mism, mism, mism, mism, mat, mism, mism, mism, mism,
            mat,
        ];
        let match_prob = vec![match_emit, [0f64; 16], [0f64; 16]];
        let ins_prob = vec![[0f64; 4], [0f64; 4], [4f64.recip(); 4]];
        let del_prob = vec![[0f64; 4], [1f64; 4], [0f64; 4]];
        let init = [1f64, 0f64, 0f64];
        Self::new(
            states,
            &transition_matrix,
            &match_prob,
            &del_prob,
            &ins_prob,
            &init,
        )
    }
    /// Return usual three state pair Hidden Markov model.
    pub fn new_three_state(match_prob: f64, gap_open: f64, gap_ext: f64, match_emit: f64) -> Self {
        let states = 3;
        let sum = match_prob + 2f64 * gap_open;
        let (match_prob, gap_open) = (match_prob / sum, gap_open / sum);
        assert!(gap_ext + gap_open < 1f64);
        let transition_matrix = [
            vec![match_prob, gap_open, gap_open],
            vec![1f64 - gap_ext - gap_open, gap_ext, gap_open],
            vec![1f64 - gap_ext - gap_open, gap_open, gap_ext],
        ];
        let (mat, mism) = (match_emit / 4f64, (1f64 - match_emit) / 12f64);
        let match_emit = [
            mat, mism, mism, mism, mism, mat, mism, mism, mism, mism, mat, mism, mism, mism, mism,
            mat,
        ];
        let match_prob = vec![match_emit, [0f64; 16], [0f64; 16]];
        let gap_prob = [(4f64).recip(); 4];
        let del_prob = vec![[0f64; 4], [0f64; 4], gap_prob];
        let ins_prob = vec![[0f64; 4], gap_prob, [0f64; 4]];
        let init = [1f64, 0f64, 0f64];
        Self::new(
            states,
            &transition_matrix,
            &match_prob,
            &del_prob,
            &ins_prob,
            &init,
        )
    }
    /// get transition probability from `from` to `to`
    pub fn transition(&self, from: usize, to: usize) -> f64 {
        let (from, to) = (from as usize, to as usize);
        // Caution: This ordering might be un-intuitive.
        self.transition_matrix[to * self.states + from]
    }
    // Convert transition matrix into log-transition matrix.
    // To obtain Pr(t->s), index by [t][s]
    fn get_log_transition(&self) -> Vec<Vec<f64>> {
        (0..self.states)
            .map(|t| {
                (0..self.states)
                    .map(|s| log(&self.transition(t, s)))
                    .collect()
            })
            .collect()
    }
    // Convert obsevation matrix into log-obsevation matrix.
    // To obtain Pr((x,y)|s), index by [s][x << 3 |  y].
    // x and y should be converted by PadSeq.
    fn get_log_observation(&self) -> Vec<Vec<f64>> {
        (0..self.states)
            .map(|s| {
                let mut slots = vec![EP; 64];
                for i in 0..5u8 {
                    for j in 0..5u8 {
                        slots[(i << 3 | j) as usize] = log(&self.observe(s, i, j));
                    }
                }
                slots
            })
            .collect()
    }
    // Return Pr((x,y)|s) or Pr((x,y)|s,x)
    fn observe(&self, s: usize, x: u8, y: u8) -> f64 {
        self.observation_matrix[s << 6 | (x << 3 | y) as usize]
    }
    /// Align `xs` and `ys` and return the maximum likelihood, its alignemnt, and its hidden states.
    /// In other words, it is traditional Viterbi algorithm.
    pub fn align(&self, xs: &[u8], ys: &[u8]) -> (f64, Vec<Op>, Vec<usize>) {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        self.align_inner(&xs, &ys)
    }
    /// Align `xs` and `ys`.
    pub fn align_inner(&self, xs: &PadSeq, ys: &PadSeq) -> (f64, Vec<Op>, Vec<usize>) {
        let mut dp = DPTable::new(xs.len() + 1, ys.len() + 1, self.states, EP);
        let log_transit = self.get_log_transition();
        let log_observe = self.get_log_observation();
        for s in 0..self.states {
            dp[(0, 0, s as isize)] = log(&self.initial_distribution[s]);
        }
        // Initial values.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            let i = i as isize;
            for s in 0..self.states {
                dp[(i, 0, s as isize)] = (0..self.states)
                    .map(|t| {
                        dp[(i - 1, 0, t as isize)]
                            + log_transit[t][s]
                            + log_observe[s][(x << 3 | GAP) as usize]
                    })
                    .max_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();
            }
        }
        for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
            let j = j as isize;
            for s in 0..self.states {
                dp[(0, j, s as isize)] = (0..self.states)
                    .map(|t| {
                        dp[(0, j - 1, t as isize)]
                            + log_transit[t][s]
                            + log_observe[s][(GAP << 3 | y) as usize]
                    })
                    .max_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();
            }
        }
        // Fill DP cells.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
                for s in 0..self.states {
                    let (i, j, s) = (i as isize, j as isize, s as isize);
                    let max_path = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i - 1, j - 1, t as isize)]
                                + log_transit[t as usize][s as usize]
                                + log_observe[s as usize][(x << 3 | y) as usize];
                            let del = dp[(i - 1, j, t as isize)]
                                + log_transit[t as usize][s as usize]
                                + log_observe[s as usize][(x << 3 | GAP) as usize];
                            let ins = dp[(i, j - 1, t as isize)]
                                + log_transit[t as usize][s as usize]
                                + log_observe[s as usize][(GAP << 3 | y) as usize];
                            mat.max(del).max(ins)
                        })
                        .max_by(|x, y| x.partial_cmp(y).unwrap())
                        .unwrap();
                    dp[(i, j, s)] = max_path;
                }
            }
        }
        let (mut i, mut j) = (xs.len() as isize, ys.len() as isize);
        let (max_lk, mut state) = (0..self.states)
            .map(|s| (dp[(i, j, s as isize)], s as isize))
            .max_by(|x, y| (x.0).partial_cmp(&(y.0)).unwrap())
            .unwrap();
        // Trace back.
        let mut ops = vec![];
        let mut states = vec![state];
        while 0 < i && 0 < j {
            let current = dp[(i, j, state)];
            let (x, y) = (xs[i - 1], ys[j - 1]);
            let (op, new_state) = (0..self.states)
                .find_map(|t| {
                    let mat = dp[(i - 1, j - 1, t as isize)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | y) as usize];
                    let del = dp[(i - 1, j, t as isize)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | GAP) as usize];
                    let ins = dp[(i, j - 1, t as isize)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(GAP << 3 | y) as usize];
                    if (current - mat).abs() < 0.00001 {
                        Some((Op::Match, t as isize))
                    } else if (current - del).abs() < 0.0001 {
                        Some((Op::Del, t as isize))
                    } else if (current - ins).abs() < 0.0001 {
                        Some((Op::Ins, t as isize))
                    } else {
                        None
                    }
                })
                .unwrap();
            state = new_state;
            match op {
                Op::Match => {
                    i -= 1;
                    j -= 1;
                }
                Op::Del => i -= 1,
                Op::Ins => j -= 1,
            }
            states.push(state);
            ops.push(op);
        }
        while 0 < i {
            let current = dp[(i, j, state)];
            let x = xs[i - 1];
            let new_state = (0..self.states)
                .find_map(|t| {
                    let del = dp[(i - 1, j, t as isize)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | GAP) as usize];
                    ((current - del) < 0.000001).then(|| t as isize)
                })
                .unwrap();
            state = new_state;
            states.push(state);
            ops.push(Op::Del);
            i -= 1;
        }
        while 0 < j {
            let current = dp[(i, j, state)];
            let y = ys[j - 1];
            let new_state = (0..self.states)
                .find_map(|t| {
                    let ins = dp[(i, j - 1, t as isize)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(GAP << 3 | y) as usize];
                    ((current - ins).abs() < 0.00001).then(|| t as isize)
                })
                .unwrap();
            state = new_state;
            states.push(state);
            ops.push(Op::Ins);
            j -= 1;
        }
        let states: Vec<_> = states.iter().rev().map(|&x| x as usize).collect();
        ops.reverse();
        (max_lk, ops, states)
    }
    /// Banded version of `align` method.
    pub fn align_banded(&self, xs: &[u8], ys: &[u8], radius: usize) -> (f64, Vec<Op>, Vec<usize>) {
        unimplemented!()
    }
    fn logsumexp(xs: &[f64]) -> f64 {
        if xs.is_empty() {
            return 0f64;
        }
        let max = xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln() + max
    }
    /// same as likelihood. A naive log-sum-exp implementation.
    pub fn likelihood_naive(&self, xs: &[u8], ys: &[u8]) -> f64 {
        let dp = self.forward_naive(xs, ys);
        let (n, m) = (xs.len() as isize, ys.len() as isize);
        let lks: Vec<_> = (0..self.states).map(|s| dp[(n, m, s as isize)]).collect();
        Self::logsumexp(&lks)
    }
    fn forward_naive(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let mut dp = DPTable::new(xs.len() + 1, ys.len() + 1, self.states, EP);
        let log_transit = self.get_log_transition();
        let log_observe = self.get_log_observation();
        for s in 0..self.states {
            dp[(0, 0, s as isize)] = log(&self.initial_distribution[s]);
        }
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            let i = i as isize;
            for s in 0..self.states {
                let lks: Vec<_> = (0..self.states)
                    .map(|t| {
                        dp[(i - 1, 0, t as isize)]
                            + log_transit[t][s]
                            + log_observe[s][(x << 3 | GAP) as usize]
                    })
                    .collect();
                dp[(i, 0, s as isize)] = Self::logsumexp(&lks);
            }
        }
        for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
            let j = j as isize;
            for s in 0..self.states {
                let lks: Vec<_> = (0..self.states)
                    .map(|t| {
                        dp[(0, j - 1, t as isize)]
                            + log_transit[t][s]
                            + log_observe[s][(GAP << 3 | y) as usize]
                    })
                    .collect();
                dp[(0, j, s as isize)] = Self::logsumexp(&lks);
            }
        }
        // Fill DP cells.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
                for s in 0..self.states {
                    let (i, j, s) = (i as isize, j as isize, s as isize);
                    let lks: Vec<_> = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i - 1, j - 1, t as isize)]
                                + log_observe[s as usize][(x << 3 | y) as usize];
                            let del = dp[(i - 1, j, t as isize)]
                                + log_observe[s as usize][(x << 3 | GAP) as usize];
                            let ins = dp[(i, j - 1, t as isize)]
                                + log_observe[s as usize][(GAP << 3 | y) as usize];
                            Self::logsumexp(&[mat, del, ins]) + log_transit[t as usize][s as usize]
                        })
                        .collect();
                    dp[(i, j, s)] = Self::logsumexp(&lks);
                }
            }
        }
        dp
    }
    /// Return the sum of the probability to observe `xs` and `ys`. In other words,
    /// it sums up the probability for all possible alignment between `xs` and `ys`.
    pub fn likelihood(&self, xs: &[u8], ys: &[u8]) -> f64 {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        self.likelihood_inner(&xs, &ys)
    }
    pub fn likelihood_inner(&self, xs: &PadSeq, ys: &PadSeq) -> f64 {
        let (dp, alphas) = self.forward(xs, ys);
        let normalized_factor = alphas.iter().map(|x| x.ln()).sum::<f64>();
        let (n, m) = (xs.len() as isize, ys.len() as isize);
        let lk: f64 = (0..self.states).map(|s| dp[(n, m, s as isize)]).sum();
        lk.ln() + normalized_factor
    }
    // Forward algorithm.
    // To obtain the original likelihood dp[(i,j,s)],
    // calculate dp[(i,j,s)] + alphas.iter().take(i+1).map(|x|x.ln()).sum::<f64>();
    fn forward(&self, xs: &PadSeq, ys: &PadSeq) -> (DPTable, Vec<f64>) {
        let mut dp = DPTable::new(xs.len() + 1, ys.len() + 1, self.states, 0f64);
        let mut norm_factors = vec![];
        // Initialize.
        for (s, &x) in self.initial_distribution.iter().enumerate() {
            dp[(0, 0, s as isize)] = x;
        }
        for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos as isize + 1, y)) {
            for s in 0..self.states {
                let trans: f64 = (0..self.states)
                    .map(|t| dp[(0, j - 1, t as isize)] * self.transition(t, s))
                    .sum();
                dp[(0, j, s as isize)] += trans * self.observe(s, GAP, y);
            }
        }
        let total = dp.total(0);
        norm_factors.push(1f64 * total);
        dp.div(0, total);
        // Fill DP cells.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            // Deletion transition from above.
            for s in 0..self.states {
                dp[(i as isize, 0, s as isize)] = (0..self.states)
                    .map(|t| dp[(i as isize - 1, 0, t as isize)] * self.transition(t, s))
                    .sum::<f64>()
                    * self.observe(s, x, GAP);
            }
            // Deletion and Match transitions.
            for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
                for s in 0..self.states {
                    dp[(i as isize, j as isize, s as isize)] = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i as isize - 1, j as isize - 1, t as isize)];
                            let del = dp[(i as isize - 1, j as isize, t as isize)];
                            (mat * self.observe(s, x, y) + del * self.observe(s, x, GAP))
                                * self.transition(t, s)
                        })
                        .sum::<f64>();
                }
            }
            let first_total = dp.total(i);
            dp.div(i, first_total);
            // Insertion transitions
            for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
                for s in 0..self.states {
                    dp[(i as isize, j as isize, s as isize)] += (0..self.states)
                        .map(|t| {
                            //second_buffer[j - 1][t]
                            dp[(i as isize, j as isize - 1, t as isize)]
                                * self.transition(t, s)
                                * self.observe(s, GAP, y)
                        })
                        .sum::<f64>();
                }
            }
            let second_total = dp.total(i);
            dp.div(i, second_total);
            norm_factors.push(first_total * second_total);
        }
        (dp, norm_factors)
    }
    // This is a test function to validate faster implementation.
    #[allow(dead_code)]
    fn backward_naive(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let mut dp = DPTable::new(xs.len() + 1, ys.len() + 1, self.states, EP);
        let log_transit = self.get_log_transition();
        let log_observe = self.get_log_observation();
        // Initialization.
        for s in 0..self.states as isize {
            dp[(xs.len() as isize, ys.len() as isize, s)] = 0f64;
        }
        for (i, &x) in xs.iter().enumerate().rev() {
            for s in 0..self.states {
                let lks: Vec<_> = (0..self.states)
                    .map(|t| {
                        log_transit[s][t]
                            + log_observe[t][(x << 3 | GAP) as usize]
                            + dp[(i as isize + 1, ys.len() as isize, t as isize)]
                    })
                    .collect();
                dp[(i as isize, ys.len() as isize, s as isize)] = Self::logsumexp(&lks);
            }
        }
        for (j, &y) in ys.iter().enumerate().rev() {
            for s in 0..self.states {
                let lks: Vec<_> = (0..self.states)
                    .map(|t| {
                        log_transit[s][t]
                            + log_observe[t][(GAP << 3 | y) as usize]
                            + dp[(xs.len() as isize, j as isize + 1, t as isize)]
                    })
                    .collect();
                dp[(xs.len() as isize, j as isize, s as isize)] = Self::logsumexp(&lks);
            }
        }
        // Loop.
        for (i, &x) in xs.iter().enumerate().rev() {
            for (j, &y) in ys.iter().enumerate().rev() {
                for s in 0..self.states {
                    let lks: Vec<_> = (0..self.states)
                        .map(|t| {
                            let (i, j) = (i as isize, j as isize);
                            let mat = log_observe[t][(x << 3 | y) as usize]
                                + dp[(i + 1, j + 1, t as isize)];
                            let del = log_observe[t][(x << 3 | GAP) as usize]
                                + dp[(i + 1, j, t as isize)];
                            let ins = log_observe[t][(GAP << 3 | y) as usize]
                                + dp[(i, j + 1, t as isize)];
                            log_transit[s][t] + Self::logsumexp(&[mat, del, ins])
                        })
                        .collect();
                    dp[(i as isize, j as isize, s as isize)] = Self::logsumexp(&lks);
                }
            }
        }
        dp
    }
    fn backward(&self, xs: &PadSeq, ys: &PadSeq) -> (DPTable, Vec<f64>) {
        let mut dp = DPTable::new(xs.len() + 1, ys.len() + 1, self.states, 0f64);
        let mut norm_factors = vec![];
        // Initialize
        let (xslen, yslen) = (xs.len() as isize, ys.len() as isize);
        for s in 0..self.states as isize {
            dp[(xslen, yslen, s)] = 1f64;
        }
        let first_total = dp.total(xs.len());
        dp.div(xs.len(), first_total);
        for (j, &y) in ys.iter().enumerate().rev() {
            for s in 0..self.states {
                let j = j as isize;
                dp[(xslen, j, s as isize)] += (0..self.states)
                    .map(|t| {
                        self.transition(s, t)
                            * self.observe(t, GAP, y)
                            * dp[(xslen, j + 1, t as isize)]
                    })
                    .sum::<f64>();
            }
        }
        let second_toral = dp.total(xs.len());
        dp.div(xs.len(), second_toral);
        norm_factors.push(first_total * second_toral);
        for (i, &x) in xs.iter().enumerate().rev() {
            // Deletion transition to below.
            for s in 0..self.states {
                dp[(i as isize, yslen, s as isize)] = (0..self.states)
                    .map(|t| {
                        self.transition(s, t)
                            * self.observe(t, x, GAP)
                            * dp[(i as isize + 1, yslen, t as isize)]
                    })
                    .sum::<f64>();
            }
            for (j, &y) in ys.iter().enumerate().rev() {
                for s in 0..self.states {
                    let (i, j) = (i as isize, j as isize);
                    dp[(i, j, s as isize)] = (0..self.states)
                        .map(|t| {
                            self.transition(s, t)
                                * (self.observe(t, x, y) * dp[(i + 1, j + 1, t as isize)]
                                    + self.observe(t, x, GAP) * dp[(i + 1, j, t as isize)])
                        })
                        .sum::<f64>();
                }
            }
            let first_total = dp.total(i);
            dp.div(i, first_total);
            for (j, &y) in ys.iter().enumerate().rev() {
                for s in 0..self.states {
                    let (i, j) = (i as isize, j as isize);
                    dp[(i, j, s as isize)] += (0..self.states)
                        .map(|t| {
                            self.transition(s, t)
                                * self.observe(t, GAP, y)
                                * dp[(i, j + 1, t as isize)]
                        })
                        .sum::<f64>();
                }
            }
            let second_total = dp.total(i);
            dp.div(i, second_total);
            norm_factors.push(first_total * second_total);
        }
        norm_factors.reverse();
        (dp, norm_factors)
    }
    pub fn likelihood_banded(&self, xs: &[u8], ys: &[u8], radius: usize) -> f64 {
        unimplemented!()
    }
    fn forward_banded(&self, xs: &PadSeq, ys: &PadSeq, radius: usize) -> (DPTable, f64) {
        unimplemented!()
    }
    pub fn correct_until_convergence<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
    ) -> Vec<u8> {
        let mut template = PadSeq::new(template);
        let mut start_position = 0;
        let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
        while let Some((seq, next)) = self.correction_inner(&template, &queries, start_position) {
            template = seq;
            start_position = next;
        }
        template.into()
    }
    /// Correct `template` by `queries`.
    pub fn correction_inner(
        &self,
        template: &PadSeq,
        queries: &[PadSeq],
        start_position: usize,
    ) -> Option<(PadSeq, usize)> {
        let profiles: Vec<_> = queries
            .iter()
            .map(|q| Profile::new(self, &template, q))
            .collect();
        let total_lk = profiles.iter().map(|prof| prof.lk()).sum::<f64>();
        (0..template.len())
            .map(|pos| (pos + start_position) % template.len())
            .find_map(|pos| {
                let subst = b"ACGT".iter().map(padseq::convert_to_twobit).find_map(|b| {
                    if b == template[pos as isize] {
                        return None;
                    }
                    let new_lk: f64 = profiles.iter().map(|pr| pr.with_mutation(pos, b)).sum();
                    (total_lk < new_lk).then(|| (pos, Op::Match, b, new_lk))
                });
                let new_lk = profiles.iter().map(|pr| pr.with_deletion(pos)).sum::<f64>();
                let deletion = (total_lk < new_lk).then(|| (pos, Op::Del, GAP, new_lk));
                let insertion = b"ACGT".iter().map(padseq::convert_to_twobit).find_map(|b| {
                    let new_lk: f64 = profiles.iter().map(|pr| pr.with_insertion(pos, b)).sum();
                    (total_lk < new_lk).then(|| (pos, Op::Ins, b, new_lk))
                });
                let ins_last = if pos + 1 == template.len() {
                    let pos = pos + 1;
                    b"ACGT".iter().map(padseq::convert_to_twobit).find_map(|b| {
                        let new_lk: f64 = profiles.iter().map(|pr| pr.with_insertion(pos, b)).sum();
                        (total_lk < new_lk).then(|| (pos, Op::Ins, b, new_lk))
                    })
                } else {
                    None
                };
                subst.or(deletion).or(insertion).or(ins_last)
            })
            .map(|(pos, op, base, new_lk)| {
                eprintln!("{:?}:{:.4}->{:.4}({})", op, total_lk, new_lk, pos);
                let mut template = template.clone();
                match op {
                    Op::Match => template[pos as isize] = base,
                    Op::Del => {
                        template.remove(pos as isize);
                    }
                    Op::Ins => {
                        template.insert(pos as isize, base);
                    }
                }
                (template, pos + 1)
            })
    }
    pub fn correct_until_converge_banded<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
        radius: usize,
    ) -> Vec<u8> {
        unimplemented!()
    }
    pub fn correction_banded(
        &self,
        template: &PadSeq,
        queries: &[PadSeq],
        radius: usize,
    ) -> Option<PadSeq> {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
struct Profile<'a, 'b, 'c> {
    template: &'a PadSeq,
    query: &'b PadSeq,
    model: &'c GPHMM,
    forward: DPTable,
    forward_factor: Vec<f64>,
    backward: DPTable,
    backward_factor: Vec<f64>,
}

impl<'a, 'b, 'c> Profile<'a, 'b, 'c> {
    fn new(model: &'c GPHMM, template: &'a PadSeq, query: &'b PadSeq) -> Self {
        let (forward, forward_factor) = model.forward(template, query);
        let (backward, backward_factor) = model.backward(template, query);
        Self {
            template,
            query,
            forward,
            forward_factor,
            backward_factor,
            backward,
            model,
        }
    }
    fn lk(&self) -> f64 {
        let n = self.forward.row as isize - 1;
        let m = self.forward.column as isize - 1;
        let state = self.forward.states as isize;
        let lk = (0..state).map(|s| self.forward[(n, m, s)]).sum::<f64>();
        lk.ln() + self.forward_factor.iter().map(log).sum::<f64>()
    }
    // Return Likelihood when the `pos` base of the template is mutated into `base`.
    fn with_mutation(&self, pos: usize, base: u8) -> f64 {
        let states = self.model.states;
        let lk = (0..self.query.len() as isize + 1)
            .map(|j| {
                let pos = pos as isize;
                let y = self.query[j];
                (0..states)
                    .map(|s| {
                        let forward: f64 = (0..states)
                            .map(|t| {
                                self.forward[(pos, j, t as isize)] * self.model.transition(t, s)
                            })
                            .sum();
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos + 1, j + 1, s as isize)]
                            + self.model.observe(s, base, GAP)
                                * self.backward[(pos + 1, j, s as isize)];
                        forward * backward
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        let forward_factor: f64 = self.forward_factor[..pos + 1].iter().map(|x| x.ln()).sum();
        let backward_factor: f64 = self.backward_factor[pos + 1..].iter().map(|x| x.ln()).sum();
        lk.ln() + forward_factor + backward_factor
    }
    // Return likelihood when the `pos` base of the template is removed.
    fn with_deletion(&self, pos: usize) -> f64 {
        let states = self.model.states;
        if pos + 1 == self.template.len() {
            let tlen = self.template.len() as isize;
            let qlen = self.query.len() as isize;
            let lk: f64 = (0..states as isize)
                .map(|s| self.forward[(tlen - 1, qlen, s)])
                .sum();
            let factor: f64 = self.forward_factor[..tlen as usize].iter().map(log).sum();
            return lk.ln() + factor;
        }
        let lk: f64 = (0..self.query.len() as isize + 1)
            .map(|j| {
                let pos = pos as isize;
                let x = self.template[pos + 1];
                let y = self.query[j];
                (0..states)
                    .map(|s| {
                        let forward: f64 = (0..states)
                            .map(|t| {
                                self.forward[(pos, j, t as isize)] * self.model.transition(t, s)
                            })
                            .sum();
                        let backward = self.model.observe(s, x, y)
                            * self.backward[(pos + 2, j + 1, s as isize)]
                            + self.model.observe(s, x, GAP)
                                * self.backward[(pos + 2, j, s as isize)];
                        forward * backward
                    })
                    .sum::<f64>()
            })
            .sum();
        let forward_factor: f64 = self.forward_factor[..pos + 1].iter().map(log).sum();
        let backward_factor: f64 = self.backward_factor[pos + 2..].iter().map(log).sum();
        lk.ln() + forward_factor + backward_factor
    }
    fn with_insertion(&self, pos: usize, base: u8) -> f64 {
        let states = self.model.states;
        let lk: f64 = (0..self.query.len() as isize + 1)
            .map(|j| {
                let y = self.query[j];
                let pos = pos as isize;
                (0..states)
                    .map(|s| {
                        let forward: f64 = (0..states)
                            .map(|t| {
                                self.forward[(pos, j, t as isize)] * self.model.transition(t, s)
                            })
                            .sum();
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos, j + 1, s as isize)]
                            + self.model.observe(s, base, GAP)
                                * self.backward[(pos, j, s as isize)];
                        forward * backward
                    })
                    .sum::<f64>()
            })
            .sum();
        let forward_factor: f64 = self.forward_factor[..pos + 1].iter().map(log).sum();
        let backward_factor: f64 = self.backward_factor[pos..].iter().map(log).sum();
        lk.ln() + forward_factor + backward_factor
    }
}

#[cfg(test)]
mod gphmm {
    use super::*;
    use crate::hmm::Op;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128PlusPlus;
    #[test]
    fn default_test() {
        GPHMM::default();
    }
    #[test]
    fn three_state_test() {
        GPHMM::new_three_state(0.9, 0.1, 0.2, 0.9);
    }
    #[test]
    fn align_test() {
        let phmm = GPHMM::default();
        let xs = b"AAAAA";
        let ys = b"AAAAA";
        let (lk, ops, states) = phmm.align(xs, ys);
        assert_eq!(ops, vec![Op::Match; 5], "{:?},{:?}", lk, states);
        let xs = b"CGTT";
        let ys = b"GTT";
        let answer = vec![Op::Del, Op::Match, Op::Match, Op::Match];
        let (lk, ops, states) = phmm.align(xs, ys);
        assert_eq!(ops, answer, "{:?},{:?}", lk, states);
        let xs = b"CGTT";
        let ys = b"CGTTA";
        let answer = vec![Op::Match, Op::Match, Op::Match, Op::Match, Op::Ins];
        let (lk, ops, states) = phmm.align(xs, ys);
        assert_eq!(ops, answer, "{:?},{:?}", lk, states);
        let xs = b"CGCGT";
        let ys = b"CGCAT";
        let answer = vec![Op::Match; 5];
        let (lk, ops, states) = phmm.align(xs, ys);
        assert_eq!(ops, answer, "{:?},{:?}", lk, states);
        let xs = b"CGCGT";
        let ys = b"CGTCAT";
        let answer = vec![
            Op::Match,
            Op::Match,
            Op::Ins,
            Op::Match,
            Op::Match,
            Op::Match,
        ];
        let (lk, ops, states) = phmm.align(xs, ys);
        assert_eq!(ops, answer, "{:?},{:?}", lk, states);
    }
    #[test]
    fn align_test_random() {
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 1000;
            let profile = &crate::gen_seq::PROFILE;
            let phmm = GPHMM::default();
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let _ = phmm.align(&xs, &ys);
            let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
            let _ = phmm.align(&xs, &ys);
        }
    }
    #[test]
    fn likelihood_logsumexp_test() {
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 1000;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
            let (lk, _, _) = phmm.align(&xs, &ys);
            let lkt = phmm.likelihood_naive(&xs, &ys);
            assert!(lk < lkt);
        }
    }
    #[test]
    fn likelihood_scaling_test() {
        let phmm = GPHMM::default();
        let xs = b"ATGC";
        let lkn = phmm.likelihood_naive(xs, xs);
        let lks = phmm.likelihood(xs, xs);
        println!("{}", phmm.align(xs, xs).0);
        assert!((lkn - lks).abs() < 0.001, "{},{}", lkn, lks);
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 100;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
            let lkn = phmm.likelihood_naive(&xs, &ys);
            let lks = phmm.likelihood(&xs, &ys);
            assert!((lkn - lks).abs() < 0.001, "{},{}", lkn, lks);
            let dpn = phmm.forward_naive(&xs, &ys);
            let (xs, ys) = (PadSeq::new(xs.as_slice()), PadSeq::new(ys.as_slice()));
            let (dps, factors) = phmm.forward(&xs, &ys);
            for i in 0..xs.len() + 1 {
                for j in 0..ys.len() + 1 {
                    for s in 0..phmm.states {
                        let naive = dpn[(i as isize, j as isize, s as isize)];
                        let scaled = log(&dps[(i as isize, j as isize, s as isize)])
                            + factors[..=i].iter().map(log).sum::<f64>();
                        if -20f64 < naive || -20f64 < scaled {
                            assert!(
                                (naive - scaled).abs() < 0.1,
                                "{}, {}, {},{}",
                                naive,
                                scaled,
                                i,
                                j
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn likelihood_backward_test() {
        let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 1000;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let lkn = phmm.likelihood_naive(&xs, &ys);
            let dp = phmm.backward_naive(&xs, &ys);
            let lkbs: Vec<_> = phmm
                .initial_distribution
                .iter()
                .enumerate()
                .map(|(s, init)| dp[(0, 0, s as isize)] + log(init))
                .collect();
            eprintln!("{:?}", lkbs);
            let lkb = GPHMM::logsumexp(&lkbs);
            assert!((lkn - lkb).abs() < 0.01, "{},{}", lkn, lkb);
        }
    }
    #[test]
    fn likelihood_backward_scaling_test() {
        let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 1000;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let dpn = phmm.backward_naive(&xs, &ys);
            let lk = phmm.likelihood_naive(&xs, &ys);
            let (xs, ys) = (PadSeq::new(xs.as_slice()), PadSeq::new(ys.as_slice()));
            let (dp, factors) = phmm.backward(&xs, &ys);
            let lkb = phmm
                .initial_distribution
                .iter()
                .enumerate()
                .map(|(s, init)| dp[(0, 0, s as isize)] * init)
                .sum::<f64>()
                .ln()
                + factors.iter().map(log).sum::<f64>();
            assert!((lk - lkb).abs() < 0.01, "{},{}", lk, lkb);
            println!("{},{}", xs.len(), ys.len());
            for i in 0..xs.len() + 1 {
                for j in 0..ys.len() + 1 {
                    for s in 0..phmm.states {
                        let naive = dpn[(i as isize, j as isize, s as isize)];
                        let scaled = log(&dp[(i as isize, j as isize, s as isize)])
                            + factors[i..].iter().map(log).sum::<f64>();
                        if -10f64 < naive || -10f64 < scaled {
                            assert!(
                                (naive - scaled).abs() < 0.1,
                                "{},{},{},{}",
                                naive,
                                scaled,
                                i,
                                j
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn profile_lk_test() {
        let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 1000;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let lk = phmm.likelihood(&xs, &ys);
            let xs = PadSeq::new(xs.as_slice());
            let ys = PadSeq::new(ys.as_slice());
            let profile = Profile::new(&phmm, &xs, &ys);
            let lkp = profile.lk();
            assert!((lk - lkp).abs() < 0.0001, "{},{}", lk, lkp);
        }
    }
    #[test]
    fn profile_mutation_test() {
        let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
        for i in 0..10u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 200;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let (_, ops, _) = phmm.align(&xs, &ys);
            let (xr, opr, yr) = crate::hmm::recover(&xs, &ys, &ops);
            for ((xr, opr), yr) in xr.chunks(200).zip(opr.chunks(200)).zip(yr.chunks(200)) {
                println!("{}", String::from_utf8_lossy(xr));
                println!("{}", String::from_utf8_lossy(opr));
                println!("{}\n", String::from_utf8_lossy(yr));
            }
            let xs = PadSeq::new(xs.as_slice());
            let ys = PadSeq::new(ys.as_slice());
            let profile = Profile::new(&phmm, &xs, &ys);
            let lk = phmm.likelihood_inner(&xs, &ys);
            for (pos, base) in xs.iter().enumerate() {
                let lkp = profile.with_mutation(pos, *base);
                assert!(
                    (lk - lkp).abs() < 0.00001,
                    "{},{},{},{}",
                    lk,
                    lkp,
                    pos,
                    base
                );
            }
            let mut xs = xs.clone();
            for pos in 0..len {
                let original = xs[pos as isize];
                for base in b"ACGT".iter().map(padseq::convert_to_twobit) {
                    let lkp = profile.with_mutation(pos, base);
                    xs[pos as isize] = base;
                    let lk = phmm.likelihood_inner(&xs, &ys);
                    assert!((lk - lkp).abs() < 0.0001, "{},{}, {}", lk, lkp, pos);
                    xs[pos as isize] = original;
                }
            }
        }
    }
    #[test]
    fn profile_insertion_test() {
        let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
        for i in 0..10u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 200;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let (_, ops, _) = phmm.align(&xs, &ys);
            let (xr, opr, yr) = crate::hmm::recover(&xs, &ys, &ops);
            for ((xr, opr), yr) in xr.chunks(200).zip(opr.chunks(200)).zip(yr.chunks(200)) {
                println!("{}", String::from_utf8_lossy(xr));
                println!("{}", String::from_utf8_lossy(opr));
                println!("{}\n", String::from_utf8_lossy(yr));
            }
            let xs = PadSeq::new(xs.as_slice());
            let ys = PadSeq::new(ys.as_slice());
            let profile = Profile::new(&phmm, &xs, &ys);
            let mut xs = xs.clone();
            for pos in 0..len + 1 {
                for base in b"ACGT".iter().map(padseq::convert_to_twobit) {
                    let lkp = profile.with_insertion(pos, base);
                    xs.insert(pos as isize, base);
                    let lk = phmm.likelihood_inner(&xs, &ys);
                    assert!((lk - lkp).abs() < 0.0001, "{},{},{},{}", lk, lkp, pos, base);
                    xs.remove(pos as isize);
                }
            }
        }
    }

    #[test]
    fn profile_deletion_test() {
        let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
        for i in 0..10u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 200;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let (_, ops, _) = phmm.align(&xs, &ys);
            let (xr, opr, yr) = crate::hmm::recover(&xs, &ys, &ops);
            for ((xr, opr), yr) in xr.chunks(200).zip(opr.chunks(200)).zip(yr.chunks(200)) {
                println!("{}", String::from_utf8_lossy(xr));
                println!("{}", String::from_utf8_lossy(opr));
                println!("{}\n", String::from_utf8_lossy(yr));
            }
            let xs = PadSeq::new(xs.as_slice());
            let ys = PadSeq::new(ys.as_slice());
            let profile = Profile::new(&phmm, &xs, &ys);
            for pos in 0..len {
                let mut xs = xs.clone();
                xs.remove(pos as isize);
                let lkp = profile.with_deletion(pos);
                let lk = phmm.likelihood_inner(&xs, &ys);
                assert!((lk - lkp).abs() < 0.0001, "{},{}, {}", lk, lkp, pos);
            }
        }
    }
    #[test]
    fn correction_test() {
        let phmm = GPHMM::new_three_state(0.8, 0.2, 0.3, 0.9);
        let coverage = 30;
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 100;
            let profile = &crate::gen_seq::PROFILE;
            let template = crate::gen_seq::generate_seq(&mut rng, len);
            let queries: Vec<_> = (0..coverage)
                .map(|_| crate::gen_seq::introduce_randomness(&template, &mut rng, &profile))
                .collect();
            let qv = crate::gen_seq::Profile {
                sub: 0.01,
                del: 0.01,
                ins: 0.01,
            };
            let draft = crate::gen_seq::introduce_randomness(&template, &mut rng, &qv);
            let polished = phmm.correct_until_convergence(&draft, &queries);
            let dist = crate::bialignment::edit_dist(&polished, &template);
            assert_eq!(0, dist);
        }
    }
}
