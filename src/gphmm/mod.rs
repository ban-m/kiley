//! This module defines generalized pair hidden Markov models.
//! In contrast to the usual pair hidden Markov model, where each state
//! should output either a match, a deletion, or an insertion,
//! a GpHMM can output any of them.
//! However, the usual algorithm such as Viterbi, Forward, Backward, and Forward-Backward algorithm could be run on a GpHMM in the
//! same time with respect to the O-notation. Specifically, all of them run in O(|X||Y||S|^2) time, where X and Y are the reference and the query, and S is the states.
//! Caution: The `fit` and its friends functions are not based on any formal theory. It is just a "thoguth-to-be-OK" implementations.
//! So, there's no garantee that the total likleihood would monotonously increase, nor some other good features.

/// A generalized pair hidden Markov model.
/// Usually, it is more stressful to use short-hand notation `GPHMM`.
/// This model generates alignment, not a pair of sequence. This distinction is very vague, but
/// one should be careful when using pair-HMM.
/// For example,
///
/// AAA-AGT-T
/// AA-CCGTGT
///
/// is an alignment between a pair of sequence `AAAAGTT` and `AACCGTGT`. Of course, there are other alignment for these two sequence.
/// So, there is possibly many alignment for a pair of sequence.
/// Currently, this function is for DNA alignment with no anbiguous bases such as N. In other words,
/// The input sequences should be strings on the alphabet b"ATGCacgt", and there is no distinction between lower-case letter and upper-case letter.
/// Note that ,for each exact algorithm, there is `banded` version approximate the result.
/// If the computational time matters, please try these method, even though it can reach sub-optimal solutions.
/// To specify the mode of the pair HMM(Full or conditional), `GPHMM::<Full>::` or `GPHMM::<Cond>::` would be OK.
#[derive(Clone, Debug)]
pub struct GeneralizedPairHiddenMarkovModel<T: HMMType> {
    // Number of states.
    states: usize,
    // Transition between each states.
    // This is transposed matrix of transition matrix. So,
    // by accessing from * self.states + to, we can get the transition probability from `from` to `to`.
    transition_matrix: Vec<f64>,
    // obseration on a state for a alignment operation.
    // By accessing [states << 6 | x << 3 | y], we can get the obvervasion probability Pr{(x,y)|x}
    observation_matrix: Vec<f64>,
    // Initial distribution. Should be normalized to 1.
    initial_distribution: Vec<f64>,
    // Mode.
    _mode: std::marker::PhantomData<T>,
}

pub mod banded;
use crate::hmm::Op;
use rayon::prelude::*;
fn get_range(radius: isize, ylen: isize, center: isize) -> std::ops::Range<isize> {
    let start = radius - center;
    let end = ylen + radius - center;
    start.max(0)..(end.min(2 * radius) + 1)
}

/// Please Do not implement this trait by your own program.
pub trait HMMType: Clone + std::marker::Send + std::marker::Sync {}

/// This is a marker type associated with Generalized Pair Hidden Markov Model.
/// If you want to use full, unconditional model, please tag GPHMM with this type, like
/// GPHMM::<FullHiddenmarkovmodel>;
#[derive(Debug, Clone)]
pub struct FullHiddenMarkovModel;
/// Alius.
pub type Full = FullHiddenMarkovModel;
impl HMMType for FullHiddenMarkovModel {}

/// This is a marker type associated with Generalized Pair Hidden Markov Model.
/// If you want to use full, unconditional model, please tag GPHMM with this type, like
/// GPHMM::<ConditionalHiddenmarkovmodel>;
#[derive(Debug, Clone)]
pub struct ConditionalHiddenMarkovModel;
/// Alius
pub type Cond = ConditionalHiddenMarkovModel;
impl HMMType for ConditionalHiddenMarkovModel {}

impl<T: HMMType> std::fmt::Display for GPHMM<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Type:{}", std::any::type_name::<T>())?;
        writeln!(f, "States:{}", self.states)?;
        writeln!(f, "Transition:")?;
        for from in 0..self.states {
            let probs: Vec<_> = (0..self.states)
                .map(|to| self.transition(from, to))
                .map(|x| format!("{:.3}", x))
                .collect();
            writeln!(f, "{}", probs.join("\t"))?;
        }
        writeln!(f, "Observation")?;
        for state in 0..self.states {
            for x in 0..5 {
                let probs: Vec<_> = (0..5)
                    .map(|y| self.observe(state, x, y))
                    .map(|x| format!("{:.2}", x))
                    .collect();
                writeln!(f, "{}", probs.join("\t"))?
            }
            writeln!(f)?;
        }
        let probs: Vec<_> = self
            .initial_distribution
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect();
        writeln!(f, "Initial:{}", probs.join("\t"))
    }
}

#[allow(clippy::upper_case_acronyms)]
pub type GPHMM<T> = GeneralizedPairHiddenMarkovModel<T>;

// We define additional structure for easy implementation.
// This is offset around a DP table for each state.
const OFFSET: usize = 3;
// This is the DP table. We have one sheet for each state.
fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return EP;
    }
    let max = xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln() + max
}
#[derive(Debug, Clone)]
struct DPTable {
    column: usize,
    row: usize,
    states: usize,
    column_size: usize,
    data: Vec<f64>,
}

impl DPTable {
    // Create new (row x column x state) DP table.
    // If you access invalid index such as -1 or [column][row][states], it would return default value..
    fn new(row: usize, column: usize, states: usize, default: f64) -> Self {
        let len = (row + 2 * OFFSET) * (column + 2 * OFFSET) * states;
        let column_size = states * (column + 2 * OFFSET);
        Self {
            column_size,
            column,
            row,
            states,
            data: vec![default; len],
        }
    }
    // Maybe I can move these implementations to std::slice::SliceIndex.
    fn get(&self, i: isize, j: isize, s: usize) -> Option<&f64> {
        let index = self.get_index(i, j, s);
        assert!(index < self.data.len());
        self.data.get(index)
    }
    fn get_index(&self, i: isize, j: isize, s: usize) -> usize {
        let column_pos = (j + OFFSET as isize) as usize * self.states;
        let row_number = (i + OFFSET as isize) as usize;
        self.column_size * row_number + column_pos + s
    }
    fn get_mut(&mut self, i: isize, j: isize, s: usize) -> Option<&mut f64> {
        let index = self.get_index(i, j, s);
        assert!(index < self.data.len());
        self.data.get_mut(index)
    }
    // Normalize the dp[i], return the divider.
    fn normalize(&mut self, i: usize) -> f64 {
        let start = self.get_index(i as isize, 0, 0);
        let length = self.column * self.states;
        let data = &mut self.data[start..start + length];
        let sum = data.iter().sum::<f64>();
        data.iter_mut().for_each(|x| *x /= sum);
        sum
    }
    // Sum dp[i][j][s] over j and s.
    fn total(&self, i: usize) -> f64 {
        let start = self.get_index(i as isize, 0, 0);
        let length = self.column * self.states;
        self.data.iter().skip(start).take(length).sum()
    }
    // Divide dp[i] by `by`
    fn div(&mut self, i: usize, by: f64) {
        let start = self.get_index(i as isize, 0, 0);
        let length = self.column * self.states;
        self.data[start..start + length]
            .iter_mut()
            .for_each(|x| *x /= by);
    }
    // Return cells in (i,j) location. The length of the returned vector
    // is `state`.
    fn get_cells(&self, i: isize, j: isize) -> &[f64] {
        let start = self.get_index(i, j, 0);
        &self.data[start..start + self.states]
    }
    fn replace_cells(&mut self, i: isize, j: isize, replace: &[f64]) {
        let start = self.get_index(i, j, 0);
        self.data[start..start + self.states]
            .iter_mut()
            .zip(replace.iter())
            .for_each(|(x, y)| *x = *y);
    }
}

impl std::ops::Index<(isize, isize, usize)> for DPTable {
    type Output = f64;
    fn index(&self, (i, j, s): (isize, isize, usize)) -> &Self::Output {
        self.get(i, j, s).unwrap()
    }
}

impl std::ops::IndexMut<(isize, isize, usize)> for DPTable {
    fn index_mut(&mut self, (i, j, s): (isize, isize, usize)) -> &mut Self::Output {
        self.get_mut(i, j, s).unwrap()
    }
}

use crate::padseq;
use crate::padseq::PadSeq;
use padseq::GAP;
const EP: f64 = -100000000000000000000000f64;
fn log(x: &f64) -> f64 {
    assert!(!x.is_sign_negative(), "{}", x);
    if f64::EPSILON < x.abs() {
        x.ln()
    } else {
        EP
    }
}

impl std::default::Default for GPHMM<FullHiddenMarkovModel> {
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
        Self::new(
            1,
            &[vec![1f64]],
            &[match_prob],
            &[del_prob],
            &[ins_prob],
            &[1f64],
        )
    }
}

impl std::default::Default for GPHMM<ConditionalHiddenMarkovModel> {
    // Return very simple single state hidden Markov model.
    fn default() -> Self {
        let mat = 0.6;
        let mism = 0.1;
        let match_prob = [
            mat, mism, mism, mism, mism, mat, mism, mism, mism, mism, mat, mism, mism, mism, mism,
            mat,
        ];
        let del_prob = [mism; 4];
        let ins_prob = [4f64.recip(); 4];
        Self::new(
            1,
            &[vec![1f64]],
            &[match_prob],
            &[del_prob],
            &[ins_prob],
            &[1f64],
        )
    }
}

impl<M: HMMType> GPHMM<M> {
    /// transition matrix: the [to *self.states + from]-th element should be Pr{from->to}
    /// observation_matrix: the [states << 6 | x << 3 | y]-th element should be Pr{(x,y)|states},
    /// or Pr{(x,y)|states, x} in the case of conditional pair-HMM.
    /// where x and y are three-bit encoded bases as follows: (A->000,C->001,G->010,T->011,'-'->100)
    /// initial_distribution: the [i]-th element should be Pr{i} at the beggining.
    pub fn from_raw_elements(
        states: usize,
        transition_matrix: Vec<f64>,
        observation_matrix: Vec<f64>,
        initial_distribution: Vec<f64>,
    ) -> Self {
        Self {
            states,
            transition_matrix,
            observation_matrix,
            initial_distribution,
            _mode: std::marker::PhantomData,
        }
    }
    pub fn states(&self) -> usize {
        self.states
    }
    /// Frobenius norm, or element-wise-square-sum.
    /// Return None if the two model have different number of states.
    pub fn dist(&self, other: &Self) -> Option<f64> {
        (self.states == other.states).then(|| {
            let init: f64 = self
                .initial_distribution
                .iter()
                .zip(other.initial_distribution.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum();
            let trans: f64 = self
                .transition_matrix
                .iter()
                .zip(other.transition_matrix.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum();
            let observe: f64 = self
                .observation_matrix
                .iter()
                .zip(other.observation_matrix.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum();
            init + trans + observe
        })
    }
}

impl GPHMM<FullHiddenMarkovModel> {
    /// Create a new generalized pair hidden Markov model.
    /// There is a few restriction on the input arguments.
    /// 1. transition matrix should be states x states matrix,
    /// all rowsum to be 1. In other words, transition_matrix[i][j] = Pr(i -> j).
    /// 2. Match_prob, del_prob, ins_prob, and initial_distribution should be the length of `states`.
    /// 3. The sum of initial_distribution should be 1.
    /// 4. The sum of match_prob[x][y] + del_prob[x] + ins_prob[x] should be 1 (summing over x and y).
    /// Each 16-length array of match probabilities is:
    /// [(A,A), (A,C), (A,G), (A,T), (C,A), ... ,(T,T)]
    /// and Each array of del_prob is [(A,-), ...,(T,-)],
    /// and each array of ins_prob is [(-,A), ... ,(-,A)]
    // TODO:Sanity check.
    pub fn new(
        states: usize,
        transition_matrix: &[Vec<f64>],
        match_prob: &[[f64; 16]],
        del_prob: &[[f64; 4]],
        ins_prob: &[[f64; 4]],
        initial_distribution: &[f64],
    ) -> Self {
        let mut observation_matrix = vec![0f64; ((states - 1) << 6) + (8 * 8)];
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
        let transition_matrix: Vec<_> = transition_matrix
            .iter()
            .flat_map(std::convert::identity)
            .copied()
            .collect();
        Self {
            states,
            transition_matrix,
            observation_matrix,
            initial_distribution: initial_distribution.to_vec(),
            _mode: std::marker::PhantomData,
        }
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
    /// Parameter estimation by set of sequences.
    /// As there *is* difference between the reference and the query,
    /// this function takes one template and several queries.
    /// For conditional pair-HMM, please use `fit_conditional`. `_banded` mode would be much faster.
    /// If this function would be called repeatedly, maybe `fit_inner` would reduce computational time.
    /// ----- This is general comment on any function like `fit_*` ----
    /// It only updates the parameters once. If you want to loop updates until convergence, like EM-algorithm,
    /// You need to call this function repeatedly until watching the total likelihood will not increase, or
    /// the total likelihood will decrease.
    /// Note that this function is based on "analogy" to EM-algorithm, not any formal theory.
    /// So there is no garantee that the total likelihood would increase by calling this function.
    pub fn fit<T: std::borrow::Borrow<[u8]>>(&self, template: &[u8], queries: &[T]) -> Self {
        let template = PadSeq::new(template);
        let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
        self.fit_inner(&template, &queries)
    }
    pub fn fit_inner(&self, xs: &PadSeq, yss: &[PadSeq]) -> Self {
        let profiles: Vec<_> = yss.iter().map(|ys| Profile::new(self, xs, ys)).collect();
        let initial_distribution = self.estimate_initial_distribution(&profiles);
        let transition_matrix = self.estimate_transition_prob(&profiles);
        let observation_matrix = self.estimate_observation_prob(&profiles);
        Self {
            states: self.states,
            initial_distribution,
            transition_matrix,
            observation_matrix,
            _mode: std::marker::PhantomData,
        }
    }
    // [states << 6 | x << 3 | y] = Pr{(x,y)|states}.
    // In other words, the length is states << 6 + 8 * 8
    fn estimate_observation_prob(&self, profiles: &[Profile<FullHiddenMarkovModel>]) -> Vec<f64> {
        let mut buffer = vec![0f64; ((self.states - 1) << 6) + 8 * 8];
        for prob in profiles.iter().map(|prf| prf.observation_probability()) {
            buffer.iter_mut().zip(prob).for_each(|(x, y)| *x += y);
        }
        // Normalize (states << 6) | 0b000_000 - (state << 6) | 0b111_111.
        for probs in buffer.chunks_mut(8 * 8) {
            let sum: f64 = probs.iter().sum();
            if 0.00001 < sum {
                probs.iter_mut().for_each(|x| *x /= sum);
            }
        }
        buffer
    }
}
impl GPHMM<ConditionalHiddenMarkovModel> {
    /// Create a new generalized pair hidden Markov model.
    /// There is a few restriction on the input arguments.
    /// 1. transition matrix should be states x states matrix,
    /// all rowsum to be 1. In other words, transition_matrix[i][j] = Pr(i -> j).
    /// 2. Match_prob, del_prob, ins_prob, and initial_distribution should be the length of `states`.
    /// 3. The sum of initial_distribution should be 1.
    /// 4. The sum of match_prob[x][y] + del_prob[x] should be 1 for each x, summing over y.
    /// 5: The sum of ins_prob[x] should be 1 (summing over x).
    /// Each 16-length array of match probabilities is:
    /// [(A|A), (C|A), (G|A), (T|A), (C,A), ... ,(T|T)]
    /// and Each array of del_prob is [(-|A), ...,(-|T)],
    /// and each array of ins_prob is [(A|-), ... ,(T|-)]
    // TODO:Sanity check.
    pub fn new(
        states: usize,
        transition_matrix: &[Vec<f64>],
        match_prob: &[[f64; 16]],
        del_prob: &[[f64; 4]],
        ins_prob: &[[f64; 4]],
        initial_distribution: &[f64],
    ) -> Self {
        let mut observation_matrix = vec![0f64; ((states - 1) << 6) + (8 * 8)];
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
        let transition_matrix: Vec<_> = transition_matrix
            .iter()
            .flat_map(std::convert::identity)
            .copied()
            .collect();
        Self {
            states,
            transition_matrix,
            observation_matrix,
            initial_distribution: initial_distribution.to_vec(),
            _mode: std::marker::PhantomData,
        }
    }
    /// Return a simple single state pair HMM for computing conditional likelihood, LK(y|x).
    /// (match_prob + del_prob) should be smaller than 1f64.
    pub fn new_single_state(match_prob: f64, del_prob: f64) -> Self {
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
    // /// CLR profile.
    pub fn clr() -> Self {
        let states = 3;
        let transition_matrix = [
            vec![0.86, 0.07, 0.07],
            vec![0.78, 0.15, 0.07],
            vec![0.78, 0.07, 0.15],
        ];
        let (mat, mism) = (0.97, 0.01);
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
    /// Return a three states pair HMM for computing conditional likelihood, LK(y|x)
    pub fn new_three_state(match_prob: f64, gap_open: f64, gap_ext: f64, match_emit: f64) -> Self {
        let sum = match_prob + 2f64 * gap_open;
        let (match_prob, gap_open) = (match_prob / sum, gap_open / sum);
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
    pub fn fit<T: std::borrow::Borrow<[u8]>>(&self, template: &[u8], queries: &[T]) -> Self {
        let template = PadSeq::new(template);
        let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
        self.fit_inner(&template, &queries)
    }
    /// Paramter estimation by set of setquence.
    /// As there *is* difference between the reference and the query,
    /// this function takes one template and several queries.
    /// For conditional pair-HMM, please use `fit_conditional`. `_banded` mode would be much faster.
    /// If this function would be called repeatedly, maybe `fit_inner` would reduce computational time.
    /// Please see [fit] function for more detailed information.
    pub fn fit_inner(&self, xs: &PadSeq, yss: &[PadSeq]) -> Self {
        let profiles: Vec<_> = yss.iter().map(|ys| Profile::new(self, xs, ys)).collect();
        let initial_distribution = self.estimate_initial_distribution(&profiles);
        let transition_matrix = self.estimate_transition_prob(&profiles);
        let observation_matrix = self.estimate_observation_prob(&profiles);
        Self {
            states: self.states,
            initial_distribution,
            transition_matrix,
            observation_matrix,
            _mode: std::marker::PhantomData,
        }
    }
    fn estimate_observation_prob(&self, profiles: &[Profile<Cond>]) -> Vec<f64> {
        let mut buffer = vec![0f64; ((self.states - 1) << 6) + 8 * 8];
        for prob in profiles.iter().map(|prf| prf.observation_probability()) {
            buffer.iter_mut().zip(prob).for_each(|(x, y)| *x += y);
        }
        // Normalize (states << 6) | 0b_i__000 - (state << 6) | 0b_i_111.
        for probs in buffer.chunks_mut(8) {
            let sum: f64 = probs.iter().sum();
            if 0.00001 < sum {
                probs.iter_mut().for_each(|x| *x /= sum);
            }
        }
        buffer
    }
}

impl<M: HMMType> GPHMM<M> {
    // The reason why these code are here is that the procedure is exactly the same between full/conditional
    // hidden Markov model.
    /// get transition probability from `from` to `to`
    pub fn transition(&self, from: usize, to: usize) -> f64 {
        self.transition_matrix[from * self.states + to]
    }
    /// Return transition probialities from `from`.
    pub fn transitions(&self, from: usize) -> &[f64] {
        &self.transition_matrix[from * self.states..(from + 1) * self.states]
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
            dp[(0, 0, s)] = log(&self.initial_distribution[s]);
        }
        // Initial values.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            let i = i as isize;
            for s in 0..self.states {
                dp[(i, 0, s)] = (0..self.states)
                    .map(|t| {
                        dp[(i - 1, 0, t)]
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
                dp[(0, j, s)] = (0..self.states)
                    .map(|t| {
                        dp[(0, j - 1, t)]
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
                    let (i, j, s) = (i as isize, j as isize, s);
                    let max_path = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i - 1, j - 1, t)]
                                + log_transit[t as usize][s as usize]
                                + log_observe[s as usize][(x << 3 | y) as usize];
                            let del = dp[(i - 1, j, t)]
                                + log_transit[t as usize][s as usize]
                                + log_observe[s as usize][(x << 3 | GAP) as usize];
                            let ins = dp[(i, j - 1, t)]
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
            .map(|s| (dp[(i, j, s)], s))
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
                    let mat = dp[(i - 1, j - 1, t)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | y) as usize];
                    let del = dp[(i - 1, j, t)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | GAP) as usize];
                    let ins = dp[(i, j - 1, t)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(GAP << 3 | y) as usize];
                    if (current - mat).abs() < 0.00001 {
                        Some((Op::Match, t))
                    } else if (current - del).abs() < 0.0001 {
                        Some((Op::Del, t))
                    } else if (current - ins).abs() < 0.0001 {
                        Some((Op::Ins, t))
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
                    let del = dp[(i - 1, j, t)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | GAP) as usize];
                    ((current - del) < 0.000001).then(|| t)
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
                    let ins = dp[(i, j - 1, t)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(GAP << 3 | y) as usize];
                    ((current - ins).abs() < 0.00001).then(|| t)
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
    /// same as likelihood. A naive log-sum-exp implementation.
    pub fn likelihood_naive(&self, xs: &[u8], ys: &[u8]) -> f64 {
        let dp = self.forward_naive(xs, ys);
        let (n, m) = (xs.len() as isize, ys.len() as isize);
        let lks: Vec<_> = (0..self.states).map(|s| dp[(n, m, s)]).collect();
        logsumexp(&lks)
    }
    fn forward_naive(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let mut dp = DPTable::new(xs.len() + 1, ys.len() + 1, self.states, EP);
        let log_transit = self.get_log_transition();
        let log_observe = self.get_log_observation();
        for s in 0..self.states {
            dp[(0, 0, s)] = log(&self.initial_distribution[s]);
        }
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            let i = i as isize;
            for s in 0..self.states {
                let lks: Vec<_> = (0..self.states)
                    .map(|t| {
                        dp[(i - 1, 0, t)]
                            + log_transit[t][s]
                            + log_observe[s][(x << 3 | GAP) as usize]
                    })
                    .collect();
                dp[(i, 0, s)] = logsumexp(&lks);
            }
        }
        for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
            let j = j as isize;
            for s in 0..self.states {
                let lks: Vec<_> = (0..self.states)
                    .map(|t| {
                        dp[(0, j - 1, t)]
                            + log_transit[t][s]
                            + log_observe[s][(GAP << 3 | y) as usize]
                    })
                    .collect();
                dp[(0, j, s)] = logsumexp(&lks);
            }
        }
        // Fill DP cells.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
                for s in 0..self.states {
                    let (i, j, s) = (i as isize, j as isize, s);
                    let lks: Vec<_> = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i - 1, j - 1, t)]
                                + log_observe[s as usize][(x << 3 | y) as usize];
                            let del = dp[(i - 1, j, t)]
                                + log_observe[s as usize][(x << 3 | GAP) as usize];
                            let ins = dp[(i, j - 1, t)]
                                + log_observe[s as usize][(GAP << 3 | y) as usize];
                            logsumexp(&[mat, del, ins]) + log_transit[t as usize][s as usize]
                        })
                        .collect();
                    dp[(i, j, s)] = logsumexp(&lks);
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
        let lk: f64 = (0..self.states).map(|s| dp[(n, m, s)]).sum();
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
            dp[(0, 0, s)] = x;
        }
        for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos as isize + 1, y)) {
            for s in 0..self.states {
                let trans: f64 = (0..self.states)
                    .map(|t| dp[(0, j - 1, t)] * self.transition(t, s))
                    .sum();
                dp[(0, j, s)] += trans * self.observe(s, GAP, y);
            }
        }
        let total = dp.total(0);
        norm_factors.push(1f64 * total);
        dp.div(0, total);
        // Fill DP cells.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            // Deletion transition from above.
            for s in 0..self.states {
                dp[(i as isize, 0, s)] = (0..self.states)
                    .map(|t| dp[(i as isize - 1, 0, t)] * self.transition(t, s))
                    .sum::<f64>()
                    * self.observe(s, x, GAP);
            }
            // Deletion and Match transitions.
            for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
                for s in 0..self.states {
                    dp[(i as isize, j as isize, s)] = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i as isize - 1, j as isize - 1, t)];
                            let del = dp[(i as isize - 1, j as isize, t)];
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
                    dp[(i as isize, j as isize, s)] += (0..self.states)
                        .map(|t| {
                            //second_buffer[j - 1][t]
                            dp[(i as isize, j as isize - 1, t)]
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
        for s in 0..self.states {
            dp[(xs.len() as isize, ys.len() as isize, s)] = 0f64;
        }
        for (i, &x) in xs.iter().enumerate().rev() {
            for s in 0..self.states {
                let lks: Vec<_> = (0..self.states)
                    .map(|t| {
                        log_transit[s][t]
                            + log_observe[t][(x << 3 | GAP) as usize]
                            + dp[(i as isize + 1, ys.len() as isize, t)]
                    })
                    .collect();
                dp[(i as isize, ys.len() as isize, s)] = logsumexp(&lks);
            }
        }
        for (j, &y) in ys.iter().enumerate().rev() {
            for s in 0..self.states {
                let lks: Vec<_> = (0..self.states)
                    .map(|t| {
                        log_transit[s][t]
                            + log_observe[t][(GAP << 3 | y) as usize]
                            + dp[(xs.len() as isize, j as isize + 1, t)]
                    })
                    .collect();
                dp[(xs.len() as isize, j as isize, s)] = logsumexp(&lks);
            }
        }
        // Loop.
        for (i, &x) in xs.iter().enumerate().rev() {
            for (j, &y) in ys.iter().enumerate().rev() {
                for s in 0..self.states {
                    let lks: Vec<_> = (0..self.states)
                        .map(|t| {
                            let (i, j) = (i as isize, j as isize);
                            let mat = log_observe[t][(x << 3 | y) as usize] + dp[(i + 1, j + 1, t)];
                            let del = log_observe[t][(x << 3 | GAP) as usize] + dp[(i + 1, j, t)];
                            let ins = log_observe[t][(GAP << 3 | y) as usize] + dp[(i, j + 1, t)];
                            log_transit[s][t] + logsumexp(&[mat, del, ins])
                        })
                        .collect();
                    dp[(i as isize, j as isize, s)] = logsumexp(&lks);
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
        for s in 0..self.states {
            dp[(xslen, yslen, s)] = 1f64;
        }
        let first_total = dp.total(xs.len());
        dp.div(xs.len(), first_total);
        for (j, &y) in ys.iter().enumerate().rev() {
            for s in 0..self.states {
                let j = j as isize;
                dp[(xslen, j, s)] += (0..self.states)
                    .map(|t| {
                        self.transition(s, t) * self.observe(t, GAP, y) * dp[(xslen, j + 1, t)]
                    })
                    .sum::<f64>();
            }
        }
        let second_total = dp.total(xs.len());
        dp.div(xs.len(), second_total);
        norm_factors.push(first_total * second_total);
        for (i, &x) in xs.iter().enumerate().rev() {
            // Deletion transition to below.
            for s in 0..self.states {
                dp[(i as isize, yslen, s)] = (0..self.states)
                    .map(|t| {
                        self.transition(s, t)
                            * self.observe(t, x, GAP)
                            * dp[(i as isize + 1, yslen, t)]
                    })
                    .sum::<f64>();
            }
            for (j, &y) in ys.iter().enumerate().rev() {
                for s in 0..self.states {
                    let (i, j) = (i as isize, j as isize);
                    dp[(i, j, s)] = (0..self.states)
                        .map(|t| {
                            self.transition(s, t)
                                * (self.observe(t, x, y) * dp[(i + 1, j + 1, t)]
                                    + self.observe(t, x, GAP) * dp[(i + 1, j, t)])
                        })
                        .sum::<f64>();
                }
            }
            let first_total = dp.total(i);
            dp.div(i, first_total);
            for (j, &y) in ys.iter().enumerate().rev() {
                for s in 0..self.states {
                    let (i, j) = (i as isize, j as isize);
                    dp[(i, j, s)] += (0..self.states)
                        .map(|t| {
                            self.transition(s, t) * self.observe(t, GAP, y) * dp[(i, j + 1, t)]
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
            .map(|(pos, op, base, _lk)| {
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
    fn estimate_initial_distribution(&self, profiles: &[Profile<M>]) -> Vec<f64> {
        let mut buffer = vec![0f64; self.states];
        for init in profiles.iter().map(|prf| prf.initial_distribution()) {
            buffer.iter_mut().zip(init).for_each(|(x, y)| *x += y);
        }
        let sum: f64 = buffer.iter().sum();
        buffer.iter_mut().for_each(|x| *x /= sum);
        buffer
    }
    fn estimate_transition_prob(&self, profiles: &[Profile<M>]) -> Vec<f64> {
        let states = self.states;
        // [from * states + to] = Pr(from->to)
        // Because it is easier to normalize.
        let mut buffer = vec![0f64; states * states];
        for prob in profiles.iter().map(|prf| prf.transition_probability()) {
            buffer.iter_mut().zip(prob).for_each(|(x, y)| *x += y);
        }
        // Normalize.
        for row in buffer.chunks_mut(states) {
            let sum: f64 = row.iter().sum();
            row.iter_mut().for_each(|x| *x /= sum);
        }
        buffer
        // Transpose. [to * states + from] = Pr{from->to}.
        // (0..states * states)
        //     .map(|idx| {
        //         let (to, from) = (idx / states, idx % states);
        //         buffer[from * states + to]
        //     })
        //     .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Profile<'a, 'b, 'c, T: HMMType> {
    template: &'a PadSeq,
    query: &'b PadSeq,
    model: &'c GPHMM<T>,
    forward: DPTable,
    pub forward_factor: Vec<f64>,
    backward: DPTable,
    pub backward_factor: Vec<f64>,
}

impl<'a, 'b, 'c, T: HMMType> Profile<'a, 'b, 'c, T> {
    pub fn new(model: &'c GPHMM<T>, template: &'a PadSeq, query: &'b PadSeq) -> Self {
        let (forward, forward_factor) = model.forward(template, query);
        let (backward, backward_factor) = model.backward(template, query);
        Self {
            template,
            query,
            model,
            forward,
            forward_factor,
            backward,
            backward_factor,
        }
    }
    pub fn lk(&self) -> f64 {
        let n = self.forward.row as isize - 1;
        let m = self.forward.column as isize - 1;
        let state = self.model.states;
        let lk = (0..state).map(|s| self.forward[(n, m, s)]).sum::<f64>();
        lk.ln() + self.forward_factor.iter().map(log).sum::<f64>()
    }
    // Return Likelihood when the `pos` base of the template is mutated into `base`.
    pub fn with_mutation(&self, pos: usize, base: u8) -> f64 {
        let states = self.model.states;
        let lk = (0..self.query.len() as isize + 1)
            .map(|j| {
                let pos = pos as isize;
                let y = self.query[j];
                (0..states)
                    .map(|s| {
                        let forward: f64 = (0..states)
                            .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                            .sum();
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos + 1, j + 1, s)]
                            + self.model.observe(s, base, GAP) * self.backward[(pos + 1, j, s)];
                        forward * backward
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        let forward_factor: f64 = self.forward_factor[..pos + 1].iter().map(|x| x.ln()).sum();
        let backward_factor: f64 = self.backward_factor[pos + 1..].iter().map(|x| x.ln()).sum();
        lk.ln() + forward_factor + backward_factor
    }
    pub fn accumlate_factors(&self) -> (Vec<f64>, Vec<f64>) {
        // pos -> [..pos+1].ln().sum()
        let (forward_acc, _) =
            self.forward_factor
                .iter()
                .fold((vec![], 0f64), |(mut result, accum), x| {
                    result.push(accum + log(x));
                    (result, accum + log(x))
                });
        // pos -> [pos..].ln().sum()
        let backward_acc = {
            let (mut backfac, _) =
                self.backward_factor
                    .iter()
                    .rev()
                    .fold((vec![], 0f64), |(mut result, accum), x| {
                        result.push(accum + log(x));
                        (result, accum + log(x))
                    });
            backfac.reverse();
            backfac
        };
        (forward_acc, backward_acc)
    }
    pub fn to_deletion_table(&self, len: usize) -> Vec<f64> {
        let (forward_acc, backward_acc) = self.accumlate_factors();
        let states = self.model.states;
        let width = len - 1;
        let mut lks = vec![EP; width * (self.template.len() - width)];
        for (pos, slots) in lks.chunks_exact_mut(width).enumerate() {
            slots.iter_mut().for_each(|x| *x = 0f64);
            let forward_acc = forward_acc[pos];
            for del_size in 2..len + 1 {
                if pos + del_size == self.template.len() {
                    let j = self.query.len() as isize;
                    let lk: f64 = (0..states)
                        .map(|s| self.forward[(pos as isize, j, s)])
                        .sum();
                    slots[del_size - 2] = lk.ln() + forward_acc;
                } else {
                    let x = self.template[(pos + del_size) as isize];
                    let mut lk_total = 0f64;
                    for (j, &y) in self.query.iter().enumerate() {
                        let pos = pos as isize;
                        let j = j as isize;
                        let forward = (0..states).map(|s| {
                            (0..states)
                                .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                                .sum::<f64>()
                        });
                        for (s, forward) in forward.enumerate() {
                            let pos_after = pos + del_size as isize + 1;
                            let backward_mat =
                                self.backward.get(pos_after, j + 1, s).unwrap_or(&0f64);
                            let backward_del = self.backward.get(pos_after, j, s).unwrap_or(&0f64);
                            lk_total += forward
                                * (backward_mat * self.model.observe(s, x, y)
                                    + backward_del * self.model.observe(s, x, GAP));
                        }
                    }
                    slots[del_size - 2] =
                        lk_total.ln() + forward_acc + backward_acc[pos + del_size + 1];
                }
            }
        }
        lks
    }
    // Return likelihood when the `pos` base of the template is removed.
    fn with_deletion(&self, pos: usize) -> f64 {
        let states = self.model.states;
        if pos + 1 == self.template.len() {
            let tlen = self.template.len() as isize;
            let qlen = self.query.len() as isize;
            let lk: f64 = (0..states).map(|s| self.forward[(tlen - 1, qlen, s)]).sum();
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
                            .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                            .sum();
                        let backward = self.model.observe(s, x, y)
                            * self.backward[(pos + 2, j + 1, s)]
                            + self.model.observe(s, x, GAP) * self.backward[(pos + 2, j, s)];
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
                            .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                            .sum();
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos, j + 1, s)]
                            + self.model.observe(s, base, GAP) * self.backward[(pos, j, s)];
                        forward * backward
                    })
                    .sum::<f64>()
            })
            .sum();
        let forward_factor: f64 = self.forward_factor[..pos + 1].iter().map(log).sum();
        let backward_factor: f64 = self.backward_factor[pos..].iter().map(log).sum();
        lk.ln() + forward_factor + backward_factor
    }
    fn initial_distribution(&self) -> Vec<f64> {
        let mut probs: Vec<_> = (0..self.model.states)
            .map(|s| {
                // We should multiply the scaling factors,
                // but it would be cancelled out by normalizing.
                self.forward[(0, 0, s)] * self.backward[(0, 0, s)]
            })
            .collect();
        let sum = probs.iter().sum::<f64>();
        probs.iter_mut().for_each(|x| *x /= sum);
        probs
    }
    // Return [from * states + to] = Pr{from->to},
    // because it is much easy to normalize.
    // TODO: This code just sucks.
    fn transition_probability(&self) -> Vec<f64> {
        let states = self.model.states;
        // Log probability.
        let mut probs: Vec<_> = vec![0f64; self.model.states.pow(2)];
        for from in 0..states {
            for to in 0..states {
                let mut log_probs = vec![];
                for (i, &x) in self.template.iter().enumerate() {
                    let forward_factor: f64 = self.forward_factor.iter().map(log).take(i + 1).sum();
                    let backward1: f64 = self.backward_factor.iter().map(log).skip(i + 1).sum();
                    let backward2: f64 = self.backward_factor.iter().map(log).skip(i).sum();
                    for (j, &y) in self.query.iter().enumerate() {
                        let (i, j) = (i as isize, j as isize);
                        let forward = log(&self.forward[(i, j, from)]) + forward_factor;
                        let transition = log(&self.model.transition(from, to));
                        let backward_match =
                            self.model.observe(from, x, y) * self.backward[(i + 1, j + 1, to)];
                        let backward_del =
                            self.model.observe(from, x, GAP) * self.backward[(i + 1, j, to)];
                        let backward_ins =
                            self.model.observe(from, GAP, y) * self.backward[(i, j + 1, to)];
                        let backward = [
                            log(&backward_match) + backward1,
                            log(&backward_del) + backward1,
                            log(&backward_ins) + backward2,
                        ];
                        log_probs.push(forward + transition + logsumexp(&backward));
                    }
                }
                probs[from * states + to] = logsumexp(&log_probs);
            }
        }
        // Normalizing.
        // These are log-probability.
        probs.chunks_mut(states).for_each(|sums| {
            let sum = logsumexp(&sums);
            // This is normal value.
            sums.iter_mut().for_each(|x| *x = (*x - sum).exp());
            assert!((1f64 - sums.iter().sum::<f64>()) < 0.001);
        });
        probs
    }
}
impl<'a, 'b, 'c> Profile<'a, 'b, 'c, Full> {
    // [state << 6 | x | y] = Pr{(x,y)|state}
    fn observation_probability(&self) -> Vec<f64> {
        // This is log_probabilities.
        let states = self.model.states;
        let mut prob = vec![vec![]; ((states - 1) << 6) + 8 * 8];
        for (i, &x) in self.template.iter().enumerate() {
            let forward_factor: f64 = self.forward_factor.iter().take(i + 1).map(log).sum();
            let backward_factor1: f64 = self.backward_factor.iter().skip(i + 1).map(log).sum();
            let backward_factor2: f64 = self.backward_factor.iter().skip(i).map(log).sum();
            for (j, &y) in self.query.iter().enumerate() {
                let (i, j) = (i as isize, j as isize);
                for state in 0..self.model.states {
                    let back_match = self.backward[(i + 1, j + 1, state)];
                    let back_del = self.backward[(i + 1, j, state)];
                    let back_ins = self.backward[(i, j + 1, state)];
                    let (mat, del, ins) = (0..self.model.states)
                        .map(|from| {
                            let forward =
                                self.forward[(i, j, from)] * self.model.transition(from, state);
                            let mat = forward * self.model.observe(state, x, y) * back_match;
                            let del = forward * self.model.observe(state, x, GAP) * back_del;
                            let ins = forward * self.model.observe(state, GAP, y) * back_ins;
                            (mat, del, ins)
                        })
                        .fold((0f64, 0f64, 0f64), |(x, y, z), (a, b, c)| {
                            (x + a, y + b, z + c)
                        });
                    let match_log_prob = log(&mat) + forward_factor + backward_factor1;
                    let del_log_prob = log(&del) + forward_factor + backward_factor1;
                    let ins_log_prob = log(&ins) + forward_factor + backward_factor2;
                    prob[state << 6 | (x << 3 | y) as usize].push(match_log_prob);
                    prob[state << 6 | (x << 3 | GAP) as usize].push(del_log_prob);
                    prob[state << 6 | (GAP << 3 | y) as usize].push(ins_log_prob);
                }
            }
        }
        // Normalizing.
        prob.chunks_mut(8 * 8)
            .flat_map(|row| {
                // These are log prob.
                let mut sums: Vec<_> = row.iter().map(|xs| logsumexp(xs)).collect();
                let sum = logsumexp(&sums);
                if EP < sum {
                    // These are usual probabilities.
                    sums.iter_mut().for_each(|x| *x = (*x - sum).exp());
                    assert!((1f64 - sums.iter().sum::<f64>()).abs() < 0.0001);
                } else {
                    sums.iter_mut().for_each(|x| *x = 0f64);
                }
                sums
            })
            .collect()
    }
}

impl<'a, 'b, 'c> Profile<'a, 'b, 'c, Cond> {
    // [state << 6 | x | y] = Pr{(x,y)|state, x}
    fn observation_probability(&self) -> Vec<f64> {
        // This is log_probabilities.
        let states = self.model.states;
        let mut prob = vec![vec![]; ((states - 1) << 6) + 8 * 8];
        for (i, &x) in self.template.iter().enumerate() {
            let forward_factor: f64 = self.forward_factor.iter().take(i + 1).map(log).sum();
            let backward_factor1: f64 = self.backward_factor.iter().skip(i + 1).map(log).sum();
            let backward_factor2: f64 = self.backward_factor.iter().skip(i).map(log).sum();
            for (j, &y) in self.query.iter().enumerate() {
                let (i, j) = (i as isize, j as isize);
                for state in 0..self.model.states {
                    let back_match = self.backward[(i + 1, j + 1, state)];
                    let back_del = self.backward[(i + 1, j, state)];
                    let back_ins = self.backward[(i, j + 1, state)];
                    let (mat, del, ins) = (0..self.model.states)
                        .map(|from| {
                            let forward =
                                self.forward[(i, j, from)] * self.model.transition(from, state);
                            let mat = forward * self.model.observe(state, x, y) * back_match;
                            let del = forward * self.model.observe(state, x, GAP) * back_del;
                            let ins = forward * self.model.observe(state, GAP, y) * back_ins;
                            (mat, del, ins)
                        })
                        .fold((0f64, 0f64, 0f64), |(x, y, z), (a, b, c)| {
                            (x + a, y + b, z + c)
                        });
                    let match_log_prob = log(&mat) + forward_factor + backward_factor1;
                    let del_log_prob = log(&del) + forward_factor + backward_factor1;
                    let ins_log_prob = log(&ins) + forward_factor + backward_factor2;
                    prob[state << 6 | (x << 3 | y) as usize].push(match_log_prob);
                    prob[state << 6 | (x << 3 | GAP) as usize].push(del_log_prob);
                    prob[state << 6 | (GAP << 3 | y) as usize].push(ins_log_prob);
                }
            }
        }
        // Normalizing.
        prob.chunks_mut(8)
            .flat_map(|row| {
                // These are log prob.
                let mut sums: Vec<_> = row.iter().map(|xs| logsumexp(xs)).collect();
                let sum = logsumexp(&sums);
                if EP < sum {
                    // These are usual probabilities. (Sometime is is for a phony row,
                    // not corresponding to any base or gap).
                    sums.iter_mut().for_each(|x| *x = (*x - sum).exp());
                } else {
                    sums.iter_mut().for_each(|x| *x = 0f64);
                }
                sums
            })
            .collect()
    }
}

#[cfg(test)]
mod gphmm {
    // TODO:Write more tests for conditional hidden Markov models.
    use super::*;
    use crate::hmm::Op;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128PlusPlus;
    #[test]
    fn default_test() {
        GPHMM::<FullHiddenMarkovModel>::default();
        GPHMM::<ConditionalHiddenMarkovModel>::default();
    }
    #[test]
    fn three_state_test() {
        GPHMM::<Full>::new_three_state(0.9, 0.1, 0.2, 0.9);
        GPHMM::<Cond>::new_three_state(0.9, 0.1, 0.2, 0.9);
    }
    #[test]
    fn align_test() {
        let phmm: GPHMM<FullHiddenMarkovModel> = GPHMM::default();
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
            let phmm: GPHMM<FullHiddenMarkovModel> = GPHMM::default();
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let _ = phmm.align(&xs, &ys);
            let phmm = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
            let _ = phmm.align(&xs, &ys);
            let phmm = GPHMM::<Cond>::new_three_state(0.8, 0.2, 0.3, 0.9);
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
            let phmm = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
            let (lk, _, _) = phmm.align(&xs, &ys);
            let lkt = phmm.likelihood_naive(&xs, &ys);
            assert!(lk < lkt);
            let phmm = GPHMM::<Cond>::new_three_state(0.8, 0.2, 0.3, 0.9);
            let (lk, _, _) = phmm.align(&xs, &ys);
            let lkt = phmm.likelihood_naive(&xs, &ys);
            assert!(lk < lkt);
        }
    }
    #[test]
    fn likelihood_scaling_test() {
        let phmm = GPHMM::<Full>::default();
        let chmm = GPHMM::<Full>::default();
        let xs = b"ATGC";
        let lkn = phmm.likelihood_naive(xs, xs);
        let lks = phmm.likelihood(xs, xs);
        println!("{}", phmm.align(xs, xs).0);
        assert!((lkn - lks).abs() < 0.001, "{},{}", lkn, lks);
        assert!((chmm.likelihood_naive(xs, xs) - chmm.likelihood(xs, xs)).abs() < 0.001);
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 100;
            let profile = &crate::gen_seq::PROFILE;
            let xs = crate::gen_seq::generate_seq(&mut rng, len);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &profile);
            let phmm = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
            let lkn = phmm.likelihood_naive(&xs, &ys);
            let lks = phmm.likelihood(&xs, &ys);
            assert!((lkn - lks).abs() < 0.001, "{},{}", lkn, lks);
            let dpn = phmm.forward_naive(&xs, &ys);
            let (xs, ys) = (PadSeq::new(xs.as_slice()), PadSeq::new(ys.as_slice()));
            let (dps, factors) = phmm.forward(&xs, &ys);
            for i in 0..xs.len() + 1 {
                for j in 0..ys.len() + 1 {
                    for s in 0..phmm.states {
                        let naive = dpn[(i as isize, j as isize, s)];
                        let scaled = log(&dps[(i as isize, j as isize, s)])
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
        let phmm: GPHMM<Full> = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
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
                .map(|(s, init)| dp[(0, 0, s)] + log(init))
                .collect();
            let lkb = logsumexp(&lkbs);
            assert!((lkn - lkb).abs() < 0.01, "{},{}", lkn, lkb);
        }
    }
    #[test]
    fn likelihood_backward_scaling_test() {
        let phmm: GPHMM<Full> = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
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
                .map(|(s, init)| dp[(0, 0, s)] * init)
                .sum::<f64>()
                .ln()
                + factors.iter().map(log).sum::<f64>();
            assert!((lk - lkb).abs() < 0.01, "{},{}", lk, lkb);
            println!("{},{}", xs.len(), ys.len());
            for i in 0..xs.len() + 1 {
                for j in 0..ys.len() + 1 {
                    for s in 0..phmm.states {
                        let naive = dpn[(i as isize, j as isize, s)];
                        let scaled = log(&dp[(i as isize, j as isize, s)])
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
        let phmm: GPHMM<Full> = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
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
        let phmm: GPHMM<Full> = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
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
        let phmm: GPHMM<Full> = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
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
        let phmm = GPHMM::<Full>::new_three_state(0.8, 0.2, 0.3, 0.9);
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
        let phmm: GPHMM<Cond> = GPHMM::<Cond>::new_three_state(0.8, 0.2, 0.3, 0.9);
        let coverage = 30;
        for i in 0..2u64 {
            let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let len = 100;
            let profile = &crate::gen_seq::PROFILE;
            let template = crate::gen_seq::generate_seq(&mut rng, len);
            let queries: Vec<_> = (0..coverage)
                .map(|_| crate::gen_seq::introduce_randomness(&template, &mut rng, &profile))
                .collect();
            let draft = crate::gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
            let polished = phmm.correct_until_convergence(&draft, &queries);
            let dist = crate::bialignment::edit_dist(&polished, &template);
            let dist_old = crate::bialignment::edit_dist(&draft, &template);
            assert!(dist < dist_old, "{},{}", dist, dist_old);
        }
    }
}
