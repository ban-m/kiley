//! This module defines generalized pair hidden Markov models.
//! In contrast to the usual pair hidden Markov model, where each state
//! should output either a match, a deletion, or an insertion,
//! a GpHMM can output any of them.
//! However, the usual algorithm such as Viterbi, Forward, Backward, and Forward-Backward algorithm could be run on a GpHMM in the
//! same time with respect to the O-notation. Specifically, all of them run in O(|X||Y||S|^2) time, where X and Y are the reference and the query, and S is the states.
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
/// To specify the mode of the pair HMM(Full or conditional), `GPHMM::<Full>::` or `GPHMM::<Cond>::` would be OK.
#[derive(Clone, Debug)]
pub struct GeneralizedPairHiddenMarkovModel<T: HMMType> {
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
    // Mode.
    _mode: std::marker::PhantomData<T>,
}

// TODO: remove `if` statements as many as possible.
// - let j_orig = foo(); if !(0..ys.len() as isize).contains(j_orig){ --- } should be removed, removed.

/// Please Do not implement this trait by your own program.
pub trait HMMType: Clone {}

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
                .map(|x| format!("{:.2}", x))
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
    pub fn fit_banded<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
        radius: usize,
    ) -> Self {
        let template = PadSeq::new(template);
        let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
        self.fit_banded_inner(&template, &queries, radius)
    }
    pub fn fit_banded_inner(&self, xs: &PadSeq, yss: &[PadSeq], radius: usize) -> Self {
        let radius = radius as isize;
        let profiles: Vec<_> = yss
            .iter()
            .filter_map(|ys| ProfileBanded::new(self, xs, ys, radius))
            .collect();
        let initial_distribution = self.estimate_initial_distribution_banded(&profiles);
        let transition_matrix = self.estimate_transition_prob_banded(&profiles);
        let observation_matrix = self.estimate_observation_prob_banded(&profiles);
        Self {
            states: self.states,
            initial_distribution,
            transition_matrix,
            observation_matrix,
            _mode: std::marker::PhantomData,
        }
    }
    fn estimate_observation_prob_banded(&self, profiles: &[ProfileBanded<Cond>]) -> Vec<f64> {
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
    /// Banded version of `align` method. Return None if the band width is too small to
    /// fail to detect a band from (0,0) to (xs.len(),ys.len()).
    pub fn align_banded(
        &self,
        xs: &[u8],
        ys: &[u8],
        radius: usize,
    ) -> Option<(f64, Vec<Op>, Vec<usize>)> {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        self.align_banded_inner(&xs, &ys, radius)
    }
    /// Banded version of `align_inner` method. Return Node if the band width is too small.
    pub fn align_banded_inner(
        &self,
        xs: &PadSeq,
        ys: &PadSeq,
        radius: usize,
    ) -> Option<(f64, Vec<Op>, Vec<usize>)> {
        if ys.len() < 2 * radius {
            return Some(self.align_inner(xs, ys));
        }
        let mut dp = DPTable::new(xs.len() + 1, 2 * radius + 1, self.states, EP);
        // The center of the i-th row.
        // In other words, it is where the `radius`-th element of in the i-th row corresponds in the
        // original DP table.
        // Note: To convert the j-th position at the DP table into the original coordinate,
        // j+center-radius would work.
        let mut centers = vec![0];
        let log_transit = self.get_log_transition();
        let log_observe = self.get_log_observation();
        let radius = radius as isize;
        // (0,0,s) is at (0, radius, s)
        for s in 0..self.states {
            dp[(0, radius, s as isize)] = log(&self.initial_distribution[s]);
        }
        // Initial values.
        // j_orig ranges in 1..radius
        for j in radius + 1..2 * radius + 1 {
            let j_orig = j - radius;
            let y = ys[j_orig - 1];
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
        centers.push(0);
        // Fill DP cells.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            assert_eq!(centers.len(), i + 1);
            let (mut max_j, mut max_lk) = (0, std::f64::NEG_INFINITY);
            let (center, prev) = (centers[i], centers[i - 1]);
            for j in 0..2 * radius as isize + 1 {
                let j_in_ys = j + center - radius;
                // The position of j_in_ys in the (i-1)-th row.
                let j_prev = j + center - prev;
                if !(0..ys.len() as isize + 1).contains(&j_in_ys) {
                    continue;
                }
                // If j_in_ys ==0, it would be -1, but it is OK. PadSeq just returns the NULL base for such access.
                let y = ys[j_in_ys - 1];
                for s in 0..self.states {
                    let i = i as isize;
                    let max_path = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i - 1, j_prev - 1, t as isize)]
                                + log_transit[t][s]
                                + log_observe[s][(x << 3 | y) as usize];
                            let del = dp[(i - 1, j_prev, t as isize)]
                                + log_transit[t][s]
                                + log_observe[s][(x << 3 | GAP) as usize];
                            // If j_in_ys == 0, this code should be skipped, but it is OK, DPTable would forgive "out-of-bound" access.
                            let ins = dp[(i, j - 1, t as isize)]
                                + log_transit[t][s]
                                + log_observe[s][(GAP << 3 | y) as usize];
                            mat.max(del).max(ins)
                        })
                        .fold(std::f64::NEG_INFINITY, |x, y| x.max(y));
                    dp[(i, j, s as isize)] = max_path;
                    if max_lk < max_path {
                        max_lk = max_path;
                        max_j = j;
                    }
                }
            }
            // Update centers.
            let next_center = if max_j < radius { center } else { center + 2 };
            centers.push(next_center);
        }
        // Debugging...
        // let mut lines: Vec<_> = vec![vec!['X'; ys.len() + 1]; xs.len() + 1];
        // for i in 0..xs.len() + 1 {
        //     let center = centers[i];
        //     // Find max j.
        //     let (_, max_j) = (0..2 * radius as isize + 1)
        //         .map(|j| {
        //             let max = (0..self.states as isize)
        //                 .map(|s| (dp[(i as isize, j, s)]))
        //                 .fold(std::f64::NEG_INFINITY, |x, y| x.max(y));
        //             (max, j)
        //         })
        //         .max_by(|x, y| (x.0).partial_cmp(&(y.0)).unwrap())
        //         .unwrap();
        //     for j in 0..2 * radius + 1 {
        //         let j_orig = j + center - radius;
        //         if (0..ys.len() as isize + 1).contains(&j_orig) {
        //             lines[i][j_orig as usize] = 'O';
        //             if i == j_orig as usize {
        //                 lines[i][j_orig as usize] = 'T';
        //             }
        //         }
        //     }
        //     lines[i][(max_j + centers[i] - radius) as usize] = 'M';
        // }
        // for line in lines.iter() {
        //     let line: String = line.iter().copied().collect();
        // }
        // Check if we have reached the terminal, (xs.len(),ys.len());
        let mut i = xs.len() as isize;
        let mut j_orig = ys.len() as isize;
        if !(0..2 * radius + 1).contains(&(j_orig + radius - centers[xs.len()])) {
            // The (xs.len(), ys.len()) is outside of the band.
            return None;
        }
        // Traceback.
        let (max_lk, mut state) = {
            let j = j_orig + radius - centers[xs.len()];
            (0..self.states)
                .map(|s| (dp[(i, j, s as isize)], s as isize))
                .max_by(|x, y| (x.0).partial_cmp(&(y.0)).unwrap())
                .unwrap()
        };
        let (mut ops, mut states) = (vec![], vec![state]);
        while 0 < i && 0 < j_orig {
            let (center, prev) = (centers[i as usize], centers[i as usize - 1]);
            let j = j_orig + radius - center;
            let j_prev = j + center - prev;
            let current = dp[(i, j, state)];
            let (x, y) = (xs[i - 1], ys[j_orig - 1]);
            let (op, new_state) = (0..self.states)
                .find_map(|t| {
                    let mat = dp[(i - 1, j_prev - 1, t as isize)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | y) as usize];
                    let del = dp[(i - 1, j_prev, t as isize)]
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
                    j_orig -= 1;
                    i -= 1;
                }
                Op::Del => i -= 1,
                Op::Ins => j_orig -= 1,
            }
            states.push(state);
            ops.push(op);
        }
        while 0 < i {
            let (center, prev) = (centers[i as usize], centers[i as usize - 1]);
            let j = j_orig + radius - center;
            let j_prev = j + center - prev;
            let current = dp[(i, j, state)];
            let x = xs[i - 1];
            let new_state = (0..self.states)
                .find_map(|t| {
                    let del = dp[(i - 1, j_prev, t as isize)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | GAP) as usize];
                    ((current - del).abs() < 0.001).then(|| t as isize)
                })
                .unwrap();
            state = new_state;
            states.push(state);
            ops.push(Op::Del);
            i -= 1;
        }
        assert_eq!(i, 0);
        while 0 < j_orig {
            assert_eq!(centers[0], 0);
            let j = j_orig + radius;
            let current = dp[(0, j, state)];
            let y = ys[j_orig - 1];
            let new_state = (0..self.states)
                .find_map(|t| {
                    let ins = dp[(0, j - 1, t as isize)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(GAP << 3 | y) as usize];
                    ((current - ins).abs() < 0.0001).then(|| t as isize)
                })
                .unwrap();
            state = new_state;
            states.push(state);
            ops.push(Op::Ins);
            j_orig -= 1;
        }
        let states: Vec<_> = states.iter().rev().map(|&x| x as usize).collect();
        ops.reverse();
        Some((max_lk, ops, states))
    }
    /// same as likelihood. A naive log-sum-exp implementation.
    pub fn likelihood_naive(&self, xs: &[u8], ys: &[u8]) -> f64 {
        let dp = self.forward_naive(xs, ys);
        let (n, m) = (xs.len() as isize, ys.len() as isize);
        let lks: Vec<_> = (0..self.states).map(|s| dp[(n, m, s as isize)]).collect();
        logsumexp(&lks)
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
                dp[(i, 0, s as isize)] = logsumexp(&lks);
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
                dp[(0, j, s as isize)] = logsumexp(&lks);
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
                dp[(i as isize, ys.len() as isize, s as isize)] = logsumexp(&lks);
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
                dp[(xs.len() as isize, j as isize, s as isize)] = logsumexp(&lks);
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
                            log_transit[s][t] + logsumexp(&[mat, del, ins])
                        })
                        .collect();
                    dp[(i as isize, j as isize, s as isize)] = logsumexp(&lks);
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
        let second_total = dp.total(xs.len());
        dp.div(xs.len(), second_total);
        norm_factors.push(first_total * second_total);
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
    pub fn likelihood_banded(&self, xs: &[u8], ys: &[u8], radius: usize) -> Option<f64> {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        self.likelihood_banded_inner(&xs, &ys, radius)
    }
    pub fn likelihood_banded_inner(&self, xs: &PadSeq, ys: &PadSeq, radius: usize) -> Option<f64> {
        let radius = radius as isize;
        let (dp, alphas, centers) = self.forward_banded(xs, ys, radius)?;
        let normalized_factor: f64 = alphas.iter().map(|x| x.ln()).sum();
        let center = centers[xs.len()];
        let n = xs.len() as isize;
        let m = ys.len() as isize - center + radius;
        (0..2 * radius + 1).contains(&m).then(|| {
            let lk: f64 = (0..self.states).map(|s| dp[(n, m, s as isize)]).sum();
            lk.ln() + normalized_factor
        })
    }
    // Forward algorithm for banded mode.
    fn forward_banded(
        &self,
        xs: &PadSeq,
        ys: &PadSeq,
        radius: isize,
    ) -> Option<(DPTable, Vec<f64>, Vec<isize>)> {
        let mut dp = DPTable::new(xs.len() + 1, 2 * radius as usize + 1, self.states, 0f64);
        let mut norm_factors = vec![];
        // The location where the radius-th element is in the original DP table.
        // In other words, if you want to convert the j-th element in the banded DP table into the original coordinate,
        // j + centers[i] - radius would be oK.
        // Inverse convertion is the sam, j_orig + radius - centers[i] would be OK.
        let mut centers = vec![0, 0];
        // Initialize.
        for (s, &x) in self.initial_distribution.iter().enumerate() {
            dp[(0, radius, s as isize)] = x;
        }
        for j in radius + 1..2 * radius + 1 {
            let y = ys[j - radius - 1];
            for s in 0..self.states {
                let trans: f64 = (0..self.states)
                    .map(|t| dp[(0, j - 1, t as isize)] * self.transition(t, s))
                    .sum();
                dp[(0, j, s as isize)] += trans * self.observe(s, GAP, y);
            }
        }
        // Normalize.
        let total = dp.total(0);
        norm_factors.push(1f64 * total);
        dp.div(0, total);
        // Fill DP cells.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            assert_eq!(centers.len(), i + 1);
            let (center, prev) = (centers[i], centers[i - 1]);
            // Deletion and Match transitions.
            // Maybe we should treat the case when j_orig == 0, but it is OK. DPTable would treat such cases.
            for j in 0..2 * radius + 1 {
                let j_orig = j + center - radius;
                if !(0..ys.len() as isize + 1).contains(&j_orig) {
                    continue;
                }
                let y = ys[j_orig - 1];
                let prev_j = j + center - prev;
                for s in 0..self.states {
                    dp[(i as isize, j as isize, s as isize)] = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i as isize - 1, prev_j - 1, t as isize)];
                            let del = dp[(i as isize - 1, prev_j, t as isize)];
                            (mat * self.observe(s, x, y) + del * self.observe(s, x, GAP))
                                * self.transition(t, s)
                        })
                        .sum::<f64>();
                }
            }
            let first_total = dp.total(i);
            dp.div(i, first_total);
            // Insertion transitions
            for j in 0..2 * radius + 1 {
                let j_orig = j + center - radius;
                if !(0..ys.len() as isize + 1).contains(&j_orig) {
                    continue;
                }
                let y = ys[j_orig - 1];
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
            // Find maximum position.
            let (_, max_j) = (0..2 * radius + 1)
                .map(|j| {
                    let max = (0..self.states as isize)
                        .map(|s| dp[(i as isize, j, s)])
                        .fold(std::f64::NEG_INFINITY, |x, y| x.max(y));
                    (max, j)
                })
                .max_by(|x, y| (x.0).partial_cmp(&(y.0)).unwrap())
                .unwrap();
            let next_center = if max_j < radius { center } else { center + 2 };
            centers.push(next_center);
        }
        let m = ys.len() as isize - centers[xs.len()] + radius;
        (0..2 * radius + 1)
            .contains(&m)
            .then(|| (dp, norm_factors, centers))
    }
    fn backward_banded(
        &self,
        xs: &PadSeq,
        ys: &PadSeq,
        radius: isize,
        centers: &[isize],
    ) -> (DPTable, Vec<f64>) {
        assert_eq!(centers.len(), xs.len() + 2);
        let mut dp = DPTable::new(xs.len() + 1, 2 * radius as usize + 1, self.states, 0f64);
        let mut norm_factors = vec![];
        // Initialize
        let (xslen, yslen) = (xs.len() as isize, ys.len() as isize);
        for s in 0..self.states as isize {
            let j = yslen - centers[xs.len()] + radius;
            dp[(xslen, j, s)] = 1f64;
        }
        let first_total = dp.total(xs.len());
        dp.div(xs.len(), first_total);
        for j in (0..2 * radius + 1).rev() {
            let j_orig = j + centers[xs.len()] - radius;
            if !(0..ys.len() as isize).contains(&j_orig) {
                continue;
            }
            let y = ys[j_orig];
            for s in 0..self.states {
                dp[(xslen, j, s as isize)] += (0..self.states)
                    .map(|t| {
                        self.transition(s, t)
                            * self.observe(t, GAP, y)
                            * dp[(xslen, j + 1, t as isize)]
                    })
                    .sum::<f64>();
            }
        }
        let second_total = dp.total(xs.len());
        dp.div(xs.len(), second_total);
        norm_factors.push(first_total * second_total);
        for (i, &x) in xs.iter().enumerate().rev() {
            let (center, next) = (centers[i], centers[i + 1]);
            // Deletion transition to below and match transition into diagonal.
            for j in (0..2 * radius + 1).rev() {
                let j_orig = j + center - radius;
                if !(0..ys.len() as isize + 1).contains(&j_orig) {
                    continue;
                }
                // You may think this would out-of-bound, but it never does.
                // this is beacuse ys is `padded` so that any out-of-bound accessing would become NULL base.
                let y = ys[j_orig];
                let i = i as isize;
                let j_next = j + center - next;
                for s in 0..self.states {
                    dp[(i, j, s as isize)] = (0..self.states)
                        .map(|t| {
                            self.transition(s, t)
                                * (self.observe(t, x, y) * dp[(i + 1, j_next + 1, t as isize)]
                                    + self.observe(t, x, GAP) * dp[(i + 1, j_next, t as isize)])
                        })
                        .sum::<f64>();
                }
            }
            let first_total = dp.total(i);
            dp.div(i, first_total);
            for j in (0..2 * radius + 1).rev() {
                let j_orig = j + center - radius;
                if !(0..ys.len() as isize + 1).contains(&j_orig) {
                    continue;
                }
                let y = ys[j_orig];
                for s in 0..self.states {
                    let i = i as isize;
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
    pub fn correct_until_convergence_banded<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
        radius: usize,
    ) -> Option<Vec<u8>> {
        let mut template = PadSeq::new(template);
        let mut start_position = 0;
        let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
        while let Some((seq, next)) =
            self.correction_inner_banded(&template, &queries, radius, start_position)
        {
            template = seq;
            start_position = next;
        }
        Some(template.into())
    }
    pub fn correction_inner_banded(
        &self,
        template: &PadSeq,
        queries: &[PadSeq],
        radius: usize,
        start_position: usize,
    ) -> Option<(PadSeq, usize)> {
        let radius = radius as isize;
        let profiles: Vec<_> = queries
            .iter()
            .filter_map(|q| ProfileBanded::new(self, &template, q, radius))
            .collect();
        if profiles.is_empty() {
            return None;
        }
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
        // Transpose. [to * states + from] = Pr{from->to}.
        (0..states * states)
            .map(|idx| {
                let (to, from) = (idx / states, idx % states);
                buffer[from * states + to]
            })
            .collect()
    }
    fn estimate_initial_distribution_banded(&self, profiles: &[ProfileBanded<M>]) -> Vec<f64> {
        let mut buffer = vec![0f64; self.states];
        for init in profiles.iter().map(|prf| prf.initial_distribution()) {
            buffer.iter_mut().zip(init).for_each(|(x, y)| *x += y);
        }
        let sum: f64 = buffer.iter().sum();
        buffer.iter_mut().for_each(|x| *x /= sum);
        buffer
    }
    fn estimate_transition_prob_banded(&self, profiles: &[ProfileBanded<M>]) -> Vec<f64> {
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
        // Transpose. [to * states + from] = Pr{from->to}.
        (0..states * states)
            .map(|idx| {
                let (to, from) = (idx / states, idx % states);
                buffer[from * states + to]
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct Profile<'a, 'b, 'c, T: HMMType> {
    template: &'a PadSeq,
    query: &'b PadSeq,
    model: &'c GPHMM<T>,
    forward: DPTable,
    forward_factor: Vec<f64>,
    backward: DPTable,
    backward_factor: Vec<f64>,
}

impl<'a, 'b, 'c, T: HMMType> Profile<'a, 'b, 'c, T> {
    fn new(model: &'c GPHMM<T>, template: &'a PadSeq, query: &'b PadSeq) -> Self {
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
        let state = self.model.states as isize;
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
    fn initial_distribution(&self) -> Vec<f64> {
        let mut probs: Vec<_> = (0..self.model.states as isize)
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
        let mut probs: Vec<_> = vec![vec![]; self.model.states.pow(2)];
        for (i, &x) in self.template.iter().enumerate() {
            let forward_factor: f64 = self.forward_factor.iter().map(log).take(i + 1).sum();
            let backward1: f64 = self.backward_factor.iter().map(log).skip(i + 1).sum();
            let backward2: f64 = self.backward_factor.iter().map(log).skip(i).sum();
            for (j, &y) in self.query.iter().enumerate() {
                let (i, j) = (i as isize, j as isize);
                for from in 0..states {
                    for to in 0..states {
                        let forward = log(&self.forward[(i, j, from as isize)]) + forward_factor;
                        let transition = log(&self.model.transition(from, to));
                        let backward_match = self.model.observe(from, x, y)
                            * self.backward[(i + 1, j + 1, to as isize)];
                        let backward_del = self.model.observe(from, x, GAP)
                            * self.backward[(i + 1, j, to as isize)];
                        let backward_ins = self.model.observe(from, GAP, y)
                            * self.backward[(i, j + 1, to as isize)];
                        let backward = [
                            log(&backward_match) + backward1,
                            log(&backward_del) + backward1,
                            log(&backward_ins) + backward2,
                        ];
                        probs[from * states + to].push(forward + transition + logsumexp(&backward));
                    }
                }
            }
        }
        // Normalizing.
        probs
            .chunks_mut(states)
            .flat_map(|row| {
                // These are log-probability.
                let mut sums: Vec<_> = row.iter().map(|xs| logsumexp(&xs)).collect();
                // This is also log.
                let sum = logsumexp(&sums);
                // This is normal value.
                sums.iter_mut().for_each(|x| *x = (*x - sum).exp());
                assert!((1f64 - sums.iter().sum::<f64>()) < 0.001);
                sums
            })
            .collect()
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
                    let back_match = self.backward[(i + 1, j + 1, state as isize)];
                    let back_del = self.backward[(i + 1, j, state as isize)];
                    let back_ins = self.backward[(i, j + 1, state as isize)];
                    let (mat, del, ins) = (0..self.model.states)
                        .map(|from| {
                            let forward = self.forward[(i, j, from as isize)]
                                * self.model.transition(from, state);
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
                    let back_match = self.backward[(i + 1, j + 1, state as isize)];
                    let back_del = self.backward[(i + 1, j, state as isize)];
                    let back_ins = self.backward[(i, j + 1, state as isize)];
                    let (mat, del, ins) = (0..self.model.states)
                        .map(|from| {
                            let forward = self.forward[(i, j, from as isize)]
                                * self.model.transition(from, state);
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

#[derive(Debug, Clone)]
struct ProfileBanded<'a, 'b, 'c, T: HMMType> {
    template: &'a PadSeq,
    query: &'b PadSeq,
    model: &'c GPHMM<T>,
    forward: DPTable,
    forward_factor: Vec<f64>,
    backward: DPTable,
    backward_factor: Vec<f64>,
    centers: Vec<isize>,
    radius: isize,
}

impl<'a, 'b, 'c, T: HMMType> ProfileBanded<'a, 'b, 'c, T> {
    fn new(
        model: &'c GPHMM<T>,
        template: &'a PadSeq,
        query: &'b PadSeq,
        radius: isize,
    ) -> Option<Self> {
        let (forward, forward_factor, centers) = model.forward_banded(template, query, radius)?;
        let (backward, backward_factor) = model.backward_banded(template, query, radius, &centers);
        if backward_factor.iter().any(|x| x.is_nan()) {
            eprintln!("{},{}", template.len(), query.len());
            eprintln!("{:?}", centers);
            eprintln!("{:?}", forward_factor);
            panic!(
                "{:?}\n{}\n{}",
                backward_factor,
                String::from_utf8(template.clone().into()).unwrap(),
                String::from_utf8(query.clone().into()).unwrap(),
            );
        }
        Some(Self {
            template,
            query,
            forward,
            forward_factor,
            backward,
            backward_factor,
            centers,
            radius,
            model,
        })
    }
    fn lk(&self) -> f64 {
        let n = self.template.len();
        let m = self.query.len() as isize - self.centers[n] + self.radius;
        let states = self.model.states as isize;
        let (n, m) = (n as isize, m as isize);
        let lk: f64 = (0..states).map(|s| self.forward[(n, m, s)]).sum();
        let factor: f64 = self.forward_factor.iter().map(log).sum();
        lk.ln() + factor
    }
    fn with_mutation(&self, pos: usize, base: u8) -> f64 {
        let states = self.model.states;
        let (center, next) = (self.centers[pos], self.centers[pos + 1]);
        let lk = (0..2 * self.radius + 1)
            .map(|j| {
                let pos = pos as isize;
                let j_orig = j + center - self.radius;
                if !(0..self.query.len() as isize + 1).contains(&j_orig) {
                    return 0f64;
                }
                let y = self.query[j_orig];
                let j_next = j + center - next;
                (0..states)
                    .map(|s| {
                        let forward: f64 = (0..states)
                            .map(|t| {
                                self.forward[(pos, j, t as isize)] * self.model.transition(t, s)
                            })
                            .sum();
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos + 1, j_next + 1, s as isize)]
                            + self.model.observe(s, base, GAP)
                                * self.backward[(pos + 1, j_next, s as isize)];
                        forward * backward
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        let forward_factor: f64 = self.forward_factor[..pos + 1].iter().map(|x| x.ln()).sum();
        let backward_factor: f64 = self.backward_factor[pos + 1..].iter().map(|x| x.ln()).sum();
        lk.ln() + forward_factor + backward_factor
    }
    fn with_deletion(&self, pos: usize) -> f64 {
        let states = self.model.states;
        let center = self.centers[pos];
        if pos + 1 == self.template.len() {
            let j = self.query.len() as isize - center + self.radius;
            let lk: f64 = (0..states as isize)
                .map(|s| self.forward[(pos as isize, j, s)])
                .sum();
            let factor: f64 = self.forward_factor[..pos + 1].iter().map(log).sum();
            return lk.ln() + factor;
        }
        let lk: f64 = (0..2 * self.radius + 1)
            .map(|j| {
                let j_orig = j + center - self.radius;
                if !(0..self.query.len() as isize + 1).contains(&j_orig) {
                    return 0f64;
                }
                let next = self.centers[pos + 2];
                let pos = pos as isize;
                let x = self.template[pos + 1];
                let y = self.query[j_orig];
                let j_next = j + center - next;
                (0..states)
                    .map(|s| {
                        let forward: f64 = (0..states)
                            .map(|t| {
                                self.forward[(pos, j, t as isize)] * self.model.transition(t, s)
                            })
                            .sum();
                        // This need to be `.get` method, as j_next might be out of range.
                        let backward_mat = self.model.observe(s, x, y)
                            * self
                                .backward
                                .get(pos + 2, j_next + 1, s as isize)
                                .unwrap_or(&0f64);
                        let backward_del = self.model.observe(s, x, GAP)
                            * self
                                .backward
                                .get(pos + 2, j_next, s as isize)
                                .unwrap_or(&0f64);
                        forward * (backward_mat + backward_del)
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
        let center = self.centers[pos];
        let lk: f64 = (0..2 * self.radius + 1)
            .map(|j| {
                let j_orig = j + center - self.radius;
                if !(0..self.query.len() as isize + 1).contains(&j_orig) {
                    return 0f64;
                }
                let y = self.query[j_orig];
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
    fn initial_distribution(&self) -> Vec<f64> {
        let mut probs: Vec<_> = (0..self.model.states as isize)
            .map(|s| {
                // We should multiply the scaling factors,
                // but it would be cancelled out by normalizing.
                self.forward[(0, self.radius, s)] * self.backward[(0, self.radius, s)]
            })
            .collect();
        let sum = probs.iter().sum::<f64>();
        probs.iter_mut().for_each(|x| *x /= sum);
        probs
    }
    // Return [from * states + to] = Pr{from->to},
    // because it is much easy to normalize.
    // TODO: This code just sucks. If requries a lot of memory, right?
    fn transition_probability(&self) -> Vec<f64> {
        let states = self.model.states;
        // Log probability.
        let mut probs: Vec<_> = vec![vec![]; self.model.states.pow(2)];
        for (i, &x) in self.template.iter().enumerate() {
            let (center, next) = (self.centers[i], self.centers[i + 1]);
            let forward_factor: f64 = self.forward_factor.iter().map(log).take(i + 1).sum();
            let backward1: f64 = self.backward_factor.iter().map(log).skip(i + 1).sum();
            let backward2: f64 = self.backward_factor.iter().map(log).skip(i).sum();
            for j in 0..2 * self.radius + 1 {
                let j_orig = j + center - self.radius;
                if !(0..self.query.len() as isize + 1).contains(&j_orig) {
                    continue;
                }
                let y = self.query[j_orig];
                let i = i as isize;
                let j_next = j + center - next;
                for from in 0..states {
                    for to in 0..states {
                        let forward = log(&self.forward[(i, j, from as isize)]) + forward_factor;
                        let transition = log(&self.model.transition(from, to));
                        let backward_match = self.model.observe(from, x, y)
                            * self.backward[(i + 1, j_next + 1, to as isize)];
                        let backward_del = self.model.observe(from, x, GAP)
                            * self.backward[(i + 1, j_next, to as isize)];
                        let backward_ins = self.model.observe(from, GAP, y)
                            * self.backward[(i, j + 1, to as isize)];
                        let backward = [
                            log(&backward_match) + backward1,
                            log(&backward_del) + backward1,
                            log(&backward_ins) + backward2,
                        ];
                        probs[from * states + to].push(forward + transition + logsumexp(&backward));
                    }
                }
            }
        }
        // Normalizing.
        probs
            .chunks_mut(states)
            .flat_map(|row| {
                // These are log-probability.
                let mut sums: Vec<_> = row.iter().map(|xs| logsumexp(&xs)).collect();
                // This is also log.
                let sum = logsumexp(&sums);
                // This is normal value.
                sums.iter_mut().for_each(|x| *x = (*x - sum).exp());
                assert!((1f64 - sums.iter().sum::<f64>()) < 0.001);
                sums
            })
            .collect()
    }
}

// impl<'a, 'b, 'c> ProfileBanded<'a, 'b, 'c, Full> {
//     fn observation_probability(&self) -> Vec<f64> {
//         unimplemented!()
//     }
// }

impl<'a, 'b, 'c> ProfileBanded<'a, 'b, 'c, Cond> {
    // [state << 6 | x | y] = Pr{(x,y)|state}
    fn observation_probability(&self) -> Vec<f64> {
        // This is log_probabilities.
        let states = self.model.states;
        let mut prob = vec![vec![]; ((states - 1) << 6) + 8 * 8];
        for (i, &x) in self.template.iter().enumerate() {
            let (center, next) = (self.centers[i], self.centers[i + 1]);
            let forward_factor: f64 = self.forward_factor.iter().take(i + 1).map(log).sum();
            let backward_factor1: f64 = self.backward_factor.iter().skip(i + 1).map(log).sum();
            let backward_factor2: f64 = self.backward_factor.iter().skip(i).map(log).sum();
            for j in 0..2 * self.radius + 1 {
                let j_orig = j + center - self.radius;
                if !(0..self.query.len() as isize + 1).contains(&j_orig) {
                    continue;
                }
                let y = self.query[j_orig];
                let i = i as isize;
                let j_next = j + center - next;
                for state in 0..self.model.states {
                    let back_match = self.backward[(i + 1, j_next + 1, state as isize)];
                    let back_del = self.backward[(i + 1, j_next, state as isize)];
                    let back_ins = self.backward[(i, j + 1, state as isize)];
                    let (mat, del, ins) = (0..self.model.states)
                        .map(|from| {
                            let forward = self.forward[(i, j, from as isize)]
                                * self.model.transition(from, state);
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
                .map(|(s, init)| dp[(0, 0, s as isize)] + log(init))
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
    fn align_banded_check<T: HMMType>(model: &GPHMM<T>, xs: &[u8], ys: &[u8], radius: usize) {
        let (lk, ops, _states) = model.align(&xs, &ys);
        let (lk_b, ops_b, _states_b) = model.align_banded(&xs, &ys, radius).unwrap();
        let diff = (lk - lk_b).abs();
        assert!(diff < 0.0001, "{},{}\n{:?}\n{:?}", lk, lk_b, ops, ops_b);
    }
    #[test]
    fn align_test_banded() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 50;
        for _ in 0..20 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            align_banded_check(&single, &xs, &ys, radius);
            align_banded_check(&single_cond, &xs, &ys, radius);
            align_banded_check(&three, &xs, &ys, radius);
            align_banded_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn likelihood_banded_check<T: HMMType>(model: &GPHMM<T>, xs: &[u8], ys: &[u8], radius: usize) {
        let lk = model.likelihood(xs, ys);
        let lkb = model.likelihood_banded(xs, ys, radius).unwrap();
        assert!((lk - lkb).abs() < 0.0001);
    }
    #[test]
    fn likelihood_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..20 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            likelihood_banded_check(&single, &xs, &ys, radius);
            likelihood_banded_check(&single_cond, &xs, &ys, radius);
            likelihood_banded_check(&three, &xs, &ys, radius);
            likelihood_banded_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn back_likelihood_check<T: HMMType>(model: &GPHMM<T>, xs: &[u8], ys: &[u8], radius: isize) {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let (dp, factors) = model.backward(&xs, &ys);
        let (_, _, centers) = model.forward_banded(&xs, &ys, radius).unwrap();
        let (dp_b, factors_b) = model.backward_banded(&xs, &ys, radius, &centers);
        // What's important is the order of each row, and the actual likelihood.
        for (i, (x, y)) in factors_b.iter().zip(factors.iter()).enumerate().rev() {
            eprintln!("{}", i);
            assert!((x - y).abs() < 0.001, "{},{},{}", x, y, i);
        }
        let lk: f64 = model
            .initial_distribution
            .iter()
            .enumerate()
            .map(|(s, init)| init * dp[(0, 0, s as isize)])
            .sum();
        let lk_b: f64 = model
            .initial_distribution
            .iter()
            .enumerate()
            .map(|(s, init)| init * dp_b[(0, radius as isize - centers[0], s as isize)])
            .sum();
        let factor_b: f64 = factors_b.iter().map(log).sum();
        let factor: f64 = factors.iter().map(log).sum();
        let lk = lk + factor;
        let lk_b = lk_b + factor_b;
        assert!((lk - lk_b).abs() < 0.001, "{},{}", lk, lk_b);
    }
    #[test]
    fn backward_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..20 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            back_likelihood_check(&single, &xs, &ys, radius);
            back_likelihood_check(&single_cond, &xs, &ys, radius);
            back_likelihood_check(&three, &xs, &ys, radius);
            back_likelihood_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn profile_lk_check<T: HMMType>(model: &GPHMM<T>, xs: &[u8], ys: &[u8], radius: isize) {
        let lk = model.likelihood(&xs, &ys);
        let lkb = model.likelihood_banded(&xs, &ys, radius as usize).unwrap();
        assert!((lk - lkb).abs() < 0.001);
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let profile = ProfileBanded::new(model, &xs, &ys, radius).unwrap();
        let lkp = profile.lk();
        assert!((lk - lkp).abs() < 0.001);
    }
    #[test]
    fn profile_lk_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..20 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            profile_lk_check(&single, &xs, &ys, radius);
            profile_lk_check(&single_cond, &xs, &ys, radius);
            profile_lk_check(&three, &xs, &ys, radius);
            profile_lk_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn profile_mutation_banded_check<T: HMMType>(
        model: &GPHMM<T>,
        xs: &[u8],
        ys: &[u8],
        radius: isize,
    ) {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let profile = ProfileBanded::new(model, &xs, &ys, radius).unwrap();
        let mut xs = xs.clone();
        let len = xs.len();
        for pos in 0..len {
            let original = xs[pos as isize];
            for base in b"ACGT".iter().map(padseq::convert_to_twobit) {
                let lkp = profile.with_mutation(pos, base);
                xs[pos as isize] = base;
                let lk = model
                    .likelihood_banded_inner(&xs, &ys, radius as usize)
                    .unwrap();
                assert!((lk - lkp).abs() < 0.001);
                xs[pos as isize] = original;
            }
        }
    }
    #[test]
    fn profile_mutation_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..20 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            profile_mutation_banded_check(&single, &xs, &ys, radius);
            profile_mutation_banded_check(&single_cond, &xs, &ys, radius);
            profile_mutation_banded_check(&three, &xs, &ys, radius);
            profile_mutation_banded_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn profile_insertion_banded_check<T: HMMType>(
        model: &GPHMM<T>,
        xs: &[u8],
        ys: &[u8],
        radius: isize,
    ) {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let profile = ProfileBanded::new(model, &xs, &ys, radius).unwrap();
        let mut xs = xs.clone();
        let len = xs.len();
        for pos in 0..len {
            for base in b"ACGT".iter().map(padseq::convert_to_twobit) {
                let lkp = profile.with_insertion(pos, base);
                xs.insert(pos as isize, base);
                let lk = model
                    .likelihood_banded_inner(&xs, &ys, radius as usize)
                    .unwrap();
                assert!((lk - lkp).abs() < 0.001);
                xs.remove(pos as isize);
            }
        }
    }
    #[test]
    fn profile_insertion_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..20 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            profile_insertion_banded_check(&single, &xs, &ys, radius);
            profile_insertion_banded_check(&single_cond, &xs, &ys, radius);
            profile_insertion_banded_check(&three, &xs, &ys, radius);
            profile_insertion_banded_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn profile_deletion_banded_check<T: HMMType>(
        model: &GPHMM<T>,
        xs: &[u8],
        ys: &[u8],
        radius: isize,
    ) {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let profile = ProfileBanded::new(model, &xs, &ys, radius).unwrap();
        let mut xs = xs.clone();
        let len = xs.len();
        for pos in 0..len {
            let original = xs[pos as isize];
            let lkp = profile.with_deletion(pos);
            xs.remove(pos as isize);
            let lk = model
                .likelihood_banded_inner(&xs, &ys, radius as usize)
                .unwrap();
            assert!((lk - lkp).abs() < 0.001);
            xs.insert(pos as isize, original);
        }
    }
    #[test]
    fn profile_deletion_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..20 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            profile_deletion_banded_check(&single, &xs, &ys, radius);
            profile_deletion_banded_check(&single_cond, &xs, &ys, radius);
            profile_deletion_banded_check(&three, &xs, &ys, radius);
            profile_deletion_banded_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn correction_banded_check<T: HMMType>(
        model: &GPHMM<T>,
        draft: &[u8],
        queries: &[Vec<u8>],
        radius: usize,
        answer: &[u8],
    ) {
        let correct = model.correct_until_convergence(&draft, queries);
        let correct_b = model
            .correct_until_convergence_banded(&draft, queries, radius)
            .unwrap();
        let dist = crate::bialignment::edit_dist(&correct, &answer);
        let dist_b = crate::bialignment::edit_dist(&correct_b, &answer);
        assert!(dist_b <= dist);
    }
    #[test]
    fn profile_correction_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        let coverage = 30;
        for _ in 0..3 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let seqs: Vec<_> = (0..coverage)
                .map(|_| crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof))
                .collect();
            let draft = crate::ternary_consensus(&seqs, 23, 3, 20);
            correction_banded_check(&single, &draft, &seqs, radius, &xs);
            correction_banded_check(&single_cond, &draft, &seqs, radius, &xs);
            correction_banded_check(&three, &draft, &seqs, radius, &xs);
            correction_banded_check(&three_cond, &draft, &seqs, radius, &xs);
        }
    }
    fn fit_model_full(model: &GPHMM<Cond>, template: &[u8], queries: &[Vec<u8>]) -> GPHMM<Cond> {
        let mut model: GPHMM<Cond> = model.clone();
        let mut lk: f64 = queries.iter().map(|q| model.likelihood(&template, q)).sum();
        loop {
            let new_m = model.fit(template, queries);
            let new_lk: f64 = queries.iter().map(|q| new_m.likelihood(&template, q)).sum();
            if lk + 1f64 < new_lk {
                eprintln!("{:.2}", new_lk);
                lk = new_lk;
                model = new_m;
            } else {
                break;
            }
        }
        model
    }
    fn fit_model_banded(
        model: &GPHMM<Cond>,
        template: &[u8],
        queries: &[Vec<u8>],
        radius: usize,
    ) -> GPHMM<Cond> {
        let mut model: GPHMM<Cond> = model.clone();
        let mut lks: Vec<Option<f64>> = queries
            .iter()
            .map(|q| model.likelihood_banded(&template, q, radius))
            .collect();
        loop {
            let new_m = model.fit_banded(template, queries, radius);
            let new_lks: Vec<Option<f64>> = queries
                .iter()
                .map(|q| new_m.likelihood_banded(&template, q, radius))
                .collect();
            let lk_gain: f64 = new_lks
                .iter()
                .zip(lks.iter())
                .map(|(x, y)| match (x, y) {
                    (Some(x), Some(y)) => (x - y),
                    _ => 0f64,
                })
                .sum();
            if 1f64 < lk_gain {
                let lk: f64 = new_lks.iter().map(|x| x.unwrap_or(0f64)).sum();
                eprintln!("{:.2}", lk);
                lks = new_lks;
                model = new_m;
            } else {
                break;
            }
        }
        model
    }
    fn fit_banded_check(model: &GPHMM<Cond>, template: &[u8], queries: &[Vec<u8>], radius: usize) {
        let model_banded = fit_model_banded(&model, template, queries, radius);
        let model_full = fit_model_full(&model, template, queries);
        let dist = model_full.dist(&model_banded).unwrap();
        assert!(dist < 0.1, "{}\t{}\n{}", dist, model_full, model_banded)
    }
    #[test]
    fn fit_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single_cond = GPHMM::<Cond>::default();
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        let coverage = 30;
        for _ in 0..3 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let seqs: Vec<_> = (0..coverage)
                .map(|_| crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof))
                .collect();
            fit_banded_check(&single_cond, &xs, &seqs, radius);
            fit_banded_check(&three_cond, &xs, &seqs, radius);
        }
    }
}
