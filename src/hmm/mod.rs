//! An tiny implementation of pair hidden Markov models.

/// A pair hidden Markov model.
/// Even though theoretically we have several degree of freedom on transition probabilities,
/// I restrict these probabilities so that there are two kind of reversibility.
/// First reversibility is the "argument revesibility." In other words,
/// for almost all the method with pair HMMs `M`, `M.foo(x,y)` should be equeal to `M.foo(y,x)`.
/// In other words, there is no distinction like "reference" versus "query."
/// This is just my opinion but if someone want to make some distinction between a reference and a query,
/// they might consider profile Markov models instead of pair HMMs.
/// The second reversibility is "time reversibility." In other words,
/// `M.foo(x,y) = M.foo(rev(x), rev(y))` holds for almost all method with `M`.
/// In other words, there is no "direction" in sequence.
/// Even though these two assumption does not, strictly speaking, hold in real biology:
/// a reference is something different from queries, and there ARE context in biological sequences such as TATA boxes.
/// However, these two reversibility would make some algorithm easier to treat.
/// To make these two reversibility hold, there is two degree of freedom remain: match probability and gap extension probability.
/// More strictly, we can compute other two value, gap open probability and gap swtich(from deletion to insertion or vise versa) as follows:
/// `gap open = (1-match)/2`, `gap switch = (1-mathch-gapextension)`.
/// Of course, `match + gap extension < 1` should hold.
/// Note that under this formalization, we have `gap extension < 2 * gap open`.
/// As always, this is not always true. Suppose that there is insertion from transposable element. If there is insertion, then we want to relax the gap extension penalty as long as we are in the inserted TE.
/// Also, there is an implicit state in a pair HMM: the end state.
/// By controling the quit probability, i.e., from any state to the end state,
/// we can control the length of the output sequence when using a pairHMM as "generative model."
/// Usually and by default, the quit probability woule be 1/length(x).
/// In contrast to the usual pair HMMs, this implementation allow
/// transition between both ins->del and del->ins, one of which is
/// usually not needed. However, by introducing these two transition,
/// a pair HMM would be reversible.
/// To access the output prbabilities,
/// call `.prob_of()` instead of direct membership access.
/// In the membership description below, `X` means some arbitrary state except end state.
#[derive(Debug, Clone)]
pub struct PairHiddenMarkovModel {
    /// log(Pr{X->Mat})
    pub log_mat: f64,
    /// log(Pr{Mat->Ins}) or log(Pr{Mat->Del}).
    pub log_gap_open: f64,
    /// log(Pr{Ins->Del}) or log(Pr{Del->Ins}).
    pub log_gap_switch: f64,
    /// log(Pr{Del->Del}) or log(Pr{Ins->Ins})
    pub log_gap_ext: f64,
    /// log(Pr{X->End})
    pub log_quit: f64,
    /// log(Pr{output base x | Del}) or log(Pr{output base x | Ins})
    pub log_gap_emit: [f64; 4],
    log_mat_emit: [f64; 4 * 4],
}

/// Shorthand for PairHiddenMarkovModel.
pub type PHMM = PairHiddenMarkovModel;

/// A dynamic programming table. It is a serialized 2-d array.
#[derive(Debug, Clone)]
pub struct DPTable {
    dp: Vec<f64>,
    column: usize,
    row: usize,
}

/// Operations.
pub enum Op {
    Match,
    Del,
    Ins,
}

impl std::default::Default for PHMM {
    fn default() -> Self {
        let match_prob = 0.8;
        let gap_ext_prob = 0.1;
        let gap_output = [(4f64).recip(); 4];
        let match_ouptut = [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ];
        let quit_prob = 0.001;
        PHMM::new(
            match_prob,
            gap_ext_prob,
            &gap_output,
            &match_output,
            quit_prob,
        )
    }
}

impl PHMM {
    /// construct a new pair HMM.
    /// # Example
    /// ```rust
    /// let match_prob = 0.8;
    /// let gap_ext_prob = 0.1;
    /// let gap_output = [(4f64).recip(); 4];
    /// let match_ouptut = [
    /// [0.7, 0.1, 0.1, 0.1],
    /// [0.1, 0.7, 0.1, 0.1],
    /// [0.1, 0.1, 0.7, 0.1],
    /// [0.1, 0.1, 0.1, 0.7],
    /// ];
    /// let quit_prob = 0.001;
    /// PHMM::new(
    ///   match_prob,
    ///   gap_ext_prob,
    ///   &gap_output,
    ///   &match_output,
    ///   quit_prob,
    /// )
    /// ```
    pub fn new(
        mat: f64,
        gap_ext: f64,
        gap_output: &[f64; 4],
        mat_ouptut: &[[f64; 4]],
        quit_prob: f64,
    ) -> Self {
        assert!(mat.is_sign_positive());
        assert!(gap_ext.is_sign_positive());
        assert!(mat + gap_ext < 1f64);
        let gap_open = (1f64 - mat) / 2f64;
        let gap_switch = (1f64 - gap_ext);
        let alive_prob = 1f64 - quit_prob;
        let log_mat_emit = {
            let mut slots = [0f64; 16];
            for i in 0..4 {
                for j in 0..4 {
                    slots[i * 4 + j] = mat_output[i][j].log();
                }
            }
            slots
        };
        let log_gap_emit = {
            let mut slots = [0f64; 4];
            for i in 0..4 {
                slots[i] = gap_output[i].log();
            }
            slots
        };
        Self {
            log_mat: (mat * alive_prob).log(),
            log_gap_start: (gap_open * alive_prob).log(),
            log_gap_ext: (gap_ext * alive_prob).log(),
            log_gap_switch: (gap_switch * alive_prob).log(),
            log_quit: quit_prob.log(),
            log_gap_emit,
            log_mat_emit,
        }
    }
    /// Return likelihood of x and y. It returns the probability to see the two sequence (x,y),
    /// summarizing all the alignment between x and y.
    /// In other words, it returns Sum_{alignment between x and y} Pr{alignment|self}.
    /// Roughly speaking, it is the value after log-sum-exp-ing all the alignment score.
    /// In HMM term, it is "forward" algorithm.
    pub fn likelihood(&self, x: &[u8], y: &[u8]) -> (DPTable, f64) {
        unimplemented!()
    }
    /// Return the alignment path between x and y.
    /// In HMM term, it is "viterbi" algorithm.
    pub fn align(&self, x: &[u8], y: &[u8]) -> (DPTable, Vec<Op>, f64) {
        unimplemented!()
    }
    /// Correcting error of template by xs. In other words, it returns a
    /// estimation of `arg max sum_{x in xs} Pr{template,x|self}`.
    /// Or, it maximize `xs.iter().map(|x|self.likelihood(x,template)).sum::<f64>()`.
    pub fn correct<T: std::borrow::Borrow<[u8]>>(&self, template: &[u8], xs: &[T]) -> Vec<u8> {
        let mut corrected_sequnce: Vec<u8> = template.to_vec();
        while let Ok(res) = self.correct_one(corrected_sequence, xs) {
            corrected_sequence = res;
        }
        res
    }
    /// Correcting error of template by xs, requiring that the distance between corrected sequence
    /// and the original sequence is 1 or less (not corrected).
    /// If we can correct the template sequence, it returns the corrected seuquence as `Ok(res)`,
    /// Otherwize it returns the original sequence as `Err(res)`
    pub fn correct_one<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        xs: &[T],
    ) -> Result<Vec<u8>, Vec<u8>> {
        unimplemented!()
    }
}
