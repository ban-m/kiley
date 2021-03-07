//! An tiny implementation of pair hidden Markov models.
use rand::Rng;
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
/// As a principle of thumb, statistics function `*` and `*_banded` (`likelihood` and `likelihood_banded`, for example)
/// would return the same type of values.
/// If you want to more fine resolution outpt, please consult more specific function such as `forward` or `forward_banded`.
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
    mat_dp: Vec<f64>,
    ins_dp: Vec<f64>,
    del_dp: Vec<f64>,
    column: usize,
    row: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum State {
    Mat,
    Del,
    Ins,
}

impl std::convert::From<State> for Op {
    fn from(state: State) -> Op {
        match state {
            State::Mat => Op::Match,
            State::Del => Op::Del,
            State::Ins => Op::Ins,
        }
    }
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
    #[allow(dead_code)]
    fn set(&mut self, val: f64) {
        self.mat_dp.iter_mut().for_each(|x| *x = val);
        self.del_dp.iter_mut().for_each(|x| *x = val);
        self.ins_dp.iter_mut().for_each(|x| *x = val);
    }
    pub fn get_mut(&mut self, i: usize, j: usize, state: State) -> &mut f64 {
        match state {
            State::Mat => self.mat_dp.get_mut(i * self.column + j).unwrap(),
            State::Del => self.del_dp.get_mut(i * self.column + j).unwrap(),
            State::Ins => self.ins_dp.get_mut(i * self.column + j).unwrap(),
        }
    }
    pub fn get(&self, i: usize, j: usize, state: State) -> f64 {
        match state {
            State::Mat => *self.mat_dp.get(i * self.column + j).unwrap(),
            State::Del => *self.del_dp.get(i * self.column + j).unwrap(),
            State::Ins => *self.ins_dp.get(i * self.column + j).unwrap(),
        }
    }
    pub fn get_total_lk(&self, i: usize, j: usize) -> f64 {
        PHMM::logsumexp(
            self.get(i, j, State::Mat),
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
            .map(|xs| logsumexp(xs))
            .collect();
        let ins: Vec<_> = self
            .ins_dp
            .chunks_exact(self.column)
            .map(|xs| logsumexp(xs))
            .collect();
        let del: Vec<_> = self
            .del_dp
            .chunks_exact(self.column)
            .map(|xs| logsumexp(xs))
            .collect();
        (mat, del, ins)
    }
}

pub fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.;
    }
    let max = xs.iter().max_by(|x, y| x.partial_cmp(&y).unwrap()).unwrap();
    let sum = xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    assert!(sum >= 0., "{:?}->{}", xs, sum);
    max + sum
}

/// Operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Match,
    Del,
    Ins,
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
    pub likelihood_trajectry: Vec<f64>,
}

const NEG_THRESHOLD: f64 = 0.10;
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
        self.likelihood_trajectry
            .iter_mut()
            .zip(other.likelihood_trajectry.iter())
            .for_each(|(x, &y)| {
                *x = if *x < y {
                    y + (1f64 + (*x - y).exp()).ln()
                } else {
                    *x + (1f64 + (y - *x).exp()).ln()
                }
            });
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
        let del_tot = self
            .deletion_prob
            .iter()
            .filter(|&&x| NEG_THRESHOLD < x)
            .sum::<f64>();
        let ins_tot = self
            .insertion_prob
            .iter()
            .filter(|&&x| NEG_THRESHOLD < x)
            .sum::<f64>();
        if (0f64 - del_tot - ins_tot).abs() < 0.000001 {
            return template;
        }
        if rng.gen_range(0f64, del_tot + ins_tot) < del_tot {
            let probe = rng.gen_range(0f64, del_tot);
            let position = {
                let mut accum = 0f64;
                self.deletion_prob
                    .iter()
                    .take_while(|&&x| {
                        if NEG_THRESHOLD < x {
                            accum += x;
                        }
                        accum <= probe
                    })
                    .count()
            };
            template.remove(position);
        } else {
            let probe = rng.gen_range(0f64, ins_tot);
            let position = {
                let mut accum = 0f64;
                self.insertion_prob
                    .iter()
                    .take_while(|&&x| {
                        if NEG_THRESHOLD < x {
                            accum += x;
                        }
                        accum <= probe
                    })
                    .count()
            };
            let total = self.insertion_bases[position]
                .iter()
                .map(|&x| x as u16)
                .sum::<u16>();
            let base = {
                let probe = rng.gen_range(0, total);
                let mut accum = 0;
                let index = self.insertion_bases[position]
                    .iter()
                    .take_while(|&&x| {
                        accum += x as u16;
                        accum < probe
                    })
                    .count();
                b"ACGT"[index]
            };
            template.insert(position + 1, base);
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
                for pos in start..end {
                    polished_seq.push(template[pos]);
                    polished_seq.push(Self::choose_max_base(&self.insertion_bases[pos]));
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

pub fn recover(xs: &[u8], ys: &[u8], ops: &[Op]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut i, mut j) = (0, 0);
    let (mut xr, mut yr, mut aln) = (vec![], vec![], vec![]);
    for &op in ops {
        match op {
            Op::Match => {
                xr.push(xs[i]);
                yr.push(ys[j]);
                if xs[i] == ys[j] {
                    aln.push(b'|');
                } else {
                    aln.push(b'X');
                }
                i += 1;
                j += 1;
            }
            Op::Del => {
                xr.push(xs[i]);
                aln.push(b' ');
                yr.push(b' ');
                i += 1;
            }
            Op::Ins => {
                xr.push(b' ');
                aln.push(b' ');
                yr.push(ys[j]);
                j += 1;
            }
        }
    }
    (xr, aln, yr)
}

impl std::default::Default for PHMM {
    fn default() -> Self {
        let match_prob = 0.9;
        let gap_ext_prob = 0.08;
        let gap_output = [(4f64).recip(); 4];
        let match_output = [
            [0.9, 0.1 / 3., 0.1 / 3., 0.1 / 3.],
            [0.1 / 3., 0.9, 0.1 / 3., 0.1 / 3.],
            [0.1 / 3., 0.1 / 3., 0.9, 0.1 / 3.],
            [0.1 / 3., 0.1 / 3., 0.1 / 3., 0.9],
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

const EP: f64 = -10000000000000000000000000000000f64;

#[derive(Debug, Clone)]
pub struct DiagonalDP {
    mat: Vec<Vec<f64>>,
    del: Vec<Vec<f64>>,
    ins: Vec<Vec<f64>>,
    column: usize,
    row: usize,
}

impl DiagonalDP {
    pub fn new(row: usize, column: usize, x: f64) -> Self {
        Self {
            mat: vec![vec![x; column]; row],
            del: vec![vec![x; column]; row],
            ins: vec![vec![x; column]; row],
            row,
            column,
        }
    }
    pub fn get(&self, k: usize, u: usize, state: State) -> f64 {
        match state {
            State::Mat => self.mat[k][u],
            State::Del => self.del[k][u],
            State::Ins => self.ins[k][u],
        }
    }
    pub fn get_check(&self, k: usize, u: usize, state: State) -> Option<f64> {
        match state {
            State::Mat => self.mat.get(k).and_then(|x| x.get(u)).copied(),
            State::Del => self.del.get(k).and_then(|x| x.get(u)).copied(),
            State::Ins => self.ins.get(k).and_then(|x| x.get(u)).copied(),
        }
    }
    pub fn get_mut(&mut self, k: usize, u: usize, state: State) -> &mut f64 {
        match state {
            State::Mat => self.mat.get_mut(k).unwrap().get_mut(u).unwrap(),
            State::Del => self.del.get_mut(k).unwrap().get_mut(u).unwrap(),
            State::Ins => self.ins.get_mut(k).unwrap().get_mut(u).unwrap(),
        }
    }
    pub fn get_row(&self, k: usize, state: State) -> &[f64] {
        match state {
            State::Mat => self.mat[k].as_slice(),
            State::Del => self.del[k].as_slice(),
            State::Ins => self.ins[k].as_slice(),
        }
    }
    pub fn add(&mut self, other: &Self) {
        self.mat
            .iter_mut()
            .zip(other.mat.iter())
            .for_each(|(xs, ys)| xs.iter_mut().zip(ys.iter()).for_each(|(x, &y)| *x += y));
        self.del
            .iter_mut()
            .zip(other.del.iter())
            .for_each(|(xs, ys)| xs.iter_mut().zip(ys.iter()).for_each(|(x, &y)| *x += y));
        self.ins
            .iter_mut()
            .zip(other.ins.iter())
            .for_each(|(xs, ys)| xs.iter_mut().zip(ys.iter()).for_each(|(x, &y)| *x += y));
    }
    pub fn sub_scalar(&mut self, a: f64) {
        self.mat
            .iter_mut()
            .for_each(|xs| xs.iter_mut().for_each(|x| *x -= a));
        self.del
            .iter_mut()
            .for_each(|xs| xs.iter_mut().for_each(|x| *x -= a));
        self.ins
            .iter_mut()
            .for_each(|xs| xs.iter_mut().for_each(|x| *x -= a));
    }
    // pub fn sum_anti_diagonal(&self, k: usize) -> f64 {
    //     self.mat[k]
    //         .iter()
    //         .chain(self.del[k].iter())
    //         .chain(self.ins[k].iter())
    //         .sum::<f64>()
    // }
    // pub fn div_anti_diagonal(&mut self, k: usize, div: f64) {
    //     self.mat[k]
    //         .iter_mut()
    //         .chain(self.del[k].iter_mut())
    //         .chain(self.ins[k].iter_mut())
    //         .for_each(|x| *x /= div);
    // }
}

impl PHMM {
    /// construct a new pair HMM.
    /// # Example
    /// ```rust
    /// use PHMM;
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
        mat_output: &[[f64; 4]],
        quit_prob: f64,
    ) -> Self {
        assert!(mat.is_sign_positive());
        assert!(gap_ext.is_sign_positive());
        assert!(mat + gap_ext < 1f64);
        let gap_open = (1f64 - mat) / 2f64;
        let gap_switch = 1f64 - gap_ext - mat;
        // let alive_prob = 1f64 - quit_prob;
        let log_mat_emit = {
            let mut slots = [0f64; 16];
            for i in 0..4 {
                for j in 0..4 {
                    slots[(i << 2) | j] = mat_output[i][j].ln();
                }
            }
            slots
        };
        let log_gap_emit = {
            let mut slots = [0f64; 4];
            for i in 0..4 {
                slots[i] = gap_output[i].ln();
            }
            slots
        };
        // Self {
        //     log_mat: (mat * alive_prob).ln(),
        //     log_gap_open: (gap_open * alive_prob).ln(),
        //     log_gap_ext: (gap_ext * alive_prob).ln(),
        //     log_gap_switch: (gap_switch * alive_prob).ln(),
        //     log_quit: quit_prob.ln(),
        //     log_gap_emit,
        //     log_mat_emit,
        // }
        Self {
            log_mat: mat.ln(),
            log_gap_open: gap_open.ln(),
            log_gap_ext: gap_ext.ln(),
            log_gap_switch: gap_switch.ln(),
            log_quit: quit_prob.ln(),
            log_gap_emit,
            log_mat_emit,
        }
    }
    /// Return likelihood of x and y.
    /// It returns the probability to see the two sequence (x,y),
    /// summarizing all the alignment between x and y.
    /// In other words, it returns Sum_{alignment between x and y} Pr{alignment|self}.
    /// Roughly speaking, it is the value after log-sum-exp-ing all the alignment score.
    /// In HMM term, it is "forward" algorithm.
    /// In contrast to usual likelihood method, it only compute only restricted range of pair.
    /// Return values are the rowwize sum and the total likelihood.
    /// For example, `let (lks, lk) = self.likelihood(xs,ys);` then `lks[10]` is
    /// the sum of the probability to see xs[0..10] and y[0..i], summing up over `i`.
    /// If the band did not reached the (xs.len(), ys.len()) cell, then this funtion
    /// return None. Please increase `radius` parameter if so.
    pub fn likelihood_banded(
        &self,
        xs: &[u8],
        ys: &[u8],
        radius: usize,
    ) -> Option<(Vec<f64>, f64)> {
        let (dp, centers) = self.forward_banded(xs, ys, radius);
        let (k, u) = (xs.len() + ys.len(), xs.len());
        let u_in_dp = u + radius - centers[k];
        let max_lk = Self::logsumexp(
            dp.get_check(k, u_in_dp, State::Mat)?,
            dp.get_check(k, u_in_dp, State::Del)?,
            dp.get_check(k, u_in_dp, State::Ins)?,
        );
        // Maybe this code is very slow...
        let mut lks: Vec<Vec<f64>> = vec![vec![]; xs.len() + 1];
        for k in 0..xs.len() + ys.len() + 1 {
            let u_center = centers[k];
            for (u, ((&x, &y), &z)) in dp
                .get_row(k, State::Mat)
                .iter()
                .zip(dp.get_row(k, State::Del).iter())
                .zip(dp.get_row(k, State::Ins).iter())
                .enumerate()
            {
                let i = if radius <= u + u_center && u + u_center - radius < xs.len() + 1 {
                    u + u_center - radius
                } else {
                    continue;
                };
                lks[i].push(x);
                lks[i].push(y);
                lks[i].push(z);
            }
        }
        let lks: Vec<_> = lks.iter().map(|xs| logsumexp(xs)).collect();
        Some((lks, max_lk))
    }
    /// Return likelihood of x and y. It returns the probability to see the two sequence (x,y),
    /// summarizing all the alignment between x and y.
    /// In other words, it returns Sum_{alignment between x and y} Pr{alignment|self}.
    /// Roughly speaking, it is the value after log-sum-exp-ing all the alignment score.
    /// In HMM term, it is "forward" algorithm.
    /// If you want to get the raw DP table, please call `forward` functionality instead.
    pub fn likelihood(&self, xs: &[u8], ys: &[u8]) -> (Vec<f64>, f64) {
        let dptable = self.forward(xs, ys);
        let lk = dptable.get_total_lk(xs.len(), ys.len());
        let lks = dptable.lks_in_row();
        (lks, lk)
    }
    /// Forward algortihm in banded manner. Return the DP tables for each state,
    /// and the ceters of each anti-diagonal.
    /// Currently, we use re-scaling method instead of log-sum-exp mode because of stability and efficiency.
    pub fn forward_banded(&self, xs: &[u8], ys: &[u8], radius: usize) -> (DiagonalDP, Vec<usize>) {
        let xs: Vec<_> = xs.iter().map(crate::alignment::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::alignment::convert_to_twobit).collect();
        // `radius` radius, plus 2 for padding.
        let mut centers: Vec<usize> = vec![0, 0, 1];
        // Sum of mat[k] + del[k] + ins[k]
        // let mut scaling_factor: Vec<f64> = Vec::with_capacity(xs.len() + ys.len() + 2);
        let mut dp = DiagonalDP::new(xs.len() + ys.len() + 1, 2 * radius + 1 + 2, EP);
        // let mut dp = DiagonalDP::new(xs.len() + ys.len() + 1, 2 * radius + 1 + 2, 0f64);
        // The first diagonal.
        *dp.get_mut(0, radius, State::Mat) = 0f64;
        // *dp.get_mut(0, radius, State::Mat) = 1f64;
        // scaling_factor.push(1f64);
        // The second diagonal.
        *dp.get_mut(1, radius, State::Ins) = self.log_gap_open + self.log_gap_emit[ys[0] as usize];
        *dp.get_mut(1, radius + 1, State::Del) =
            self.log_gap_open + self.log_gap_emit[xs[0] as usize];
        // let sum = dp.sum_anti_diagonal(1);
        // dp.div_anti_diagonal(1, sum);
        // scaling_factor.push(sum);
        for k in 2..xs.len() + ys.len() + 1 {
            let center = *centers.last().unwrap();
            let matdiff = center as isize - centers[k - 2] as isize;
            let gapdiff = center as isize - centers[k - 1] as isize;
            // TODO: remove `if` statements as many as possible.
            for pos in 0..2 * radius + 1 {
                let u = (pos + center) as isize - radius as isize;
                let (i, j) = (u, k as isize - u);
                if !((0..xs.len() as isize + 1).contains(&i)
                    && (0..ys.len() as isize + 1).contains(&j))
                {
                    continue;
                }
                let u = u as usize;
                let prev_mat = pos as isize + matdiff;
                let prev_gap = pos as isize + gapdiff;
                let mat = if 0 < prev_mat && 0 < u && u < k {
                    let (x, y) = (xs[u - 1], ys[k - u - 1]);
                    Self::logsumexp(
                        dp.get(k - 2, prev_mat as usize - 1, State::Mat),
                        dp.get(k - 2, prev_mat as usize - 1, State::Del),
                        dp.get(k - 2, prev_mat as usize - 1, State::Ins),
                    ) + self.log_mat
                        + self.log_mat_emit[(x << 2 | y) as usize]
                } else {
                    EP
                };
                *dp.get_mut(k, pos, State::Mat) = mat;
                let del = if 0 < prev_gap && 0 < u {
                    Self::logsumexp(
                        dp.get(k - 1, prev_gap as usize - 1, State::Mat) + self.log_gap_open,
                        dp.get(k - 1, prev_gap as usize - 1, State::Del) + self.log_gap_ext,
                        dp.get(k - 1, prev_gap as usize - 1, State::Ins) + self.log_gap_switch,
                    ) + self.log_gap_emit[xs[u - 1] as usize]
                } else {
                    EP
                };
                *dp.get_mut(k, pos, State::Del) = del;
                let ins = if 0 <= prev_gap && u < k {
                    Self::logsumexp(
                        dp.get(k - 1, prev_gap as usize, State::Mat) + self.log_gap_open,
                        dp.get(k - 1, prev_gap as usize, State::Del) + self.log_gap_switch,
                        dp.get(k - 1, prev_gap as usize, State::Ins) + self.log_gap_ext,
                    ) + self.log_gap_emit[ys[k - u - 1] as usize]
                } else {
                    EP
                };
                *dp.get_mut(k, pos, State::Ins) = ins;
            }
            let (max_u, _max_lk) = dp
                .get_row(k, State::Mat)
                .iter()
                .zip(dp.get_row(k, State::Del).iter())
                .zip(dp.get_row(k, State::Ins).iter())
                .map(|((&x, &y), &z)| Self::logsumexp(x, y, z))
                .enumerate()
                .max_by(|x, y| (x.1).partial_cmp(&(y.1)).unwrap())
                .unwrap();
            let max_u = max_u + center - radius;
            if center < max_u {
                centers.push(center + 1);
            } else {
                centers.push(center);
            };
        }
        (dp, centers)
    }
    /// Forward algorithm. Return the raw DP table.
    pub fn forward(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let xs: Vec<_> = xs.iter().map(crate::alignment::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::alignment::convert_to_twobit).collect();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        let mut gap = 0f64;
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            gap += self.log_gap_emit[x as usize];
            *dptable.get_mut(i, 0, State::Mat) = EP;
            *dptable.get_mut(i, 0, State::Del) =
                self.log_gap_open + self.log_gap_ext * (i - 1) as f64 + gap;
            *dptable.get_mut(i, 0, State::Ins) = EP;
        }
        let mut gap = 0f64;
        for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
            gap += self.log_gap_emit[y as usize];
            *dptable.get_mut(0, j, State::Mat) = EP;
            *dptable.get_mut(0, j, State::Del) = EP;
            *dptable.get_mut(0, j, State::Ins) =
                self.log_gap_open + self.log_gap_ext * (j - 1) as f64 + gap;
        }
        *dptable.get_mut(0, 0, State::Ins) = EP;
        *dptable.get_mut(0, 0, State::Del) = EP;
        for (i, &x) in xs.iter().enumerate().map(|(p, x)| (p + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(p, y)| (p + 1, y)) {
                let mat = Self::logsumexp(
                    dptable.get(i - 1, j - 1, State::Mat),
                    dptable.get(i - 1, j - 1, State::Del),
                    dptable.get(i - 1, j - 1, State::Ins),
                ) + self.log_mat
                    + self.log_mat_emit[((x << 2) | y) as usize];
                *dptable.get_mut(i, j, State::Mat) = mat;
                let del = Self::logsumexp(
                    dptable.get(i - 1, j, State::Mat) + self.log_gap_open,
                    dptable.get(i - 1, j, State::Del) + self.log_gap_ext,
                    dptable.get(i - 1, j, State::Ins) + self.log_gap_switch,
                ) + self.log_gap_emit[x as usize];
                *dptable.get_mut(i, j, State::Del) = del;
                let ins = Self::logsumexp(
                    dptable.get(i, j - 1, State::Mat) + self.log_gap_open,
                    dptable.get(i, j - 1, State::Del) + self.log_gap_switch,
                    dptable.get(i, j - 1, State::Ins) + self.log_gap_ext,
                ) + self.log_gap_emit[y as usize];
                *dptable.get_mut(i, j, State::Ins) = ins;
            }
        }
        dptable
    }
    /// Banded backward algorithm. The filling cells are determined by `center`.
    /// In other words, for i in 0..2*radius+1,
    /// dp[k][i] = the (i - radius + center, k - (i -radius + center)) position in the usual DP matrix.
    /// So, for example, the (0,0) cell would be dp[0][radius].
    pub fn backward_banded(
        &self,
        xs: &[u8],
        ys: &[u8],
        radius: usize,
        centers: &[usize],
    ) -> DiagonalDP {
        let xs: Vec<_> = xs.iter().map(crate::alignment::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::alignment::convert_to_twobit).collect();
        let mut dp = DiagonalDP::new(xs.len() + ys.len() + 1, 2 * radius + 1 + 2, EP);
        // Calc the boundary score for each sequence, in other words,
        // calc DP[xs.len()][j] and DP[i][ys.len()] in the usual DP.
        // For DP[i][ys.len()].
        let gap_emit_xs = {
            let (mut gap_emit_xs_rev, _) = xs
                .iter()
                .rev()
                .map(|&x| self.log_gap_emit[x as usize])
                .fold((Vec::new(), 0f64), |(mut logs, acc), x| {
                    logs.push(x + acc);
                    (logs, x + acc)
                });
            gap_emit_xs_rev.reverse();
            gap_emit_xs_rev
        };
        let gap_emit_ys = {
            let (mut gap_emit_ys_rev, _) = ys
                .iter()
                .rev()
                .map(|&y| self.log_gap_emit[y as usize])
                .fold((Vec::new(), 0f64), |(mut logs, acc), y| {
                    logs.push(y + acc);
                    (logs, y + acc)
                });
            gap_emit_ys_rev.reverse();
            gap_emit_ys_rev
        };
        // Get the location corresponding to [xs.len()][ys.len()].
        // Fill the last DP call.
        {
            let (k, u) = (xs.len() + ys.len(), xs.len());
            let u_in_dp = u + radius - centers[k];
            *dp.get_mut(k, u_in_dp, State::Mat) = 0f64;
            *dp.get_mut(k, u_in_dp, State::Del) = 0f64;
            *dp.get_mut(k, u_in_dp, State::Ins) = 0f64;
        }
        // Filling the 2nd-last DP cell.
        {
            let k = xs.len() + ys.len() - 1;
            for u in vec![xs.len() - 1, xs.len()] {
                let (i, j) = (u, k - u);
                let u = u + radius - centers[k];
                if i == xs.len() {
                    *dp.get_mut(k, u, State::Mat) = gap_emit_ys[j]
                        + self.log_gap_open
                        + self.log_gap_ext * (ys.len() - j - 1) as f64;
                    *dp.get_mut(k, u, State::Del) = gap_emit_ys[j]
                        + self.log_gap_switch
                        + self.log_gap_ext * (ys.len() - j - 1) as f64;
                    *dp.get_mut(k, u, State::Ins) =
                        gap_emit_ys[j] + self.log_gap_ext * (ys.len() - j) as f64;
                } else if j == ys.len() {
                    *dp.get_mut(k, u, State::Mat) = gap_emit_xs[i]
                        + self.log_gap_open
                        + self.log_gap_ext * (xs.len() - i - 1) as f64;
                    *dp.get_mut(k, u, State::Del) =
                        gap_emit_xs[i] + self.log_gap_ext * (xs.len() - i) as f64;
                    *dp.get_mut(k, u, State::Ins) = gap_emit_xs[i]
                        + self.log_gap_switch
                        + self.log_gap_ext * (xs.len() - i - 1) as f64;
                } else {
                    unreachable!();
                }
            }
        }
        for k in (0..xs.len() + ys.len() - 1).rev() {
            let center = centers[k];
            let gapdiff = center as isize - centers[k + 1] as isize;
            let matdiff = center as isize - centers[k + 2] as isize;
            for pos in 0..2 * radius + 1 {
                let u = (pos + center) as isize - radius as isize;
                let (i, j) = (u, k as isize - u);
                if !((0..xs.len() as isize + 1).contains(&i)
                    && (0..ys.len() as isize + 1).contains(&j))
                {
                    continue;
                }
                let (i, j) = (i as usize, j as usize);
                // Previous, prev-previous position.
                let u_mat = pos as isize + matdiff + 1;
                let u_gap = pos as isize + gapdiff;
                let mat = if 0 <= u_mat && i < xs.len() && j < ys.len() {
                    let (x, y) = (xs[i], ys[j]);
                    self.log_mat
                        + self.log_mat_emit[(x << 2 | y) as usize]
                        + dp.get(k + 2, u_mat as usize, State::Mat)
                } else {
                    EP
                };
                let del = if -1 <= u_gap && i < xs.len() {
                    self.log_gap_open
                        + self.log_gap_emit[xs[i] as usize]
                        + dp.get(k + 1, (u_gap + 1) as usize, State::Del)
                } else {
                    EP
                };
                let ins = if 0 <= u_gap && j < ys.len() {
                    self.log_gap_open
                        + self.log_gap_emit[ys[j] as usize]
                        + dp.get(k + 1, u_gap as usize, State::Ins)
                } else {
                    EP
                };
                *dp.get_mut(k, pos, State::Mat) = Self::logsumexp(mat, del, ins);
                let del = del - self.log_gap_open + self.log_gap_ext;
                let ins = ins - self.log_gap_open + self.log_gap_switch;
                *dp.get_mut(k, pos, State::Del) = Self::logsumexp(mat, del, ins);
                let del = del - self.log_gap_ext + self.log_gap_switch;
                let ins = ins - self.log_gap_switch + self.log_gap_ext;
                *dp.get_mut(k, pos, State::Ins) = Self::logsumexp(mat, del, ins);
            }
        }
        dp
    }
    /// Naive implementation of backward algorithm.
    pub fn backward(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let xs: Vec<_> = xs.iter().map(crate::alignment::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::alignment::convert_to_twobit).collect();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        *dptable.get_mut(xs.len(), ys.len(), State::Mat) = 0f64;
        *dptable.get_mut(xs.len(), ys.len(), State::Del) = 0f64;
        *dptable.get_mut(xs.len(), ys.len(), State::Ins) = 0f64;
        let mut gap = 0f64;
        for (i, &x) in xs.iter().enumerate().rev() {
            gap += self.log_gap_emit[x as usize];
            *dptable.get_mut(i, ys.len(), State::Del) =
                self.log_gap_ext * (xs.len() - i) as f64 + gap;
            *dptable.get_mut(i, ys.len(), State::Ins) =
                self.log_gap_switch + self.log_gap_ext * (xs.len() - i - 1) as f64 + gap;
            *dptable.get_mut(i, ys.len(), State::Mat) =
                self.log_gap_open + self.log_gap_ext * (xs.len() - i - 1) as f64 + gap;
        }
        gap = 0f64;
        for (j, &y) in ys.iter().enumerate().rev() {
            gap += self.log_gap_emit[y as usize];
            *dptable.get_mut(xs.len(), j, State::Ins) =
                self.log_gap_ext * (ys.len() - j) as f64 + gap;
            *dptable.get_mut(xs.len(), j, State::Del) =
                self.log_gap_switch + self.log_gap_ext * (ys.len() - j - 1) as f64 + gap;
            *dptable.get_mut(xs.len(), j, State::Mat) =
                self.log_gap_open + self.log_gap_ext * (ys.len() - j - 1) as f64 + gap;
        }
        for (i, &x) in xs.iter().enumerate().rev() {
            for (j, &y) in ys.iter().enumerate().rev() {
                // Match state;
                let mat = self.log_mat
                    + self.log_mat_emit[(x << 2 | y) as usize]
                    + dptable.get(i + 1, j + 1, State::Mat);
                let del = self.log_gap_open
                    + self.log_gap_emit[x as usize]
                    + dptable.get(i + 1, j, State::Del);
                let ins = self.log_gap_open
                    + self.log_gap_emit[y as usize]
                    + dptable.get(i, j + 1, State::Ins);
                *dptable.get_mut(i, j, State::Mat) = Self::logsumexp(mat, del, ins);
                // Del state.
                let del = del - self.log_gap_open + self.log_gap_ext;
                let ins = ins - self.log_gap_open + self.log_gap_switch;
                *dptable.get_mut(i, j, State::Del) = Self::logsumexp(mat, del, ins);
                // Ins state
                let del = del - self.log_gap_ext + self.log_gap_switch;
                let ins = ins - self.log_gap_switch + self.log_gap_ext;
                *dptable.get_mut(i, j, State::Ins) = Self::logsumexp(mat, del, ins);
            }
        }
        dptable
    }
    /// Return erorr profile, use banded "forward/backward" algorithm.
    /// If the band did not reach to the corner, then this function returns None.
    /// otherwize, it returns the summary of forward backward algorithm.
    pub fn get_profile_banded(
        &self,
        xs: &[u8],
        ys: &[u8],
        radius: usize,
    ) -> Option<LikelihoodSummary> {
        let (fwd, centers) = self.forward_banded(xs, ys, radius);
        let lk = {
            let (k, u) = (xs.len() + ys.len(), xs.len());
            let u_in_dp = u + radius - centers[k];
            Self::logsumexp(
                fwd.get_check(k, u_in_dp, State::Mat)?,
                fwd.get_check(k, u_in_dp, State::Del)?,
                fwd.get_check(k, u_in_dp, State::Ins)?,
            )
        };
        let likelihood_trajectry: Vec<_> = {
            let mut lk_trajectry: Vec<Vec<f64>> = vec![vec![]; xs.len() + 1];
            for (k, center) in centers.iter().take(xs.len() + ys.len() + 1).enumerate() {
                let mat = fwd.get_row(k, State::Mat);
                for (u, &m) in mat.iter().enumerate() {
                    let pos = (u + center) as isize - radius as isize;
                    if (0..xs.len() as isize + 1).contains(&pos) {
                        lk_trajectry[pos as usize].push(m);
                    }
                }
            }
            lk_trajectry.iter().map(|xs| logsumexp(&xs)).collect()
        };
        // forward-backward.
        let mut fbwd = self.backward_banded(xs, ys, radius, &centers);
        fbwd.add(&fwd);
        fbwd.sub_scalar(lk);
        // Allocate each prob of state
        let mut match_max_prob = vec![(EP, 0); xs.len()];
        let mut ins_max_prob = vec![(EP, 0); xs.len()];
        let mut match_prob: Vec<Vec<f64>> = vec![vec![]; xs.len()];
        let mut del_prob: Vec<Vec<f64>> = vec![vec![]; xs.len()];
        let mut ins_prob: Vec<Vec<f64>> = vec![vec![]; xs.len()];
        for (k, center) in centers.iter().take(xs.len() + ys.len() + 1).enumerate() {
            let mat = fbwd.get_row(k, State::Mat);
            let del = fbwd.get_row(k, State::Del);
            let ins = fbwd.get_row(k, State::Ins);
            for (u, ((&mat, &del), &ins)) in mat.iter().zip(del).zip(ins).enumerate() {
                let i = (u + center) as isize - radius as isize;
                let j = k as isize - i;
                if (1..xs.len() as isize + 1).contains(&i)
                    && (0..ys.len() as isize + 1).contains(&j)
                {
                    let i = i as usize;
                    match_prob[i - 1].push(mat);
                    del_prob[i - 1].push(del);
                    ins_prob[i - 1].push(ins);
                    let y = if j == 0 { b'A' } else { ys[j as usize - 1] };
                    if match_max_prob[i - 1].0 < mat {
                        match_max_prob[i - 1] = (mat, y);
                    }
                    if ins_max_prob[i - 1].0 < ins {
                        ins_max_prob[i - 1] = (ins, y);
                    }
                }
            }
        }
        let match_prob: Vec<_> = match_prob.iter().map(|xs| logsumexp(xs).exp()).collect();
        let deletion_prob: Vec<_> = del_prob.iter().map(|xs| logsumexp(xs).exp()).collect();
        let insertion_prob: Vec<_> = ins_prob.iter().map(|xs| logsumexp(xs).exp()).collect();
        let match_bases: Vec<_> = match_max_prob
            .iter()
            .map(|x| {
                let mut slot = [0u8; 4];
                slot[crate::alignment::convert_to_twobit(&x.1) as usize] += 1;
                slot
            })
            .collect();
        let insertion_bases: Vec<_> = ins_max_prob
            .iter()
            .map(|x| {
                let mut slot = [0u8; 4];
                slot[crate::alignment::convert_to_twobit(&x.1) as usize] += 1;
                slot
            })
            .collect();
        let summary = LikelihoodSummary {
            match_prob,
            match_bases,
            insertion_prob,
            insertion_bases,
            deletion_prob,
            total_likelihood: lk,
            likelihood_trajectry,
        };
        Some(summary)
    }
    /// Return error profile.
    pub fn get_profile(&self, xs: &[u8], ys: &[u8]) -> LikelihoodSummary {
        // Forward backward profile
        let f_dp = self.forward(xs, ys);
        let b_dp = self.backward(xs, ys);
        let lk = f_dp.get_total_lk(xs.len(), ys.len());
        let mut dptable = DPTable::new(xs.len(), ys.len() + 1);
        for i in 0..xs.len() {
            for j in 0..ys.len() + 1 {
                for s in vec![State::Mat, State::Del, State::Ins] {
                    *dptable.get_mut(i, j, s) = f_dp.get(i + 1, j, s) + b_dp.get(i + 1, j, s) - lk;
                }
            }
        }
        let (mut match_prob, mut deletion_prob, mut insertion_prob) = dptable.lks_in_row_by_state();
        match_prob.iter_mut().for_each(|x| *x = x.exp());
        insertion_prob.iter_mut().for_each(|x| *x = x.exp());
        deletion_prob.iter_mut().for_each(|x| *x = x.exp());
        let match_bases: Vec<_> = dptable
            .mat_dp
            .chunks_exact(dptable.column)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|x, y| (x.1).partial_cmp(&(y.1)).unwrap())
                    .map(|(j, _)| {
                        let mut slot = [0; 4];
                        slot[crate::alignment::convert_to_twobit(&ys[j - 1]) as usize] += 1;
                        slot
                    })
                    .unwrap()
            })
            .collect();
        let insertion_bases: Vec<_> = dptable
            .ins_dp
            .chunks_exact(dptable.column)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|x, y| (x.1).partial_cmp(&(y.1)).unwrap())
                    .map(|(j, _)| {
                        let mut slot = [0u8; 4];
                        slot[crate::alignment::convert_to_twobit(&ys[j - 1]) as usize] += 1;
                        slot
                    })
                    .unwrap()
            })
            .collect();
        let (likelihood_trajectry, _, _) = f_dp.lks_in_row_by_state();
        LikelihoodSummary {
            match_prob,
            match_bases,
            insertion_prob,
            insertion_bases,
            deletion_prob,
            total_likelihood: lk,
            likelihood_trajectry,
        }
    }
    /// Correcting the template sequence by queries, calculate the
    /// new likelihoods and alignment summary, which can be used for futher correction.
    pub fn correct_step<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
        summary: &LikelihoodSummary,
    ) -> (Vec<u8>, LikelihoodSummary) {
        assert!(!queries.is_empty());
        let new_template = summary.correct(&template);
        let summary: Option<LikelihoodSummary> = queries
            .iter()
            .map(|x| self.get_profile(&new_template, x.borrow()))
            .fold(None, |summary, x| match summary {
                Some(mut summary) => {
                    summary.add(&x);
                    Some(summary)
                }
                None => Some(x),
            });
        let mut summary = summary.unwrap();
        summary.div_probs(queries.len() as f64);
        (new_template, summary)
    }
    pub fn correct<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
    ) -> (Vec<u8>, LikelihoodSummary) {
        let lks: Option<LikelihoodSummary> = queries
            .iter()
            .map(|x| self.get_profile(&template, x.borrow()))
            .fold(None, |summary, x| match summary {
                Some(mut summary) => {
                    summary.add(&x);
                    Some(summary)
                }
                None => Some(x),
            });
        let mut lks = lks.unwrap();
        lks.div_probs(queries.len() as f64);
        let mut corrected = template.to_vec();
        loop {
            let (new_corrected, new_lks) = self.correct_step(&corrected, queries, &lks);
            println!(
                "LK:{:.3}->{:.3}",
                lks.total_likelihood, new_lks.total_likelihood
            );
            println!("CurrentSQ:{}", String::from_utf8_lossy(&corrected));
            if lks.total_likelihood < new_lks.total_likelihood {
                corrected = new_corrected;
                lks = new_lks;
            } else {
                break;
            }
        }
        (corrected, lks)
    }
    /// Batch function for `get_profiles.`
    pub fn get_profiles_banded<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
        radius: usize,
    ) -> LikelihoodSummary {
        assert!(!queries.is_empty());
        let mut ok_sequences = 0;
        let lks: Option<LikelihoodSummary> = queries
            .iter()
            .filter_map(|x| self.get_profile_banded(&template, x.borrow(), radius))
            .fold(None, |summary, x| {
                ok_sequences += 1;
                match summary {
                    Some(mut summary) => {
                        summary.add(&x);
                        Some(summary)
                    }
                    None => Some(x),
                }
            });
        let mut lks = lks.unwrap();
        lks.div_probs(ok_sequences as f64);
        lks
    }
    /// Batch function for `get_profile`
    pub fn get_profiles<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
    ) -> LikelihoodSummary {
        assert!(!queries.is_empty());
        let lks: Option<LikelihoodSummary> = queries
            .iter()
            .map(|x| self.get_profile(&template, x.borrow()))
            .fold(None, |summary, x| match summary {
                Some(mut summary) => {
                    summary.add(&x);
                    Some(summary)
                }
                None => Some(x),
            });
        let mut lks = lks.unwrap();
        lks.div_probs(queries.len() as f64);
        lks
    }
    /// Correction by sampling, use banded forward-backward algorithm inside.
    pub fn correct_flip_banded<T: std::borrow::Borrow<[u8]>, R: Rng>(
        &self,
        template: &[u8],
        queries: &[T],
        rng: &mut R,
        repeat_time: usize,
        radius: usize,
    ) -> (Vec<u8>, LikelihoodSummary) {
        let mut lks = self.get_profiles_banded(&template, queries, radius);
        let mut template = template.to_vec();
        for _ in 0..repeat_time {
            let new_template = lks.correct_flip(rng);
            let new_lks = self.get_profiles_banded(&new_template, queries, radius);
            let ratio = (new_lks.total_likelihood - lks.total_likelihood)
                .exp()
                .min(1f64);
            if rng.gen_bool(ratio) {
                template = new_template;
                lks = new_lks;
            }
        }
        (template, lks)
    }
    pub fn correct_flip<T: std::borrow::Borrow<[u8]>, R: Rng>(
        &self,
        template: &[u8],
        queries: &[T],
        rng: &mut R,
        repeat_time: usize,
    ) -> (Vec<u8>, LikelihoodSummary) {
        let mut lks = self.get_profiles(&template, queries);
        let mut template = template.to_vec();
        for _ in 0..repeat_time {
            let new_template = lks.correct_flip(rng);
            let new_lks = self.get_profiles(&new_template, queries);
            let ratio = (new_lks.total_likelihood - lks.total_likelihood)
                .exp()
                .min(1f64);
            if rng.gen_bool(ratio) {
                template = new_template;
                lks = new_lks;
            }
        }
        (template, lks)
    }
    fn logsumexp(x: f64, y: f64, z: f64) -> f64 {
        let max = x.max(y).max(z);
        max + ((x - max).exp() + (y - max).exp() + (z - max).exp()).ln()
    }
    /// Return the alignment path between x and y.
    /// In HMM term, it is "viterbi" algorithm.
    pub fn align(&self, xs: &[u8], ys: &[u8]) -> (DPTable, Vec<Op>, f64) {
        let xs: Vec<_> = xs.iter().map(crate::alignment::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::alignment::convert_to_twobit).collect();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        *dptable.get_mut(0, 0, State::Ins) = EP;
        *dptable.get_mut(0, 0, State::Del) = EP;
        for i in 1..xs.len() + 1 {
            *dptable.get_mut(i, 0, State::Mat) = EP;
            *dptable.get_mut(i, 0, State::Ins) = EP;
            *dptable.get_mut(i, 0, State::Del) = EP;
        }
        for j in 1..ys.len() + 1 {
            *dptable.get_mut(0, j, State::Mat) = EP;
            *dptable.get_mut(0, j, State::Ins) = EP;
            *dptable.get_mut(0, j, State::Del) = EP;
        }
        for (i, &x) in xs.iter().enumerate().map(|(p, x)| (p + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(p, y)| (p + 1, y)) {
                let mat = dptable
                    .get(i - 1, j - 1, State::Mat)
                    .max(dptable.get(i - 1, j - 1, State::Ins))
                    .max(dptable.get(i - 1, j - 1, State::Del))
                    + self.log_mat
                    + self.log_mat_emit[((x << 2) | y) as usize];
                *dptable.get_mut(i, j, State::Mat) = mat;
                let del = (dptable.get(i - 1, j, State::Mat) + self.log_gap_open)
                    .max(dptable.get(i - 1, j, State::Del) + self.log_gap_ext)
                    .max(dptable.get(i - 1, j, State::Ins) + self.log_gap_switch)
                    + self.log_gap_emit[x as usize];
                *dptable.get_mut(i, j, State::Del) = del;
                let ins = (dptable.get(i, j - 1, State::Mat) + self.log_gap_open)
                    .max(dptable.get(i, j - 1, State::Del) + self.log_gap_switch)
                    .max(dptable.get(i, j - 1, State::Ins) + self.log_gap_ext)
                    + self.log_gap_emit[y as usize];
                *dptable.get_mut(i, j, State::Ins) = ins;
            }
        }
        let (max_state, max_lk) = vec![State::Mat, State::Ins, State::Del]
            .iter()
            .map(|&s| (s, dptable.get(xs.len(), ys.len(), s)))
            .max_by(|x, y| (x.1).partial_cmp(&(y.1)).unwrap())
            .unwrap();
        let (mut i, mut j, mut state) = (xs.len(), ys.len(), max_state);
        let mut ops: Vec<Op> = vec![];
        while i > 0 && j > 0 {
            let diff = 0.00000000001;
            let (x, y) = (xs[i - 1], ys[j - 1]);
            let lk = dptable.get(i, j, state);
            ops.push(state.into());
            match state {
                State::Mat => {
                    let mat_lk = lk - (self.log_mat + self.log_mat_emit[((x << 2) | y) as usize]);
                    let mat = dptable.get(i - 1, j - 1, State::Mat);
                    let del = dptable.get(i - 1, j - 1, State::Del);
                    let ins = dptable.get(i - 1, j - 1, State::Ins);
                    if (mat_lk - mat).abs() < diff {
                        state = State::Mat;
                    } else if (mat_lk - del).abs() < diff {
                        state = State::Del;
                    } else {
                        assert!((mat_lk - ins).abs() < diff);
                        state = State::Ins;
                    }
                    i -= 1;
                    j -= 1;
                }
                State::Del => {
                    let del_lk = lk - self.log_gap_emit[x as usize];
                    let mat = dptable.get(i - 1, j, State::Mat) + self.log_gap_open;
                    let del = dptable.get(i - 1, j, State::Del) + self.log_gap_ext;
                    let ins = dptable.get(i - 1, j, State::Ins) + self.log_gap_switch;
                    if (del_lk - mat).abs() < diff {
                        state = State::Mat;
                    } else if (del_lk - del).abs() < diff {
                        state = State::Del;
                    } else {
                        assert!((del_lk - ins).abs() < diff);
                        state = State::Ins;
                    }
                    i -= 1;
                }
                State::Ins => {
                    let ins_lk = lk - self.log_gap_emit[y as usize];
                    let mat = dptable.get(i, j - 1, State::Mat) + self.log_gap_open;
                    let del = dptable.get(i, j - 1, State::Del) + self.log_gap_switch;
                    let ins = dptable.get(i, j - 1, State::Ins) + self.log_gap_ext;
                    if (ins_lk - mat).abs() < diff {
                        state = State::Mat;
                    } else if (ins_lk - del).abs() < diff {
                        state = State::Del;
                    } else {
                        assert!((ins_lk - ins).abs() < diff);
                        state = State::Ins;
                    }
                    j -= 1;
                }
            }
        }
        while i > 0 {
            i -= 1;
            ops.push(Op::Del);
        }
        while j > 0 {
            j -= 1;
            ops.push(Op::Ins);
        }
        ops.reverse();
        (dptable, ops, max_lk)
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
    #[test]
    fn forward() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 300);
        let profile = gen_seq::PROFILE;
        let hmm = PHMM::default();
        for i in 0..10 {
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let (_, lkb) = hmm.likelihood_banded(&template, &seq, 100).unwrap();
            let (_, lk) = hmm.likelihood(&template, &seq);
            assert!((lkb - lk).abs() < 10., "{},{},{}", i, lkb, lk);
        }
    }
    #[test]
    fn forward_short() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 10);
        let hmm = PHMM::default();
        for i in 0..10 {
            let seq = gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
            let (_, lkb) = hmm.likelihood_banded(&template, &seq, 5).unwrap();
            let (_, lk) = hmm.likelihood(&template, &seq);
            if (lkb - lk).abs() > 0.1 {
                eprintln!("{}", String::from_utf8_lossy(&template));
                eprintln!("{}", String::from_utf8_lossy(&seq));
            }
            assert!((lkb - lk).abs() < 1f64, "{},{},{}", i, lkb, lk);
        }
    }
    #[test]
    fn forward_banded_test() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 30);
        let hmm = PHMM::default();
        let radius = 4;
        for _ in 0..10 {
            let seq = gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
            let (fwd, centers) = hmm.forward_banded(&template, &seq, radius);
            let k = template.len() + seq.len();
            let u_in_dp = template.len() + radius - centers[k];
            assert!(fwd.get_check(k, u_in_dp, State::Mat).is_some());
            let table = hmm.forward(&template, &seq);
            let lk_banded = PHMM::logsumexp(
                fwd.get(k, u_in_dp, State::Mat),
                fwd.get(k, u_in_dp, State::Del),
                fwd.get(k, u_in_dp, State::Ins),
            );
            let lk = table.get_total_lk(template.len(), seq.len());
            assert!((lk - lk_banded).abs() < 0.001, "{},{}", lk, lk_banded);
            for i in 0..template.len() + 1 {
                for j in 0..seq.len() + 1 {
                    let x = table.get(i, j, State::Del);
                    if EP < x {
                        print!("{:.2}\t", x);
                    } else {
                        print!("{:.2}\t", 1f64);
                    }
                }
                println!();
            }
            for k in 0..template.len() + seq.len() + 1 {
                let center = centers[k];
                for (u, ((&mat, &del), &ins)) in fwd
                    .get_row(k, State::Mat)
                    .iter()
                    .zip(fwd.get_row(k, State::Del).iter())
                    .zip(fwd.get_row(k, State::Ins).iter())
                    .enumerate()
                    .take(2 * radius + 1)
                {
                    let u = (u + center) as isize - radius as isize;
                    let i = u;
                    let j = k as isize - u;
                    if 0 <= u && u <= template.len() as isize && 0 <= j && j <= seq.len() as isize {
                        let (i, j) = (i as usize, j as usize);
                        if 3f64 + EP + 1. < mat {
                            assert!((table.get(i, j, State::Mat) - mat).abs() < 2.);
                        }
                        if EP < del {
                            let del_exact = table.get(i, j, State::Del);
                            if 2f64 < (del_exact - del).abs() {
                                println!("{},{}", i, j);
                            }
                            assert!((del_exact - del).abs() < 2., "E{},B{}", del_exact, del);
                        }
                        if 3f64 + EP < ins {
                            let ins_exact = table.get(i, j, State::Ins);
                            assert!((ins_exact - ins).abs() < 2., "{},{}", ins_exact, ins);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn backward_banded_test() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 30);
        let hmm = PHMM::default();
        let radius = 10;
        for _ in 0..10 {
            let seq = gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
            let (_, centers) = hmm.forward_banded(&template, &seq, radius);
            let table = hmm.backward(&template, &seq);
            let bwd = hmm.backward_banded(&template, &seq, radius, &centers);
            println!();
            for i in 0..template.len() + 1 {
                for j in 0..seq.len() + 1 {
                    print!("{:.2}\t", table.get(i, j, State::Mat));
                }
                println!();
            }
            println!();
            let mut dump_mat = vec![vec![0f64; seq.len() + 1]; template.len() + 1];
            for k in 0..template.len() + seq.len() + 1 {
                let center = centers[k];
                //for (pos, &mat) in b_mat[k].iter().enumerate() {
                for (pos, &mat) in bwd.get_row(k, State::Mat).iter().enumerate() {
                    let u = (pos + center) as isize - radius as isize;
                    let (i, j) = (u, k as isize - u);
                    if (0..template.len() as isize + 1).contains(&i)
                        && (0..seq.len() as isize + 1).contains(&j)
                    {
                        dump_mat[i as usize][j as usize] = mat
                    }
                }
            }
            for line in dump_mat {
                for x in line {
                    if EP < x {
                        print!("{:.2}\t", x);
                    } else {
                        print!("{:.2}\t", 1f64);
                    }
                }
                println!();
            }

            for k in 0..template.len() + seq.len() + 1 {
                let center = centers[k];
                for (u, ((&mat, &del), &ins)) in bwd
                    .get_row(k, State::Mat)
                    .iter()
                    .zip(bwd.get_row(k, State::Del).iter())
                    .zip(bwd.get_row(k, State::Ins).iter())
                    .enumerate()
                    .take(2 * radius + 1)
                {
                    let u = (u + center) as isize - radius as isize;
                    let (i, j) = (u, k as isize - u);
                    if 0 <= u && u <= template.len() as isize && 0 <= j && j <= seq.len() as isize {
                        let (i, j) = (i as usize, j as usize);
                        if EP < mat {
                            let mat_exact = table.get(i, j, State::Mat);
                            assert!(
                                (mat_exact - mat).abs() < 2.,
                                "{},{},{},{}",
                                mat_exact,
                                mat,
                                i,
                                j
                            );
                        }
                        if 3f64 + EP < del {
                            assert!((table.get(i, j, State::Del) - del).abs() < 2.);
                        }
                        if 3f64 + EP < ins {
                            assert!((table.get(i, j, State::Ins) - ins).abs() < 2.);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn forward_backward_test() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 100);
        let hmm = PHMM::default();
        let radius = 10;
        let profile = gen_seq::Profile {
            sub: 0.01,
            del: 0.01,
            ins: 0.01,
        };
        for _ in 0..100 {
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let profile_exact = hmm.get_profile(&template, &seq);
            let profile_banded = hmm.get_profile_banded(&template, &seq, radius).unwrap();
            assert!((profile_banded.total_likelihood - profile_exact.total_likelihood).abs() < 0.1);
            for (x, y) in profile_banded
                .match_prob
                .iter()
                .zip(profile_exact.match_prob.iter())
            {
                assert!((x - y).abs() < 0.1f64);
            }
            for (x, y) in profile_banded
                .deletion_prob
                .iter()
                .zip(profile_exact.deletion_prob.iter())
            {
                assert!((x - y).abs() < 0.1f64);
            }
            for (x, y) in profile_banded
                .insertion_prob
                .iter()
                .zip(profile_exact.insertion_prob.iter())
            {
                assert!((x - y).abs() < 0.1f64);
            }
            for (x, y) in profile_banded
                .match_bases
                .iter()
                .zip(profile_exact.match_bases.iter())
            {
                let diff = x
                    .iter()
                    .zip(y.iter())
                    .map(|(x, y)| if x == y { 0 } else { 1 })
                    .sum::<u8>();
                assert_eq!(diff, 0, "{:?},{:?}", x, y);
            }
            //     let mut total = 0;
            //     for (x, y) in profile_banded
            //         .insertion_bases
            //         .iter()
            //         .zip(profile_exact.insertion_bases.iter())
            //     {
            //         println!("{:?}\t{:?}", x, y);
            //         let diff = x
            //             .iter()
            //             .zip(y.iter())
            //             .map(|(x, y)| if x == y { 0 } else { 1 })
            //             .sum::<u32>();
            //         total += diff;
            //     }
            //     println!(
            //         "{}\n{}",
            //         String::from_utf8_lossy(&template),
            //         String::from_utf8_lossy(&seq)
            //     );
            //     assert_eq!(total, 0, "{}", template.len());
        }
    }
}
