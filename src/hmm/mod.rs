//! An tiny implementation of pair hidden Markov models.
use crate::op::Op;
use crate::padseq;
use rand::Rng;
pub mod guided;
/// A pair hidden Markov model.
/// To access the output prbabilities,
/// call `.prob_of()` instead of direct membership access.
/// In the membership description below, `X` means some arbitrary state except end state.
/// As a principle of thumb, statistics function `*` and `*_banded`
/// (`likelihood` and `likelihood_banded`, for example) would return the same type of values.
/// If you want to more fine resolution outpt, please consult more specific function such as `forward` or `forward_banded`.
#[derive(Debug, Clone)]
pub struct PairHiddenMarkovModel {
    /// Pr{X->Mat}
    pub mat_ext: f64,
    pub mat_from_ins: f64,
    pub mat_from_del: f64,
    /// Pr{Mat->Ins}
    pub ins_open: f64,
    /// Pr{Mat->Del}
    pub del_open: f64,
    /// Pr{Ins->Del}
    pub del_from_ins: f64,
    /// Pr{Del->Ins}.
    pub ins_from_del: f64,
    /// Pr{Del->Del}
    pub del_ext: f64,
    /// Pr{Ins->Ins}
    pub ins_ext: f64,
    // Pr{(-,base)|Del}. Bases are A,C,G,T, '-' and NULL. The last two "bases" are defined just for convinience.
    del_emit: [f64; 6],
    // Pr{(base,-)|Ins}
    ins_emit: [f64; 6],
    // Pr{(base,base)|Mat}
    mat_emit: [f64; 64],
}

/// Shorthand for PairHiddenMarkovModel.
#[allow(clippy::upper_case_acronyms)]
pub type PHMM = PairHiddenMarkovModel;

// Samll value.
const EP: f64 = -10000000000000000000000000000000f64;
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

#[derive(Debug, Clone, Copy)]
pub enum State {
    Mat,
    Del,
    Ins,
}

impl std::convert::From<State> for usize {
    fn from(state: State) -> usize {
        match state {
            State::Mat => 0,
            State::Del => 1,
            State::Ins => 2,
        }
    }
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
        PHMM::as_reversible(
            match_prob,
            gap_ext_prob,
            &gap_output,
            &match_output,
            quit_prob,
        )
    }
}

impl PHMM {
    /// construct a new pair reversible HMM.
    /// In reversible, I mean that the HMM is synmetric with respect to switching the reference and the query,
    /// and w.r.t reversing the reference sequence and the query sequence.
    /// In other words, it has the same probability to transit from deletion/insertion <-> matches,
    /// same emittion probability on the deletion/insertion states, and so on.
    /// # Example
    /// ```rust
    /// use kiley::hmm::PHMM;
    /// let match_prob = 0.8;
    /// let gap_ext_prob = 0.1;
    /// let gap_output = [(4f64).recip(); 4];
    /// let match_output = [
    /// [0.7, 0.1, 0.1, 0.1],
    /// [0.1, 0.7, 0.1, 0.1],
    /// [0.1, 0.1, 0.7, 0.1],
    /// [0.1, 0.1, 0.1, 0.7],
    /// ];
    /// let quit_prob = 0.001;
    /// PHMM::as_reversible(
    ///   match_prob,
    ///   gap_ext_prob,
    ///   &gap_output,
    ///   &match_output,
    ///   quit_prob,
    /// );
    /// ```
    #[allow(clippy::wrong_self_convention)]
    pub fn as_reversible(
        mat: f64,
        gap_ext: f64,
        gap_output: &[f64; 4],
        mat_output: &[[f64; 4]],
        _quit_prob: f64,
    ) -> Self {
        assert!(mat.is_sign_positive());
        assert!(gap_ext.is_sign_positive());
        assert!(mat + gap_ext <= 1f64);
        let gap_open = (1f64 - mat) / 2f64;
        let gap_switch = 1f64 - gap_ext - mat;
        // let alive_prob = 1f64 - quit_prob;
        let mut gap_emit = [0f64; 6];
        gap_emit[..4].clone_from_slice(&gap_output[..4]);
        // Maybe we should have the matching function to compute this matrix...
        let mat_emit = {
            let mut slots = [0f64; 8 * 8];
            for i in 0..4 {
                for j in 0..4 {
                    slots[(i << 3) | j] = mat_output[i][j];
                }
            }
            slots
        };
        Self {
            mat_ext: mat,
            mat_from_del: mat,
            mat_from_ins: mat,
            ins_open: gap_open,
            del_open: gap_open,
            ins_from_del: gap_switch,
            del_from_ins: gap_switch,
            del_ext: gap_ext,
            ins_ext: gap_ext,
            del_emit: gap_emit,
            ins_emit: gap_emit,
            mat_emit,
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
        let (k, u) = ((xs.len() + ys.len()) as isize, xs.len());
        let u_in_dp = (u + radius) as isize - centers[k as usize];
        let max_lk = Self::logsumexp(
            dp.get_check(k, u_in_dp, State::Mat)?,
            dp.get_check(k, u_in_dp, State::Del)?,
            dp.get_check(k, u_in_dp, State::Ins)?,
        );
        // Maybe this code is very slow...
        let mut lks: Vec<Vec<f64>> = vec![vec![]; xs.len() + 1];
        for k in 0..(xs.len() + ys.len() + 1) as isize {
            let u_center = centers[k as usize];
            for (u, ((&x, &y), &z)) in dp
                .get_row(k, State::Mat)
                .iter()
                .zip(dp.get_row(k, State::Del).iter())
                .zip(dp.get_row(k, State::Ins).iter())
                .enumerate()
            {
                let u = u as isize;
                let radius = radius as isize;
                let i = if radius <= u + u_center && u + u_center - radius < xs.len() as isize + 1 {
                    u + u_center - radius as isize
                } else {
                    continue;
                };
                lks[i as usize].push(x);
                lks[i as usize].push(y);
                lks[i as usize].push(z);
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
    pub fn forward_banded(&self, xs: &[u8], ys: &[u8], radius: usize) -> (DiagonalDP, Vec<isize>) {
        let xs = padseq::PadSeq::new(xs);
        let ys = padseq::PadSeq::new(ys);
        let mut centers: Vec<isize> = vec![0, 0, 1];
        let mut dp = DiagonalDP::new(xs.len() + ys.len() + 1, 2 * radius + 1, 0f64);
        let mut scaling_factor: Vec<f64> = Vec::with_capacity(xs.len() + ys.len() + 2);
        let radius = radius as isize;
        // The first diagonal.
        {
            *dp.get_mut(0, radius, State::Mat) = 1f64;
            scaling_factor.push(1f64);
        }
        // The second diagonal.
        {
            *dp.get_mut(1, radius, State::Ins) = self.ins_open * self.ins_emit[ys[0] as usize];
            *dp.get_mut(1, radius + 1, State::Del) = self.del_open * self.del_emit[xs[0] as usize];
            let sum = dp.sum_anti_diagonal(1);
            dp.div_anti_diagonal(1, sum);
            scaling_factor.push(sum);
        }
        for k in 2..(xs.len() + ys.len() + 1) as isize {
            let center = centers[k as usize];
            let matdiff = center - centers[k as usize - 2];
            let gapdiff = center - centers[k as usize - 1];
            let (start, end) = {
                let k = k as usize;
                let radius = radius as usize;
                let center = center as usize;
                let start = radius.saturating_sub(center);
                let end = (xs.len() + 1 - center + radius).min(2 * radius + 1);
                // With respect to the k-coordinate.
                let start = start.max((k - center + radius).saturating_sub(ys.len()));
                let end = end.min(k + 1 - center + radius);
                (start as isize, end as isize)
            };
            let (mut max, mut max_pos) = (0f64, end);
            for pos in start..end {
                let u = pos + center - radius;
                let prev_mat = pos as isize + matdiff;
                let prev_gap = pos as isize + gapdiff;
                // let (i, j) = (u, k - u);
                let (x, y) = (xs[u - 1], ys[k - u - 1]);
                let mat = self.mat_emit[(x << 3 | y) as usize]
                    * (dp.get(k - 2, prev_mat - 1, State::Mat) * self.mat_ext
                        + dp.get(k - 2, prev_mat - 1, State::Del) * self.mat_from_del
                        + dp.get(k - 2, prev_mat - 1, State::Ins) * self.mat_from_ins)
                    / scaling_factor[k as usize - 1];
                let del = (dp.get(k - 1, prev_gap - 1, State::Mat) * self.del_open
                    + dp.get(k - 1, prev_gap - 1, State::Del) * self.del_ext
                    + dp.get(k - 1, prev_gap - 1, State::Ins) * self.del_from_ins)
                    * self.del_emit[x as usize];
                let ins = (dp.get(k - 1, prev_gap, State::Mat) * self.ins_open
                    + dp.get(k - 1, prev_gap, State::Del) * self.ins_from_del
                    + dp.get(k - 1, prev_gap, State::Ins) * self.ins_ext)
                    * self.ins_emit[y as usize];
                *dp.get_mut(k, pos, State::Mat) = mat;
                *dp.get_mut(k, pos, State::Del) = del;
                *dp.get_mut(k, pos, State::Ins) = ins;
                let sum = mat + del + ins;
                if max <= sum {
                    max = sum;
                    max_pos = pos;
                }
            }
            let max_u = max_pos as isize + center - radius;
            if center < max_u {
                centers.push(center + 1);
            } else {
                centers.push(center);
            };
            let sum = dp.sum_anti_diagonal(k);
            dp.div_anti_diagonal(k, sum);
            scaling_factor.push(sum);
        }
        let mut log_factor = 0f64;
        for (k, factor) in scaling_factor
            .iter()
            .enumerate()
            .take(xs.len() + ys.len() + 1)
        {
            log_factor += factor.ln();
            dp.convert_to_ln(k as isize, log_factor);
        }
        (dp, centers)
    }
    fn log(x: &f64) -> f64 {
        assert!(!x.is_sign_negative());
        if x.abs() > 0.00000001 {
            x.ln()
        } else {
            EP
        }
    }
    /// Forward algorithm. Return the raw DP table.
    pub fn forward(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let xs: Vec<_> = xs.iter().map(crate::padseq::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::padseq::convert_to_twobit).collect();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        let log_del_emit: Vec<_> = self.del_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.del_open.ln(), self.ins_open.ln());
        let (log_del_ext, log_ins_ext) = (self.del_ext.ln(), self.ins_ext.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.del_from_ins.ln(), self.ins_from_del.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.mat_from_del.ln(), self.mat_from_ins.ln());
        let log_mat_ext = self.mat_ext.ln();
        let mut del_accum = 0f64;
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            *dptable.get_mut(i, 0, State::Mat) = EP;
            del_accum += log_del_emit[x as usize];
            *dptable.get_mut(i, 0, State::Del) =
                log_del_open + log_del_ext * (i - 1) as f64 + del_accum;
            *dptable.get_mut(i, 0, State::Ins) = EP;
        }
        let mut ins_accum = 0f64;
        for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
            *dptable.get_mut(0, j, State::Mat) = EP;
            *dptable.get_mut(0, j, State::Del) = EP;
            ins_accum += log_ins_emit[y as usize];
            *dptable.get_mut(0, j, State::Ins) =
                log_ins_open + log_ins_ext * (j - 1) as f64 + ins_accum;
        }
        *dptable.get_mut(0, 0, State::Ins) = EP;
        *dptable.get_mut(0, 0, State::Del) = EP;
        for (i, &x) in xs.iter().enumerate().map(|(p, x)| (p + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(p, y)| (p + 1, y)) {
                let mat = Self::logsumexp(
                    dptable.get(i - 1, j - 1, State::Mat) + log_mat_ext,
                    dptable.get(i - 1, j - 1, State::Del) + log_mat_from_del,
                    dptable.get(i - 1, j - 1, State::Ins) + log_mat_from_ins,
                ) + log_mat_emit[((x << 3) | y) as usize];
                *dptable.get_mut(i, j, State::Mat) = mat;
                let del = Self::logsumexp(
                    dptable.get(i - 1, j, State::Mat) + log_del_open,
                    dptable.get(i - 1, j, State::Del) + log_del_ext,
                    dptable.get(i - 1, j, State::Ins) + log_del_from_ins,
                ) + log_del_emit[x as usize];
                *dptable.get_mut(i, j, State::Del) = del;
                let ins = Self::logsumexp(
                    dptable.get(i, j - 1, State::Mat) + log_ins_open,
                    dptable.get(i, j - 1, State::Del) + log_ins_from_del,
                    dptable.get(i, j - 1, State::Ins) + log_ins_ext,
                ) + log_ins_emit[y as usize];
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
        centers: &[isize],
    ) -> DiagonalDP {
        let xs = padseq::PadSeq::new(xs);
        let ys = padseq::PadSeq::new(ys);
        let mut dp = DiagonalDP::new(xs.len() + ys.len() + 1, 2 * radius + 1, 0f64);
        let mut scaling_factors = vec![1f64; xs.len() + ys.len() + 1];
        let radius = radius as isize;
        // Calc the boundary score for each sequence, in other words,
        // calc DP[xs.len()][j] and DP[i][ys.len()] in the usual DP.
        // For DP[i][ys.len()].
        // Get the location corresponding to [xs.len()][ys.len()].
        // Fill the last DP call.
        {
            let (k, u) = ((xs.len() + ys.len()) as isize, xs.len() as isize);
            let u_in_dp = u + radius - centers[k as usize];
            *dp.get_mut(k, u_in_dp, State::Mat) = 1f64;
            *dp.get_mut(k, u_in_dp, State::Del) = 1f64;
            *dp.get_mut(k, u_in_dp, State::Ins) = 1f64;
            let sum = dp.sum_anti_diagonal(k);
            dp.div_anti_diagonal(k, sum);
            scaling_factors[k as usize] = sum;
        }
        // Filling the 2nd-last DP cell.
        {
            let k = (xs.len() + ys.len() - 1) as isize;
            for &u in &[xs.len() - 1, xs.len()] {
                let (i, j) = (u as isize, k as isize - u as isize);
                let u = u as isize + radius - centers[k as usize];
                if i == xs.len() as isize {
                    let emit = self.ins_emit[ys[j] as usize];
                    *dp.get_mut(k, u, State::Mat) = emit * self.ins_open;
                    *dp.get_mut(k, u, State::Del) = emit * self.ins_from_del;
                    *dp.get_mut(k, u, State::Ins) = emit * self.ins_ext;
                } else if j == ys.len() as isize {
                    let emit = self.del_emit[xs[i] as usize];
                    *dp.get_mut(k, u, State::Mat) = emit * self.del_open;
                    *dp.get_mut(k, u, State::Del) = emit * self.del_ext;
                    *dp.get_mut(k, u, State::Ins) = emit * self.del_from_ins;
                } else {
                    unreachable!();
                }
            }
            let sum = dp.sum_anti_diagonal(k);
            dp.div_anti_diagonal(k, sum);
            scaling_factors[k as usize] = sum / scaling_factors[k as usize + 1];
        }
        for k in (0..(xs.len() + ys.len() - 1) as isize).rev() {
            let center = centers[k as usize];
            let matdiff = center as isize - centers[k as usize + 2] as isize;
            let gapdiff = center as isize - centers[k as usize + 1] as isize;
            let (start, end) = {
                let k = k as usize;
                let radius = radius as usize;
                let center = center as usize;
                let start = radius.saturating_sub(center);
                let end = (xs.len() + 1 - center + radius).min(2 * radius + 1);
                // With respect to the k-coordinate.
                let start = start.max((k - center + radius).saturating_sub(ys.len()));
                let end = end.min(k + 1 - center + radius);
                (start as isize, end as isize)
            };
            for pos in start..end {
                let u = pos + center - radius;
                // let (i, j) = (u, k - u);
                // Previous, prev-previous position.
                let u_mat = pos as isize + matdiff + 1;
                let u_gap = pos as isize + gapdiff;
                let (x, y) = (xs[u], ys[k - u]);
                let ins = self.ins_open
                    * self.ins_emit[ys[k - u] as usize]
                    * dp.get(k + 1, u_gap, State::Ins);
                let del = self.del_open
                    * self.del_emit[xs[u] as usize]
                    * dp.get(k + 1, u_gap + 1, State::Del);
                let mat = self.mat_ext
                    * self.mat_emit[(x << 3 | y) as usize]
                    * dp.get(k + 2, u_mat, State::Mat)
                    / scaling_factors[k as usize + 1];
                *dp.get_mut(k, pos, State::Mat) = mat + del + ins;
                {
                    let ins = ins / self.ins_open * self.ins_from_del;
                    let del = del / self.del_open * self.del_ext;
                    let mat = mat / self.mat_ext * self.mat_from_del;
                    *dp.get_mut(k, pos, State::Del) = mat + del + ins;
                }
                {
                    let ins = ins / self.ins_open * self.ins_ext;
                    let del = del / self.del_open * self.del_from_ins;
                    let mat = mat / self.mat_ext * self.mat_from_ins;
                    *dp.get_mut(k, pos, State::Ins) = mat + del + ins;
                }
            }
            let sum = dp.sum_anti_diagonal(k);
            dp.div_anti_diagonal(k, sum);
            scaling_factors[k as usize] = sum;
        }
        let mut log_factor = 0f64;
        for k in (0..xs.len() + ys.len() + 1).rev() {
            log_factor += scaling_factors[k].ln();
            dp.convert_to_ln(k as isize, log_factor);
        }
        dp
    }
    /// Naive implementation of backward algorithm.
    pub fn backward(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let xs: Vec<_> = xs.iter().map(crate::padseq::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::padseq::convert_to_twobit).collect();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        *dptable.get_mut(xs.len(), ys.len(), State::Mat) = 0f64;
        *dptable.get_mut(xs.len(), ys.len(), State::Del) = 0f64;
        *dptable.get_mut(xs.len(), ys.len(), State::Ins) = 0f64;
        let mut gap = 0f64;
        let log_del_emit: Vec<_> = self.del_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.del_open.ln(), self.ins_open.ln());
        let (log_del_ext, log_ins_ext) = (self.del_ext.ln(), self.ins_ext.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.del_from_ins.ln(), self.ins_from_del.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.mat_from_del.ln(), self.mat_from_ins.ln());
        let log_mat_ext = self.mat_ext.ln();
        for (i, &x) in xs.iter().enumerate().rev() {
            gap += log_del_emit[x as usize];
            *dptable.get_mut(i, ys.len(), State::Del) = log_del_ext * (xs.len() - i) as f64 + gap;
            *dptable.get_mut(i, ys.len(), State::Ins) =
                log_del_from_ins + log_del_ext * (xs.len() - i - 1) as f64 + gap;
            *dptable.get_mut(i, ys.len(), State::Mat) =
                log_del_open + log_del_ext * (xs.len() - i - 1) as f64 + gap;
        }
        gap = 0f64;
        for (j, &y) in ys.iter().enumerate().rev() {
            gap += log_ins_emit[y as usize];
            *dptable.get_mut(xs.len(), j, State::Ins) = log_ins_ext * (ys.len() - j) as f64 + gap;
            *dptable.get_mut(xs.len(), j, State::Del) =
                log_ins_from_del + log_ins_ext * (ys.len() - j - 1) as f64 + gap;
            *dptable.get_mut(xs.len(), j, State::Mat) =
                log_ins_open + log_ins_ext * (ys.len() - j - 1) as f64 + gap;
        }
        for (i, &x) in xs.iter().enumerate().rev() {
            for (j, &y) in ys.iter().enumerate().rev() {
                // Match state;
                let mat = log_mat_ext
                    + log_mat_emit[(x << 3 | y) as usize]
                    + dptable.get(i + 1, j + 1, State::Mat);
                let del =
                    log_del_open + log_del_emit[x as usize] + dptable.get(i + 1, j, State::Del);
                let ins =
                    log_ins_open + log_ins_emit[y as usize] + dptable.get(i, j + 1, State::Ins);
                *dptable.get_mut(i, j, State::Mat) = Self::logsumexp(mat, del, ins);
                // Del state.
                {
                    let mat = mat - log_mat_ext + log_mat_from_del;
                    let del = del - log_del_open + log_del_ext;
                    let ins = ins - log_ins_open + log_del_from_ins;
                    *dptable.get_mut(i, j, State::Del) = Self::logsumexp(mat, del, ins);
                }
                // Ins state
                {
                    let mat = mat - log_mat_ext + log_mat_from_ins;
                    let del = del - log_del_open + log_del_from_ins;
                    let ins = ins - log_ins_open + log_ins_ext;
                    *dptable.get_mut(i, j, State::Ins) = Self::logsumexp(mat, del, ins);
                }
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
            let (k, u) = ((xs.len() + ys.len()) as isize, xs.len());
            let u_in_dp = (u + radius) as isize - centers[k as usize];
            Self::logsumexp(
                fwd.get_check(k, u_in_dp, State::Mat)?,
                fwd.get_check(k, u_in_dp, State::Del)?,
                fwd.get_check(k, u_in_dp, State::Ins)?,
            )
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
        for (k, &center) in centers.iter().take(xs.len() + ys.len() + 1).enumerate() {
            let (k, center) = (k as isize, center as usize);
            // With respect to the u-corrdinate.
            let k = k as usize;
            let start = (radius + 1).saturating_sub(center);
            let end = (xs.len() + 1 - center + radius).min(2 * radius + 1);
            // With respect to the k-coordinate.
            let start = start.max((k - center + radius).saturating_sub(ys.len()));
            let end = end.min(k + 1 - center + radius);
            for u in start..end {
                let mat = fbwd.get(k as isize, u as isize, State::Mat);
                let del = fbwd.get(k as isize, u as isize, State::Del);
                let ins = fbwd.get(k as isize, u as isize, State::Ins);
                let i = u + center - radius;
                let j = k - i;
                match_prob[i - 1].push(mat);
                del_prob[i - 1].push(del);
                ins_prob[i - 1].push(ins);
                let y = if j == 0 { b'A' } else { ys[j - 1] };
                if match_max_prob[i - 1].0 < mat {
                    match_max_prob[i - 1] = (mat, y);
                }
                if ins_max_prob[i - 1].0 < ins {
                    ins_max_prob[i - 1] = (ins, y);
                }
            }
        }
        let exp_sum = |xs: &Vec<f64>| {
            let max = xs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            max.exp() * xs.iter().map(|x| (x - max).exp()).sum::<f64>()
        };
        let match_prob: Vec<_> = match_prob.iter().map(exp_sum).collect();
        let deletion_prob: Vec<_> = del_prob.iter().map(exp_sum).collect();
        let insertion_prob: Vec<_> = ins_prob.iter().map(exp_sum).collect();
        let match_bases: Vec<_> = match_max_prob
            .iter()
            .map(|x| {
                let mut slot = [0u8; 4];
                slot[crate::padseq::LOOKUP_TABLE[x.1 as usize] as usize] = 1;
                slot
            })
            .collect();
        let insertion_bases: Vec<_> = ins_max_prob
            .iter()
            .map(|x| {
                let mut slot = [0u8; 4];
                slot[crate::padseq::LOOKUP_TABLE[x.1 as usize] as usize] = 1;
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
            // likelihood_trajectry,
        };
        // let forward_time = (forward - start).as_nanos();
        // let backward_time = (backward - forward).as_nanos();
        // let alloc_time = (alloc - backward).as_nanos();
        // let rest_time = (rest - alloc).as_nanos();
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
                for &s in &[State::Mat, State::Del, State::Ins] {
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
                    .max_by(|x, y| (x.1).partial_cmp(y.1).unwrap())
                    .map(|(j, _)| {
                        let mut slot = [0; 4];
                        slot[crate::padseq::LOOKUP_TABLE[ys[j - 1] as usize] as usize] += 1;
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
                    .max_by(|x, y| (x.1).partial_cmp(y.1).unwrap())
                    .map(|(j, _)| {
                        let mut slot = [0u8; 4];
                        slot[crate::padseq::LOOKUP_TABLE[ys[j - 1] as usize] as usize] += 1;
                        slot
                    })
                    .unwrap()
            })
            .collect();
        // let (likelihood_trajectry, _, _) = f_dp.lks_in_row_by_state();
        LikelihoodSummary {
            match_prob,
            match_bases,
            insertion_prob,
            insertion_bases,
            deletion_prob,
            total_likelihood: lk,
            // likelihood_trajectry,
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
        let new_template = summary.correct(template);
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
    /// Correct input until convergence.
    pub fn correct<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
    ) -> (Vec<u8>, LikelihoodSummary) {
        let lks: Option<LikelihoodSummary> = queries
            .iter()
            .map(|x| self.get_profile(template, x.borrow()))
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
            .filter_map(|x| self.get_profile_banded(template, x.borrow(), radius))
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
            .map(|x| self.get_profile(template, x.borrow()))
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
        let mut lks = self.get_profiles_banded(template, queries, radius);
        let mut template = template.to_vec();
        println!("{:.0}", lks.total_likelihood);
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
        let mut lks = self.get_profiles(template, queries);
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
        let xs: Vec<_> = xs.iter().map(crate::padseq::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::padseq::convert_to_twobit).collect();
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
        let log_del_emit: Vec<_> = self.del_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.del_open.ln(), self.ins_open.ln());
        let (log_del_ext, log_ins_ext) = (self.del_ext.ln(), self.ins_ext.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.del_from_ins.ln(), self.ins_from_del.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.mat_from_del.ln(), self.mat_from_ins.ln());
        let log_mat_ext = self.mat_ext.ln();
        for (i, &x) in xs.iter().enumerate().map(|(p, x)| (p + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(p, y)| (p + 1, y)) {
                let mat = (dptable.get(i - 1, j - 1, State::Mat) + log_mat_ext)
                    .max(dptable.get(i - 1, j - 1, State::Ins) + log_mat_from_ins)
                    .max(dptable.get(i - 1, j - 1, State::Del) + log_mat_from_del)
                    + log_mat_emit[((x << 3) | y) as usize];
                *dptable.get_mut(i, j, State::Mat) = mat;
                let del = (dptable.get(i - 1, j, State::Mat) + log_del_open)
                    .max(dptable.get(i - 1, j, State::Del) + log_del_ext)
                    .max(dptable.get(i - 1, j, State::Ins) + log_del_from_ins)
                    + log_del_emit[x as usize];
                *dptable.get_mut(i, j, State::Del) = del;
                let ins = (dptable.get(i, j - 1, State::Mat) + log_ins_open)
                    .max(dptable.get(i, j - 1, State::Del) + log_ins_from_del)
                    .max(dptable.get(i, j - 1, State::Ins) + log_ins_ext)
                    + log_ins_emit[y as usize];
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
                    let mat_lk = lk - log_mat_emit[((x << 3) | y) as usize];
                    let mat = dptable.get(i - 1, j - 1, State::Mat) + log_mat_ext;
                    let del = dptable.get(i - 1, j - 1, State::Del) + log_mat_from_del;
                    let ins = dptable.get(i - 1, j - 1, State::Ins) + log_mat_from_ins;
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
                    let del_lk = lk - log_del_emit[x as usize];
                    let mat = dptable.get(i - 1, j, State::Mat) + log_del_open;
                    let del = dptable.get(i - 1, j, State::Del) + log_del_ext;
                    let ins = dptable.get(i - 1, j, State::Ins) + log_del_from_ins;
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
                    let ins_lk = lk - log_ins_emit[y as usize];
                    let mat = dptable.get(i, j - 1, State::Mat) + log_ins_open;
                    let del = dptable.get(i, j - 1, State::Del) + log_ins_from_del;
                    let ins = dptable.get(i, j - 1, State::Ins) + log_ins_ext;
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

#[derive(Debug, Clone)]
pub struct DiagonalDP {
    // Mat,Del,Ins, Mat,Del,Ins,....,
    // 1-dimensionalized 2D array.
    data: Vec<f64>,
    column: usize,
    // row: usize,
}

const OFFSET: usize = 3;
impl DiagonalDP {
    pub fn new(row: usize, column: usize, x: f64) -> Self {
        Self {
            data: vec![x; 3 * (column + 2 * OFFSET) * (row + 2 * OFFSET)],
            // row,
            column,
        }
    }
    fn location(&self, k: isize, u: isize, state: State) -> usize {
        let k = (k + OFFSET as isize) as usize;
        let u = (u + OFFSET as isize) as usize;
        let diff: usize = state.into();
        3 * (k * (self.column + 2 * OFFSET) + u) + diff
    }
    pub fn get(&self, k: isize, u: isize, state: State) -> f64 {
        let pos = self.location(k, u, state);
        self.data[pos]
    }
    pub fn get_check(&self, k: isize, u: isize, state: State) -> Option<f64> {
        self.data.get(self.location(k, u, state)).copied()
    }
    pub fn get_mut(&mut self, k: isize, u: isize, state: State) -> &mut f64 {
        let location = self.location(k, u, state);
        self.data.get_mut(location).unwrap()
    }
    pub fn get_row(&self, k: isize, state: State) -> Vec<f64> {
        let k = (k + OFFSET as isize) as usize;
        let collen = self.column + 2 * OFFSET;
        let state: usize = state.into();
        let start = 3 * (k * collen + OFFSET) + state;
        let end = 3 * ((k + 1) * collen - OFFSET) + state;
        self.data[start..end].iter().step_by(3).copied().collect()
    }
    /// Return the sum over state for each cell in the k-th anti-diagonal.
    pub fn get_row_statesum(&self, k: isize) -> Vec<f64> {
        let k = (k + OFFSET as isize) as usize;
        let collen = self.column + 2 * OFFSET;
        let start = 3 * (k * collen + OFFSET);
        let end = 3 * ((k + 1) * collen - OFFSET);
        self.data[start..end]
            .chunks_exact(3)
            .map(|x| x[0] + x[1] + x[2])
            .collect()
    }
    pub fn add(&mut self, other: &Self) {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, &y)| *x += y);
    }
    pub fn sub_scalar(&mut self, a: f64) {
        self.data.iter_mut().for_each(|x| *x -= a);
    }
    pub fn sum_anti_diagonal(&self, k: isize) -> f64 {
        let k = (k + OFFSET as isize) as usize;
        let collen = self.column + 2 * OFFSET;
        let range = 3 * (k * collen + OFFSET)..3 * ((k + 1) * collen - OFFSET);
        self.data[range].iter().sum::<f64>()
    }
    pub fn div_anti_diagonal(&mut self, k: isize, div: f64) {
        let k = (k + OFFSET as isize) as usize;
        let collen = self.column + 2 * OFFSET;
        let range = 3 * (k * collen + OFFSET)..3 * ((k + 1) * collen - OFFSET);
        self.data[range].iter_mut().for_each(|x| *x /= div);
    }
    /// Convert each cell in each state in the anti-diagonal into x.ln() + lk.
    pub fn convert_to_ln(&mut self, k: isize, lk: f64) {
        let k = (k + OFFSET as isize) as usize;
        let collen = self.column + 2 * OFFSET;
        let start = 3 * (k * collen + OFFSET);
        let end = 3 * ((k + 1) * collen - OFFSET);
        let elements = self.data[start..end].iter_mut();
        for elm in elements {
            *elm = if std::f64::EPSILON < *elm {
                elm.ln() + lk
            } else {
                EP
            };
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

pub fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.;
    }
    let max = xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    let sum = xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    assert!(sum >= 0., "{:?}->{}", xs, sum);
    max + sum
}

// /// Operations.
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum Op {
//     Match,
//     Del,
//     Ins,
// }

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
        // self.likelihood_trajectry
        //     .iter_mut()
        //     .zip(other.likelihood_trajectry.iter())
        //     .for_each(|(x, &y)| {
        //         *x = if *x < y {
        //             y + (1f64 + (*x - y).exp()).ln()
        //         } else {
        //             *x + (1f64 + (y - *x).exp()).ln()
        //         }
        //     });
    }
    pub fn div_probs(&mut self, r: f64) {
        self.match_prob.iter_mut().for_each(|x| *x /= r);
        self.deletion_prob.iter_mut().for_each(|x| *x /= r);
        self.insertion_prob.iter_mut().for_each(|x| *x /= r);
    }
    /// Flipping a suspicious erroneous position.
    // TODO: Homopolymer-compression is needed.
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
            let k = (template.len() + seq.len()) as isize;
            let u_in_dp = (template.len() + radius) as isize - centers[k as usize];
            assert!(fwd.get_check(k, u_in_dp, State::Mat).is_some());
            let table = hmm.forward(&template, &seq);
            let lk_banded = PHMM::logsumexp(
                fwd.get(k, u_in_dp, State::Mat),
                fwd.get(k, u_in_dp, State::Del),
                fwd.get(k, u_in_dp, State::Ins),
            );
            let lk = table.get_total_lk(template.len(), seq.len());
            assert!((lk - lk_banded).abs() < 0.001, "{},{}", lk, lk_banded);
            let state = State::Del;
            for i in 0..template.len() + 1 {
                for j in 0..seq.len() + 1 {
                    let x = table.get(i, j, state);
                    if EP < x {
                        print!("{:.1}\t", x);
                    } else {
                        print!("{:.1}\t", 1f64);
                    }
                }
                println!();
            }
            println!();
            let mut dump = vec![vec![EP; seq.len() + 1]; template.len() + 1];
            // for k in 0..template.len() + seq.len() + 1 {
            for (k, center) in centers
                .iter()
                .enumerate()
                .take(template.len() + seq.len() + 1)
            {
                // let center = centers[k];
                for (pos, &lk) in fwd.get_row(k as isize, state).iter().enumerate() {
                    let u = pos as isize + center - radius as isize;
                    let (i, j) = (u, k as isize - u);
                    if (0..template.len() as isize + 1).contains(&i)
                        && (0..seq.len() as isize + 1).contains(&j)
                    {
                        dump[i as usize][j as usize] = lk;
                    }
                }
            }
            for line in dump {
                for x in line {
                    if EP < x {
                        print!("{:.1}\t", x);
                    } else {
                        print!("{:.1}\t", 1f64);
                    }
                }
                println!();
            }
            println!();
            for (k, center) in centers
                .iter()
                .enumerate()
                .take(template.len() + seq.len() + 1)
            {
                // for k in 0..template.len() + seq.len() + 1 {
                //     let center = centers[k];
                let k = k as isize;
                for (u, ((&mat, &del), &ins)) in fwd
                    .get_row(k, State::Mat)
                    .iter()
                    .zip(fwd.get_row(k, State::Del).iter())
                    .zip(fwd.get_row(k, State::Ins).iter())
                    .enumerate()
                    .take(2 * radius - 2)
                    .skip(2)
                {
                    let u = u as isize + center - radius as isize;
                    let i = u;
                    let j = k as isize - u;
                    if 0 <= u && u <= template.len() as isize && 0 <= j && j <= seq.len() as isize {
                        let (i, j) = (i as usize, j as usize);
                        assert!((table.get(i, j, State::Mat) - mat).abs() < 2.);
                        let del_exact = table.get(i, j, State::Del);
                        if 2f64 < (del_exact - del).abs() {
                            println!("{},{}", i, j);
                        }
                        assert!((del_exact - del).abs() < 2., "E{},B{}", del_exact, del);
                        let ins_exact = table.get(i, j, State::Ins);
                        assert!((ins_exact - ins).abs() < 2., "{},{}", ins_exact, ins);
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
        let radius = 5;
        for _ in 0..10 {
            let seq = gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
            let (_, centers) = hmm.forward_banded(&template, &seq, radius);
            let table = hmm.backward(&template, &seq);
            let bwd = hmm.backward_banded(&template, &seq, radius, &centers);
            println!();
            let state = State::Del;
            for i in 0..template.len() + 1 {
                for j in 0..seq.len() + 1 {
                    print!("{:.1}\t", table.get(i, j, state));
                }
                println!();
            }
            println!();
            // for k in 0..template.len() + seq.len() + 1 {
            //     for &x in fwd.get_row(k, state) {
            //         if EP < x {
            //             print!("{:.1}\t", x);
            //         } else {
            //             print!("{:.1}\t", 1f64);
            //         }
            //     }
            //     println!();
            // }
            // for k in 0..template.len() + seq.len() + 1 {
            //     for &x in bwd.get_row(k, state) {
            //         if EP < x {
            //             print!("{:.1}\t", x);
            //         } else {
            //             print!("{:.1}\t", 1f64);
            //         }
            //     }
            //     println!();
            // }
            // let mut dump = vec![vec![EP; seq.len() + 1]; template.len() + 1];
            // for k in 0..template.len() + seq.len() + 1 {
            //     let center = centers[k];
            //     for (pos, &lk) in fwd.get_row(k, state).iter().enumerate() {
            //         let u = (pos + center) as isize - radius as isize;
            //         let (i, j) = (u, k as isize - u);
            //         if (0..template.len() as isize + 1).contains(&i)
            //             && (0..seq.len() as isize + 1).contains(&j)
            //         {
            //             dump[i as usize][j as usize] = lk;
            //         }
            //     }
            // }
            // for line in dump {
            //     for x in line {
            //         if EP < x {
            //             print!("{:.2}\t", x);
            //         } else {
            //             print!("{:.2}\t", 1f64);
            //         }
            //     }
            //     println!();
            // }
            println!();
            let mut dump = vec![vec![EP; seq.len() + 1]; template.len() + 1];
            for (k, center) in centers
                .iter()
                .enumerate()
                .take(template.len() + seq.len() + 1)
            {
                // for k in 0..template.len() + seq.len() + 1 {
                //     let center = centers[k];
                let k = k as isize;
                for (pos, &lk) in bwd.get_row(k, state).iter().enumerate() {
                    let u = pos as isize + center - radius as isize;
                    let (i, j) = (u, k as isize - u);
                    if (0..template.len() as isize + 1).contains(&i)
                        && (0..seq.len() as isize + 1).contains(&j)
                    {
                        dump[i as usize][j as usize] = lk;
                    }
                }
            }
            for line in dump {
                for x in line {
                    if EP < x {
                        print!("{:.1}\t", x);
                    } else {
                        print!("{:.1}\t", 1f64);
                    }
                }
                println!();
            }
            for (k, center) in centers
                .iter()
                .enumerate()
                .take(template.len() + seq.len() + 1)
            {
                // for k in 0..template.len() + seq.len() + 1 {
                //     let center = centers[k];
                let k = k as isize;
                for (u, ((&mat, &del), &ins)) in bwd
                    .get_row(k, State::Mat)
                    .iter()
                    .zip(bwd.get_row(k, State::Del).iter())
                    .zip(bwd.get_row(k, State::Ins).iter())
                    .enumerate()
                    .take(2 * radius - 1)
                    .skip(2)
                {
                    let u = u as isize + center - radius as isize;
                    let (i, j) = (u, k as isize - u);
                    if 0 <= u && u <= template.len() as isize && 0 <= j && j <= seq.len() as isize {
                        let (i, j) = (i as usize, j as usize);
                        let mat_exact = table.get(i, j, State::Mat);
                        assert!(
                            (mat_exact - mat).abs() < 2.,
                            "{},{},{},{}",
                            mat_exact,
                            mat,
                            i,
                            j
                        );
                        let diff = (table.get(i, j, State::Del) - del).abs() < 2f64;
                        assert!(diff, "{},{},{}", diff, i, j);
                        let diff = (table.get(i, j, State::Ins) - ins).abs() < 2.;
                        assert!(diff, "{},{}", i, j);
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
        }
    }
}
