use super::DPTable;
use super::LikelihoodSummary;
use super::PairHiddenMarkovModel;
use super::State;
use crate::op::Op;
use crate::EP;
impl PairHiddenMarkovModel {
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
    /// Forward algorithm. Return the raw DP table.
    pub(crate) fn forward(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let xs: Vec<_> = xs.iter().map(crate::padseq::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::padseq::convert_to_twobit).collect();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        let log_del_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        let mut del_accum = 0f64;
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            *dptable.get_mut(i, 0, State::Match) = EP;
            del_accum += log_del_emit[x as usize];
            *dptable.get_mut(i, 0, State::Del) =
                log_del_open + log_del_ext * (i - 1) as f64 + del_accum;
            *dptable.get_mut(i, 0, State::Ins) = EP;
        }
        let mut ins_accum = 0f64;
        for (j, &y) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
            *dptable.get_mut(0, j, State::Match) = EP;
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
                    dptable.get(i - 1, j - 1, State::Match) + log_mat_ext,
                    dptable.get(i - 1, j - 1, State::Del) + log_mat_from_del,
                    dptable.get(i - 1, j - 1, State::Ins) + log_mat_from_ins,
                ) + log_mat_emit[((x << 2) | y) as usize];
                *dptable.get_mut(i, j, State::Match) = mat;
                let del = Self::logsumexp(
                    dptable.get(i - 1, j, State::Match) + log_del_open,
                    dptable.get(i - 1, j, State::Del) + log_del_ext,
                    dptable.get(i - 1, j, State::Ins) + log_del_from_ins,
                ) + log_del_emit[x as usize];
                *dptable.get_mut(i, j, State::Del) = del;
                let ins = Self::logsumexp(
                    dptable.get(i, j - 1, State::Match) + log_ins_open,
                    dptable.get(i, j - 1, State::Del) + log_ins_from_del,
                    dptable.get(i, j - 1, State::Ins) + log_ins_ext,
                ) + log_ins_emit[y as usize];
                *dptable.get_mut(i, j, State::Ins) = ins;
            }
        }
        dptable
    }

    /// Naive implementation of backward algorithm.
    pub fn backward(&self, xs: &[u8], ys: &[u8]) -> DPTable {
        let xs: Vec<_> = xs.iter().map(crate::padseq::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::padseq::convert_to_twobit).collect();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        *dptable.get_mut(xs.len(), ys.len(), State::Match) = 0f64;
        *dptable.get_mut(xs.len(), ys.len(), State::Del) = 0f64;
        *dptable.get_mut(xs.len(), ys.len(), State::Ins) = 0f64;
        let mut gap = 0f64;
        let log_del_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        for (i, &x) in xs.iter().enumerate().rev() {
            gap += log_del_emit[x as usize];
            *dptable.get_mut(i, ys.len(), State::Del) = log_del_ext * (xs.len() - i) as f64 + gap;
            *dptable.get_mut(i, ys.len(), State::Ins) =
                log_del_from_ins + log_del_ext * (xs.len() - i - 1) as f64 + gap;
            *dptable.get_mut(i, ys.len(), State::Match) =
                log_del_open + log_del_ext * (xs.len() - i - 1) as f64 + gap;
        }
        gap = 0f64;
        for (j, &y) in ys.iter().enumerate().rev() {
            gap += log_ins_emit[y as usize];
            *dptable.get_mut(xs.len(), j, State::Ins) = log_ins_ext * (ys.len() - j) as f64 + gap;
            *dptable.get_mut(xs.len(), j, State::Del) =
                log_ins_from_del + log_ins_ext * (ys.len() - j - 1) as f64 + gap;
            *dptable.get_mut(xs.len(), j, State::Match) =
                log_ins_open + log_ins_ext * (ys.len() - j - 1) as f64 + gap;
        }
        for (i, &x) in xs.iter().enumerate().rev() {
            for (j, &y) in ys.iter().enumerate().rev() {
                // Match state;
                let mat = log_mat_ext
                    + log_mat_emit[(x << 2 | y) as usize]
                    + dptable.get(i + 1, j + 1, State::Match);
                let del =
                    log_del_open + log_del_emit[x as usize] + dptable.get(i + 1, j, State::Del);
                let ins =
                    log_ins_open + log_ins_emit[y as usize] + dptable.get(i, j + 1, State::Ins);
                *dptable.get_mut(i, j, State::Match) = Self::logsumexp(mat, del, ins);
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
    /// Return error profile.
    pub fn get_profile(&self, xs: &[u8], ys: &[u8]) -> LikelihoodSummary {
        // Forward backward profile
        let f_dp = self.forward(xs, ys);
        let b_dp = self.backward(xs, ys);
        let lk = f_dp.get_total_lk(xs.len(), ys.len());
        let mut dptable = DPTable::new(xs.len(), ys.len() + 1);
        for i in 0..xs.len() {
            for j in 0..ys.len() + 1 {
                for &s in &[State::Match, State::Del, State::Ins] {
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
    pub fn correct_flip<T: std::borrow::Borrow<[u8]>, R: rand::Rng>(
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
    /// Return the alignment path between x and y.
    /// In HMM term, it is "viterbi" algorithm.
    pub fn align(&self, xs: &[u8], ys: &[u8]) -> (DPTable, Vec<Op>, f64) {
        let xs: Vec<_> = xs.iter().map(crate::padseq::convert_to_twobit).collect();
        let ys: Vec<_> = ys.iter().map(crate::padseq::convert_to_twobit).collect();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        *dptable.get_mut(0, 0, State::Ins) = EP;
        *dptable.get_mut(0, 0, State::Del) = EP;
        for i in 1..xs.len() + 1 {
            *dptable.get_mut(i, 0, State::Match) = EP;
            *dptable.get_mut(i, 0, State::Ins) = EP;
            *dptable.get_mut(i, 0, State::Del) = EP;
        }
        for j in 1..ys.len() + 1 {
            *dptable.get_mut(0, j, State::Match) = EP;
            *dptable.get_mut(0, j, State::Ins) = EP;
            *dptable.get_mut(0, j, State::Del) = EP;
        }
        let log_del_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        for (i, &x) in xs.iter().enumerate().map(|(p, x)| (p + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(p, y)| (p + 1, y)) {
                let mat = (dptable.get(i - 1, j - 1, State::Match) + log_mat_ext)
                    .max(dptable.get(i - 1, j - 1, State::Ins) + log_mat_from_ins)
                    .max(dptable.get(i - 1, j - 1, State::Del) + log_mat_from_del)
                    + log_mat_emit[((x << 2) | y) as usize];
                *dptable.get_mut(i, j, State::Match) = mat;
                let del = (dptable.get(i - 1, j, State::Match) + log_del_open)
                    .max(dptable.get(i - 1, j, State::Del) + log_del_ext)
                    .max(dptable.get(i - 1, j, State::Ins) + log_del_from_ins)
                    + log_del_emit[x as usize];
                *dptable.get_mut(i, j, State::Del) = del;
                let ins = (dptable.get(i, j - 1, State::Match) + log_ins_open)
                    .max(dptable.get(i, j - 1, State::Del) + log_ins_from_del)
                    .max(dptable.get(i, j - 1, State::Ins) + log_ins_ext)
                    + log_ins_emit[y as usize];
                *dptable.get_mut(i, j, State::Ins) = ins;
            }
        }
        let (max_state, max_lk) = vec![State::Match, State::Ins, State::Del]
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
                State::Match => {
                    let mat_lk = lk - log_mat_emit[((x << 2) | y) as usize];
                    let mat = dptable.get(i - 1, j - 1, State::Match) + log_mat_ext;
                    let del = dptable.get(i - 1, j - 1, State::Del) + log_mat_from_del;
                    let ins = dptable.get(i - 1, j - 1, State::Ins) + log_mat_from_ins;
                    if (mat_lk - mat).abs() < diff {
                        state = State::Match;
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
                    let mat = dptable.get(i - 1, j, State::Match) + log_del_open;
                    let del = dptable.get(i - 1, j, State::Del) + log_del_ext;
                    let ins = dptable.get(i - 1, j, State::Ins) + log_del_from_ins;
                    if (del_lk - mat).abs() < diff {
                        state = State::Match;
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
                    let mat = dptable.get(i, j - 1, State::Match) + log_ins_open;
                    let del = dptable.get(i, j - 1, State::Del) + log_ins_from_del;
                    let ins = dptable.get(i, j - 1, State::Ins) + log_ins_ext;
                    if (ins_lk - mat).abs() < diff {
                        state = State::Match;
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
