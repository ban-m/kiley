// use super::DPTable;
// use super::LikelihoodSummary;
// use super::PairHiddenMarkovModel;
// use super::State;
// use super::BASE_TABLE;
// use crate::logsumexp;
// use crate::EP;

// impl PairHiddenMarkovModel {
//     /// Return likelihood of x and y.
//     /// It returns the probability to see the two sequence (x,y),
//     /// summarizing all the alignment between x and y.
//     /// In other words, it returns Sum_{alignment between x and y} Pr{alignment|self}.
//     /// Roughly speaking, it is the value after log-sum-exp-ing all the alignment score.
//     /// In HMM term, it is "forward" algorithm.
//     /// In contrast to usual likelihood method, it only compute only restricted range of pair.
//     /// Return values are the rowwize sum and the total likelihood.
//     /// For example, `let (lks, lk) = self.likelihood(xs,ys);` then `lks[10]` is
//     /// the sum of the probability to see xs[0..10] and y[0..i], summing up over `i`.
//     /// If the band did not reached the (xs.len(), ys.len()) cell, then this funtion
//     /// return None. Please increase `radius` parameter if so.
//     pub fn likelihood_banded(
//         &self,
//         xs: &[u8],
//         ys: &[u8],
//         radius: usize,
//     ) -> Option<(Vec<f64>, f64)> {
//         let (dp, centers) = self.forward_banded(xs, ys, radius);
//         let (k, u) = ((xs.len() + ys.len()) as isize, xs.len());
//         let u_in_dp = (u + radius) as isize - centers[k as usize];
//         let max_lk = Self::logsumexp(
//             dp.get_check(k, u_in_dp, State::Match)?,
//             dp.get_check(k, u_in_dp, State::Del)?,
//             dp.get_check(k, u_in_dp, State::Ins)?,
//         );
//         // Maybe this code is very slow...
//         let mut lks: Vec<Vec<f64>> = vec![vec![]; xs.len() + 1];
//         for k in 0..(xs.len() + ys.len() + 1) as isize {
//             let u_center = centers[k as usize];
//             for (u, ((&x, &y), &z)) in dp
//                 .get_row(k, State::Match)
//                 .iter()
//                 .zip(dp.get_row(k, State::Del).iter())
//                 .zip(dp.get_row(k, State::Ins).iter())
//                 .enumerate()
//             {
//                 let u = u as isize;
//                 let radius = radius as isize;
//                 let i = if radius <= u + u_center && u + u_center - radius < xs.len() as isize + 1 {
//                     u + u_center - radius as isize
//                 } else {
//                     continue;
//                 };
//                 lks[i as usize].push(x);
//                 lks[i as usize].push(y);
//                 lks[i as usize].push(z);
//             }
//         }
//         let lks: Vec<_> = lks.iter().map(|xs| logsumexp(xs)).collect();
//         Some((lks, max_lk))
//     }

//     // Forward algortihm in banded manner. Return the DP tables for each state,
//     // and the ceters of each anti-diagonal.
//     // Currently, we use re-scaling method instead of log-sum-exp mode because of stability and efficiency.
//     pub(crate) fn forward_banded(
//         &self,
//         xs: &[u8],
//         ys: &[u8],
//         radius: usize,
//     ) -> (DiagonalDP, Vec<isize>) {
//         let mut centers: Vec<isize> = vec![0, 0, 1];
//         let mut dp = DiagonalDP::new(xs.len() + ys.len() + 1, 2 * radius + 1, 0f64);
//         let mut scaling_factor: Vec<f64> = Vec::with_capacity(xs.len() + ys.len() + 2);
//         let radius = radius as isize;
//         // The first diagonal.
//         {
//             *dp.get_mut(0, radius, State::Match) = 1f64;
//             scaling_factor.push(1f64);
//         }
//         // The second diagonal.
//         {
//             let ins = self.mat_ins * self.ins_emit[BASE_TABLE[ys[0] as usize]];
//             let del = self.mat_del * self.ins_emit[BASE_TABLE[xs[0] as usize]];
//             *dp.get_mut(1, radius, State::Ins) = ins;
//             *dp.get_mut(1, radius + 1, State::Del) = del;
//             let sum = dp.sum_anti_diagonal(1);
//             dp.div_anti_diagonal(1, sum);
//             scaling_factor.push(sum);
//         }
//         for k in 2..(xs.len() + ys.len() + 1) as isize {
//             let center = centers[k as usize];
//             let matdiff = center - centers[k as usize - 2];
//             let gapdiff = center - centers[k as usize - 1];
//             let (start, end) = {
//                 let k = k as usize;
//                 let radius = radius as usize;
//                 let center = center as usize;
//                 let start = radius.saturating_sub(center);
//                 let end = (xs.len() + 1 - center + radius).min(2 * radius + 1);
//                 // With respect to the k-coordinate.
//                 let start = start.max((k - center + radius).saturating_sub(ys.len()));
//                 let end = end.min(k + 1 - center + radius);
//                 (start as isize, end as isize)
//             };
//             let (mut max, mut max_pos) = (0f64, end);
//             for pos in start..end {
//                 let u = pos + center - radius;
//                 let prev_mat = pos as isize + matdiff;
//                 let prev_gap = pos as isize + gapdiff;
//                 if u == 0 {
//                     assert!(k - u > 0);
//                     *dp.get_mut(k, pos, State::Match) = 0f64;
//                     *dp.get_mut(k, pos, State::Del) = 0f64;
//                     let y = BASE_TABLE[ys[(k - u - 1) as usize] as usize];
//                     let ins = (dp.get(k - 1, prev_gap, State::Match) * self.mat_ins
//                         + dp.get(k - 1, prev_gap, State::Del) * self.del_ins
//                         + dp.get(k - 1, prev_gap, State::Ins) * self.ins_ins)
//                         * self.ins_emit[y as usize];
//                     *dp.get_mut(k, pos, State::Ins) = ins;
//                 } else if k - u == 0 {
//                     assert!(u > 0);
//                     *dp.get_mut(k, pos, State::Match) = 0f64;
//                     *dp.get_mut(k, pos, State::Ins) = 0f64;
//                     let x = BASE_TABLE[xs[(u - 1) as usize] as usize];
//                     let del = (dp.get(k - 1, prev_gap - 1, State::Match) * self.mat_del
//                         + dp.get(k - 1, prev_gap - 1, State::Del) * self.del_del
//                         + dp.get(k - 1, prev_gap - 1, State::Ins) * self.ins_del)
//                         * self.ins_emit[x as usize];
//                     *dp.get_mut(k, pos, State::Del) = del;
//                 } else {
//                     let y = BASE_TABLE[ys[(k - u - 1) as usize] as usize];
//                     let x = BASE_TABLE[xs[(u - 1) as usize] as usize];
//                     let mat = self.mat_emit[(x << 2 | y) as usize]
//                         * (dp.get(k - 2, prev_mat - 1, State::Match) * self.mat_mat
//                             + dp.get(k - 2, prev_mat - 1, State::Del) * self.del_mat
//                             + dp.get(k - 2, prev_mat - 1, State::Ins) * self.ins_mat)
//                         / scaling_factor[k as usize - 1];
//                     let del = (dp.get(k - 1, prev_gap - 1, State::Match) * self.mat_del
//                         + dp.get(k - 1, prev_gap - 1, State::Del) * self.del_del
//                         + dp.get(k - 1, prev_gap - 1, State::Ins) * self.ins_del)
//                         * self.ins_emit[x as usize];
//                     let ins = (dp.get(k - 1, prev_gap, State::Match) * self.mat_ins
//                         + dp.get(k - 1, prev_gap, State::Del) * self.del_ins
//                         + dp.get(k - 1, prev_gap, State::Ins) * self.ins_ins)
//                         * self.ins_emit[y as usize];
//                     *dp.get_mut(k, pos, State::Match) = mat;
//                     *dp.get_mut(k, pos, State::Del) = del;
//                     *dp.get_mut(k, pos, State::Ins) = ins;
//                 }
//                 let sum: f64 = [State::Match, State::Del, State::Ins]
//                     .iter()
//                     .map(|&s| dp.get(k, pos, s))
//                     .sum();
//                 if max <= sum {
//                     max = sum;
//                     max_pos = pos;
//                 }
//             }
//             let max_u = max_pos as isize + center - radius;
//             if center < max_u {
//                 centers.push(center + 1);
//             } else {
//                 centers.push(center);
//             };
//             let sum = dp.sum_anti_diagonal(k);
//             dp.div_anti_diagonal(k, sum);
//             scaling_factor.push(sum);
//         }
//         let mut log_factor = 0f64;
//         for (k, factor) in scaling_factor
//             .iter()
//             .enumerate()
//             .take(xs.len() + ys.len() + 1)
//         {
//             log_factor += factor.ln();
//             dp.convert_to_ln(k as isize, log_factor);
//         }
//         (dp, centers)
//     }
//     /// Banded backward algorithm. The filling cells are determined by `center`.
//     /// In other words, for i in 0..2*radius+1,
//     /// dp[k][i] = the (i - radius + center, k - (i -radius + center)) position in the usual DP matrix.
//     /// So, for example, the (0,0) cell would be dp[0][radius].
//     pub fn backward_banded(
//         &self,
//         xs: &[u8],
//         ys: &[u8],
//         radius: usize,
//         centers: &[isize],
//     ) -> DiagonalDP {
//         let mut dp = DiagonalDP::new(xs.len() + ys.len() + 1, 2 * radius + 1, 0f64);
//         let mut scaling_factors = vec![1f64; xs.len() + ys.len() + 1];
//         let radius = radius as isize;
//         // Calc the boundary score for each sequence, in other words,
//         // calc DP[xs.len()][j] and DP[i][ys.len()] in the usual DP.
//         // For DP[i][ys.len()].
//         // Get the location corresponding to [xs.len()][ys.len()].
//         // Fill the last DP call.
//         {
//             let (k, u) = ((xs.len() + ys.len()) as isize, xs.len() as isize);
//             let u_in_dp = u + radius - centers[k as usize];
//             *dp.get_mut(k, u_in_dp, State::Match) = 1f64;
//             *dp.get_mut(k, u_in_dp, State::Del) = 1f64;
//             *dp.get_mut(k, u_in_dp, State::Ins) = 1f64;
//             let sum = dp.sum_anti_diagonal(k);
//             dp.div_anti_diagonal(k, sum);
//             scaling_factors[k as usize] = sum;
//         }
//         // Filling the 2nd-last DP cell.
//         {
//             let k = (xs.len() + ys.len() - 1) as isize;
//             for &u in &[xs.len() - 1, xs.len()] {
//                 let (i, j) = (u as isize, k as isize - u as isize);
//                 let u = u as isize + radius - centers[k as usize];
//                 if i == xs.len() as isize {
//                     let y = BASE_TABLE[ys[j as usize] as usize];
//                     let emit = self.ins_emit[y];
//                     *dp.get_mut(k, u, State::Match) = emit * self.mat_ins;
//                     *dp.get_mut(k, u, State::Del) = emit * self.del_ins;
//                     *dp.get_mut(k, u, State::Ins) = emit * self.ins_ins;
//                 } else if j == ys.len() as isize {
//                     let x = BASE_TABLE[xs[i as usize] as usize];
//                     let emit = self.ins_emit[x];
//                     *dp.get_mut(k, u, State::Match) = emit * self.mat_del;
//                     *dp.get_mut(k, u, State::Del) = emit * self.del_del;
//                     *dp.get_mut(k, u, State::Ins) = emit * self.ins_del;
//                 } else {
//                     unreachable!();
//                 }
//             }
//             let sum = dp.sum_anti_diagonal(k);
//             dp.div_anti_diagonal(k, sum);
//             scaling_factors[k as usize] = sum / scaling_factors[k as usize + 1];
//         }
//         for k in (0..(xs.len() + ys.len() - 1) as isize).rev() {
//             let center = centers[k as usize];
//             let matdiff = center as isize - centers[k as usize + 2] as isize;
//             let gapdiff = center as isize - centers[k as usize + 1] as isize;
//             let (start, end) = {
//                 let k = k as usize;
//                 let radius = radius as usize;
//                 let center = center as usize;
//                 let start = radius.saturating_sub(center);
//                 let end = (xs.len() + 1 - center + radius).min(2 * radius + 1);
//                 // With respect to the k-coordinate.
//                 let start = start.max((k - center + radius).saturating_sub(ys.len()));
//                 let end = end.min(k + 1 - center + radius);
//                 (start as isize, end as isize)
//             };
//             for pos in start..end {
//                 let u = pos + center - radius;
//                 // Previous, prev-previous position.
//                 let u_mat = pos as isize + matdiff + 1;
//                 let u_gap = pos as isize + gapdiff;
//                 let (ins_emit, del_emit, mat_emit) = if !(0 <= k - u && k - u < xs.len() as isize) {
//                     let y = BASE_TABLE[ys[u as usize] as usize];
//                     (self.ins_emit[y as usize], 0f64, 0f64)
//                 } else if !(0 <= u && u < ys.len() as isize) {
//                     let x = BASE_TABLE[xs[(k - u) as usize] as usize];
//                     (0f64, self.ins_emit[x as usize], 0f64)
//                 } else {
//                     let y = BASE_TABLE[ys[u as usize] as usize];
//                     let x = BASE_TABLE[xs[(k - u) as usize] as usize];
//                     (
//                         self.ins_emit[y as usize],
//                         self.ins_emit[x as usize],
//                         self.mat_emit[(x << 2 | y) as usize],
//                     )
//                 };
//                 let ins = self.mat_ins * ins_emit * dp.get(k + 1, u_gap, State::Ins);
//                 let del = self.mat_del * del_emit * dp.get(k + 1, u_gap + 1, State::Del);
//                 let mat = self.mat_mat * mat_emit * dp.get(k + 2, u_mat, State::Match)
//                     / scaling_factors[k as usize + 1];
//                 *dp.get_mut(k, pos, State::Match) = mat + del + ins;
//                 {
//                     let ins = ins / self.mat_ins * self.del_ins;
//                     let del = del / self.mat_del * self.del_del;
//                     let mat = mat / self.mat_mat * self.del_mat;
//                     *dp.get_mut(k, pos, State::Del) = mat + del + ins;
//                 }
//                 {
//                     let ins = ins / self.mat_ins * self.ins_ins;
//                     let del = del / self.mat_del * self.ins_del;
//                     let mat = mat / self.mat_mat * self.ins_mat;
//                     *dp.get_mut(k, pos, State::Ins) = mat + del + ins;
//                 }
//             }
//             let sum = dp.sum_anti_diagonal(k);
//             dp.div_anti_diagonal(k, sum);
//             scaling_factors[k as usize] = sum;
//         }
//         let mut log_factor = 0f64;
//         for k in (0..xs.len() + ys.len() + 1).rev() {
//             log_factor += scaling_factors[k].ln();
//             dp.convert_to_ln(k as isize, log_factor);
//         }
//         dp
//     }

//     /// Naive implementation of backward algorithm.
//     pub fn backward(&self, xs: &[u8], ys: &[u8]) -> DPTable {
//         let xs: Vec<_> = xs.iter().map(crate::padseq::convert_to_twobit).collect();
//         let ys: Vec<_> = ys.iter().map(crate::padseq::convert_to_twobit).collect();
//         let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
//         *dptable.get_mut(xs.len(), ys.len(), State::Match) = 0f64;
//         *dptable.get_mut(xs.len(), ys.len(), State::Del) = 0f64;
//         *dptable.get_mut(xs.len(), ys.len(), State::Ins) = 0f64;
//         let mut gap = 0f64;
//         let log_del_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
//         let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
//         let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
//         let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
//         let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
//         let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
//         let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
//         let log_mat_ext = self.mat_mat.ln();
//         for (i, &x) in xs.iter().enumerate().rev() {
//             gap += log_del_emit[x as usize];
//             *dptable.get_mut(i, ys.len(), State::Del) = log_del_ext * (xs.len() - i) as f64 + gap;
//             *dptable.get_mut(i, ys.len(), State::Ins) =
//                 log_del_from_ins + log_del_ext * (xs.len() - i - 1) as f64 + gap;
//             *dptable.get_mut(i, ys.len(), State::Match) =
//                 log_del_open + log_del_ext * (xs.len() - i - 1) as f64 + gap;
//         }
//         gap = 0f64;
//         for (j, &y) in ys.iter().enumerate().rev() {
//             gap += log_ins_emit[y as usize];
//             *dptable.get_mut(xs.len(), j, State::Ins) = log_ins_ext * (ys.len() - j) as f64 + gap;
//             *dptable.get_mut(xs.len(), j, State::Del) =
//                 log_ins_from_del + log_ins_ext * (ys.len() - j - 1) as f64 + gap;
//             *dptable.get_mut(xs.len(), j, State::Match) =
//                 log_ins_open + log_ins_ext * (ys.len() - j - 1) as f64 + gap;
//         }
//         for (i, &x) in xs.iter().enumerate().rev() {
//             for (j, &y) in ys.iter().enumerate().rev() {
//                 // Match state;
//                 let mat = log_mat_ext
//                     + log_mat_emit[(x << 2 | y) as usize]
//                     + dptable.get(i + 1, j + 1, State::Match);
//                 let del =
//                     log_del_open + log_del_emit[x as usize] + dptable.get(i + 1, j, State::Del);
//                 let ins =
//                     log_ins_open + log_ins_emit[y as usize] + dptable.get(i, j + 1, State::Ins);
//                 *dptable.get_mut(i, j, State::Match) = Self::logsumexp(mat, del, ins);
//                 // Del state.
//                 {
//                     let mat = mat - log_mat_ext + log_mat_from_del;
//                     let del = del - log_del_open + log_del_ext;
//                     let ins = ins - log_ins_open + log_del_from_ins;
//                     *dptable.get_mut(i, j, State::Del) = Self::logsumexp(mat, del, ins);
//                 }
//                 // Ins state
//                 {
//                     let mat = mat - log_mat_ext + log_mat_from_ins;
//                     let del = del - log_del_open + log_del_from_ins;
//                     let ins = ins - log_ins_open + log_ins_ext;
//                     *dptable.get_mut(i, j, State::Ins) = Self::logsumexp(mat, del, ins);
//                 }
//             }
//         }
//         dptable
//     }

//     /// Return erorr profile, use banded "forward/backward" algorithm.
//     /// If the band did not reach to the corner, then this function returns None.
//     /// otherwize, it returns the summary of forward backward algorithm.
//     pub fn get_profile_banded(
//         &self,
//         xs: &[u8],
//         ys: &[u8],
//         radius: usize,
//     ) -> Option<LikelihoodSummary> {
//         let (fwd, centers) = self.forward_banded(xs, ys, radius);
//         let lk = {
//             let (k, u) = ((xs.len() + ys.len()) as isize, xs.len());
//             let u_in_dp = (u + radius) as isize - centers[k as usize];
//             Self::logsumexp(
//                 fwd.get_check(k, u_in_dp, State::Match)?,
//                 fwd.get_check(k, u_in_dp, State::Del)?,
//                 fwd.get_check(k, u_in_dp, State::Ins)?,
//             )
//         };
//         // forward-backward.
//         let mut fbwd = self.backward_banded(xs, ys, radius, &centers);
//         fbwd.add(&fwd);
//         fbwd.sub_scalar(lk);
//         // Allocate each prob of state
//         let mut match_max_prob = vec![(EP, 0); xs.len()];
//         let mut ins_max_prob = vec![(EP, 0); xs.len()];
//         let mut match_prob: Vec<Vec<f64>> = vec![vec![]; xs.len()];
//         let mut del_prob: Vec<Vec<f64>> = vec![vec![]; xs.len()];
//         let mut ins_prob: Vec<Vec<f64>> = vec![vec![]; xs.len()];
//         for (k, &center) in centers.iter().take(xs.len() + ys.len() + 1).enumerate() {
//             let (k, center) = (k as isize, center as usize);
//             // With respect to the u-corrdinate.
//             let k = k as usize;
//             let start = (radius + 1).saturating_sub(center);
//             let end = (xs.len() + 1 - center + radius).min(2 * radius + 1);
//             // With respect to the k-coordinate.
//             let start = start.max((k - center + radius).saturating_sub(ys.len()));
//             let end = end.min(k + 1 - center + radius);
//             for u in start..end {
//                 let mat = fbwd.get(k as isize, u as isize, State::Match);
//                 let del = fbwd.get(k as isize, u as isize, State::Del);
//                 let ins = fbwd.get(k as isize, u as isize, State::Ins);
//                 let i = u + center - radius;
//                 let j = k - i;
//                 match_prob[i - 1].push(mat);
//                 del_prob[i - 1].push(del);
//                 ins_prob[i - 1].push(ins);
//                 let y = if j == 0 { b'A' } else { ys[j - 1] };
//                 if match_max_prob[i - 1].0 < mat {
//                     match_max_prob[i - 1] = (mat, y);
//                 }
//                 if ins_max_prob[i - 1].0 < ins {
//                     ins_max_prob[i - 1] = (ins, y);
//                 }
//             }
//         }
//         let exp_sum = |xs: &Vec<f64>| {
//             let max = xs.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
//             max.exp() * xs.iter().map(|x| (x - max).exp()).sum::<f64>()
//         };
//         let match_prob: Vec<_> = match_prob.iter().map(exp_sum).collect();
//         let deletion_prob: Vec<_> = del_prob.iter().map(exp_sum).collect();
//         let insertion_prob: Vec<_> = ins_prob.iter().map(exp_sum).collect();
//         let match_bases: Vec<_> = match_max_prob
//             .iter()
//             .map(|x| {
//                 let mut slot = [0u8; 4];
//                 slot[crate::padseq::LOOKUP_TABLE[x.1 as usize] as usize] = 1;
//                 slot
//             })
//             .collect();
//         let insertion_bases: Vec<_> = ins_max_prob
//             .iter()
//             .map(|x| {
//                 let mut slot = [0u8; 4];
//                 slot[crate::padseq::LOOKUP_TABLE[x.1 as usize] as usize] = 1;
//                 slot
//             })
//             .collect();
//         let summary = LikelihoodSummary {
//             match_prob,
//             match_bases,
//             insertion_prob,
//             insertion_bases,
//             deletion_prob,
//             total_likelihood: lk,
//             // likelihood_trajectry,
//         };
//         Some(summary)
//     }
//     /// Batch function for `get_profiles.`
//     pub fn get_profiles_banded<T: std::borrow::Borrow<[u8]>>(
//         &self,
//         template: &[u8],
//         queries: &[T],
//         radius: usize,
//     ) -> LikelihoodSummary {
//         assert!(!queries.is_empty());
//         let mut ok_sequences = 0;
//         let lks: Option<LikelihoodSummary> = queries
//             .iter()
//             .filter_map(|x| self.get_profile_banded(template, x.borrow(), radius))
//             .fold(None, |summary, x| {
//                 ok_sequences += 1;
//                 match summary {
//                     Some(mut summary) => {
//                         summary.add(&x);
//                         Some(summary)
//                     }
//                     None => Some(x),
//                 }
//             });
//         let mut lks = lks.unwrap();
//         lks.div_probs(ok_sequences as f64);
//         lks
//     }
//     /// Correction by sampling, use banded forward-backward algorithm inside.
//     pub fn correct_flip_banded<T: std::borrow::Borrow<[u8]>, R: rand::Rng>(
//         &self,
//         template: &[u8],
//         queries: &[T],
//         rng: &mut R,
//         repeat_time: usize,
//         radius: usize,
//     ) -> (Vec<u8>, LikelihoodSummary) {
//         let mut lks = self.get_profiles_banded(template, queries, radius);
//         let mut template = template.to_vec();
//         println!("{:.0}", lks.total_likelihood);
//         for _ in 0..repeat_time {
//             let new_template = lks.correct_flip(rng);
//             let new_lks = self.get_profiles_banded(&new_template, queries, radius);
//             let ratio = (new_lks.total_likelihood - lks.total_likelihood)
//                 .exp()
//                 .min(1f64);
//             if rng.gen_bool(ratio) {
//                 template = new_template;
//                 lks = new_lks;
//             }
//         }
//         (template, lks)
//     }
// }

// #[derive(Debug, Clone)]
// pub struct DiagonalDP {
//     // Mat,Del,Ins, Mat,Del,Ins,....,
//     // 1-dimensionalized 2D array.
//     data: Vec<f64>,
//     column: usize,
// }

// const OFFSET: usize = 3;
// impl DiagonalDP {
//     pub fn new(row: usize, column: usize, x: f64) -> Self {
//         Self {
//             data: vec![x; 3 * (column + 2 * OFFSET) * (row + 2 * OFFSET)],
//             column,
//         }
//     }
//     pub(crate) fn location(&self, k: isize, u: isize, state: State) -> usize {
//         let k = (k + OFFSET as isize) as usize;
//         let u = (u + OFFSET as isize) as usize;
//         let diff: usize = state.into();
//         3 * (k * (self.column + 2 * OFFSET) + u) + diff
//     }
//     pub(crate) fn get(&self, k: isize, u: isize, state: State) -> f64 {
//         let pos = self.location(k, u, state);
//         self.data[pos]
//     }
//     pub(crate) fn get_check(&self, k: isize, u: isize, state: State) -> Option<f64> {
//         self.data.get(self.location(k, u, state)).copied()
//     }
//     pub(crate) fn get_mut(&mut self, k: isize, u: isize, state: State) -> &mut f64 {
//         let location = self.location(k, u, state);
//         self.data.get_mut(location).unwrap()
//     }
//     pub(crate) fn get_row(&self, k: isize, state: State) -> Vec<f64> {
//         let k = (k + OFFSET as isize) as usize;
//         let collen = self.column + 2 * OFFSET;
//         let state: usize = state.into();
//         let start = 3 * (k * collen + OFFSET) + state;
//         let end = 3 * ((k + 1) * collen - OFFSET) + state;
//         self.data[start..end].iter().step_by(3).copied().collect()
//     }
//     /// Return the sum over state for each cell in the k-th anti-diagonal.
//     pub fn get_row_statesum(&self, k: isize) -> Vec<f64> {
//         let k = (k + OFFSET as isize) as usize;
//         let collen = self.column + 2 * OFFSET;
//         let start = 3 * (k * collen + OFFSET);
//         let end = 3 * ((k + 1) * collen - OFFSET);
//         self.data[start..end]
//             .chunks_exact(3)
//             .map(|x| x[0] + x[1] + x[2])
//             .collect()
//     }
//     pub fn add(&mut self, other: &Self) {
//         self.data
//             .iter_mut()
//             .zip(other.data.iter())
//             .for_each(|(x, &y)| *x += y);
//     }
//     pub fn sub_scalar(&mut self, a: f64) {
//         self.data.iter_mut().for_each(|x| *x -= a);
//     }
//     pub fn sum_anti_diagonal(&self, k: isize) -> f64 {
//         let k = (k + OFFSET as isize) as usize;
//         let collen = self.column + 2 * OFFSET;
//         let range = 3 * (k * collen + OFFSET)..3 * ((k + 1) * collen - OFFSET);
//         self.data[range].iter().sum::<f64>()
//     }
//     pub fn div_anti_diagonal(&mut self, k: isize, div: f64) {
//         let k = (k + OFFSET as isize) as usize;
//         let collen = self.column + 2 * OFFSET;
//         let range = 3 * (k * collen + OFFSET)..3 * ((k + 1) * collen - OFFSET);
//         self.data[range].iter_mut().for_each(|x| *x /= div);
//     }
//     /// Convert each cell in each state in the anti-diagonal into x.ln() + lk.
//     pub fn convert_to_ln(&mut self, k: isize, lk: f64) {
//         let k = (k + OFFSET as isize) as usize;
//         let collen = self.column + 2 * OFFSET;
//         let start = 3 * (k * collen + OFFSET);
//         let end = 3 * ((k + 1) * collen - OFFSET);
//         let elements = self.data[start..end].iter_mut();
//         for elm in elements {
//             *elm = if std::f64::EPSILON < *elm {
//                 elm.ln() + lk
//             } else {
//                 EP
//             };
//         }
//     }
// }
