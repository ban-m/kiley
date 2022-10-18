//! Banded implementations.
//! Caution2: The `_banded` functions determine the "center-path" of the DP-table before any computation. In other words,
//! they do not track the most promissing DPcell or employ some other heuristics.
//! The rationale is that the `_banded` algorithm should be a banded version of "global" algorithm, consuming the
//! entire sequences of the reference and the query.
/// Default maximum deletion length to consider to polish a draft sequence.
pub const DEL_SIZE: usize = 4;
/// Default maximum copy length to consider to polish a draft sequence.
pub const REP_SIZE: usize = 4;
use super::*;
impl GPHMM<ConditionalHiddenMarkovModel> {
    pub fn polish_banded_batch<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
        radius: usize,
        skip_size: usize,
    ) -> Vec<u8> {
        let mut template = PadSeq::new(template);
        let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
        while let Some(imp) = self.correct_banded_batch(&template, &queries, radius, skip_size) {
            template = imp;
        }
        template.into()
    }
    pub fn correct_banded_batch(
        &self,
        template: &PadSeq,
        queries: &[PadSeq],
        radius: usize,
        skip_size: usize,
    ) -> Option<PadSeq> {
        let radius = radius as isize;
        let profiles: Vec<_> = queries
            .iter()
            .filter_map(|q| ProfileBanded::new(self, template, q, radius))
            .collect();
        if profiles.is_empty() {
            return None;
        }
        let total_lk = profiles.iter().map(|prof| prof.lk()).sum::<f64>();
        // debug!("LK\t{}\t{}", total_lk, queries.len());
        let diff = 0.001;
        fn merge(mut xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
            xs.iter_mut().zip(ys).for_each(|(x, y)| *x += y);
            xs
        }
        // Single base edits.
        {
            let profile_with_diff = profiles
                .iter()
                .map(|prf| prf.to_modification_table())
                .reduce(merge)
                .unwrap();
            let mut improved = template.clone();
            let mut offset = 0;
            let mut profile_with_diff = profile_with_diff.chunks_exact(9).enumerate();
            let mut changed_positions = vec![];
            while let Some((pos, with_diff)) = profile_with_diff.next() {
                // diff = [A,C,G,T,A,C,G,T,-], first four element is for mutation,
                // second four element is for insertion.
                let (op, &lk) = with_diff
                    .iter()
                    .enumerate()
                    .max_by(|x, y| (x.1).partial_cmp(y.1).unwrap())
                    .unwrap();
                if total_lk + diff < lk {
                    let pos = pos as isize + offset;
                    changed_positions.push(pos);
                    let (op, base) = (op / 4, (op % 4) as u8);
                    let op = [Op::Match, Op::Ins, Op::Del][op];
                    match op {
                        Op::Mismatch | Op::Match => improved[pos as isize] = base,
                        Op::Del => {
                            offset -= 1;
                            improved.remove(pos as isize);
                        }
                        Op::Ins => {
                            offset += 1;
                            improved.insert(pos as isize, base);
                        }
                    }
                    profile_with_diff.nth(skip_size);
                }
            }
            if !changed_positions.is_empty() {
                return Some(improved);
            }
        }
        // multiple-base copy-edits.
        {
            let mut improved: Vec<u8> = vec![];
            let profile_with_rep = profiles
                .iter()
                .map(|prf| prf.to_copy_table(REP_SIZE))
                .reduce(merge)
                .unwrap();
            let mut changed_positions = vec![];
            let mut inactive = 0;
            for (i, &base) in template.iter().enumerate() {
                if 0 < inactive {
                    inactive -= 1;
                } else {
                    let suggested = profile_with_rep
                        .iter()
                        .skip(i * REP_SIZE)
                        .take(REP_SIZE)
                        .enumerate()
                        .filter(|(_, &lk)| total_lk + diff < lk)
                        .max_by(|x, y| (x.1).partial_cmp(y.1).unwrap())
                        .map(|(len, _)| len + 1);
                    if let Some(len) = suggested {
                        changed_positions.push((i, len));
                        inactive = len + skip_size;
                        improved.extend(template.get_range(i as isize, (i + len) as isize));
                    }
                }
                improved.push(base);
            }
            if !changed_positions.is_empty() {
                return Some(PadSeq::from_raw_parts(improved));
            }
        }
        // Multiple-base deletion.
        {
            let mut improved = vec![];
            let profile_with_del = profiles
                .iter()
                .map(|prf| prf.to_deletion_table(DEL_SIZE))
                .reduce(merge)
                .unwrap();
            let mut changed_positions = vec![];
            let mut inactive = 0;
            let mut template = template.iter().enumerate();
            while let Some((i, &base)) = template.next() {
                if 0 < inactive {
                    inactive -= 1;
                } else {
                    let suggested = profile_with_del
                        .iter()
                        .skip(i * (DEL_SIZE - 1))
                        .take(DEL_SIZE - 1)
                        .enumerate()
                        .filter(|(_, &lk)| total_lk + diff < lk)
                        .max_by(|x, y| (x.1).partial_cmp(y.1).unwrap())
                        .map(|(len, _)| len + 1);
                    // This is 1, as we already removed the i-th base if
                    // we omit improved.push(base) below.
                    if let Some(len) = suggested {
                        changed_positions.push((i, len));
                        inactive = len + skip_size;
                        template.nth(len);
                        continue;
                    }
                }
                improved.push(base);
            }
            if !changed_positions.is_empty() {
                return Some(PadSeq::from_raw_parts(improved));
            }
        }
        None
    }
    pub fn fit_banded<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
        radius: usize,
    ) -> Self {
        let template = PadSeq::new(template);
        let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
        self.fit_banded_inner(&template, &queries, radius).0
    }
    pub fn fit_banded_inner(&self, xs: &PadSeq, yss: &[PadSeq], radius: usize) -> (Self, f64) {
        let radius = radius as isize;
        let profiles: Vec<_> = yss
            .iter()
            .filter_map(|ys| ProfileBanded::new(self, xs, ys, radius))
            .collect();
        let initial_distribution = self.estimate_initial_distribution_banded(&profiles);
        let transition_matrix = self.estimate_transition_prob_banded(&profiles);
        let observation_matrix = self.estimate_observation_prob_banded(&profiles);
        let updated = Self {
            states: self.states,
            initial_distribution,
            transition_matrix,
            observation_matrix,
            _mode: std::marker::PhantomData,
        };
        let lk: f64 = profiles.iter().map(|x| x.lk()).sum();
        (updated, lk)
    }
    pub fn estimate_observation_prob_banded(&self, profiles: &[ProfileBanded<Cond>]) -> Vec<f64> {
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
    pub fn par_estimate_observation_prob_banded(
        &self,
        profiles: &[ProfileBanded<Cond>],
    ) -> Vec<f64> {
        let mut buffer = profiles
            .par_iter()
            .map(|prf| prf.observation_probability())
            .fold(|| vec![0f64; ((self.states - 1) << 6) + 8 * 8], Self::merge)
            .reduce(|| vec![0f64; ((self.states - 1) << 6) + 8 * 8], Self::merge);
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
    fn merge(mut acc: Vec<f64>, xs: Vec<f64>) -> Vec<f64> {
        acc.iter_mut().zip(xs.iter()).for_each(|(a, x)| *a += x);
        acc
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
            dp[(0, radius, s)] = log(&self.initial_distribution[s]);
        }
        // Initial values.
        // j_orig ranges in 1..radius
        for j in radius + 1..2 * radius + 1 {
            let j_orig = j - radius;
            let y = ys[j_orig - 1];
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
        centers.push(0);
        // Fill DP cells.
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            assert_eq!(centers.len(), i + 1);
            // let (mut max_j, mut max_lk) = (0, std::f64::NEG_INFINITY);
            let (center, prev) = (centers[i], centers[i - 1]);
            for j in get_range(radius, ys.len() as isize, center) {
                let j_in_ys = j + center - radius;
                // The position of j_in_ys in the (i-1)-th row.
                let j_prev = j + center - prev;
                // If j_in_ys ==0, it would be -1, but it is OK. PadSeq just returns the NULL base for such access.
                let y = ys[j_in_ys - 1];
                for s in 0..self.states {
                    let i = i as isize;
                    let max_path = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i - 1, j_prev - 1, t)]
                                + log_transit[t][s]
                                + log_observe[s][(x << 3 | y) as usize];
                            let del = dp[(i - 1, j_prev, t)]
                                + log_transit[t][s]
                                + log_observe[s][(x << 3 | GAP) as usize];
                            // If j_in_ys == 0, this code should be skipped, but it is OK, DPTable would forgive "out-of-bound" access.
                            let ins = dp[(i, j - 1, t)]
                                + log_transit[t][s]
                                + log_observe[s][(GAP << 3 | y) as usize];
                            mat.max(del).max(ins)
                        })
                        .fold(std::f64::NEG_INFINITY, |x, y| x.max(y));
                    dp[(i, j, s)] = max_path;
                    // if max_lk < max_path {
                    //     max_lk = max_path;
                    //     max_j = j;
                    // }
                }
            }
            // Update centers.
            //let next_center = if max_j < radius { center } else { center + 2 };
            // TODO: Which is better?
            let next_center = (ys.len() * i / xs.len()) as isize;
            centers.push(next_center);
        }
        // Debugging...
        // Check if we have reached the terminal, (xs.len(),ys.len());
        let mut i = xs.len() as isize;
        let mut j_orig = ys.len() as isize;
        // Traceback.
        let (max_lk, mut state) = {
            let j = j_orig + radius - centers[xs.len()];
            (0..self.states)
                .map(|s| (dp[(i, j, s)], s))
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
                    let mat = dp[(i - 1, j_prev - 1, t)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | y) as usize];
                    let del = dp[(i - 1, j_prev, t)]
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
                Op::Mismatch | Op::Match => {
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
                    let del = dp[(i - 1, j_prev, t)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(x << 3 | GAP) as usize];
                    ((current - del).abs() < 0.001).then_some(t)
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
                    let ins = dp[(0, j - 1, t)]
                        + log_transit[t][state as usize]
                        + log_observe[state as usize][(GAP << 3 | y) as usize];
                    ((current - ins).abs() < 0.0001).then_some(t)
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
            let lk: f64 = (0..self.states).map(|s| dp[(n, m, s)]).sum();
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
        let mut norm_factors = Vec::with_capacity(xs.len() + 2);
        // The location where the radius-th element is in the original DP table.
        // In other words, if you want to convert the j-th element in the banded DP table into the original coordinate,
        // j + centers[i] - radius would be oK.
        // Inverse convertion is the sam, j_orig + radius - centers[i] would be OK.
        // Find maximum position? No, use naive band.
        let mut centers = Vec::with_capacity(xs.len() + 2);
        centers.push(0);
        centers.push(0);
        for i in 0..xs.len() {
            let next_center = (i * ys.len() / xs.len()) as isize;
            centers.push(next_center);
        }
        // Initialize.
        for (s, &x) in self.initial_distribution.iter().enumerate() {
            dp[(0, radius, s)] = x;
        }
        for j in radius + 1..2 * radius + 1 {
            let j_orig = j - radius - 1;
            if !(0..ys.len() as isize).contains(&j_orig) {
                continue;
            }
            let y = ys[j - radius - 1];
            for s in 0..self.states {
                let trans: f64 = (0..self.states)
                    .map(|t| dp[(0, j - 1, t)] * self.transition(t, s))
                    .sum();
                dp[(0, j, s)] += trans * self.observe(s, GAP, y);
            }
        }
        // Normalize.
        norm_factors.push(1f64 * dp.normalize(0));
        // Transposed matrix.
        let transitions: Vec<Vec<_>> = (0..self.states)
            .map(|to| {
                (0..self.states)
                    .map(|from| self.transition(from, to))
                    .collect()
            })
            .collect();
        // Fill DP cells.
        let mut buffer = vec![0f64; self.states];
        for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
            let (center, prev) = (centers[i], centers[i - 1]);
            // Deletion and Match transitions.
            // Maybe we should treat the case when j_orig == 0, but it is OK. DPTable would treat such cases.
            for j in get_range(radius, ys.len() as isize, center) {
                let j_orig = j + center - radius;
                let y = ys[j_orig - 1];
                let prev_j = j + center - prev;
                let mat = dp.get_cells(i as isize - 1, prev_j - 1);
                let del = dp.get_cells(i as isize - 1, prev_j);
                for (s, (transition, buffer)) in
                    transitions.iter().zip(buffer.iter_mut()).enumerate()
                {
                    let (mat_obs, del_obs) = (self.observe(s, x, y), self.observe(s, x, GAP));
                    let mat = mat
                        .iter()
                        .zip(transition.iter())
                        .fold(0f64, |x, (y, z)| x + y * z);
                    let del = del
                        .iter()
                        .zip(transition.iter())
                        .fold(0f64, |x, (y, z)| x + y * z);
                    *buffer = del * del_obs + mat * mat_obs;
                }
                dp.replace_cells(i as isize, j as isize, &buffer);
            }
            let first_total = dp.normalize(i);
            // Insertion transitions
            for j in get_range(radius, ys.len() as isize, center) {
                let j_orig = j + center - radius;
                let y = ys[j_orig - 1];
                let (i, j) = (i as isize, j as isize);
                let ins = dp.get_cells(i, j - 1);
                for (s, (transition, buffer)) in
                    transitions.iter().zip(buffer.iter_mut()).enumerate()
                {
                    let ins_obs = self.observe(s, GAP, y);
                    let ins = ins
                        .iter()
                        .zip(transition.iter())
                        .fold(0f64, |x, (y, z)| x + y * z);
                    *buffer = ins * ins_obs;
                }
                for s in 0..self.states {
                    dp[(i, j, s)] += buffer[s];
                }
            }
            let second_total = dp.normalize(i);
            norm_factors.push(first_total * second_total);
        }
        let m = ys.len() as isize - centers[xs.len()] + radius;
        (0..2 * radius + 1)
            .contains(&m)
            .then_some((dp, norm_factors, centers))
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
        for s in 0..self.states {
            let j = yslen - centers[xs.len()] + radius;
            dp[(xslen, j, s)] = 1f64;
        }
        let first_total = dp.total(xs.len());
        dp.div(xs.len(), first_total);
        for j in get_range(radius, ys.len() as isize, centers[xs.len()]).rev() {
            let j_orig = j + centers[xs.len()] - radius;
            let y = ys[j_orig];
            for s in 0..self.states {
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
            let (center, next) = (centers[i], centers[i + 1]);
            // Deletion transition to below and match transition into diagonal.
            for j in get_range(radius, ys.len() as isize, center) {
                let j_orig = j + center - radius;
                let y = ys[j_orig];
                let i = i as isize;
                let j_next = j + center - next;
                for s in 0..self.states {
                    dp[(i, j, s)] = (0..self.states)
                        .map(|t| {
                            self.transition(s, t)
                                * (self.observe(t, x, y) * dp[(i + 1, j_next + 1, t)]
                                    + self.observe(t, x, GAP) * dp[(i + 1, j_next, t)])
                        })
                        .sum::<f64>();
                }
            }
            let first_total = dp.total(i);
            dp.div(i, first_total);
            for j in get_range(radius, ys.len() as isize, center).rev() {
                let j_orig = j + center - radius;
                let y = ys[j_orig];
                for s in 0..self.states {
                    let i = i as isize;
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
    pub fn correct_until_convergence_banded<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        queries: &[T],
        radius: usize,
    ) -> Option<Vec<u8>> {
        let template = PadSeq::new(template);
        let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
        self.correction_until_convergence_banded_inner(&template, &queries, radius)
            .map(|x| x.into())
    }
    pub fn correction_until_convergence_banded_inner(
        &self,
        template: &PadSeq,
        queries: &[PadSeq],
        radius: usize,
    ) -> Option<PadSeq> {
        let mut template = template.clone();
        while let Some((seq, _)) = self.correction_inner_banded(&template, queries, radius, 0) {
            template = seq;
        }
        Some(template)
    }
    pub fn correction_inner_banded(
        &self,
        template: &PadSeq,
        queries: &[PadSeq],
        radius: usize,
        _start_position: usize,
    ) -> Option<(PadSeq, usize)> {
        let radius = radius as isize;
        let profiles: Vec<_> = queries
            .iter()
            .filter_map(|q| ProfileBanded::new(self, template, q, radius))
            .collect();
        if profiles.is_empty() {
            return None;
        }
        let total_lk = profiles.iter().map(|prof| prof.lk()).sum::<f64>();
        let diff = 0.001;
        fn merge(mut xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
            xs.iter_mut().zip(ys).for_each(|(x, y)| *x += y);
            xs
        }
        let profile_with_diff = profiles
            .iter()
            .map(|prf| prf.to_modification_table())
            .reduce(merge)
            .unwrap();
        profile_with_diff
            .chunks_exact(9)
            .enumerate()
            .filter_map(|(pos, with_diff)| {
                // diff = [A,C,G,T,A,C,G,T,-], first four element is for mutation,
                // second four element is for insertion.
                with_diff
                    .iter()
                    .enumerate()
                    .filter(|&(_, &lk)| total_lk + diff < lk)
                    .max_by(|x, y| (x.1).partial_cmp(y.1).unwrap())
                    .map(|(op, lk)| {
                        let (op, base) = (op / 4, op % 4);
                        let op = [Op::Match, Op::Ins, Op::Del][op];
                        (pos, op, base as u8, lk)
                    })
            })
            .max_by(|x, y| (x.3).partial_cmp(y.3).unwrap())
            .map(|(pos, op, base, _lk)| {
                let mut template = template.clone();
                match op {
                    Op::Mismatch | Op::Match => template[pos as isize] = base,
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
    pub fn estimate_initial_distribution_banded(&self, profiles: &[ProfileBanded<M>]) -> Vec<f64> {
        let mut buffer = vec![0f64; self.states];
        for init in profiles.iter().map(|prf| prf.initial_distribution()) {
            buffer.iter_mut().zip(init).for_each(|(x, y)| *x += y);
        }
        let sum: f64 = buffer.iter().sum();
        buffer.iter_mut().for_each(|x| *x /= sum);
        buffer
    }
    pub fn par_estimate_initial_distribution_banded(
        &self,
        profiles: &[ProfileBanded<M>],
    ) -> Vec<f64> {
        let mut buffer = profiles
            .par_iter()
            .map(|prf| prf.initial_distribution())
            .fold(|| vec![0f64; self.states], Self::merge)
            .reduce(|| vec![0f64; self.states], Self::merge);
        let sum: f64 = buffer.iter().sum();
        buffer.iter_mut().for_each(|x| *x /= sum);
        buffer
    }
    pub fn estimate_transition_prob_banded(&self, profiles: &[ProfileBanded<M>]) -> Vec<f64> {
        let states = self.states;
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
    }
    pub fn par_estimate_transition_prob_banded(&self, profiles: &[ProfileBanded<M>]) -> Vec<f64> {
        let states = self.states;
        let mut buffer = profiles
            .par_iter()
            .map(|prf| prf.transition_probability())
            .fold(|| vec![0f64; states * states], Self::merge)
            .reduce(|| vec![0f64; states * states], Self::merge);
        // Normalize.
        for row in buffer.chunks_mut(states) {
            let sum: f64 = row.iter().sum();
            row.iter_mut().for_each(|x| *x /= sum);
        }
        buffer
    }
}

#[derive(Debug, Clone)]
pub struct ProfileBanded<'a, 'b, 'c, T: HMMType> {
    template: &'a PadSeq,
    query: &'b PadSeq,
    model: &'c GPHMM<T>,
    forward: DPTable,
    pub forward_factor: Vec<f64>,
    backward: DPTable,
    pub backward_factor: Vec<f64>,
    centers: Vec<isize>,
    radius: isize,
}

impl<'a, 'b, 'c, T: HMMType> ProfileBanded<'a, 'b, 'c, T> {
    pub fn new(
        model: &'c GPHMM<T>,
        template: &'a PadSeq,
        query: &'b PadSeq,
        radius: isize,
    ) -> Option<Self> {
        let (forward, forward_factor, centers) = model.forward_banded(template, query, radius)?;
        let (backward, backward_factor) = model.backward_banded(template, query, radius, &centers);
        if backward_factor.iter().any(|x| x.is_nan()) {
            error!(
                "TEMPLATE\t{}",
                String::from_utf8(template.clone().into()).unwrap()
            );
            error!(
                "QUERY\t{}",
                String::from_utf8(query.clone().into()).unwrap()
            );
            return None;
        }
        Some(Self {
            template,
            query,
            model,
            forward,
            forward_factor,
            backward,
            backward_factor,
            centers,
            radius,
        })
    }
    pub fn lk(&self) -> f64 {
        let n = self.template.len();
        let m = self.query.len() as isize - self.centers[n] + self.radius;
        let states = self.model.states;
        let (n, m) = (n as isize, m as isize);
        let lk: f64 = (0..states).map(|s| self.forward[(n, m, s)]).sum();
        let factor: f64 = self.forward_factor.iter().map(log).sum();
        lk.ln() + factor
    }
    #[allow(dead_code)]
    pub fn with_mutation(&self, pos: usize, base: u8) -> f64 {
        let states = self.model.states;
        let (center, next) = (self.centers[pos], self.centers[pos + 1]);
        let lk = get_range(self.radius, self.query.len() as isize, center)
            .map(|j| {
                let pos = pos as isize;
                let j_orig = j + center - self.radius;
                let y = self.query[j_orig];
                let j_next = j + center - next;
                (0..states)
                    .map(|s| {
                        let forward: f64 = (0..states)
                            .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                            .sum();
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos + 1, j_next + 1, s)]
                            + self.model.observe(s, base, GAP)
                                * self.backward[(pos + 1, j_next, s)];
                        forward * backward
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        let forward_factor: f64 = self.forward_factor[..pos + 1].iter().map(|x| x.ln()).sum();
        let backward_factor: f64 = self.backward_factor[pos + 1..].iter().map(|x| x.ln()).sum();
        lk.ln() + forward_factor + backward_factor
    }
    #[allow(dead_code)]
    fn with_deletion(&self, pos: usize) -> f64 {
        let states = self.model.states;
        let center = self.centers[pos];
        if pos + 1 == self.template.len() {
            let j = self.query.len() as isize - center + self.radius;
            let lk: f64 = (0..states)
                .map(|s| self.forward[(pos as isize, j, s)])
                .sum();
            let factor: f64 = self.forward_factor[..pos + 1].iter().map(log).sum();
            return lk.ln() + factor;
        }
        let lk: f64 = get_range(self.radius, self.query.len() as isize, center)
            .map(|j| {
                let j_orig = j + center - self.radius;
                let next = self.centers[pos + 2];
                let pos = pos as isize;
                let x = self.template[pos + 1];
                let y = self.query[j_orig];
                let j_next = j + center - next;
                (0..states)
                    .map(|s| {
                        let forward: f64 = (0..states)
                            .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                            .sum();
                        // This need to be `.get` method, as j_next might be out of range.
                        let backward_mat = self.model.observe(s, x, y)
                            * self.backward.get(pos + 2, j_next + 1, s).unwrap_or(&0f64);
                        let backward_del = self.model.observe(s, x, GAP)
                            * self.backward.get(pos + 2, j_next, s).unwrap_or(&0f64);
                        forward * (backward_mat + backward_del)
                    })
                    .sum::<f64>()
            })
            .sum();
        let forward_factor: f64 = self.forward_factor[..pos + 1].iter().map(log).sum();
        let backward_factor: f64 = self.backward_factor[pos + 2..].iter().map(log).sum();
        lk.ln() + forward_factor + backward_factor
    }
    #[allow(dead_code)]
    fn with_insertion(&self, pos: usize, base: u8) -> f64 {
        let states = self.model.states;
        let center = self.centers[pos];
        let lk: f64 = get_range(self.radius, self.query.len() as isize, center)
            .map(|j| {
                let j_orig = j + center - self.radius;
                let y = self.query[j_orig];
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
    fn fill_modification_slots(
        &self,
        slots: &mut [f64],
        pos: usize,
        forward_acc: &[f64],
        backward_acc: &[f64],
    ) {
        let states = self.model.states;
        slots.iter_mut().for_each(|x| *x = 0f64);
        let (center, next) = (self.centers[pos], self.centers[pos + 1]);
        let forward_acc = forward_acc[pos];
        let x = self.template[pos as isize + 1];
        for j in get_range(self.radius, self.query.len() as isize, center) {
            let pos = pos as isize;
            let forward = (0..states).map(|s| {
                (0..states)
                    .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                    .sum::<f64>()
            });
            let j_orig = j + center - self.radius;
            let j_next = j + center - next;
            let y = self.query[j_orig];
            for (s, forward) in forward.enumerate() {
                for base in 0..4u8 {
                    let backward = self.model.observe(s, base, y) * self.backward[(pos, j + 1, s)]
                        + self.model.observe(s, base, GAP) * self.backward[(pos, j, s)];
                    slots[4 + base as usize] += forward * backward;
                    let backward = self.model.observe(s, base, y)
                        * self.backward[(pos + 1, j_next + 1, s)]
                        + self.model.observe(s, base, GAP) * self.backward[(pos + 1, j_next, s)];
                    slots[base as usize] += forward * backward;
                }
                if let Some(gap_next) = self.centers.get(pos as usize + 2) {
                    let j_gap_next = j + center - gap_next;
                    let backward_mat =
                        self.model.observe(s, x, y) * self.backward[(pos + 2, j_gap_next + 1, s)];
                    let backward_del =
                        self.model.observe(s, x, GAP) * self.backward[(pos + 2, j_gap_next, s)];
                    slots[8] += forward * (backward_mat + backward_del);
                }
            }
        }
        slots[..4]
            .iter_mut()
            .for_each(|x| *x = x.ln() + forward_acc + backward_acc[pos + 1]);
        slots[4..8]
            .iter_mut()
            .for_each(|x| *x = x.ln() + forward_acc + backward_acc[pos]);
        if pos + 1 != self.template.len() {
            slots[8] = slots[8].ln() + forward_acc + backward_acc[pos + 2];
        } else {
            let j = self.query.len() as isize - center + self.radius;
            let lk: f64 = (0..states)
                .map(|s| self.forward[(pos as isize, j, s)])
                .sum();
            slots[8] = lk.ln() + forward_acc;
        }
    }
    /// Return 2,3,...,len-size deletion table.
    /// Specifically, it returns an array xs,
    /// where xs[(len-1) * i + d - 1] = the likelihood when
    /// we remove the i..i+d bases from the reference.
    /// So, len should be larger than 1 and
    /// the access is valid as long as i is less than |template|-len.
    /// The reason why 1-size deletion is not in this array
    /// is because it is calculated by `to_modification_table` method.
    pub fn to_deletion_table(&self, len: usize) -> Vec<f64> {
        let (forward_acc, backward_acc) = self.accumlate_factors();
        let states = self.model.states;
        let width = len - 1;
        let mut lks = vec![EP; width * (self.template.len() - len)];
        for (pos, slots) in lks.chunks_exact_mut(width).enumerate() {
            slots.iter_mut().for_each(|x| *x = 0f64);
            let center = self.centers[pos];
            let forward_acc = forward_acc[pos];
            for del_size in 2..len + 1 {
                if pos + del_size == self.template.len() {
                    let j = self.query.len() as isize - center + self.radius;
                    let lk: f64 = (0..states)
                        .map(|s| self.forward[(pos as isize, j, s)])
                        .sum();
                    slots[del_size - 2] = lk.ln() + forward_acc;
                } else {
                    let x = self.template[(pos + del_size) as isize];
                    let mut lk_total = 0f64;
                    for j in get_range(self.radius, self.query.len() as isize, center) {
                        let pos = pos as isize;
                        let gap_next = self.centers[pos as usize + del_size + 1];
                        let j_gap_next = match j + center - gap_next {
                            x if 0 <= x => x,
                            _ => continue,
                        };
                        let forward = (0..states).map(|s| {
                            (0..states)
                                .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                                .sum::<f64>()
                        });
                        let j_orig = j + center - self.radius;
                        let y = self.query[j_orig];
                        for (s, forward) in forward.enumerate() {
                            let pos_after = pos + del_size as isize + 1;
                            let backward_mat = self
                                .backward
                                .get(pos_after, j_gap_next + 1, s)
                                .unwrap_or(&0f64);
                            let backward_del =
                                self.backward.get(pos_after, j_gap_next, s).unwrap_or(&0f64);
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
    /// Return an array xs,
    /// where xs[i * len + k] is the likleihood
    /// when we duplicate xs[i..i+k] right after xs[i+k]. In other words,
    /// the likelihood between  xs[..i+k] + xs[i..] and ys.
    /// So, the i is bounded by i + k < |xs|
    pub fn to_copy_table(&self, len: usize) -> Vec<f64> {
        let (forward_acc, backward_acc) = self.accumlate_factors();
        let states = self.model.states;
        let mut lks = vec![EP; len * (self.template.len() - len)];
        for (pos, slots) in lks.chunks_exact_mut(len).enumerate() {
            slots.iter_mut().for_each(|x| *x = 0f64);
            for dup_size in 1..len + 1 {
                let from_pos = pos + dup_size;
                let center = self.centers[from_pos];
                let forward_acc = forward_acc[from_pos];
                let x = self.template[pos as isize];
                let mut lk_total = 0f64;
                for j in get_range(self.radius, self.query.len() as isize, center) {
                    let gap_next = self.centers[pos + 1];
                    let pos = pos as isize;
                    let from_pos = from_pos as isize;
                    let forward = (0..states).map(|s| {
                        (0..states)
                            .map(|t| self.forward[(from_pos, j, t)] * self.model.transition(t, s))
                            .sum::<f64>()
                    });
                    let j_orig = j + center - self.radius;
                    let y = self.query[j_orig];
                    for (s, forward) in forward.enumerate() {
                        let j_gap_next = j + center - gap_next;
                        let backward_mat = match self.backward.get(pos + 1, j_gap_next + 1, s) {
                            Some(mat) => mat * self.model.observe(s, x, y),
                            None => 0f64,
                        };
                        let backward_del = match self.backward.get(pos + 1, j_gap_next, s) {
                            Some(del) => del * self.model.observe(s, x, GAP),
                            None => 0f64,
                        };
                        lk_total += forward * (backward_mat + backward_del);
                    }
                }
                slots[dup_size - 1] = lk_total.ln() + forward_acc + backward_acc[pos + 1];
            }
        }
        lks
    }
    // 9-element window.
    // Position->[A,C,G,T,A,C,G,T,-]
    pub fn to_modification_table(&self) -> Vec<f64> {
        let (forward_acc, backward_acc) = self.accumlate_factors();
        let states = self.model.states;
        let mut lks = vec![EP; 9 * (self.template.len() + 1)];
        for (pos, slots) in lks
            .chunks_exact_mut(9)
            .enumerate()
            .take(self.template.len())
        {
            self.fill_modification_slots(slots, pos, &forward_acc, &backward_acc);
        }
        // Last insertion.
        let (pos, slots) = lks.chunks_exact_mut(9).enumerate().last().unwrap();
        let center = self.centers[pos];
        slots[4..8].iter_mut().for_each(|x| *x = 0f64);
        for j in get_range(self.radius, self.query.len() as isize, center) {
            let pos = pos as isize;
            let forward: Vec<f64> = (0..states)
                .map(|s| {
                    (0..states)
                        .map(|t| self.forward[(pos, j, t)] * self.model.transition(t, s))
                        .sum()
                })
                .collect();
            for base in 0..4u8 {
                let j_orig = j + center - self.radius;
                let y = self.query[j_orig];
                slots[4 + base as usize] += forward
                    .iter()
                    .enumerate()
                    .map(|(s, forward)| {
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos, j + 1, s)]
                            + self.model.observe(s, base, GAP) * self.backward[(pos, j, s)];
                        forward * backward
                    })
                    .sum::<f64>();
            }
        }
        slots[4..8]
            .iter_mut()
            .for_each(|x| *x = x.ln() + forward_acc[pos] + backward_acc[pos]);
        lks
    }
    pub fn initial_distribution(&self) -> Vec<f64> {
        let mut probs: Vec<_> = (0..self.model.states)
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
    pub fn transition_probability(&self) -> Vec<f64> {
        let states = self.model.states;
        // Log probability.
        let (forward_acc, backward_acc) = self.accumlate_factors();
        let offset_factor = forward_acc
            .iter()
            .zip(backward_acc.iter())
            .fold(std::f64::MIN, |x, (f, b)| x.max(f + b));
        let factors: Vec<_> = (0..self.template.len())
            .map(|i| {
                let forward_factor: f64 = forward_acc[i + 1];
                let backward1: f64 = backward_acc[i + 1];
                let backward2: f64 = backward_acc[i];
                let backward = (backward2 - backward1).exp();
                let factor = (backward1 + forward_factor - offset_factor).exp();
                (backward, factor)
            })
            .collect();
        let mut probs: Vec<_> = vec![0f64; self.model.states.pow(2)];
        for from in 0..states {
            for to in 0..states {
                let mut lks = 0f64;
                for (i, &x) in self.template.iter().enumerate() {
                    let (center, next) = (self.centers[i], self.centers[i + 1]);
                    let (backward, offset) = factors[i];
                    let i = i as isize;
                    for j in get_range(self.radius, self.query.len() as isize, center) {
                        let transition = &self.model.transition(from, to);
                        let j_orig = j + center - self.radius;
                        let y = self.query[j_orig];
                        let j_next = j + center - next;
                        let forward = &self.forward[(i, j, from)];
                        let backward_match =
                            self.model.observe(from, x, y) * self.backward[(i + 1, j_next + 1, to)];
                        let backward_del =
                            self.model.observe(from, x, GAP) * self.backward[(i + 1, j_next, to)];
                        let backward_ins =
                            self.model.observe(from, GAP, y) * self.backward[(i, j + 1, to)];
                        let backward = backward_match + backward_del + backward_ins * backward;
                        lks += forward * transition * backward * offset;
                    }
                }
                probs[from * states + to] += lks;
            }
        }
        probs.chunks_mut(states).for_each(|sums| {
            let sum: f64 = sums.iter().sum();
            if 1000f64 * std::f64::EPSILON < sum {
                sums.iter_mut().for_each(|x| *x /= sum);
                assert!((1f64 - sums.iter().sum::<f64>()) < 0.001);
            }
        });
        probs
    }
    // Return [from * states + to] = Pr{from->to},
    // because it is much easy to normalize.
    // Please do not use log so frequently.
    pub fn transition_probability_old(&self) -> Vec<f64> {
        let states = self.model.states;
        // Log probability.
        let (forward_acc, backward_acc) = self.accumlate_factors();
        let mut probs: Vec<_> = vec![0f64; self.model.states.pow(2)];
        for from in 0..states {
            for to in 0..states {
                let transition = log(&self.model.transition(from, to));
                let mut lks = vec![];
                for (i, &x) in self.template.iter().enumerate() {
                    let (center, next) = (self.centers[i], self.centers[i + 1]);
                    let forward_factor: f64 = forward_acc[i + 1];
                    let backward1: f64 = backward_acc[i + 1];
                    let backward2: f64 = backward_acc[i];
                    let i = i as isize;
                    for j in get_range(self.radius, self.query.len() as isize, center) {
                        let j_orig = j + center - self.radius;
                        let y = self.query[j_orig];
                        let j_next = j + center - next;
                        let forward = log(&self.forward[(i, j, from)]) + forward_factor;
                        let backward_match =
                            self.model.observe(from, x, y) * self.backward[(i + 1, j_next + 1, to)];
                        let backward_del =
                            self.model.observe(from, x, GAP) * self.backward[(i + 1, j_next, to)];
                        let backward_ins =
                            self.model.observe(from, GAP, y) * self.backward[(i, j + 1, to)];
                        let backward = backward_match
                            + backward_del
                            + backward_ins * (backward2 - backward1).exp();
                        let backward = backward.ln() + backward1;
                        if EP < forward + transition + backward {
                            lks.push(forward + transition + backward);
                        }
                    }
                }
                probs[from * states + to] = logsumexp(&lks);
            }
        }
        // Normalizing.
        // These are log-probability.
        probs.chunks_mut(states).for_each(|sums| {
            let sum = logsumexp(sums);
            // This is normal value.
            if EP < sum {
                sums.iter_mut().for_each(|x| *x = (*x - sum).exp());
                assert!((1f64 - sums.iter().sum::<f64>()) < 0.001);
            } else {
                sums.iter_mut().for_each(|x| *x = 0f64);
            }
        });
        probs
    }
}

impl<'a, 'b, 'c> ProfileBanded<'a, 'b, 'c, Cond> {
    pub fn observation_probability(&self) -> Vec<f64> {
        let states = self.model.states;
        let (forward_acc, backward_acc) = self.accumlate_factors();
        let offset_factor = forward_acc
            .iter()
            .zip(backward_acc.iter())
            .fold(std::f64::MIN, |x, (f, b)| x.max(f + b));
        let factors: Vec<_> = (0..self.template.len())
            .map(|i| {
                let current_fac = (forward_acc[i] + backward_acc[i] - offset_factor).exp();
                let next_fac = (forward_acc[i] + backward_acc[i + 1] - offset_factor).exp();
                (current_fac, next_fac)
            })
            .collect();
        let mut prob = vec![0f64; ((states - 1) << 6) + 8 * 8];
        for (i, &x) in self.template.iter().enumerate() {
            let (center, next) = (self.centers[i], self.centers[i + 1]);
            let (current_factor, next_factor) = factors[i];
            for j in get_range(self.radius, self.query.len() as isize, center) {
                let j_orig = j + center - self.radius;
                let y = self.query[j_orig];
                let i = i as isize;
                let j_next = j + center - next;
                for state in 0..self.model.states {
                    let forward: f64 = (0..self.model.states)
                        .map(|from| self.forward[(i, j, from)] * self.model.transition(from, state))
                        .sum();
                    let ins = self.model.observe(state, GAP, y) * self.backward[(i, j + 1, state)];
                    let del =
                        self.model.observe(state, x, GAP) * self.backward[(i + 1, j_next, state)];
                    let mat =
                        self.model.observe(state, x, y) * self.backward[(i + 1, j_next + 1, state)];
                    prob[state << 6 | (x << 3 | y) as usize] += forward * mat * next_factor;
                    prob[state << 6 | (x << 3 | GAP) as usize] += forward * del * next_factor;
                    prob[state << 6 | (GAP << 3 | y) as usize] += forward * ins * current_factor;
                }
            }
        }
        // Normalizing.
        prob.chunks_mut(8).for_each(|sums| {
            let sum: f64 = sums.iter().sum();
            if 1000f64 * std::f64::EPSILON < sum {
                sums.iter_mut().for_each(|x| *x /= sum);
                let tot: f64 = sums.iter().sum();
                assert!((1f64 - tot).abs() < 0.0001, "{:?}\t{}", sums, sum);
            }
        });
        prob
    }
    // [state << 6 | x | y] = Pr{(x,y)|state}
    pub fn observation_probability_old(&self) -> Vec<f64> {
        // This is log_probabilities.
        let states = self.model.states;
        let (forward_acc, backward_acc) = self.accumlate_factors();
        let mut prob = vec![vec![]; ((states - 1) << 6) + 8 * 8];
        for (i, &x) in self.template.iter().enumerate() {
            let (center, next) = (self.centers[i], self.centers[i + 1]);
            let forward_factor: f64 = forward_acc[i];
            let backward_factor1: f64 = backward_acc[i + 1];
            let backward_factor2: f64 = backward_acc[i];
            for j in get_range(self.radius, self.query.len() as isize, center) {
                let j_orig = j + center - self.radius;
                let y = self.query[j_orig];
                let i = i as isize;
                let j_next = j + center - next;
                for state in 0..self.model.states {
                    let back_match = self.backward[(i + 1, j_next + 1, state)];
                    let back_del = self.backward[(i + 1, j_next, state)];
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
mod gphmm_banded {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128PlusPlus;
    fn align_banded_check<T: HMMType>(model: &GPHMM<T>, xs: &[u8], ys: &[u8], radius: usize) {
        let (lk, ops, _states) = model.align(xs, ys);
        let (lk_b, ops_b, _states_b) = model.align_banded(xs, ys, radius).unwrap();
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
        assert!((lk - lkb).abs() < 0.0001, "{},{}", lk, lkb);
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
            .map(|(s, init)| init * dp[(0, 0, s)])
            .sum();
        let lk_b: f64 = model
            .initial_distribution
            .iter()
            .enumerate()
            .map(|(s, init)| init * dp_b[(0, radius as isize - centers[0], s)])
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
        let lk = model.likelihood(xs, ys);
        let lkb = model.likelihood_banded(xs, ys, radius as usize).unwrap();
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
    fn profile_modification_banded_check<T: HMMType>(
        model: &GPHMM<T>,
        xs: &[u8],
        ys: &[u8],
        radius: isize,
    ) {
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let profile = ProfileBanded::new(model, &xs, &ys, radius).unwrap();
        let mut xs = xs.clone();
        let difftable = profile.to_modification_table();
        for (pos, diffs) in difftable.chunks(9).enumerate() {
            // Insertion.
            for base in b"ACGT".iter().map(padseq::convert_to_twobit) {
                xs.insert(pos as isize, base);
                let lk = model
                    .likelihood_banded_inner(&xs, &ys, radius as usize)
                    .unwrap();
                assert!((diffs[4 + base as usize] - lk).abs() < 0.001, "{}", pos);
                xs.remove(pos as isize);
            }
            if pos == xs.len() {
                continue;
            }
            let original = xs[pos as isize];
            // Mutation.
            for base in b"ACGT".iter().map(padseq::convert_to_twobit) {
                xs[pos as isize] = base;
                let lk = model
                    .likelihood_banded_inner(&xs, &ys, radius as usize)
                    .unwrap();
                assert!((diffs[base as usize] - lk).abs() < 0.001, "{}", pos);
                xs[pos as isize] = original;
            }
            // Deletion.
            xs.remove(pos as isize);
            let lk = model
                .likelihood_banded_inner(&xs, &ys, radius as usize)
                .unwrap();
            assert!((lk - diffs[8]).abs() < 0.001);
            xs.insert(pos as isize, original);
        }
    }
    #[test]
    fn profile_modification_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..2 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            profile_modification_banded_check(&single, &xs, &ys, radius);
            profile_modification_banded_check(&single_cond, &xs, &ys, radius);
            profile_modification_banded_check(&three, &xs, &ys, radius);
            profile_modification_banded_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn profile_multi_deletion_banded_check<T: HMMType>(
        model: &GPHMM<T>,
        xs: &[u8],
        ys: &[u8],
        radius: isize,
    ) {
        let orig_xs: Vec<_> = xs.to_vec();
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let profile = ProfileBanded::new(model, &xs, &ys, radius).unwrap();
        let len = 6;
        let difftable = profile.to_deletion_table(len);
        for (pos, diffs) in difftable.chunks(len - 1).enumerate() {
            let mut xs: Vec<_> = orig_xs.clone();
            xs.remove(pos);
            for (i, lkd) in diffs.iter().enumerate() {
                xs.remove(pos);
                let xs = PadSeq::new(xs.as_slice());
                let lk = model
                    .likelihood_banded_inner(&xs, &ys, radius as usize)
                    .unwrap();
                assert!((lk - lkd).abs() < 0.001, "{},{},{},{}", lk, lkd, pos, i);
            }
        }
    }
    #[test]
    fn profile_multi_deletion_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..2 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            profile_multi_deletion_banded_check(&single, &xs, &ys, radius);
            profile_multi_deletion_banded_check(&single_cond, &xs, &ys, radius);
            profile_multi_deletion_banded_check(&three, &xs, &ys, radius);
            profile_multi_deletion_banded_check(&three_cond, &xs, &ys, radius);
        }
    }
    fn profile_copy_banded_check<T: HMMType>(
        model: &GPHMM<T>,
        xs: &[u8],
        ys: &[u8],
        radius: isize,
    ) {
        let orig_xs: Vec<_> = xs.to_vec();
        let (xs, ys) = (PadSeq::new(xs), PadSeq::new(ys));
        let profile = ProfileBanded::new(model, &xs, &ys, radius).unwrap();
        let len = 4;
        let difftable = profile.to_copy_table(len);
        for (pos, diffs) in difftable.chunks(len).enumerate() {
            let latter = orig_xs.iter().skip(pos);
            for (i, lkd) in diffs.iter().enumerate() {
                let xs: Vec<_> = orig_xs
                    .iter()
                    .take(pos + i + 1)
                    .chain(latter.clone())
                    .copied()
                    .collect();
                assert_eq!(xs.len(), orig_xs.len() + i + 1);
                let xs = PadSeq::from(xs);
                let lk = model
                    .likelihood_banded_inner(&xs, &ys, radius as usize)
                    .unwrap();
                assert!((lk - lkd).abs() < 0.001, "{},{},{},{}", lk, lkd, pos, i);
            }
        }
    }
    #[test]
    fn profile_copy_banded_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let single = GPHMM::<Full>::default();
        let single_cond = GPHMM::<Cond>::default();
        let three = GPHMM::<Full>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..2 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 200);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            profile_copy_banded_check(&single, &xs, &ys, radius);
            profile_copy_banded_check(&single_cond, &xs, &ys, radius);
            profile_copy_banded_check(&three, &xs, &ys, radius);
            profile_copy_banded_check(&three_cond, &xs, &ys, radius);
        }
    }

    fn correction_banded_check<T: HMMType>(
        model: &GPHMM<T>,
        draft: &[u8],
        queries: &[Vec<u8>],
        radius: usize,
        answer: &[u8],
    ) {
        let correct = model.correct_until_convergence(draft, queries);
        let correct_b = model
            .correct_until_convergence_banded(draft, queries, radius)
            .unwrap();
        let dist = crate::bialignment::edit_dist(&correct, answer);
        let dist_b = crate::bialignment::edit_dist(&correct_b, answer);
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
    fn fit_model_banded(
        model: &GPHMM<Cond>,
        template: &[u8],
        queries: &[Vec<u8>],
        radius: usize,
    ) -> GPHMM<Cond> {
        let mut model: GPHMM<Cond> = model.clone();
        let mut lks: Vec<Option<f64>> = queries
            .iter()
            .map(|q| model.likelihood_banded(template, q, radius))
            .collect();
        loop {
            let new_m = model.fit_banded(template, queries, radius);
            let new_lks: Vec<Option<f64>> = queries
                .iter()
                .map(|q| new_m.likelihood_banded(template, q, radius))
                .collect();
            let lk_gain: f64 = new_lks
                .iter()
                .zip(lks.iter())
                .map(|(x, y)| match (x, y) {
                    (Some(x), Some(y)) => x - y,
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
        let model_banded = fit_model_banded(model, template, queries, radius);
        let model_full = fit_model_full(model, template, queries);
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
    fn fit_model_full(model: &GPHMM<Cond>, template: &[u8], queries: &[Vec<u8>]) -> GPHMM<Cond> {
        let mut model: GPHMM<Cond> = model.clone();
        let mut lk: f64 = queries.iter().map(|q| model.likelihood(template, q)).sum();
        loop {
            let new_m = model.fit(template, queries);
            let new_lk: f64 = queries.iter().map(|q| new_m.likelihood(template, q)).sum();
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
    #[test]
    fn transition_prob_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        let xs = crate::gen_seq::generate_seq(&mut rng, 200);
        let seq = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
        let xs = PadSeq::new(xs.as_slice());
        let seq = PadSeq::new(seq.as_slice());
        let profile = ProfileBanded::new(&three_cond, &xs, &seq, radius).unwrap();
        let trans1 = profile.transition_probability();
        let trans2 = profile.transition_probability_old();
        for (x, y) in trans1.iter().zip(trans2.iter()) {
            assert!((x - y).abs() < 0.0001, "{},{}", x, y)
        }
    }
    #[test]
    fn observation_prob_test() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(4280);
        let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        let xs = crate::gen_seq::generate_seq(&mut rng, 200);
        let seq = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
        let xs = PadSeq::new(xs.as_slice());
        let seq = PadSeq::new(seq.as_slice());
        let profile = ProfileBanded::new(&three_cond, &xs, &seq, radius).unwrap();
        let trans1 = profile.observation_probability();
        let trans2 = profile.observation_probability_old();
        for (x, y) in trans1.iter().zip(trans2.iter()) {
            assert!((x - y).abs() < 0.0001, "{},{}", x, y)
        }
    }
}
