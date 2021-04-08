//! Banded implementations.
//! Caution2: The `_banded` functions determine the "center-path" of the DP-table before any computation. In other words,
//! they do not track the most promissing DPcell or employ some other heuristics.
//! The rationale is that the `_banded` algorithm should be a banded version of "global" algorithm, consuming the
//! entire sequences of the reference and the query.
use super::*;
impl GPHMM<ConditionalHiddenMarkovModel> {
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
        let start = std::time::Instant::now();
        let initial_distribution = self.estimate_initial_distribution_banded(&profiles);
        let init = std::time::Instant::now();
        let transition_matrix = self.estimate_transition_prob_banded(&profiles);
        let trans = std::time::Instant::now();
        let observation_matrix = self.estimate_observation_prob_banded(&profiles);
        let obs = std::time::Instant::now();
        debug!(
            "{}\t{}\t{}",
            (init - start).as_millis(),
            (trans - init).as_millis(),
            (obs - trans).as_millis()
        );
        Self {
            states: self.states,
            initial_distribution,
            transition_matrix,
            observation_matrix,
            _mode: std::marker::PhantomData,
        }
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
}

impl<M: HMMType> GPHMM<M> {
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
            let j_orig = j - radius - 1;
            if !(0..ys.len() as isize).contains(&j_orig) {
                continue;
            }
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
            for j in get_range(radius, ys.len() as isize, center) {
                let j_orig = j + center - radius;
                let y = ys[j_orig - 1];
                let prev_j = j + center - prev;
                for s in 0..self.states {
                    let (mat_obs, del_obs) = (self.observe(s, x, y), self.observe(s, x, GAP));
                    dp[(i as isize, j as isize, s as isize)] = (0..self.states)
                        .map(|t| {
                            let mat = dp[(i as isize - 1, prev_j - 1, t as isize)];
                            let del = dp[(i as isize - 1, prev_j, t as isize)];
                            (mat * mat_obs + del * del_obs) * self.transition(t, s)
                        })
                        .sum::<f64>();
                }
            }
            let first_total = dp.total(i);
            dp.div(i, first_total);
            // Insertion transitions
            for j in get_range(radius, ys.len() as isize, center) {
                let j_orig = j + center - radius;
                let y = ys[j_orig - 1];
                for s in 0..self.states {
                    let ins_obs = self.observe(s, GAP, y);
                    dp[(i as isize, j as isize, s as isize)] += (0..self.states)
                        .map(|t| {
                            dp[(i as isize, j as isize - 1, t as isize)]
                                * self.transition(t, s)
                                * ins_obs
                        })
                        .sum::<f64>();
                }
            }
            let second_total = dp.total(i);
            dp.div(i, second_total);
            norm_factors.push(first_total * second_total);
            // Find maximum position.
            let next_center = (i * ys.len() / xs.len()) as isize;
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
        for j in get_range(radius, ys.len() as isize, centers[xs.len()]).rev() {
            let j_orig = j + centers[xs.len()] - radius;
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
            for j in get_range(radius, ys.len() as isize, center) {
                let j_orig = j + center - radius;
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
            for j in get_range(radius, ys.len() as isize, center).rev() {
                let j_orig = j + center - radius;
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
        let mut start_position = 0;
        while let Some((seq, next)) =
            self.correction_inner_banded(&template, &queries, radius, start_position)
        {
            template = seq;
            start_position = next;
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
        // let start = std::time::Instant::now();
        let profiles: Vec<_> = queries
            .iter()
            .filter_map(|q| ProfileBanded::new(self, &template, q, radius))
            .collect();
        // let prof = std::time::Instant::now();
        if profiles.is_empty() {
            return None;
        }
        let total_lk = profiles.iter().map(|prof| prof.lk()).sum::<f64>();
        let diff = 0.001;
        let profile_with_diff = profiles
            .iter()
            .map(|prf| prf.to_modification_table())
            .reduce(|mut x, y| {
                x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
                x
            })
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
                    .max_by(|x, y| (x.1).partial_cmp(&(y.1)).unwrap())
                    .map(|(op, lk)| {
                        let (op, base) = (op / 4, op % 4);
                        let op = [Op::Match, Op::Ins, Op::Del][op];
                        (pos, op, base as u8, lk)
                    })
            })
            .max_by(|x, y| (x.3).partial_cmp(&(y.3)).unwrap())
            .map(|(pos, op, base, lk)| {
                debug!("LK\t{:.3}->{:.3}", total_lk, lk);
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
    pub fn estimate_initial_distribution_banded(&self, profiles: &[ProfileBanded<M>]) -> Vec<f64> {
        let mut buffer = vec![0f64; self.states];
        for init in profiles.iter().map(|prf| prf.initial_distribution()) {
            buffer.iter_mut().zip(init).for_each(|(x, y)| *x += y);
        }
        let sum: f64 = buffer.iter().sum();
        buffer.iter_mut().for_each(|x| *x /= sum);
        buffer
    }
    pub fn estimate_transition_prob_banded(&self, profiles: &[ProfileBanded<M>]) -> Vec<f64> {
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
pub struct ProfileBanded<'a, 'b, 'c, T: HMMType> {
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
    pub fn new(
        model: &'c GPHMM<T>,
        template: &'a PadSeq,
        query: &'b PadSeq,
        radius: isize,
    ) -> Option<Self> {
        let (forward, forward_factor, centers) = model.forward_banded(template, query, radius)?;
        let (backward, backward_factor) = model.backward_banded(template, query, radius, &centers);
        if backward_factor.iter().any(|x| x.is_nan()) {
            panic!("{:?}\n{}", backward_factor, model,);
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
        let states = self.model.states as isize;
        let (n, m) = (n as isize, m as isize);
        let lk: f64 = (0..states).map(|s| self.forward[(n, m, s)]).sum();
        let factor: f64 = self.forward_factor.iter().map(log).sum();
        lk.ln() + factor
    }
    #[allow(dead_code)]
    fn with_mutation(&self, pos: usize, base: u8) -> f64 {
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
    #[allow(dead_code)]
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
    fn accumlate_factors(&self) -> (Vec<f64>, Vec<f64>) {
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
            slots.iter_mut().for_each(|x| *x = 0f64);
            let (center, next) = (self.centers[pos], self.centers[pos + 1]);
            let range = get_range(self.radius, self.query.len() as isize, center);
            let forward_acc = forward_acc[pos];
            let x = self.template[pos as isize + 1];
            for j in range.clone() {
                let forward = (0..states).map(|s| {
                    let pos = pos as isize;
                    (0..states)
                        .map(|t| self.forward[(pos, j, t as isize)] * self.model.transition(t, s))
                        .sum::<f64>()
                });
                let j_orig = j + center - self.radius;
                let j_next = j + center - next;
                let y = self.query[j_orig];
                let pos = pos as isize;
                for (s, forward) in forward.enumerate() {
                    for base in b"ACGT".iter().map(padseq::convert_to_twobit) {
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos, j + 1, s as isize)]
                            + self.model.observe(s, base, GAP)
                                * self.backward[(pos, j, s as isize)];
                        slots[4 + base as usize] += forward * backward;
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos + 1, j_next + 1, s as isize)]
                            + self.model.observe(s, base, GAP)
                                * self.backward[(pos + 1, j_next, s as isize)];
                        slots[base as usize] += forward * backward;
                    }
                    if pos + 1 != self.template.len() as isize {
                        let gap_next = self.centers[pos as usize + 2];
                        let j_gap_next = j + center - gap_next;
                        let pos = pos as isize;
                        let backward_mat = self.model.observe(s, x, y)
                            * self
                                .backward
                                .get(pos + 2, j_gap_next + 1, s as isize)
                                .unwrap_or(&0f64);
                        let backward_del = self.model.observe(s, x, GAP)
                            * self
                                .backward
                                .get(pos + 2, j_gap_next, s as isize)
                                .unwrap_or(&0f64);
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
            }
            // Deletion
            if pos + 1 == self.template.len() {
                let j = self.query.len() as isize - center + self.radius;
                let lk: f64 = (0..states as isize)
                    .map(|s| self.forward[(pos as isize, j, s)])
                    .sum();
                slots[8] = lk.ln() + forward_acc;
            }
        }
        // Last insertion.
        let (pos, slots) = lks.chunks_exact_mut(9).enumerate().last().unwrap();
        let center = self.centers[pos];
        slots[4..8].iter_mut().for_each(|x| *x = 0f64);
        for j in get_range(self.radius, self.query.len() as isize, center) {
            let forward: Vec<f64> = (0..states)
                .map(|s| {
                    let pos = pos as isize;
                    (0..states)
                        .map(|t| self.forward[(pos, j, t as isize)] * self.model.transition(t, s))
                        .sum()
                })
                .collect();
            for base in b"ACGT".iter().map(padseq::convert_to_twobit) {
                let j_orig = j + center - self.radius;
                let y = self.query[j_orig];
                let pos = pos as isize;
                slots[4 + base as usize] += forward
                    .iter()
                    .enumerate()
                    .map(|(s, forward)| {
                        let backward = self.model.observe(s, base, y)
                            * self.backward[(pos, j + 1, s as isize)]
                            + self.model.observe(s, base, GAP)
                                * self.backward[(pos, j, s as isize)];
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
    pub fn transition_probability(&self) -> Vec<f64> {
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
                        let forward = log(&self.forward[(i, j, from as isize)]) + forward_factor;
                        let backward_match = self.model.observe(from, x, y)
                            * self.backward[(i + 1, j_next + 1, to as isize)];
                        let backward_del = self.model.observe(from, x, GAP)
                            * self.backward[(i + 1, j_next, to as isize)];
                        let backward_ins = self.model.observe(from, GAP, y)
                            * self.backward[(i, j + 1, to as isize)];
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
            let sum = logsumexp(&sums);
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

// impl<'a, 'b, 'c> ProfileBanded<'a, 'b, 'c, Full> {
//     fn observation_probability(&self) -> Vec<f64> {
//         unimplemented!()
//     }
// }

impl<'a, 'b, 'c> ProfileBanded<'a, 'b, 'c, Cond> {
    // [state << 6 | x | y] = Pr{(x,y)|state}
    pub fn observation_probability(&self) -> Vec<f64> {
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
}
