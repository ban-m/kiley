use super::DPTable;
use super::PairHiddenMarkovModel;
use super::State;
use super::COPY_SIZE;
use super::DEL_SIZE;
use super::NUM_ROW;
use crate::logsumexp;
use crate::op::Op;
use crate::EP;
impl PairHiddenMarkovModel {
    /// Return likelihood of x and y. It returns the probability to see the two sequence (x,y),
    /// summarizing all the alignment between x and y.
    /// In other words, it returns Sum_{alignment between x and y} Pr{alignment|self}.
    /// Roughly speaking, it is the value after log-sum-exp-ing all the alignment score.
    /// In HMM term, it is "forward" algorithm.
    /// If you want to get the raw DP table, please call `forward` functionality instead.
    pub fn likelihood(&self, reference: &[u8], query: &[u8]) -> f64 {
        let dptable = self.forward(reference, query);
        dptable.get_total_lk(query.len(), reference.len())
    }
    /// Forward algorithm. Return the raw DP table.
    pub(crate) fn forward(&self, reference: &[u8], query: &[u8]) -> DPTable {
        let xs = query;
        let ys = reference;
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        use super::BASE_TABLE;
        {
            let mut ins_accum = 0f64;
            for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
                let x = BASE_TABLE[x as usize];
                *dptable.get_mut(i, 0, State::Match) = EP;
                ins_accum += log_ins_emit[x];
                *dptable.get_mut(i, 0, State::Ins) =
                    log_ins_open + log_ins_ext * (i - 1) as f64 + ins_accum;
                *dptable.get_mut(i, 0, State::Del) = EP;
            }
        }
        {
            for (j, _) in ys.iter().enumerate().map(|(pos, y)| (pos + 1, y)) {
                *dptable.get_mut(0, j, State::Match) = EP;
                *dptable.get_mut(0, j, State::Del) = log_del_open + log_del_ext * (j - 1) as f64;
                *dptable.get_mut(0, j, State::Ins) = EP;
            }
        }
        *dptable.get_mut(0, 0, State::Ins) = EP;
        *dptable.get_mut(0, 0, State::Del) = EP;
        for (i, &x) in xs.iter().enumerate().map(|(p, x)| (p + 1, x)) {
            for (j, &y) in ys.iter().enumerate().map(|(p, y)| (p + 1, y)) {
                let x = BASE_TABLE[x as usize];
                let y = BASE_TABLE[y as usize];
                let mat = Self::logsumexp(
                    dptable.get(i - 1, j - 1, State::Match) + log_mat_ext,
                    dptable.get(i - 1, j - 1, State::Del) + log_mat_from_del,
                    dptable.get(i - 1, j - 1, State::Ins) + log_mat_from_ins,
                ) + log_mat_emit[(x << 2) | y];
                *dptable.get_mut(i, j, State::Match) = mat;
                let del = Self::logsumexp(
                    dptable.get(i, j - 1, State::Match) + log_del_open,
                    dptable.get(i, j - 1, State::Del) + log_del_ext,
                    dptable.get(i, j - 1, State::Ins) + log_del_from_ins,
                );
                *dptable.get_mut(i, j, State::Del) = del;
                let ins = Self::logsumexp(
                    dptable.get(i - 1, j, State::Match) + log_ins_open,
                    dptable.get(i - 1, j, State::Del) + log_ins_from_del,
                    dptable.get(i - 1, j, State::Ins) + log_ins_ext,
                ) + log_ins_emit[x];
                *dptable.get_mut(i, j, State::Ins) = ins;
            }
        }
        dptable
    }
    // Naive implementation of backward algorithm.
    pub(crate) fn backward(&self, reference: &[u8], query: &[u8]) -> DPTable {
        let xs = query;
        let ys = reference;
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        *dptable.get_mut(xs.len(), ys.len(), State::Match) = 0f64;
        *dptable.get_mut(xs.len(), ys.len(), State::Del) = 0f64;
        *dptable.get_mut(xs.len(), ys.len(), State::Ins) = 0f64;
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        use super::BASE_TABLE;
        {
            let mut gap = 0f64;
            for (i, &x) in xs.iter().enumerate().rev() {
                let prev = match 0 < i {
                    true => BASE_TABLE[xs[i - 1] as usize] << 2,
                    false => 16,
                };
                let x = BASE_TABLE[x as usize];
                gap += log_ins_emit[prev | x];
                *dptable.get_mut(i, ys.len(), State::Ins) =
                    log_ins_ext * (xs.len() - i) as f64 + gap;
                *dptable.get_mut(i, ys.len(), State::Del) =
                    log_ins_from_del + log_ins_ext * (xs.len() - i - 1) as f64 + gap;
                *dptable.get_mut(i, ys.len(), State::Match) =
                    log_ins_open + log_ins_ext * (xs.len() - i - 1) as f64 + gap;
            }
        }
        {
            for (j, _) in ys.iter().enumerate().rev() {
                *dptable.get_mut(xs.len(), j, State::Del) = log_del_ext * (ys.len() - j) as f64;
                *dptable.get_mut(xs.len(), j, State::Ins) =
                    log_del_from_ins + log_del_ext * (ys.len() - j - 1) as f64;
                *dptable.get_mut(xs.len(), j, State::Match) =
                    log_del_open + log_del_ext * (ys.len() - j - 1) as f64;
            }
        }
        for (i, &x) in xs.iter().enumerate().rev() {
            let prev = match 0 < i {
                true => BASE_TABLE[xs[i - 1] as usize] << 2,
                false => 16,
            };
            for (j, &y) in ys.iter().enumerate().rev() {
                // Match state;
                let x = BASE_TABLE[x as usize];
                let y = BASE_TABLE[y as usize] << 2;
                let mat =
                    log_mat_ext + log_mat_emit[y | x] + dptable.get(i + 1, j + 1, State::Match);
                let del = log_del_open + dptable.get(i, j + 1, State::Del);
                let ins = log_ins_open + log_ins_emit[prev | x] + dptable.get(i + 1, j, State::Ins);
                *dptable.get_mut(i, j, State::Match) = Self::logsumexp(mat, del, ins);
                // Del state.
                {
                    let mat = mat - log_mat_ext + log_mat_from_del;
                    let del = del - log_del_open + log_del_ext;
                    let ins = ins - log_ins_open + log_ins_from_del;
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
    /// Return the alignment path between x and y.
    /// In HMM term, it is "viterbi" algorithm.
    pub fn align(&self, reference: &[u8], query: &[u8]) -> (f64, Vec<Op>) {
        let xs = query;
        let ys = reference;
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        let mut dptable = DPTable::new(xs.len() + 1, ys.len() + 1);
        *dptable.get_mut(0, 0, State::Ins) = EP;
        *dptable.get_mut(0, 0, State::Del) = EP;
        {
            let mut ins_accum = log_ins_open;
            for i in 1..xs.len() + 1 {
                *dptable.get_mut(i, 0, State::Match) = EP;
                ins_accum += self.ins(xs[i - 1], None).ln();
                *dptable.get_mut(i, 0, State::Ins) = ins_accum;
                ins_accum += log_ins_ext;
                *dptable.get_mut(i, 0, State::Del) = EP;
            }
        }
        {
            for j in 1..ys.len() + 1 {
                *dptable.get_mut(0, j, State::Match) = EP;
                *dptable.get_mut(0, j, State::Ins) = EP;
                *dptable.get_mut(0, j, State::Del) = log_del_open + (j - 1) as f64 * log_del_ext;
            }
        }
        for (i, &x) in xs.iter().enumerate().map(|(p, x)| (p + 1, x)) {
            let x = super::BASE_TABLE[x as usize];
            for (j, &y) in ys.iter().enumerate().map(|(p, y)| (p + 1, y)) {
                let y = super::BASE_TABLE[y as usize];
                let mat = (dptable.get(i - 1, j - 1, State::Match) + log_mat_ext)
                    .max(dptable.get(i - 1, j - 1, State::Ins) + log_mat_from_ins)
                    .max(dptable.get(i - 1, j - 1, State::Del) + log_mat_from_del)
                    + log_mat_emit[(y << 2) | x];
                *dptable.get_mut(i, j, State::Match) = mat;
                let del = (dptable.get(i, j - 1, State::Match) + log_del_open)
                    .max(dptable.get(i, j - 1, State::Del) + log_del_ext)
                    .max(dptable.get(i, j - 1, State::Ins) + log_del_from_ins);
                *dptable.get_mut(i, j, State::Del) = del;
                let ins_lk = match (0 < i).then(|| xs[i - 1]) {
                    Some(b) => {
                        let prev = super::BASE_TABLE[b as usize];
                        log_ins_emit[(prev << 2) | x]
                    }
                    None => log_ins_emit[16 + x],
                };
                let ins = (dptable.get(i - 1, j, State::Match) + log_ins_open)
                    .max(dptable.get(i - 1, j, State::Del) + log_ins_from_del)
                    .max(dptable.get(i - 1, j, State::Ins) + log_ins_ext)
                    + ins_lk;
                // + log_ins_emit[x as usize];
                *dptable.get_mut(i, j, State::Ins) = ins;
            }
        }
        let (max_state, max_lk) = [State::Match, State::Ins, State::Del]
            .iter()
            .map(|&s| (s, dptable.get(xs.len(), ys.len(), s)))
            .max_by(|x, y| (x.1).partial_cmp(&(y.1)).unwrap())
            .unwrap();
        let (mut i, mut j, mut state) = (xs.len(), ys.len(), max_state);
        let mut ops: Vec<Op> = vec![];
        while i > 0 && j > 0 {
            let diff = 0.00000000001;
            let x = super::BASE_TABLE[xs[i - 1] as usize];
            let y = super::BASE_TABLE[ys[j - 1] as usize];
            let lk = dptable.get(i, j, state);
            ops.push(state.into());
            match state {
                State::Match => {
                    let mat_lk = lk - log_mat_emit[(y << 2) | x];
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
                    let mat = dptable.get(i, j - 1, State::Match) + log_del_open;
                    let del = dptable.get(i, j - 1, State::Del) + log_del_ext;
                    let ins = dptable.get(i, j - 1, State::Ins) + log_del_from_ins;
                    if (lk - mat).abs() < diff {
                        state = State::Match;
                    } else if (lk - del).abs() < diff {
                        state = State::Del;
                    } else {
                        assert!((lk - ins).abs() < diff);
                        state = State::Ins;
                    }
                    j -= 1;
                }
                State::Ins => {
                    let ins_lk = lk - log_ins_emit[x];
                    let mat = dptable.get(i - 1, j, State::Match) + log_ins_open;
                    let del = dptable.get(i - 1, j, State::Del) + log_ins_from_del;
                    let ins = dptable.get(i - 1, j, State::Ins) + log_ins_ext;
                    if (ins_lk - mat).abs() < diff {
                        state = State::Match;
                    } else if (ins_lk - del).abs() < diff {
                        state = State::Del;
                    } else {
                        assert!((ins_lk - ins).abs() < diff);
                        state = State::Ins;
                    }
                    i -= 1;
                }
            }
        }
        while i > 0 {
            i -= 1;
            ops.push(Op::Ins);
        }
        while j > 0 {
            j -= 1;
            ops.push(Op::Del);
        }
        ops.reverse();
        (max_lk, ops)
    }
    /// Return the modification table. `Tab[i * NUM_ROW..(i + 1)*NUM_ROW]` record the likelihood when changing the `i` th base of the `rs`.
    pub fn modification_table(&self, rs: &[u8], qs: &[u8]) -> (Vec<f64>, f64) {
        use crate::LogSumExp;
        let pre = self.forward(rs, qs);
        let post = self.backward(rs, qs);
        let total_len = NUM_ROW * (rs.len() + 1);
        let mut lse_lks = vec![LogSumExp::new(); total_len];
        pre.get(qs.len(), rs.len(), State::Match);
        post.get(qs.len(), rs.len(), State::Match);
        for (i, &q) in qs.iter().enumerate() {
            assert!(i <= qs.len());
            for (j, slots) in lse_lks.chunks_exact_mut(NUM_ROW).enumerate() {
                assert!(j <= rs.len());
                let mat_mat = pre.get(i, j, State::Match) + self.mat_mat.ln();
                let del_mat = pre.get(i, j, State::Del) + self.del_mat.ln();
                let ins_mat = pre.get(i, j, State::Ins) + self.ins_mat.ln();
                let mat_del = pre.get(i, j, State::Match) + self.mat_del.ln();
                let del_del = pre.get(i, j, State::Del) + self.del_del.ln();
                let ins_del = pre.get(i, j, State::Ins) + self.ins_del.ln();
                let post_no_del = post.get(i, j, State::Del);
                let post_del_del = match j < rs.len() {
                    true => post.get(i, j + 1, State::Del),
                    false => EP,
                };
                let post_mat_mat = match j < rs.len() {
                    true => post.get(i + 1, j + 1, State::Match),
                    false => EP,
                };
                let post_ins_mat = post.get(i + 1, j, State::Match);
                b"ACGT".iter().zip(slots.iter_mut()).for_each(|(&b, y)| {
                    let mat = self.obs(b, q).ln() + post_mat_mat;
                    let del = self.del(b).ln() + post_del_del;
                    *y += mat_mat + mat;
                    *y += del_mat + mat;
                    *y += ins_mat + mat;
                    *y += mat_del + del;
                    *y += del_del + del;
                    *y += ins_del + del;
                });
                b"ACGT"
                    .iter()
                    .zip(slots.iter_mut().skip(4))
                    .for_each(|(&b, y)| {
                        let mat = self.obs(b, q).ln() + post_ins_mat;
                        let del = self.del(b).ln() + post_no_del;
                        *y += mat_mat + mat;
                        *y += del_mat + mat;
                        *y += ins_mat + mat;
                        *y += mat_del + del;
                        *y += del_del + del;
                        *y += ins_del + del;
                    });
                // Copying the j..j+c bases ...
                (0..COPY_SIZE)
                    .filter(|len| j + len < rs.len())
                    .zip(slots.iter_mut().skip(8))
                    .for_each(|(len, y)| {
                        let pos = j + len + 1;
                        let r = rs[j];
                        let mat = self.obs(r, q).ln() + post_mat_mat;
                        let del = self.del(r).ln() + post_del_del;
                        {
                            *y += pre.get(i, pos, State::Match) + self.mat_mat.ln() + mat;
                            *y += pre.get(i, pos, State::Del) + self.del_mat.ln() + mat;
                            *y += pre.get(i, pos, State::Ins) + self.ins_mat.ln() + mat;
                        };
                        {
                            *y += pre.get(i, pos, State::Match) + self.mat_del.ln() + del;
                            *y += pre.get(i, pos, State::Del) + self.del_del.ln() + del;
                            *y += pre.get(i, pos, State::Ins) + self.ins_del.ln() + del;
                        };
                    });
                // deleting the j..j+d bases..
                (0..DEL_SIZE)
                    .filter(|d| j + d + 1 < rs.len())
                    .zip(slots.iter_mut().skip(8 + COPY_SIZE))
                    .for_each(|(len, y)| {
                        let post_pos = j + len + 1;
                        let r = rs[post_pos];
                        let post_mat_mat = post.get(i + 1, post_pos + 1, State::Match);
                        let mat = self.obs(r, q).ln() + post_mat_mat;
                        let post_del_del = post.get(i, post_pos + 1, State::Del);
                        let del = self.del(r).ln() + post_del_del;
                        *y += mat_mat + mat;
                        *y += del_mat + mat;
                        *y += ins_mat + mat;
                        *y += mat_del + del;
                        *y += del_del + del;
                        *y += ins_del + del;
                    });
            }
        }
        {
            let i = qs.len();
            for (j, slots) in lse_lks.chunks_exact_mut(NUM_ROW).enumerate() {
                assert!(j <= rs.len());
                let to_del = {
                    let mat_to_del = pre.get(i, j, State::Match) + self.mat_del.ln();
                    let del_to_del = pre.get(i, j, State::Del) + self.del_del.ln();
                    let ins_to_del = pre.get(i, j, State::Ins) + self.ins_del.ln();
                    logsumexp(&[mat_to_del, del_to_del, ins_to_del])
                };
                let post_no_del = post.get(i, j, State::Del);
                let post_del_del = match j < rs.len() {
                    true => post.get(i, j + 1, State::Del),
                    false => EP,
                };
                // Change the j-th base into ...
                b"ACGT".iter().zip(slots.iter_mut()).for_each(|(&b, y)| {
                    *y += to_del + post_del_del + self.del(b).ln();
                });
                // Insertion before the j-th base ...
                b"ACGT"
                    .iter()
                    .zip(slots.iter_mut().skip(4))
                    .for_each(|(&b, y)| {
                        *y += to_del + self.del(b).ln() + post_no_del;
                    });
                // Copying the j..j+c bases....
                (0..COPY_SIZE)
                    .filter(|len| j + len < rs.len())
                    .zip(slots.iter_mut().skip(8))
                    .for_each(|(len, y)| {
                        let r = rs[j];
                        let postpos = j + len + 1;
                        let del = self.del(r).ln() + post_del_del;
                        *y += pre.get(i, postpos, State::Match) + self.mat_del.ln() + del;
                        *y += pre.get(i, postpos, State::Del) + self.del_del.ln() + del;
                        *y += pre.get(i, postpos, State::Ins) + self.ins_del.ln() + del;
                    });
                // Deleting the j..j+d bases
                (0..DEL_SIZE)
                    .filter(|len| j + len < rs.len())
                    .zip(slots.iter_mut().skip(8 + COPY_SIZE))
                    .for_each(|(len, y)| {
                        let postpos = j + len + 1;
                        *y += match rs.get(postpos) {
                            Some(&r) => {
                                let post_del_del = post.get(i, postpos + 1, State::Del);
                                to_del + self.del(r).ln() + post_del_del
                            }
                            None => pre.get_total_lk(i, j),
                        };
                    });
            }
        }
        let slots: Vec<f64> = lse_lks.into_iter().map(|x| x.into()).collect();
        (slots, pre.get_total_lk(qs.len(), rs.len()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gen_seq;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    #[test]
    fn align() {
        let hmm = PairHiddenMarkovModel::default();
        for seed in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198 + seed);
            let template = gen_seq::generate_seq(&mut rng, 70);
            let profile = gen_seq::PROFILE;
            let query = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let (lk, ops) = hmm.align(&template, &query);
            let lk_l = hmm.eval_ln(&template, &query, &ops);
            assert!((lk - lk_l).abs() < 0.0001, "{},{}", lk, lk_l);
        }
    }
    #[test]
    fn forward() {
        let template = b"AAAAA";
        let hmm = PairHiddenMarkovModel::default();
        let lk = hmm.likelihood(template, template);
        let len = template.len() as f64;
        let expected = hmm.mat_mat.ln() * len + hmm.mat_emit[0].ln() * len;
        // eprintln!("{lk},{expected}");
        assert!(lk > expected);
        for seed in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198 + seed);
            let template = gen_seq::generate_seq(&mut rng, 70);
            let profile = gen_seq::PROFILE;
            let query = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let (lk, _) = hmm.align(&template, &query);
            let lk_f = hmm.likelihood(&template, &query);
            assert!(lk < lk_f, "{},{}", lk, lk_f);
        }
    }
    #[test]
    fn modification_table_test() {
        const ERR_THR: f64 = 1f64;
        for seed in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198 + seed);
            let template = gen_seq::generate_seq(&mut rng, 70);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let query = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let (modif_table, _) = hmm.modification_table(&template, &query);
            let mut mod_version = template.clone();
            println!("{}", seed);
            // Mutation error
            for (j, modif_table) in modif_table
                .chunks_exact(NUM_ROW)
                .take(template.len())
                .enumerate()
            {
                println!("{}", j);
                let orig = mod_version[j];
                for (&base, lk_m) in b"ACGT".iter().zip(modif_table) {
                    mod_version[j] = base;
                    let lk = hmm.likelihood(&mod_version, &query);
                    assert!((lk - lk_m).abs() < ERR_THR, "{},{},mod", lk, lk_m);
                    // println!("M\t{}\t{}", j, (lk - lk_m).abs());
                    mod_version[j] = orig;
                }
                // Insertion error
                for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                    mod_version.insert(j, base);
                    let lk = hmm.likelihood(&mod_version, &query);
                    assert!((lk - lk_m).abs() < ERR_THR, "{},{}", lk, lk_m);
                    // println!("I\t{}\t{}", j, (lk - lk_m).abs());
                    mod_version.remove(j);
                }
                // Copying mod
                for len in (0..COPY_SIZE).filter(|c| j + c < template.len()) {
                    let lk_m = modif_table[8 + len];
                    let mod_version: Vec<_> = template[..j + len + 1]
                        .iter()
                        .chain(template[j..].iter())
                        .copied()
                        .collect();
                    let lk = hmm.likelihood(&mod_version, &query);
                    // println!("C\t{}\t{}\t{}", j, len, (lk - lk_m).abs());
                    assert!((lk - lk_m).abs() < ERR_THR, "{},{}", lk, lk_m);
                }
                // Deletion error
                for len in (0..DEL_SIZE).filter(|d| j + d + 1 < template.len()) {
                    let lk_m = modif_table[8 + COPY_SIZE + len];
                    let mod_version: Vec<_> = template[..j]
                        .iter()
                        .chain(template[j + len + 1..].iter())
                        .copied()
                        .collect();
                    let lk = hmm.likelihood(&mod_version, &query);
                    // println!("D\t{}\t{}\t{}", j, len, lk - lk_m);
                    assert!(
                        (lk - lk_m).abs() < ERR_THR,
                        "{},{},{}",
                        lk,
                        lk_m,
                        template.len(),
                    );
                }
            }
            let modif_table = modif_table.chunks_exact(NUM_ROW).last().unwrap();
            for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                mod_version.push(base);
                let lk = hmm.likelihood(&mod_version, &query);
                assert!((lk - lk_m).abs() < 1.0001, "{},{}", lk, lk_m);
                mod_version.pop();
            }
        }
    }
}
