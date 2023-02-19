use super::BASE_TABLE;
use super::{COPY_SIZE, DEL_SIZE, NUM_ROW};
use crate::op::Op;
// Anti diagonal version of the HMM.
// The index is (a, i) where a is the anti diagonal (i + j = a) and i is the position of the *query*.
impl super::PairHiddenMarkovModel {
    pub fn likelihood_antidiagonal_bootstrap(&self, rs: &[u8], qs: &[u8], radius: usize) -> f64 {
        let ops = crate::op::bootstrap_ops(rs.len(), qs.len());
        self.likelihood_antidiagonal(rs, qs, &ops, radius)
    }
    pub fn likelihood_antidiagonal(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: usize) -> f64 {
        let block_size = 32;
        let (dptable, scaling) = self.pre(rs, qs, ops, radius, block_size);
        let last_ad = rs.len() + qs.len();
        let (mat, ins, del) = dptable[(last_ad, qs.len())];
        let scale: f64 = scaling.iter().map(|scl| scl.ln()).sum();
        (mat + ins + del).ln() + scale
    }
    pub fn modification_table_antidiagonal(
        &self,
        rs: &[u8],
        qs: &[u8],
        radius: usize,
        ops: &[Op],
    ) -> Vec<f64> {
        let block_size = 7;
        assert!(COPY_SIZE * 2 < block_size);
        assert!(DEL_SIZE * 2 < block_size);
        let pre = self.pre(rs, qs, ops, radius, block_size);
        let post = self.post(rs, qs, ops, radius, block_size);
        let total_len = NUM_ROW * (rs.len() + 1);
        let mut mod_table = vec![0f64; total_len];
        self.modification_table_ad_inner(rs, qs, &pre, &post, &mut mod_table, block_size);
        mod_table
    }
    fn modification_table_ad_inner(
        &self,
        rs: &[u8],
        qs: &[u8],
        &(ref pre_dp, ref pre_scl): &FBTable,
        &(ref post_dp, ref post_scl): &FBTable,
        mod_table: &mut [f64],
        block_size: usize,
    ) {
        // Let S[T] = \prod_{t=0}^{t=T} pre_scl[t] \prod_{t=T}^{t=L} post_scl[t].
        // Then, this returns max_scale = M = max_T S[T] and S[T]/M.
        let (max_scale, combined_scl) = {
            let pre_scl_ln = pre_scl.iter().map(|x| x.ln());
            let post_scl_ln = post_scl.iter().map(|x| x.ln());
            let post_scl_prod_ln: f64 = post_scl_ln.clone().sum();
            let max_scale: f64 = std::iter::zip(pre_scl_ln.clone(), post_scl_ln.clone())
                .scan(post_scl_prod_ln, |lk, (pre, post)| {
                    *lk += pre;
                    *lk -= post;
                    Some(*lk + post)
                })
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();
            let combined_scl: Vec<_> = std::iter::zip(pre_scl_ln, post_scl_ln)
                .scan(post_scl_prod_ln, |lk, (pre, post)| {
                    *lk += pre;
                    *lk -= post;
                    Some((*lk + post - max_scale).exp())
                })
                .collect();
            (max_scale, combined_scl)
        };
        assert_eq!(pre_dp.filling_regions, post_dp.filling_regions);
        for (ad, &(start, end)) in pre_dp.filling_regions.iter().enumerate() {
            for q_idx in start..end {
                let r_idx = ad - q_idx;
                let slot_start = r_idx * NUM_ROW;
                // Mutate the `rs[r_idx]` into ACGT
                let slots = std::iter::zip(mod_table.iter_mut().skip(slot_start), b"ACGT");
                for (s, &b) in slots.filter(|_| r_idx < rs.len()) {
                    // If it matches the `q_idx`.
                    if q_idx < qs.len() {
                        let mat = self.to_mat(pre_dp[(ad, q_idx)])
                            * self.obs(b, qs[q_idx])
                            * post_dp[(ad + 2, q_idx + 1)].0;
                        let normalize_factor = combined_scl[ad / block_size];
                        let additional_factor = match ad / block_size < (ad + 2) / block_size {
                            true => post_scl[ad / block_size].recip(),
                            false => 1f64,
                        };
                        *s += mat * normalize_factor * additional_factor;
                    }
                    // If the base is consumed by deletion *before* the `q_idx` base.
                    let del =
                        self.to_del(pre_dp[(ad, q_idx)]) * self.del(b) * post_dp[(ad + 1, q_idx)].2;
                    let normalize_factor = combined_scl[ad / block_size];
                    let additional_factor = match ad / block_size < (ad + 1) / block_size {
                        true => post_scl[ad / block_size].recip(),
                        false => 1f64,
                    };
                    *s += del * normalize_factor * additional_factor;
                }
                // Insert a base *before* the `r_idx`.
                for (s, &b) in std::iter::zip(mod_table.iter_mut().skip(slot_start + 4), b"ACGT") {
                    // If it matches the `q_idx`
                    if q_idx < qs.len() {
                        let mat = self.to_mat(pre_dp[(ad, q_idx)])
                            * self.obs(b, qs[q_idx])
                            * post_dp[(ad + 1, q_idx + 1)].0;
                        let normalize_factor = combined_scl[ad / block_size];
                        let additional_factor = match ad / block_size < (ad + 1) / block_size {
                            true => post_scl[ad / block_size].recip(),
                            false => 1f64,
                        };
                        *s += mat * normalize_factor * additional_factor;
                    }
                    // If the base is consumed by the deletion *before* the `q_idx` base.
                    let del =
                        self.to_del(pre_dp[(ad, q_idx)]) * self.del(b) * post_dp[(ad, q_idx)].2;
                    let normalize_factor = combined_scl[ad / block_size];
                    *s += del * normalize_factor;
                }
                // Copy the `r_idx`..`r_idx`+c bases of `rs`
                for (len, s) in mod_table
                    .iter_mut()
                    .skip(slot_start + 8)
                    .take(COPY_SIZE)
                    .enumerate()
                {
                    let len = len + 1;
                    if qs.len() + rs.len() < ad + len || rs.len() < r_idx + len {
                        continue;
                    }
                    let (pre_mat, _pre_ins, pre_del) = pre_dp[(ad + len, q_idx)];
                    let (post_mat, _post_ins, post_del) = post_dp[(ad, q_idx)];
                    let lk = pre_mat * post_mat + pre_del * post_del;
                    let normalize_factor = combined_scl[(ad + len) / block_size];
                    let additional_factor = match ad / block_size < (ad + len) / block_size {
                        true => post_scl[ad / block_size],
                        false => 1f64,
                    };
                    *s += lk * normalize_factor * additional_factor;
                }
                // Deleting the `r_idx`..`r_idx + d` bases.
                for (len, s) in mod_table
                    .iter_mut()
                    .skip(slot_start + 8 + COPY_SIZE)
                    .take(DEL_SIZE)
                    .enumerate()
                {
                    let len = len + 1;
                    if qs.len() + rs.len() < ad + len || rs.len() < r_idx + len {
                        continue;
                    }
                    let (pre_mat, _pre_ins, pre_del) = pre_dp[(ad, q_idx)];
                    let (post_mat, _post_ins, post_del) = post_dp[(ad + len, q_idx)];
                    let lk = pre_mat * post_mat + pre_del * post_del;
                    let normalize_factor = combined_scl[(ad + len) / block_size];
                    let additional_factor = match ad / block_size < (ad + len) / block_size {
                        true => pre_scl[(ad + len) / block_size].recip(),
                        false => 1f64,
                    };
                    *s += lk * normalize_factor * additional_factor;
                }
            }
        }
        // Scaling.
        mod_table.iter_mut().for_each(|x| {
            *x = x.ln() + max_scale;
        });
    }
    fn pre(
        &self,
        rs: &[u8],
        qs: &[u8],
        ops: &[Op],
        radius: usize,
        block_size: usize,
    ) -> (DPTable<(f64, f64, f64)>, Vec<f64>) {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let mut scaling = Vec::with_capacity(filling_regions.len() / block_size + 1);
        let mut dptable = DPTable::new(filling_regions, (0f64, 0f64, 0f64));
        self.pre_fill(rs, qs, &mut dptable, &mut scaling, block_size);
        (dptable, scaling)
    }
    fn pre_fill(
        &self,
        rs: &[u8],
        qs: &[u8],
        dptable: &mut DPTable<(f64, f64, f64)>,
        scaling: &mut Vec<f64>,
        block_size: usize,
    ) {
        dptable[(0, 0)] = (1f64, 0f64, 0f64);
        for ad in 1..rs.len() + qs.len() + 1 {
            let (start, end) = dptable.filling_regions[ad];
            for q_idx in start..end {
                let r_idx = ad - q_idx;
                if r_idx == 0 {
                    let q = qs[q_idx - 1];
                    let prev = (1 < q_idx).then(|| qs[q_idx - 2]);
                    let ins_lk = self.to_ins(dptable[(ad - 1, q_idx - 1)]) * self.ins(q, prev);
                    dptable[(ad, q_idx)] = (0f64, ins_lk, 0f64);
                } else if q_idx == 0 {
                    let del_lk = self.to_del(dptable[(ad - 1, q_idx)]) * self.del(rs[r_idx - 1]);
                    dptable[(ad, q_idx)] = (0f64, 0f64, del_lk);
                } else {
                    let (q, r) = (qs[q_idx - 1], rs[r_idx - 1]);
                    let prev = (1 < q_idx).then(|| qs[q_idx - 2]);
                    let mat_lk = self.to_mat(dptable[(ad - 2, q_idx - 1)]) * self.obs(r, q);
                    let ins_lk = self.to_ins(dptable[(ad - 1, q_idx - 1)]) * self.ins(q, prev);
                    let del_lk = self.to_del(dptable[(ad - 1, q_idx)]) * self.del(r);
                    dptable[(ad, q_idx)] = (mat_lk, ins_lk, del_lk);
                }
            }
            if (ad + 1) % block_size == 0 || ad == rs.len() + qs.len() {
                // Normalize.
                let bucket = ad / block_size;
                let b_start = bucket * block_size;
                let b_end = ((bucket + 1) * block_size).min(rs.len() + qs.len() + 1);
                let sum: f64 = dptable.filling_regions[b_start..b_end]
                    .iter()
                    .enumerate()
                    .map(|(ad, &(start, end))| {
                        (start..end).fold(0f64, |sum, qpos| {
                            let (mat, ins, del) = dptable[(ad, qpos)];
                            sum + mat + ins + del
                        })
                    })
                    .sum();
                for ad in b_start..b_end {
                    let (start, end) = dptable.filling_regions[ad];
                    for qpos in start..end {
                        let (mat, ins, del) = dptable[(ad, qpos)];
                        dptable[(ad, qpos)] = (mat / sum, ins / sum, del / sum);
                    }
                }
                scaling.push(sum);
            }
        }
    }
    fn post(
        &self,
        rs: &[u8],
        qs: &[u8],
        ops: &[Op],
        radius: usize,
        block_size: usize,
    ) -> (DPTable<(f64, f64, f64)>, Vec<f64>) {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let mut scaling = Vec::with_capacity(filling_regions.len() / block_size + 1);
        let mut dptable = DPTable::new(filling_regions, (0f64, 0f64, 0f64));
        self.post_fill(rs, qs, &mut dptable, &mut scaling, block_size);
        (dptable, scaling)
    }
    fn post_fill(
        &self,
        rs: &[u8],
        qs: &[u8],
        dptable: &mut DPTable<(f64, f64, f64)>,
        scaling: &mut Vec<f64>,
        block_size: usize,
    ) {
        dptable[(qs.len() + rs.len(), qs.len())] = (1f64, 1f64, 1f64);
        if (rs.len() + qs.len()) % block_size == 0 {
            scaling.push(1f64);
        }
        for ad in (0..rs.len() + qs.len()).rev() {
            let (start, end) = dptable.filling_regions[ad];
            for q_idx in (start..end).rev() {
                let r_idx = ad - q_idx;
                if r_idx == rs.len() {
                    let q = qs[q_idx];
                    let prev = (0 < q_idx).then(|| qs[q_idx - 1]);
                    let ins = self.ins(q, prev) * dptable[(ad + 1, q_idx + 1)].1;
                    let elm = (self.mat_ins * ins, self.ins_ins * ins, self.del_ins * ins);
                    dptable[(ad, q_idx)] = elm;
                } else if q_idx == qs.len() {
                    let r = rs[r_idx];
                    let del = self.del(r) * dptable[(ad + 1, q_idx)].2;
                    let elm = (self.mat_del * del, self.ins_del * del, self.del_del * del);
                    dptable[(ad, q_idx)] = elm;
                } else {
                    let r = rs[r_idx];
                    let q = qs[q_idx];
                    let prev = (0 < q_idx).then(|| qs[q_idx - 1]);
                    let af_mat = self.obs(r, q) * dptable[(ad + 2, q_idx + 1)].0;
                    let af_ins = self.ins(q, prev) * dptable[(ad + 1, q_idx + 1)].1;
                    let af_del = self.del(r) * dptable[(ad + 1, q_idx)].2;
                    let mat = self.mat_mat * af_mat + self.mat_ins * af_ins + self.mat_del * af_del;
                    let ins = self.ins_mat * af_mat + self.ins_ins * af_ins + self.ins_del * af_del;
                    let del = self.del_mat * af_mat + self.del_ins * af_ins + self.del_del * af_del;
                    dptable[(ad, q_idx)] = (mat, ins, del);
                }
            }
            if ad % block_size == 0 {
                // Normalize
                let bucket = ad / block_size;
                let b_start = bucket * block_size;
                let b_end = ((bucket + 1) * block_size).min(rs.len() + qs.len() + 1);
                let sum: f64 = dptable
                    .filling_regions
                    .iter()
                    .enumerate()
                    .take(b_end)
                    .skip(b_start)
                    .map(|(ad, &(start, end))| {
                        (start..end).fold(0f64, |sum, qpos| {
                            let (mat, ins, del) = dptable[(ad, qpos)];
                            sum + mat + ins + del
                        })
                    })
                    .sum();
                for ad in b_start..b_end {
                    let (start, end) = dptable.filling_regions[ad];
                    for qpos in start..end {
                        let (mat, ins, del) = dptable[(ad, qpos)];
                        dptable[(ad, qpos)] = (mat / sum, ins / sum, del / sum);
                    }
                }
                scaling.push(sum);
            }
        }
        scaling.reverse();
    }
    pub fn align_antidiagonal_bootstrap(
        &self,
        rs: &[u8],
        qs: &[u8],
        radius: usize,
    ) -> (f64, Vec<Op>) {
        let ops = crate::op::bootstrap_ops(rs.len(), qs.len());
        self.align_antidiagonal(rs, qs, &ops, radius)
    }
    pub fn align_antidiagonal(
        &self,
        rs: &[u8],
        qs: &[u8],
        ops: &[Op],
        radius: usize,
    ) -> (f64, Vec<Op>) {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        self.align_antidiagonal_filling(rs, qs, &filling_regions)
    }
    fn align_antidiagonal_filling(
        &self,
        rs: &[u8],
        qs: &[u8],
        filling_regions: &[(usize, usize)],
    ) -> (f64, Vec<Op>) {
        let neg_inf = (rs.len() + qs.len()) as f64 * 3f64 * -100000000000000000000f64;
        let last_ad = filling_regions.len() - 1;
        // (Match, Ins, Del)
        let mut dptable = DPTable::new(filling_regions.to_vec(), (neg_inf, neg_inf, neg_inf));
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        dptable[(0, 0)] = (0f64, neg_inf, neg_inf);
        for ad in 1..rs.len() + qs.len() + 1 {
            let (start, end) = dptable.filling_regions[ad];
            let mut prev = 16;
            if start == 0 {
                let (q_idx, r_idx) = (0, ad);
                let del_lk = log_del_open + (r_idx - 1) as f64 * log_del_ext;
                dptable[(ad, q_idx)] = (neg_inf, neg_inf, del_lk);
            }
            if ad == end - 1 {
                let q_idx = end - 1;
                let q = BASE_TABLE[qs[q_idx - 1] as usize];
                let ins_obs = log_ins_emit[prev | q];
                prev = q << 2;
                let (mat, ins, del) = dptable[(ad - 1, q_idx - 1)];
                let ins_lk = (mat + log_ins_open)
                    .max(ins + log_ins_ext)
                    .max(del + log_ins_from_del)
                    + ins_obs;
                dptable[(ad, q_idx)] = (neg_inf, ins_lk, neg_inf);
            }
            let (start, end) = (start.max(1), end.min(ad));
            for q_idx in start..end {
                let r_idx = ad - q_idx;
                assert!(0 < q_idx);
                assert!(0 < r_idx);
                let q = BASE_TABLE[qs[q_idx - 1] as usize];
                let r = BASE_TABLE[rs[r_idx - 1] as usize];
                assert!(1 < ad);
                let mat_lk = {
                    let (mat, ins, del) = dptable[(ad - 2, q_idx - 1)];
                    (mat + log_mat_ext)
                        .max(del + log_mat_from_del)
                        .max(ins + log_mat_from_ins)
                        + log_mat_emit[(r << 2) | q]
                };
                let ins_lk = {
                    let ins_lk = log_ins_emit[prev | q];
                    let (mat, ins, del) = dptable[(ad - 1, q_idx - 1)];
                    (mat + log_ins_open)
                        .max(ins + log_ins_ext)
                        .max(del + log_ins_from_del)
                        + ins_lk
                };
                let del_lk = {
                    let (mat, ins, del) = dptable[(ad - 1, q_idx)];
                    (mat + log_del_open)
                        .max(ins + log_del_from_ins)
                        .max(del + log_del_ext)
                };
                prev = q << 2;
                dptable[(ad, q_idx)] = (mat_lk, ins_lk, del_lk);
            }
        }
        // Traceback.
        let (lk, mut state) = {
            let (mat, ins, del) = dptable[(last_ad, qs.len())];
            let max = mat.max(ins).max(del);
            if max == mat {
                (max, 0)
            } else if max == ins {
                (max, 1)
            } else {
                assert!(max == del);
                (max, 2)
            }
        };
        let mut ops = vec![];
        let (mut r_idx, mut q_idx) = (rs.len(), qs.len());
        while 0 < r_idx && 0 < q_idx {
            let ad = r_idx + q_idx;
            state = match state {
                0 => {
                    assert!(1 < ad);
                    ops.push(Op::Match);
                    let (mat, ins, del) = dptable[(ad - 2, q_idx - 1)];
                    let max = (mat + log_mat_ext)
                        .max(del + log_mat_from_del)
                        .max(ins + log_mat_from_ins);
                    q_idx -= 1;
                    r_idx -= 1;
                    if max == mat + log_mat_ext {
                        0
                    } else if max == ins + log_mat_from_ins {
                        1
                    } else {
                        assert!(max == del + log_mat_from_del);
                        2
                    }
                }
                1 => {
                    ops.push(Op::Ins);
                    let (mat, ins, del) = dptable[(ad - 1, q_idx - 1)];
                    let max = (mat + log_ins_open)
                        .max(ins + log_ins_ext)
                        .max(del + log_ins_from_del);
                    q_idx -= 1;
                    if max == mat + log_ins_open {
                        0
                    } else if max == ins + log_ins_ext {
                        1
                    } else {
                        assert!(max == del + log_del_from_ins);
                        2
                    }
                }
                2 => {
                    ops.push(Op::Del);
                    let (mat, ins, del) = dptable[(ad - 1, q_idx)];
                    let max = (mat + log_del_open)
                        .max(ins + log_del_from_ins)
                        .max(del + log_del_ext);
                    r_idx -= 1;
                    if max == mat + log_del_open {
                        0
                    } else if max == ins + log_del_from_ins {
                        1
                    } else {
                        assert!(max == del + log_del_ext);
                        2
                    }
                }
                _ => unreachable!(),
            };
        }
        ops.extend(std::iter::repeat(Op::Del).take(r_idx));
        ops.extend(std::iter::repeat(Op::Ins).take(q_idx));
        ops.reverse();
        (lk, ops)
    }
    pub fn polish_until_converge_antidiagonal<T, O>(
        &self,
        draft: &[u8],
        xss: &[T],
        opss: &mut [O],
        config: &super::HMMPolishConfig,
    ) -> Vec<u8>
    where
        T: std::borrow::Borrow<[u8]>,
        O: std::borrow::BorrowMut<Vec<Op>>,
    {
        let &super::HMMPolishConfig {
            radius,
            take_num,
            ignore_edge,
        } = config;
        let default_radius = radius;
        let take_num = take_num.min(xss.len());
        let mut rs = draft.to_vec();
        let mut radius_reads: Vec<Vec<_>> = opss
            .iter()
            .map(|ops| {
                let ops = ops.borrow();
                let center_line = center_line(ops);
                center_line.iter().map(|&qpos| (qpos, radius)).collect()
            })
            .collect();
        // let block_size = 12;
        //        let mut memory = Memory::new(block_size);
        for t in 0..100 {
            let inactive = INACTIVE_TIME + (t * INACTIVE_TIME) % rs.len();
            let mut modif_table = vec![0f64; (rs.len() + 1) * NUM_ROW];
            let mut lk = 0f64;
            for (seq, radius) in std::iter::zip(xss, &radius_reads).take(take_num) {
                assert_eq!(radius.len(), seq.borrow().len() + rs.len() + 1);
                let (m_lk, mt) = self.dp(&rs, seq.borrow(), radius);
                lk += m_lk;
                modif_table.iter_mut().zip(&mt).for_each(|(x, y)| *x += y);
            }
            eprintln!("LK\t{lk}");
            let changed_pos =
                super::polish_by_modification_table(&mut rs, &modif_table, lk, inactive);
            let changed_pos: Vec<_> = changed_pos
                .iter()
                .map(|&(pos, op)| (pos, super::usize_to_edit_op(op)))
                .collect();
            println!("{changed_pos:?}");
            let ops_seq = opss.iter_mut().zip(xss.iter());
            for (radius_per, (ops, xs)) in std::iter::zip(radius_reads.iter_mut(), ops_seq) {
                let (ops, xs) = (ops.borrow_mut(), xs.borrow());
                let (qlen, rlen) = (xs.len(), rs.len());
                let changed_iter = changed_pos.iter().copied();
                // Fix path.
                crate::op::fix_alignment_path(ops, changed_iter, qlen, rlen);
                // Fix radius.
                update_radius(ops, &changed_pos, radius_per, default_radius, qlen, rlen);
                assert_eq!(radius_per.len(), xs.len() + rs.len() + 1);
                // Fix operations.
                let filling = radius_to_filling(&radius_per, rs.len(), xs.len());
                *ops = self.align_antidiagonal_filling(&rs, xs, &filling).1;
            }
            if changed_pos.is_empty() {
                break;
            }
            println!("Loop\t{t}");
        }
        rs
    }
    fn dp(&self, rs: &[u8], qs: &[u8], radius: &[(usize, usize)]) -> (f64, Vec<f64>) {
        assert_eq!(radius.len(), rs.len() + qs.len() + 1);
        let block_size = 12;
        let filling_regions = radius_to_filling(radius, rs.len(), qs.len());
        for (start, end) in filling_regions.iter() {
            assert!(start < end);
        }
        let post = {
            let mut post_scl = Vec::with_capacity(filling_regions.len() / block_size + 1);
            let mut post_dp = DPTable::new(filling_regions.clone(), (0f64, 0f64, 0f64));
            self.post_fill(rs, qs, &mut post_dp, &mut post_scl, block_size);
            (post_dp, post_scl)
        };
        let (pre, lk) = {
            let mut pre_scl = Vec::with_capacity(filling_regions.len() / block_size + 1);
            let mut pre_dp = DPTable::new(filling_regions, (0f64, 0f64, 0f64));
            self.pre_fill(rs, qs, &mut pre_dp, &mut pre_scl, block_size);
            let last_ad = rs.len() + qs.len();
            let (mat, ins, del) = pre_dp[(last_ad, qs.len())];
            let scale: f64 = pre_scl.iter().map(|scl| scl.ln()).sum();
            let lk = (mat + ins + del).ln() + scale;
            ((pre_dp, pre_scl), lk)
        };
        let total_len = NUM_ROW * (rs.len() + 1);
        let mut mod_table = vec![0f64; total_len];
        self.modification_table_ad_inner(rs, qs, &pre, &post, &mut mod_table, block_size);
        (lk, mod_table)
    }
}

const INACTIVE_TIME: usize = 4;
const MIN_RADIUS: usize = 4;
const RAD_SCALE: usize = 2;
use crate::op::Edit;
fn update_radius(
    ops: &[Op],
    changed_pos: &[(usize, Edit)],
    radius: &mut Vec<(usize, usize)>,
    _default: usize,
    qlen: usize,
    rlen: usize,
) {
    let mut changed_pos = changed_pos.iter().peekable();
    let orig_len = radius.len();
    let (mut qpos, mut rpos) = (0, 0);
    let mut offset: isize = 0;
    if let Some(&&(pos, ed)) = changed_pos.peek() {
        if pos == rpos {
            match ed {
                Edit::Subst => {}
                Edit::Insertion => offset -= 1,
                Edit::Copy(l) => offset -= l as isize,
                Edit::Deletion(l) => offset += l as isize,
            }
        }
        changed_pos.next();
    }
    for &op in ops {
        let diag = ((qpos + rpos) as isize + offset).max(0) as usize;
        match op {
            Op::Mismatch | Op::Match => {
                let new_r = radius[diag].1;
                radius.push((qpos, new_r));
                rpos += 1;
                let new_r = radius[diag + 1].1;
                radius.push((qpos, new_r));
                qpos += 1;
            }
            Op::Ins => {
                let new_r = radius[diag].1;
                radius.push((qpos, new_r));
                qpos += 1
            }
            Op::Del => {
                let new_r = radius[diag].1;
                radius.push((qpos, new_r));
                rpos += 1;
            }
        }
    }
    let diag = ((qpos + rpos) as isize + offset) as usize;
    let new_r = radius[diag].1;
    radius.push((qpos, new_r));
    assert_eq!(rpos, rlen);
    assert_eq!(qpos, qlen);
    {
        let mut idx = 0;
        radius.retain(|_| {
            idx += 1;
            orig_len < idx
        });
    }
    assert_eq!(radius.len(), qlen + rlen + 1);
}

fn radius_to_filling(radius: &[(usize, usize)], rlen: usize, qlen: usize) -> Vec<(usize, usize)> {
    assert_eq!(radius.len(), rlen + qlen + 1);
    radius
        .iter()
        .enumerate()
        .map(|(ad, &(qpos, radius))| cap(ad, qpos, radius, rlen, qlen))
        .collect()
}

fn cap(diag: usize, qpos: usize, radius: usize, rlen: usize, qlen: usize) -> (usize, usize) {
    let start = qpos.saturating_sub(radius).max(diag.saturating_sub(rlen));
    let end = (qpos + radius).min(qlen).min(diag);
    assert!(diag < rlen + qlen + 1);
    if end < start {
        println!("{diag}\t{qpos}\t{radius}\t{rlen}\t{qlen}");
    }
    assert!(start < end + 1, "{},{}", start, end);
    (start, end + 1)
}

fn center_line(ops: &[Op]) -> Vec<usize> {
    let mut q_positions = vec![];
    let mut qpos = 0;
    for &op in ops {
        match op {
            Op::Match | Op::Mismatch => {
                q_positions.push(qpos);
                qpos += 1;
                q_positions.push(qpos);
            }
            Op::Ins => {
                q_positions.push(qpos);
                qpos += 1;
            }
            Op::Del => q_positions.push(qpos),
        }
    }
    q_positions.push(qpos);
    q_positions
}

// Index of the anti-diagonal (a = 0, ..., qs.len() + rs.len())
fn filling_region(ops: &[Op], radius: usize, rlen: usize, qlen: usize) -> Vec<(usize, usize)> {
    let center_line = center_line(ops);
    let radius: Vec<_> = center_line
        .into_iter()
        .enumerate()
        .map(|(ad, qpos)| cap(ad, qpos, radius, rlen, qlen))
        .collect();
    assert_eq!(radius.len(), rlen + qlen + 1);
    radius
}

// #[derive(Debug, Clone)]
// struct Memory {
//     pre: DPTable<(f64, f64, f64)>,
//     post: DPTable<(f64, f64, f64)>,
//     mod_table: Vec<f64>,
//     block_size: usize,
// }

// impl Memory {
//     fn new(block_size: usize) -> Self {
//         let ub = (0f64, 0f64, 0f64);
//         Self {
//             pre: DPTable::new(vec![], ub),
//             post: DPTable::new(vec![], ub),
//             mod_table: Vec::new(),
//             block_size,
//         }
//     }
//     fn mod_table(&self) -> &[f64] {
//         &self.mod_table
//     }
// }

#[derive(Debug, Clone)]
struct DPTable<T: std::fmt::Debug + Clone + Copy> {
    inner: Vec<T>,
    accum_count: Vec<usize>,
    filling_regions: Vec<(usize, usize)>,
}

type FBTable = (DPTable<(f64, f64, f64)>, Vec<f64>);

use std::fmt::Debug;
impl<T: Debug + Copy> DPTable<T> {
    fn new(filling_regions: Vec<(usize, usize)>, ub: T) -> Self {
        let total_cells: usize = filling_regions.iter().map(|(s, e)| e - s).sum();
        let accum_count: Vec<_> = filling_regions
            .iter()
            .map(|(s, e)| e - s)
            .scan(0, |accum, len| {
                *accum += len;
                Some(*accum - len)
            })
            .collect();
        let inner = vec![ub; total_cells];
        Self {
            filling_regions,
            inner,
            accum_count,
        }
    }
}
impl<T: Debug + Copy> std::ops::Index<(usize, usize)> for DPTable<T> {
    type Output = T;
    fn index(&self, (anti_diag, qpos): (usize, usize)) -> &Self::Output {
        let start = self.accum_count[anti_diag];
        let (offset, _) = self.filling_regions[anti_diag];
        self.inner.get(start + qpos - offset).unwrap()
    }
}

impl<T: Debug + Copy> std::ops::IndexMut<(usize, usize)> for DPTable<T> {
    fn index_mut(&mut self, (anti_diag, qpos): (usize, usize)) -> &mut Self::Output {
        let start = self.accum_count[anti_diag];
        let (offset, _) = self.filling_regions[anti_diag];
        self.inner.get_mut(start + qpos - offset).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::gen_seq::{self, Generate};
    use crate::hmm::PairHiddenMarkovModel;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    #[test]
    fn align() {
        let model = PairHiddenMarkovModel::default();
        let (_lk, ops) = model.align_antidiagonal_bootstrap(b"ACCG", b"ACCG", 5);
        assert_eq!(ops, vec![Op::Match; 4]);
        let (_lk, ops) = model.align_antidiagonal_bootstrap(b"ACCG", b"", 2);
        assert_eq!(ops, vec![Op::Del; 4]);
        let (_lk, ops) = model.align_antidiagonal_bootstrap(b"", b"ACCG", 2);
        assert_eq!(ops, vec![Op::Ins; 4]);
        let (_lk, ops) = model.align_antidiagonal_bootstrap(b"ATGCCGCACAGTCGAT", b"ATCCGC", 5);
        use Op::*;
        let answer = vec![vec![Match; 2], vec![Del], vec![Match; 4], vec![Del; 9]].concat();
        assert_eq!(ops, answer);
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 300);
        let profile = gen_seq::PROFILE;
        let hmm = PairHiddenMarkovModel::default();
        let radius = 50;
        let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
        hmm.align_antidiagonal_bootstrap(&template, &seq, radius);
        let (lk, ops) = model.align_antidiagonal_bootstrap(b"CCG", b"ACCG", 3);
        let (lk_f, _) = model.align(b"CCG", b"ACCG");
        assert_eq!(ops, vec![Op::Ins, Op::Match, Op::Match, Op::Match]);
        assert!((lk - lk_f).abs() < 0.001, "{},{}", lk, lk_f);
        eprintln!("{:?}\t{:.3}", ops, lk);
    }
    #[test]
    fn align_test() {
        for i in 0..200 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let template = gen_seq::generate_seq(&mut rng, 300);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let radius = 50;
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let (lk, _ops) = hmm.align_antidiagonal_bootstrap(&template, &seq, radius);
            let (lk_f, _) = hmm.align(&template, &seq);
            assert!((lk - lk_f).abs() < 0.001, "{},{}", lk, lk_f);
        }
    }
    #[test]
    fn lk_test() {
        for i in 0..200 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let template = gen_seq::generate_seq(&mut rng, 300);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let radius = 50;
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let lk = hmm.likelihood_antidiagonal_bootstrap(&template, &seq, radius);
            let lk_f = hmm.likelihood(&template, &seq);
            assert!((lk - lk_f).abs() < 0.001, "{},{}", lk, lk_f);
        }
    }
    fn post_lk(dptable: &DPTable<(f64, f64, f64)>, scalings: &[f64]) -> f64 {
        let mat = dptable[(0, 0)].0;
        let scale: f64 = scalings.iter().map(|s| s.ln()).sum();
        //  eprintln!("{mat},{scale}");
        mat.ln() + scale
    }
    #[test]
    fn lk_post_test() {
        let block_size = 32;
        for i in 0..100 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let template = gen_seq::generate_seq(&mut rng, 300);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let radius = 50;
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let ops = crate::op::bootstrap_ops(template.len(), seq.len());
            let (dptable, scale) = hmm.post(&template, &seq, &ops, radius, block_size);
            let lk = post_lk(&dptable, &scale);
            let lk_f = hmm.likelihood(&template, &seq);
            assert!((lk - lk_f).abs() < 0.001, "{},{}", lk, lk_f);
        }
    }
    #[test]
    fn lk_post() {
        for i in 0..200 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let rs = gen_seq::generate_seq(&mut rng, 300);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let radius = 30;
            let qs = gen_seq::introduce_randomness(&rs, &mut rng, &profile);
            let ops = crate::op::bootstrap_ops(rs.len(), qs.len());
            let block_size = 15;
            let (post_dp, scaling) = hmm.post(&rs, &qs, &ops, radius, block_size);
            let answer = hmm.backward(&rs, &qs);
            //  eprintln!("{scaling:?}");
            let block_num = match (rs.len() + qs.len()) % block_size == 0 {
                true => (rs.len() + qs.len()) / block_size + 1,
                false => (rs.len() + qs.len()) / block_size + 1,
            };
            assert_eq!(scaling.len(), block_num);
            for (ad, &(start, end)) in post_dp.filling_regions.iter().enumerate().rev() {
                let block = ad / block_size;
                for i in (start..end).rev() {
                    let j = ad - i;
                    if (j.max(i) - j.min(i)) > radius / 2 {
                        continue;
                    }
                    let scale: f64 = scaling.iter().skip(block).map(|x| x.ln()).sum();
                    let (mat, ins, del) = post_dp[(ad, i)];
                    // eprintln!("{i},{j},{block},{scale},{mat}");
                    let mat = mat.ln() + scale;
                    let ins = ins.ln() + scale;
                    let del = del.ln() + scale;
                    let mat_answer = answer.get(i, j, crate::hmm::State::Match);
                    let ins_answer = answer.get(i, j, crate::hmm::State::Ins);
                    let del_answer = answer.get(i, j, crate::hmm::State::Del);
                    if crate::hmm::EP < mat_answer {
                        assert!((mat - mat_answer).abs() < 0.1, "{},{}", mat, mat_answer);
                    }
                    if crate::hmm::EP < ins_answer {
                        assert!((ins - ins_answer).abs() < 0.1, "{},{}", ins, ins_answer);
                    }
                    if crate::hmm::EP < del_answer {
                        assert!((del - del_answer).abs() < 0.1, "{},{}", del, del_answer);
                    }
                }
            }
        }
    }
    #[test]
    fn lk_pre() {
        for i in 0..200 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let rs = gen_seq::generate_seq(&mut rng, 300);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let radius = 30;
            let qs = gen_seq::introduce_randomness(&rs, &mut rng, &profile);
            //let qs = rs.clone();
            let ops = crate::op::bootstrap_ops(rs.len(), qs.len());
            let block_size = 15;
            let (pre_dp, scaling) = hmm.pre(&rs, &qs, &ops, radius, block_size);
            let answer = hmm.forward(&rs, &qs);
            let block_num = (qs.len() + rs.len()) / block_size + 1;
            assert_eq!(block_num, scaling.len());
            for (ad, &(start, end)) in pre_dp.filling_regions.iter().enumerate() {
                let block = ad / block_size + 1;
                for i in start..end {
                    let j = ad - i;
                    if (j.max(i) - j.min(i)) > radius / 2 {
                        continue;
                    }
                    let scale: f64 = scaling.iter().take(block).map(|x| x.ln()).sum();
                    let (mat, ins, del) = pre_dp[(ad, i)];
                    let mat = mat.ln() + scale;
                    let ins = ins.ln() + scale;
                    let del = del.ln() + scale;
                    let mat_answer = answer.get(i, j, crate::hmm::State::Match);
                    let ins_answer = answer.get(i, j, crate::hmm::State::Ins);
                    let del_answer = answer.get(i, j, crate::hmm::State::Del);
                    if crate::hmm::EP < mat_answer {
                        assert!((mat - mat_answer).abs() < 0.1, "{},{}", mat, mat_answer);
                    }
                    if crate::hmm::EP < ins_answer {
                        assert!((ins - ins_answer).abs() < 0.1, "{},{}", ins, ins_answer);
                    }
                    if crate::hmm::EP < del_answer {
                        assert!((del - del_answer).abs() < 0.1, "{},{}", del, del_answer);
                    }
                }
            }
        }
    }
    #[test]
    fn modification_table_test() {
        let length = 100;
        for i in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let template = gen_seq::generate_seq(&mut rng, length);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let radius = 50;
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let ops = crate::op::bootstrap_ops(template.len(), seq.len());
            let modif_table = hmm.modification_table_antidiagonal(&template, &seq, radius, &ops);
            let mut mod_version = template.clone();
            // Mutation error
            let query = &seq;
            for (j, modif_table) in modif_table
                .chunks_exact(NUM_ROW)
                .take(template.len())
                .enumerate()
            {
                //  println!("{j}");
                assert_eq!(mod_version, template);
                let orig = mod_version[j];
                for (&base, lk_m) in b"ACGT".iter().zip(modif_table) {
                    mod_version[j] = base;
                    let lk = hmm.likelihood(&mod_version, query);
                    assert!((lk - lk_m).abs() < 0.001, "{},{},mod", lk, lk_m);
                    //                    println!("M\t{}\t{}", j, (lk - lk_m).abs());
                    mod_version[j] = orig;
                }
                // Insertion error
                for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                    mod_version.insert(j, base);
                    let lk = hmm.likelihood(&mod_version, query);
                    assert!((lk - lk_m).abs() < 0.001, "{},{}", lk, lk_m);
                    //                    println!("I\t{}\t{}", j, (lk - lk_m).abs());
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
                    let lk = hmm.likelihood(&mod_version, query);
                    //                    println!("C\t{}\t{}\t{}", j, len, (lk - lk_m).abs());
                    assert!((lk - lk_m).abs() < 0.001, "{},{}", lk, lk_m);
                }
                // Deletion error
                for len in (0..DEL_SIZE).filter(|d| j + d < template.len()) {
                    let lk_m = modif_table[8 + COPY_SIZE + len];
                    let mod_version: Vec<_> = template[..j]
                        .iter()
                        .chain(template[j + len + 1..].iter())
                        .copied()
                        .collect();
                    let lk = hmm.likelihood(&mod_version, query);
                    // println!("D\t{}\t{}\t{}", j, len, lk - lk_m);
                    assert!((lk - lk_m).abs() < 0.01, "{},{}", lk, lk_m,);
                }
            }
            let modif_table = modif_table
                .chunks_exact(NUM_ROW)
                .nth(template.len())
                .unwrap();
            for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                mod_version.push(base);
                let lk = hmm.likelihood(&mod_version, query);
                assert!((lk - lk_m).abs() < 0.001);
                mod_version.pop();
            }
        }
    }
    #[test]
    fn polish() {
        let radius = 30;
        let profile = gen_seq::PROFILE;
        let hmm = PairHiddenMarkovModel::default();
        let coverage = 20;
        let config = crate::hmm::HMMPolishConfig::new(radius, coverage, 0);
        for i in 0..100 {
            println!("SEED\t{i}");
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let rs = gen_seq::generate_seq(&mut rng, 300);
            let draft = profile.gen(&rs, &mut rng);
            let qss: Vec<_> = (0..coverage).map(|_| profile.gen(&rs, &mut rng)).collect();
            let mut opss: Vec<_> = qss
                .iter()
                .map(|qs| crate::op::bootstrap_ops(draft.len(), qs.len()))
                .collect();
            let polished = hmm.polish_until_converge_antidiagonal(&draft, &qss, &mut opss, &config);
            assert_eq!(rs, polished);
        }
    }
}
