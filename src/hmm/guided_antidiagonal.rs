use super::{BASE_TABLE, COPY_DEL_MAX};
use super::{COPY_SIZE, DEL_SIZE, NUM_ROW};
use crate::op::Op;
use crate::EP;
// Anti diagonal version of the HMM.
// The index is (a, i) where a is the anti diagonal (i + j = a) and i is the position of the *query*.
impl super::PairHiddenMarkovModel {
    pub fn likelihood_antidiagonal_bootstrap(&self, rs: &[u8], qs: &[u8], radius: usize) -> f64 {
        let ops = crate::op::bootstrap_ops(rs.len(), qs.len());
        self.likelihood_antidiagonal(rs, qs, &ops, radius)
    }
    pub fn likelihood_antidiagonal(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: usize) -> f64 {
        let block_size = 5;
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let fr = FillingRegions::new(&filling_regions);
        let (dptable, scaling) = self.pre(rs, qs, ops, radius, block_size);
        let (mat, ins, del) = dptable[fr.idx(rs.len() + qs.len(), qs.len())];
        (mat + ins + del).ln() + scaling.iter().map(|s| s.ln()).sum::<f64>()
        // let (dptable, scaling) = self.post(rs, qs, ops, radius, block_size);
        // dptable[fr.idx(0, 0)].0.ln() + scaling.iter().map(|scl| scl.ln()).sum::<f64>()
    }
    pub fn modification_table_antidiagonal(
        &self,
        rs: &[u8],
        qs: &[u8],
        ops: &[Op],
        radius: usize,
    ) -> Vec<f64> {
        let block_size = 5;
        assert!(COPY_SIZE < block_size);
        assert!(DEL_SIZE < block_size);
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let fr = FillingRegions::new(&filling_regions);
        let pre = {
            let mut scaling = Vec::with_capacity(filling_regions.len() / block_size + 3);
            let mut dptable = vec![(0f64, 0f64, 0f64); fr.total_cells];
            self.pre_fill(rs, qs, &fr, &mut dptable, &mut scaling, block_size);
            (dptable, scaling)
        };
        let post = {
            let mut scaling = Vec::with_capacity(filling_regions.len() / block_size + 1);
            let mut dptable = vec![(0f64, 0f64, 0f64); fr.total_cells];
            self.post_fill(rs, qs, &fr, &mut dptable, &mut scaling, block_size);
            (dptable, scaling)
        };
        let total_len = NUM_ROW * (rs.len() + 1);
        let mut mod_table = vec![0f64; total_len];
        let seqs = (rs, qs);
        self.modification_table_ad_inner(seqs, &fr, &pre, &post, &mut mod_table, block_size);
        mod_table
    }
    fn modification_table_ad_inner(
        &self,
        (rs, qs): (&[u8], &[u8]),
        fr: &FillingRegions,
        &(ref pre_dp, ref pre_scl): &FBTable,
        &(ref post_dp, ref post_scl): &FBTable,
        mod_table: &mut [f64],
        block_size: usize,
    ) {
        let mut slots = [0f64; NUM_ROW];
        let (max_scale, combined_scl) = combine_scaling_factors(pre_scl, post_scl);
        for (ad, &(offset, start, end)) in fr.offsets.iter().enumerate() {
            // Len -> additional factor.
            let post_add = {
                let mut post_add = [1f64; COPY_DEL_MAX];
                for (i, s) in post_add.iter_mut().enumerate() {
                    if (ad + i + 1) % block_size < i + 1 {
                        *s = post_scl.get(ad / block_size).unwrap_or(&1f64).recip();
                    }
                }
                post_add
            };
            let normalize_factors = {
                let mut slots = [0f64; COPY_SIZE + 1];
                for (c, s) in slots.iter_mut().enumerate() {
                    *s = *combined_scl.get((ad + c) / block_size).unwrap_or(&1f64);
                }
                slots
            };
            for q_idx in start..end {
                // slots.iter_mut().for_each(|x| *x = 0f64);
                let dp_idx = offset + q_idx - start;
                let r_idx = ad - q_idx;
                // Caching values.
                let (pre_mat, pre_ins, pre_del) = pre_dp[dp_idx];
                let (post_mat, _, post_del) = post_dp[dp_idx];
                let (ins_consume_mat, q) = match q_idx < qs.len() {
                    true => (post_dp[fr.idx(ad + 1, q_idx + 1)].0, qs[q_idx]),
                    false => (0f64, 0),
                };
                let mut_consume_mat = match q_idx < qs.len() && r_idx < rs.len() {
                    true => post_dp[fr.idx(ad + 2, q_idx + 1)].0,
                    false => 0f64,
                };
                let mut_consum_del = match r_idx < rs.len() {
                    true => post_dp[fr.idx(ad + 1, q_idx)].2,
                    false => 0f64,
                };
                let skip_idx = {
                    let mut skip_idx = [0; COPY_DEL_MAX];
                    for (len, s) in skip_idx.iter_mut().enumerate() {
                        if ad + len + 1 < fr.offsets.len() {
                            *s = fr.idx(ad + len + 1, q_idx);
                        }
                    }
                    skip_idx
                };
                let to_mat = self.to_mat((pre_mat, pre_ins, pre_del));
                let to_del = self.to_del((pre_mat, pre_ins, pre_del));
                // Mutate the `rs[r_idx]` into ACGT
                b"ACGT".iter().zip(slots.iter_mut()).for_each(|(&b, s)| {
                    let mat = to_mat * self.obs(b, q) * mut_consume_mat;
                    let del = to_del * self.del(b) * mut_consum_del;
                    *s = mat * normalize_factors[0] * post_add[1]
                        + del * normalize_factors[0] * post_add[0];
                });
                // Insert a base *before* the `r_idx`.
                b"ACGT"
                    .iter()
                    .zip(slots.iter_mut().skip(4))
                    .for_each(|(&b, s)| {
                        let mat = to_mat * self.obs(b, q) * ins_consume_mat;
                        let additional_factor = post_add[0];
                        *s = mat * normalize_factors[0] * additional_factor
                            + to_del * self.del(b) * post_del * normalize_factors[0];
                    });
                // Copy the `r_idx`..`r_idx`+c bases of `rs`
                let copy_size = COPY_SIZE.min(rs.len().saturating_sub(r_idx));
                (1..copy_size + 1)
                    .zip(slots.iter_mut().skip(8))
                    .zip(post_add.iter())
                    .zip(skip_idx.iter())
                    .for_each(|(((len, s), additional_factor), &idx)| {
                        let (pre_mat, _pre_ins, pre_del) = pre_dp[idx];
                        let lk = pre_mat * post_mat + pre_del * post_del;
                        let normalize_factor = normalize_factors[len];
                        *s = lk * normalize_factor / additional_factor;
                    });
                // Deleting the `r_idx`..`r_idx + d` bases.
                let del_size = DEL_SIZE.min(rs.len().saturating_sub(r_idx));
                slots
                    .iter_mut()
                    .skip(8 + COPY_SIZE)
                    .zip(post_add.iter())
                    .zip(skip_idx.iter())
                    .take(del_size)
                    .for_each(|((s, additional_factor), &idx)| {
                        let (post_mat, _post_ins, post_del) = post_dp[idx];
                        let lk = pre_mat * post_mat + pre_del * post_del;
                        *s = lk * normalize_factors[0] * additional_factor;
                    });
                let slot_start = r_idx * NUM_ROW;
                slots
                    .iter_mut()
                    .zip(mod_table.iter_mut().skip(slot_start))
                    .for_each(|(s, m)| {
                        *m += *s;
                        *s = 0f64;
                    });
            }
        }
        // Scaling.
        mod_table.iter_mut().for_each(|x| {
            *x = x.ln() + max_scale;
        });
    }
    #[allow(dead_code)]
    fn pre(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: usize, block_size: usize) -> FBTable {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let mut scaling = Vec::with_capacity(filling_regions.len() / block_size + 1);
        let fr = FillingRegions::new(&filling_regions);
        let mut dptable = vec![(0f64, 0f64, 0f64); fr.total_cells];
        self.pre_fill(rs, qs, &fr, &mut dptable, &mut scaling, block_size);
        (dptable, scaling)
    }
    fn pre_fill(
        &self,
        rs: &[u8],
        qs: &[u8],
        fr: &FillingRegions,
        dptable: &mut [(f64, f64, f64)],
        scaling: &mut Vec<f64>,
        block_size: usize,
    ) {
        dptable[fr.idx(0, 0)] = (1f64, 0f64, 0f64);
        let mut sum = 0f64;
        for (ad, &(ofs, start, end)) in fr.offsets.iter().enumerate().skip(1) {
            if start == 0 {
                let (q_idx, r_idx) = (0, ad);
                let del_lk = self.to_del(dptable[fr.idx(ad - 1, q_idx)]) * self.del(rs[r_idx - 1]);
                dptable[ofs + q_idx - start] = (0f64, 0f64, del_lk);
                sum += del_lk;
            }
            if ad == end - 1 {
                let q_idx = end - 1;
                let q = qs[q_idx - 1];
                let prev = (1 < q_idx).then(|| qs[q_idx - 2]);
                let ins_lk = self.to_ins(dptable[fr.idx(ad - 1, q_idx - 1)]) * self.ins(q, prev);
                dptable[ofs + q_idx - start] = (0f64, ins_lk, 0f64);
                sum += ins_lk;
            }
            let r_start = start.max(1);
            let r_end = end.min(ad);
            let mut prev = None;
            for (q_idx, &q) in qs.iter().enumerate().take(r_end - 1).skip(r_start - 1) {
                let q_idx = q_idx + 1;
                let r = rs[ad - q_idx - 1];
                let mat_lk = self.to_mat(dptable[fr.idx(ad - 2, q_idx - 1)]) * self.obs(r, q);
                let ins_lk = self.to_ins(dptable[fr.idx(ad - 1, q_idx - 1)]) * self.ins(q, prev);
                let del_lk = self.to_del(dptable[fr.idx(ad - 1, q_idx)]) * self.del(r);
                dptable[ofs + q_idx - start] = (mat_lk, ins_lk, del_lk);
                sum += mat_lk + ins_lk + del_lk;
                prev = Some(q);
            }
            if (ad + 1) % block_size == 0 || ad == rs.len() + qs.len() {
                if sum < 1f64 {
                    div_range(ad, block_size, sum, fr, dptable);
                    scaling.push(sum);
                } else {
                    scaling.push(1f64);
                }
                sum = 0f64;
            }
        }
    }
    #[allow(dead_code)]
    fn post(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: usize, block_size: usize) -> FBTable {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let mut scaling = Vec::with_capacity(filling_regions.len() / block_size + 1);
        let fr = FillingRegions::new(&filling_regions);
        let mut dptable = vec![(0f64, 0f64, 0f64); fr.total_cells];
        self.post_fill(rs, qs, &fr, &mut dptable, &mut scaling, block_size);
        (dptable, scaling)
    }
    fn post_fill(
        &self,
        rs: &[u8],
        qs: &[u8],
        filling_regions: &FillingRegions,
        dptable: &mut [(f64, f64, f64)],
        scaling: &mut Vec<f64>,
        block_size: usize,
    ) {
        dptable[filling_regions.idx(qs.len() + rs.len(), qs.len())] = (1f64, 1f64, 1f64);
        if (rs.len() + qs.len()) % block_size == 0 {
            scaling.push(1f64);
        }
        // TODO:We can fasten the array access by using *previous offsets*.
        let mut sum = 3f64;
        let fr = filling_regions;
        for (ad, &(ofs, start, end)) in filling_regions.offsets.iter().enumerate().rev().skip(1) {
            for q_idx in (start..end).rev() {
                let r_idx = ad - q_idx;
                if r_idx == rs.len() {
                    let q = qs[q_idx];
                    let prev = (0 < q_idx).then(|| qs[q_idx - 1]);
                    let ins = self.ins(q, prev) * dptable[fr.idx(ad + 1, q_idx + 1)].1;
                    let elm = (self.mat_ins * ins, self.ins_ins * ins, self.del_ins * ins);
                    dptable[ofs + q_idx - start] = elm;
                    sum += elm.0 + elm.1 + elm.2;
                } else if q_idx == qs.len() {
                    let r = rs[r_idx];
                    let del = self.del(r) * dptable[fr.idx(ad + 1, q_idx)].2;
                    let elm = (self.mat_del * del, self.ins_del * del, self.del_del * del);
                    dptable[ofs + q_idx - start] = elm;
                    sum += elm.0 + elm.1 + elm.2;
                } else {
                    let r = rs[r_idx];
                    let q = qs[q_idx];
                    let prev = (0 < q_idx).then(|| qs[q_idx - 1]);
                    let af_mat = self.obs(r, q) * dptable[fr.idx(ad + 2, q_idx + 1)].0;
                    let af_ins = self.ins(q, prev) * dptable[fr.idx(ad + 1, q_idx + 1)].1;
                    let af_del = self.del(r) * dptable[fr.idx(ad + 1, q_idx)].2;
                    let mat = self.mat_mat * af_mat + self.mat_ins * af_ins + self.mat_del * af_del;
                    let ins = self.ins_mat * af_mat + self.ins_ins * af_ins + self.ins_del * af_del;
                    let del = self.del_mat * af_mat + self.del_ins * af_ins + self.del_del * af_del;
                    dptable[ofs + q_idx - start] = (mat, ins, del);
                    sum += mat + ins + del;
                }
            }
            if ad % block_size == 0 {
                if sum < 1f64 {
                    div_range(ad, block_size, sum, fr, dptable);
                    scaling.push(sum);
                } else {
                    scaling.push(1f64);
                }
                sum = 0f64;
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
        let fr = FillingRegions::new(&filling_regions);
        let mut dptable = vec![(EP, EP, EP); fr.total_cells];
        self.align_antidiagonal_filling(rs, qs, &fr, &mut dptable)
    }
    fn align_antidiagonal_filling(
        &self,
        rs: &[u8],
        qs: &[u8],
        fr: &FillingRegions,
        dptable: &mut [(f64, f64, f64)],
    ) -> (f64, Vec<Op>) {
        let last_ad = fr.offsets.len() - 1;
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        dptable[fr.idx(0, 0)] = (0f64, EP, EP);
        for ad in 1..rs.len() + qs.len() + 1 {
            let (_offset, start, end) = fr.offsets[ad];
            let mut prev = 16;
            if start == 0 {
                let (q_idx, r_idx) = (0, ad);
                let del_lk = log_del_open + (r_idx - 1) as f64 * log_del_ext;
                dptable[fr.idx(ad, q_idx)] = (EP, EP, del_lk);
            }
            if ad == end - 1 {
                let q_idx = end - 1;
                let q = BASE_TABLE[qs[q_idx - 1] as usize];
                let ins_obs = log_ins_emit[prev | q];
                prev = q << 2;
                let (mat, ins, del) = dptable[fr.idx(ad - 1, q_idx - 1)];
                let ins_lk = (mat + log_ins_open)
                    .max(ins + log_ins_ext)
                    .max(del + log_ins_from_del)
                    + ins_obs;
                dptable[fr.idx(ad, q_idx)] = (EP, ins_lk, EP);
            }
            let (start, end) = (start.max(1), end.min(ad));
            for q_idx in start..end {
                let r_idx = ad - q_idx;
                let q = BASE_TABLE[qs[q_idx - 1] as usize];
                let r = BASE_TABLE[rs[r_idx - 1] as usize];
                assert!(1 < ad);
                let mat_lk = {
                    let (mat, ins, del) = dptable[fr.idx(ad - 2, q_idx - 1)];
                    (mat + log_mat_ext)
                        .max(del + log_mat_from_del)
                        .max(ins + log_mat_from_ins)
                        + log_mat_emit[(r << 2) | q]
                };
                let ins_lk = {
                    let ins_lk = log_ins_emit[prev | q];
                    let (mat, ins, del) = dptable[fr.idx(ad - 1, q_idx - 1)];
                    (mat + log_ins_open)
                        .max(ins + log_ins_ext)
                        .max(del + log_ins_from_del)
                        + ins_lk
                };
                let del_lk = {
                    let (mat, ins, del) = dptable[fr.idx(ad - 1, q_idx)];
                    (mat + log_del_open)
                        .max(ins + log_del_from_ins)
                        .max(del + log_del_ext)
                };
                prev = q << 2;
                dptable[fr.idx(ad, q_idx)] = (mat_lk, ins_lk, del_lk);
            }
        }
        // Traceback.
        let (lk, mut state) = {
            let (mat, ins, del) = dptable[fr.idx(last_ad, qs.len())];
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
                    let (mat, ins, del) = dptable[fr.idx(ad - 2, q_idx - 1)];
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
                    let (mat, ins, del) = dptable[fr.idx(ad - 1, q_idx - 1)];
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
                    let (mat, ins, del) = dptable[fr.idx(ad - 1, q_idx)];
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
        // ToDo: Use memory to speed up? (Compare real experiments!)
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
            modif_table
                .iter_mut()
                .take(ignore_edge)
                .for_each(|x| *x = lk - 100f64);
            let len = modif_table.len();
            modif_table
                .iter_mut()
                .skip(len.saturating_sub(ignore_edge))
                .for_each(|x| *x = lk - 100f64);
            let changed_pos =
                super::polish_by_modification_table(&mut rs, &modif_table, lk, inactive);
            let changed_pos: Vec<_> = changed_pos
                .iter()
                .map(|&(pos, op)| (pos, super::usize_to_edit_op(op)))
                .collect();
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
                let filling = radius_to_filling(radius_per, rs.len(), xs.len());
                let fr = FillingRegions::new(&filling);
                let mut dptable = vec![(EP, EP, EP); fr.total_cells];
                *ops = self
                    .align_antidiagonal_filling(&rs, xs, &fr, &mut dptable)
                    .1;
            }
            if changed_pos.is_empty() {
                break;
            }
        }
        rs
    }
    fn dp(&self, rs: &[u8], qs: &[u8], radius: &[(usize, usize)]) -> (f64, Vec<f64>) {
        assert_eq!(radius.len(), rs.len() + qs.len() + 1);
        let block_size = 12;
        let filling_regions = radius_to_filling(radius, rs.len(), qs.len());
        let fr = FillingRegions::new(&filling_regions);
        let post = {
            let mut post_scl = Vec::with_capacity(fr.total_cells / block_size + 1);
            let mut post_dp = vec![(0f64, 0f64, 0f64); fr.total_cells];
            self.post_fill(rs, qs, &fr, &mut post_dp, &mut post_scl, block_size);
            (post_dp, post_scl)
        };
        let (pre, lk) = {
            let mut pre_scl = Vec::with_capacity(fr.total_cells / block_size + 1);
            let mut pre_dp = vec![(0f64, 0f64, 0f64); fr.total_cells];
            self.pre_fill(rs, qs, &fr, &mut pre_dp, &mut pre_scl, block_size);
            let last_ad = rs.len() + qs.len();
            let (mat, ins, del) = pre_dp[fr.idx(last_ad, qs.len())];
            let scale: f64 = pre_scl.iter().map(|scl| scl.ln()).sum();
            let lk = (mat + ins + del).ln() + scale;
            ((pre_dp, pre_scl), lk)
        };
        let total_len = NUM_ROW * (rs.len() + 1);
        let mut mod_table = vec![0f64; total_len];
        self.modification_table_ad_inner((rs, qs), &fr, &pre, &post, &mut mod_table, block_size);
        (lk, mod_table)
    }
}

fn div_range(
    ad: usize,
    block_size: usize,
    sum: f64,
    fr: &FillingRegions,
    dptable: &mut [(f64, f64, f64)],
) {
    let bucket = ad / block_size;
    let b_start = bucket * block_size;
    let b_end = ((bucket + 1) * block_size).min(fr.offsets.len());
    let (start_offset, _, _) = fr.offsets[b_start];
    let (end_offset, start, end) = fr.offsets[b_end - 1];
    let end_offset = end_offset + end - start;
    dptable
        .iter_mut()
        .take(end_offset)
        .skip(start_offset)
        .for_each(|(m, i, d)| {
            *m /= sum;
            *i /= sum;
            *d /= sum;
        });
}

// Let S[T] = \prod_{t=0}^{t=T} pre_scl[t] \prod_{t=T}^{t=L} post_scl[t].
// Then, this returns max_scale = M = max_T S[T] and S[]/M.
fn combine_scaling_factors(pre_scl: &[f64], post_scl: &[f64]) -> (f64, Vec<f64>) {
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
}

const INACTIVE_TIME: usize = 4;
// const MIN_RADIUS: usize = 4;
// const RAD_SCALE: usize = 2;
use crate::op::Edit;
fn update_radius(
    ops: &[Op],
    changed_pos: &[(usize, Edit)],
    radius: &mut Vec<(usize, usize)>,
    _default: usize,
    qlen: usize,
    rlen: usize,
) {
    // TODO: Shrink the radius.
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
    assert!(start < end + 1, "{},{}", start, end);
    (start, end + 1)
}

fn center_line(ops: &[Op]) -> Vec<usize> {
    let len: usize = ops
        .iter()
        .map(|&op| match op {
            Op::Mismatch | Op::Match => 2,
            Op::Ins => 1,
            Op::Del => 1,
        })
        .sum();
    let mut q_positions = Vec::with_capacity(len);
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
    radius
}

#[derive(Debug, Clone)]
struct FillingRegions {
    offsets: Vec<(usize, usize, usize)>,
    total_cells: usize,
}

impl FillingRegions {
    fn new(filling_regions: &[(usize, usize)]) -> Self {
        let mut offsets = Vec::with_capacity(filling_regions.len());
        let mut total_cells = 0;
        for &(s, e) in filling_regions {
            offsets.push((total_cells, s, e));
            total_cells += e - s;
        }
        Self {
            offsets,
            total_cells,
        }
    }
    fn idx(&self, ad: usize, qpos: usize) -> usize {
        let (offset, start, _end) = self.offsets[ad];
        offset + qpos - start
    }
}

type FBTable = (Vec<(f64, f64, f64)>, Vec<f64>);

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
    fn post_lk(dptable: &[(f64, f64, f64)], scalings: &[f64]) -> f64 {
        let mat = dptable[0].0;
        let scale: f64 = scalings.iter().map(|s| s.ln()).sum();
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

            let filling_regions = filling_region(&ops, radius, rs.len(), qs.len());
            let mut scaling = Vec::with_capacity(filling_regions.len() / block_size + 1);
            let fr = FillingRegions::new(&filling_regions);
            let mut post_dp = vec![(0f64, 0f64, 0f64); fr.total_cells];
            hmm.post_fill(&rs, &qs, &fr, &mut post_dp, &mut scaling, block_size);

            let answer = hmm.backward(&rs, &qs);
            let block_num = match (rs.len() + qs.len()) % block_size == 0 {
                true => (rs.len() + qs.len()) / block_size + 1,
                false => (rs.len() + qs.len()) / block_size + 1,
            };
            assert_eq!(scaling.len(), block_num);
            for (ad, &(_ofs, start, end)) in fr.offsets.iter().enumerate().rev() {
                let block = ad / block_size;
                for i in (start..end).rev() {
                    let j = ad - i;
                    if (j.max(i) - j.min(i)) > radius / 2 {
                        continue;
                    }
                    let scale: f64 = scaling.iter().skip(block).map(|x| x.ln()).sum();
                    let (mat, ins, del) = post_dp[fr.idx(ad, i)];
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
            let ops = crate::op::bootstrap_ops(rs.len(), qs.len());
            let block_size = 15;
            let filling_regions = filling_region(&ops, radius, rs.len(), qs.len());
            let fr = FillingRegions::new(&filling_regions);
            let mut scaling = Vec::with_capacity(filling_regions.len() / block_size + 3);
            let mut pre_dp = vec![(0f64, 0f64, 0f64); fr.total_cells];
            hmm.pre_fill(&rs, &qs, &fr, &mut pre_dp, &mut scaling, block_size);
            let answer = hmm.forward(&rs, &qs);
            let block_num = (qs.len() + rs.len()) / block_size + 1;
            assert_eq!(block_num, scaling.len());
            for (ad, &(_ofs, start, end)) in fr.offsets.iter().enumerate().rev() {
                let block = ad / block_size + 1;
                for i in start..end {
                    let j = ad - i;
                    if (j.max(i) - j.min(i)) > radius / 2 {
                        continue;
                    }
                    let scale: f64 = scaling.iter().take(block).map(|x| x.ln()).sum();
                    let (mat, ins, del) = pre_dp[fr.idx(ad, i)];
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
            let modif_table = hmm.modification_table_antidiagonal(&template, &seq, &ops, radius);
            let mut mod_version = template.clone();
            // Mutation error
            let query = &seq;
            for (j, modif_table) in modif_table
                .chunks_exact(NUM_ROW)
                .take(template.len())
                .enumerate()
            {
                assert_eq!(mod_version, template);
                let orig = mod_version[j];
                for (&base, lk_m) in b"ACGT".iter().zip(modif_table) {
                    mod_version[j] = base;
                    let lk = hmm.likelihood(&mod_version, query);
                    assert!((lk - lk_m).abs() < 0.001, "{},{},mod", lk, lk_m);
                    mod_version[j] = orig;
                }
                // Insertion error
                for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                    mod_version.insert(j, base);
                    let lk = hmm.likelihood(&mod_version, query);
                    assert!((lk - lk_m).abs() < 0.001, "{},{}", lk, lk_m);
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
