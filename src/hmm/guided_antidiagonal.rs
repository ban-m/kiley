use std::fmt::Debug;

// TODO: Fallback method when facing with large deletion in a very low error model.
use super::PairHiddenMarkovModel;
use super::PairHiddenMarkovModelOnStrands;
use super::TrainingDataPack;
use super::{BASE_TABLE, COPY_DEL_MAX, COPY_SIZE, DEL_SIZE, NUM_ROW};
const BLOCK_SIZE: usize = COPY_DEL_MAX + 1;
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
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let fr = FillingRegions::new(&filling_regions);
        let (dptable, scaling) = self.pre(rs, qs, ops, radius);
        let (mat, ins, del) = dptable[fr.idx(rs.len() + qs.len(), qs.len()).unwrap()];
        (mat + ins + del).ln() + scaling.iter().map(|s| s.ln()).sum::<f64>()
    }
    pub fn modification_table_antidiagonal(
        &self,
        rs: &[u8],
        qs: &[u8],
        ops: &[Op],
        radius: usize,
    ) -> (Vec<f64>, f64) {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let fr = FillingRegions::new(&filling_regions);
        let (pre, lk) = {
            let mut scaling = Vec::with_capacity(filling_regions.len() / BLOCK_SIZE + 3);
            let mut dptable = vec![(0f64, 0f64, 0f64); fr.total_cells];
            self.pre_fill(rs, qs, &fr, &mut dptable, &mut scaling);
            let (mat, ins, del) = dptable[fr.idx(rs.len() + qs.len(), qs.len()).unwrap()];
            let lk = (mat + ins + del).ln() + scaling.iter().map(|s| s.ln()).sum::<f64>();
            ((dptable, scaling), lk)
        };
        let pre = (pre.0.as_slice(), pre.1.as_slice());
        let post = {
            let mut scaling = Vec::with_capacity(filling_regions.len() / BLOCK_SIZE + 1);
            let mut dptable = vec![(0f64, 0f64, 0f64); fr.total_cells];
            self.post_fill(rs, qs, &fr, &mut dptable, &mut scaling);
            (dptable, scaling)
        };
        let post = (post.0.as_slice(), post.1.as_slice());
        let total_len = NUM_ROW * (rs.len() + 1);
        let mut mod_table = vec![0f64; total_len];
        let seqs = (rs, qs);
        self.modification_table_ad_inner(seqs, &fr, pre, post, &mut mod_table);
        (mod_table, lk)
    }
    fn modification_table_ad_inner(
        &self,
        (rs, qs): (&[u8], &[u8]),
        fr: &FillingRegions,
        (pre_dp, pre_scl): FBSlice,
        (post_dp, post_scl): FBSlice,
        mod_table: &mut [f64],
    ) {
        let rslen = rs.len();
        let (max_scale, combined_scl) = combine_scaling_factors(pre_scl, post_scl);
        for (ad, &(offset, start, end)) in fr.offsets.iter().enumerate() {
            let post_add = Self::post_add(ad, post_scl);
            let normalize_factors = Self::normalize_factors(ad, &combined_scl);
            for q_idx in start..end {
                let idx = (ad, q_idx, offset - start);
                let seq = (qs, rslen);
                let dp = (pre_dp, post_dp);
                let factors = (&normalize_factors, &post_add);
                self.fill_one(idx, seq, dp, fr, mod_table, factors);
            }
        }
        // Scaling.
        Self::scaling(mod_table, max_scale);
    }
    fn scaling(mod_table: &mut [f64], max_scale: f64) {
        mod_table.iter_mut().for_each(|x| {
            *x = x.max(std::f64::MIN_POSITIVE).ln() + max_scale;
        });
    }
    fn normalize_factors(ad: usize, combined_scl: &[f64]) -> [f64; COPY_SIZE + 1] {
        let mut slots = [0f64; COPY_SIZE + 1];
        for (c, s) in slots.iter_mut().enumerate() {
            *s = *combined_scl.get((ad + c) / BLOCK_SIZE).unwrap_or(&1f64);
        }
        slots
    }
    fn post_add(ad: usize, post_scl: &[f64]) -> [f64; COPY_DEL_MAX] {
        let mut post_add = [1f64; COPY_DEL_MAX];
        for (i, s) in post_add.iter_mut().enumerate() {
            if (ad + i + 1) % BLOCK_SIZE < i + 1 {
                *s = post_scl.get(ad / BLOCK_SIZE).unwrap_or(&1f64).recip();
            }
        }
        post_add
    }
    #[allow(clippy::type_complexity)]
    fn fill_one(
        &self,
        (ad, q_idx, offset): (usize, usize, usize),
        (qs, rslen): (&[u8], usize),
        (pre_dp, post_dp): (&[(f64, f64, f64)], &[(f64, f64, f64)]),
        fr: &FillingRegions,
        mod_table: &mut [f64],
        (normalize_factors, post_add): (&[f64; COPY_SIZE + 1], &[f64; COPY_DEL_MAX]),
    ) {
        let dp_idx = offset + q_idx;
        let r_idx = ad - q_idx;
        let slot_start = (ad - q_idx) * NUM_ROW;
        let slots = &mut mod_table[slot_start..slot_start + NUM_ROW];
        let (pre_mat, pre_ins, pre_del) = pre_dp[dp_idx];
        let (post_mat, _, post_del) = post_dp[dp_idx];
        let fb = (0f64, 0f64, 0f64);
        let (ins_consume_mat, q) = match q_idx < qs.len() {
            true => (fr.get(ad + 1, q_idx + 1, post_dp, fb).0, qs[q_idx]),
            false => (0f64, 0),
        };
        let mut_consume_del = match r_idx < rslen {
            true => fr.get(ad + 1, q_idx, post_dp, fb).2,
            false => 0f64,
        };
        let mut_consume_mat = match q_idx < qs.len() && r_idx < rslen {
            true => fr.get(ad + 2, q_idx + 1, post_dp, fb).0,
            false => 0f64,
        };
        let skip_idx = {
            let mut skip_idx = [0; COPY_DEL_MAX];
            for (len, s) in skip_idx.iter_mut().enumerate() {
                if let Some(idx) = fr.idx(ad + len + 1, q_idx) {
                    *s = idx;
                }
            }
            skip_idx
        };
        let to_mat = self.to_mat((pre_mat, pre_ins, pre_del));
        let to_del = self.to_del((pre_mat, pre_ins, pre_del));
        // Mutate the `rs[r_idx]` into ACGT
        b"ACGT".iter().zip(slots.iter_mut()).for_each(|(&b, s)| {
            let mat = to_mat * self.obs(b, q) * mut_consume_mat;
            let del = to_del * self.del(b) * mut_consume_del;
            *s +=
                mat * normalize_factors[0] * post_add[1] + del * normalize_factors[0] * post_add[0];
        });
        // Insert a base *before* the `r_idx`.
        b"ACGT"
            .iter()
            .zip(slots.iter_mut().skip(4))
            .for_each(|(&b, s)| {
                let mat = to_mat * self.obs(b, q) * ins_consume_mat;
                let additional_factor = post_add[0];
                *s += mat * normalize_factors[0] * additional_factor
                    + to_del * self.del(b) * post_del * normalize_factors[0];
            });
        // Copy the `r_idx`..`r_idx`+c bases of `rs`
        let copy_size = COPY_SIZE.min(rslen.saturating_sub(r_idx));
        (1..copy_size + 1)
            .zip(slots.iter_mut().skip(8))
            .zip(post_add.iter())
            .zip(skip_idx.iter())
            .filter(|x| *x.1 != 0)
            .for_each(|(((len, s), additional_factor), &idx)| {
                let (pre_mat, _pre_ins, pre_del) = pre_dp[idx];
                let lk = pre_mat * post_mat + pre_del * post_del;
                let normalize_factor = normalize_factors[len];
                *s += lk * normalize_factor / additional_factor;
            });
        // Deleting the `r_idx`..`r_idx + d` bases.
        let del_size = DEL_SIZE.min(rslen.saturating_sub(r_idx));
        slots
            .iter_mut()
            .skip(8 + COPY_SIZE)
            .zip(post_add.iter())
            .zip(skip_idx.iter())
            .take(del_size)
            .filter(|x| *x.1 != 0)
            .for_each(|((s, additional_factor), &idx)| {
                let (post_mat, _post_ins, post_del) = post_dp[idx];
                let lk = pre_mat * post_mat + pre_del * post_del;
                *s += lk * normalize_factors[0] * additional_factor;
            });
    }
    #[allow(dead_code)]
    fn pre(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: usize) -> FBTable {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let mut scaling = Vec::with_capacity(filling_regions.len() / BLOCK_SIZE + 1);
        let fr = FillingRegions::new(&filling_regions);
        let mut dptable = vec![(0f64, 0f64, 0f64); fr.total_cells];
        self.pre_fill(rs, qs, &fr, &mut dptable, &mut scaling);
        (dptable, scaling)
    }
    fn pre_fill(
        &self,
        rs: &[u8],
        qs: &[u8],
        fr: &FillingRegions,
        dptable: &mut [(f64, f64, f64)],
        scaling: &mut Vec<f64>,
    ) {
        fr.set(0, 0, dptable, (1f64, 0f64, 0f64));
        let fb = (0f64, 0f64, 0f64);
        let mut sum = 0f64;
        for (ad, &(ofs, start, end)) in fr.offsets.iter().enumerate().skip(1) {
            if start == 0 {
                let (q_idx, r_idx) = (0, ad);
                let del_lk =
                    self.to_del(fr.get(ad - 1, q_idx, dptable, fb)) * self.del(rs[r_idx - 1]);
                dptable[ofs + q_idx - start] = (0f64, 0f64, del_lk);
                sum += del_lk;
            }
            if ad == end - 1 {
                let q_idx = end - 1;
                let q = qs[q_idx - 1];
                let prev = (1 < q_idx).then(|| qs[q_idx - 2]);
                let ins_lk =
                    self.to_ins(fr.get(ad - 1, q_idx - 1, dptable, fb)) * self.ins(q, prev);
                dptable[ofs + q_idx - start] = (0f64, ins_lk, 0f64);
                sum += ins_lk;
            }
            let r_start = start.max(1);
            let r_end = end.min(ad);
            let mut prev = None;
            for (q_idx, &q) in qs.iter().enumerate().take(r_end - 1).skip(r_start - 1) {
                let q_idx = q_idx + 1;
                let r = rs[ad - q_idx - 1];
                let mat = self.to_mat(fr.get(ad - 2, q_idx - 1, dptable, fb)) * self.obs(r, q);
                let ins = self.to_ins(fr.get(ad - 1, q_idx - 1, dptable, fb)) * self.ins(q, prev);
                let del = self.to_del(fr.get(ad - 1, q_idx, dptable, fb)) * self.del(r);
                dptable[ofs + q_idx - start] = (mat, ins, del);
                sum += mat + ins + del;
                prev = Some(q);
            }
            if (ad + 1) % BLOCK_SIZE == 0 || ad == rs.len() + qs.len() {
                if sum < 1f64 {
                    div_range(ad, sum, fr, dptable);
                    scaling.push(sum);
                } else {
                    scaling.push(1f64);
                }
                sum = 0f64;
            }
        }
    }
    #[allow(dead_code)]
    fn post(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: usize) -> FBTable {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        let mut scaling = Vec::with_capacity(filling_regions.len() / BLOCK_SIZE + 1);
        let fr = FillingRegions::new(&filling_regions);
        let mut dptable = vec![(0f64, 0f64, 0f64); fr.total_cells];
        self.post_fill(rs, qs, &fr, &mut dptable, &mut scaling);
        (dptable, scaling)
    }
    fn post_fill(
        &self,
        rs: &[u8],
        qs: &[u8],
        fr: &FillingRegions,
        dptable: &mut [(f64, f64, f64)],
        scaling: &mut Vec<f64>,
    ) {
        let fb = (0f64, 0f64, 0f64);
        fr.set(qs.len() + rs.len(), qs.len(), dptable, (1f64, 1f64, 1f64));
        if (rs.len() + qs.len()) % BLOCK_SIZE == 0 {
            scaling.push(1f64);
        }
        let mut sum = 3f64;
        let fr = fr;
        for (ad, &(ofs, start, end)) in fr.offsets.iter().enumerate().rev().skip(1) {
            for q_idx in (start..end).rev() {
                let r_idx = ad - q_idx;
                if r_idx == rs.len() {
                    let q = qs[q_idx];
                    let prev = (0 < q_idx).then(|| qs[q_idx - 1]);
                    let ins = self.ins(q, prev) * fr.get(ad + 1, q_idx + 1, dptable, fb).1;
                    let elm = (self.mat_ins * ins, self.ins_ins * ins, self.del_ins * ins);
                    dptable[ofs + q_idx - start] = elm;
                    sum += elm.0 + elm.1 + elm.2;
                } else if q_idx == qs.len() {
                    let r = rs[r_idx];
                    let del = self.del(r) * fr.get(ad + 1, q_idx, dptable, fb).2;
                    let elm = (self.mat_del * del, self.ins_del * del, self.del_del * del);
                    dptable[ofs + q_idx - start] = elm;
                    sum += elm.0 + elm.1 + elm.2;
                } else {
                    let r = rs[r_idx];
                    let q = qs[q_idx];
                    let prev = (0 < q_idx).then(|| qs[q_idx - 1]);
                    let af_mat = self.obs(r, q) * fr.get(ad + 2, q_idx + 1, dptable, fb).0;
                    let af_ins = self.ins(q, prev) * fr.get(ad + 1, q_idx + 1, dptable, fb).1;
                    let af_del = self.del(r) * fr.get(ad + 1, q_idx, dptable, fb).2;
                    let mat = self.mat_mat * af_mat + self.mat_ins * af_ins + self.mat_del * af_del;
                    let ins = self.ins_mat * af_mat + self.ins_ins * af_ins + self.ins_del * af_del;
                    let del = self.del_mat * af_mat + self.del_ins * af_ins + self.del_del * af_del;
                    dptable[ofs + q_idx - start] = (mat, ins, del);
                    sum += mat + ins + del;
                }
            }
            if ad % BLOCK_SIZE == 0 {
                if sum < 1f64 {
                    div_range(ad, sum, fr, dptable);
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
        let log_mat_emit: Vec<_> = self.mat_emit.iter().map(Self::log).collect();
        let log_ins_emit: Vec<_> = self.ins_emit.iter().map(Self::log).collect();
        let (log_del_open, log_ins_open) = (self.mat_del.ln(), self.mat_ins.ln());
        let (log_del_ext, log_ins_ext) = (self.del_del.ln(), self.ins_ins.ln());
        let (log_del_from_ins, log_ins_from_del) = (self.ins_del.ln(), self.del_ins.ln());
        let (log_mat_from_del, log_mat_from_ins) = (self.del_mat.ln(), self.ins_mat.ln());
        let log_mat_ext = self.mat_mat.ln();
        let fb = (EP, EP, EP);
        fr.set(0, 0, dptable, (0f64, EP, EP));
        for ad in 1..rs.len() + qs.len() + 1 {
            let (offset, start, end) = fr.offsets[ad];
            let mut prev = 16;
            if start == 0 {
                let (_q_idx, r_idx) = (0, ad);
                let del_lk = log_del_open + (r_idx - 1) as f64 * log_del_ext;
                dptable[offset] = (EP, EP, del_lk);
            }
            if ad + 1 == end {
                let q_idx = ad;
                let q = BASE_TABLE[qs[q_idx - 1] as usize];
                let ins_obs = log_ins_emit[prev | q];
                prev = q << 2;
                let (mat, ins, del) = fr.get(ad - 1, q_idx - 1, dptable, fb);
                let ins_lk = (mat + log_ins_open)
                    .max(ins + log_ins_ext)
                    .max(del + log_ins_from_del)
                    + ins_obs;
                dptable[offset + q_idx - start] = (EP, ins_lk, EP);
            }
            for q_idx in start.max(1)..end.min(ad) {
                let r_idx = ad - q_idx;
                let q = BASE_TABLE[qs[q_idx - 1] as usize];
                let r = BASE_TABLE[rs[r_idx - 1] as usize];
                let mat_lk = {
                    let (mat, ins, del) = fr.get(ad - 2, q_idx - 1, dptable, fb);
                    (mat + log_mat_ext)
                        .max(del + log_mat_from_del)
                        .max(ins + log_mat_from_ins)
                        + log_mat_emit[(r << 2) | q]
                };
                let ins_lk = {
                    let ins_lk = log_ins_emit[prev | q];
                    let (mat, ins, del) = fr.get(ad - 1, q_idx - 1, dptable, fb);
                    (mat + log_ins_open)
                        .max(ins + log_ins_ext)
                        .max(del + log_ins_from_del)
                        + ins_lk
                };
                let del_lk = {
                    let (mat, ins, del) = fr.get(ad - 1, q_idx, dptable, fb);
                    (mat + log_del_open)
                        .max(ins + log_del_from_ins)
                        .max(del + log_del_ext)
                };
                prev = q << 2;
                dptable[offset + q_idx - start] = (mat_lk, ins_lk, del_lk);
            }
        }
        // Traceback.
        let (lk, mut state) = {
            let (mat, ins, del) = dptable[fr.idx(qs.len() + rs.len(), qs.len()).unwrap()];
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
                    ops.push(Op::Match);
                    let (mat, ins, del) = dptable[fr.idx(ad - 2, q_idx - 1).unwrap()];
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
                    let (mat, ins, del) = dptable[fr.idx(ad - 1, q_idx - 1).unwrap()];
                    let max = (mat + log_ins_open)
                        .max(ins + log_ins_ext)
                        .max(del + log_ins_from_del);
                    q_idx -= 1;
                    if max == mat + log_ins_open {
                        0
                    } else if max == ins + log_ins_ext {
                        1
                    } else {
                        assert!(max == del + log_ins_from_del, "{}", max);
                        2
                    }
                }
                2 => {
                    ops.push(Op::Del);
                    let (mat, ins, del) = dptable[fr.idx(ad - 1, q_idx).unwrap()];
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
        let qmax = xss
            .iter()
            .map(|x| x.borrow().len())
            .max()
            .expect("Reads empty.");
        let mut memory = Memory::new(draft.len(), qmax, radius);
        for t in 0..100 {
            let inactive = INACTIVE_TIME + (t * INACTIVE_TIME) % rs.len();
            let mut modif_table = vec![0f64; (rs.len() + 1) * NUM_ROW];
            let mut lk = 0f64;
            for (seq, radius) in std::iter::zip(xss, &radius_reads).take(take_num) {
                assert_eq!(radius.len(), seq.borrow().len() + rs.len() + 1);
                let m_lk = self.dp_memory(&rs, seq.borrow(), radius, &mut memory);
                let mt = memory.modif_table.as_slice();
                if m_lk.is_finite() && mt.iter().all(|x| x.is_finite()) {
                    lk += m_lk;
                    modif_table.iter_mut().zip(mt).for_each(|(x, y)| *x += y);
                } else {
                    println!("{m_lk}");
                    for (i, w) in mt
                        .chunks(NUM_ROW)
                        .enumerate()
                        .filter(|(_, w)| w.iter().any(|x| !x.is_finite()))
                    {
                        println!("{i}\t{w:?}");
                    }
                }
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
                crate::op::fix_alignment_path(ops, changed_iter, qlen, rlen);
                update_radius(ops, &changed_pos, radius_per, default_radius, qlen, rlen);
                assert_eq!(radius_per.len(), xs.len() + rs.len() + 1);
                memory.fr.update_by_radius(radius_per, rs.len(), xs.len());
                memory.initialize_aln();
                *ops = self
                    .align_antidiagonal_filling(&rs, xs, &memory.fr, &mut memory.pre_dp)
                    .1;
            }
            if changed_pos.is_empty() {
                break;
            }
        }
        rs
    }
    fn dp_memory(
        &self,
        rs: &[u8],
        qs: &[u8],
        radius: &[(usize, usize)],
        memory: &mut Memory,
    ) -> f64 {
        assert_eq!(radius.len(), rs.len() + qs.len() + 1);
        memory.fr.update_by_radius(radius, rs.len(), qs.len());
        memory.initialize(rs.len());
        let lk = {
            let (post_dp, post_scl) = (&mut memory.post_dp, &mut memory.post_scl);
            self.post_fill(rs, qs, &memory.fr, post_dp, post_scl);
            let (pre_dp, pre_scl) = (&mut memory.pre_dp, &mut memory.pre_scl);
            self.pre_fill(rs, qs, &memory.fr, pre_dp, pre_scl);
            let last_ad = rs.len() + qs.len();
            let (mat, ins, del) = pre_dp[memory.fr.idx(last_ad, qs.len()).unwrap()];
            let scale: f64 = pre_scl.iter().map(|scl| scl.ln()).sum();
            (mat + ins + del).ln() + scale
        };
        let post = (memory.post_dp.as_slice(), memory.post_scl.as_slice());
        let pre = (memory.pre_dp.as_slice(), memory.pre_scl.as_slice());
        let mod_table = &mut memory.modif_table;
        self.modification_table_ad_inner((rs, qs), &memory.fr, pre, post, mod_table);
        lk
    }
    fn register_antidiagonal(&self, memory: &Memory, rs: &[u8], qs: &[u8], next: &mut Self) {
        use crate::LogSumExp;
        // Scaling factors T => sum_{t = 0}^{t=T} ln pre_scl[t]
        let pre_scl: Vec<_> = memory
            .pre_scl
            .iter()
            .scan(0f64, |acc, scl| {
                *acc += scl.ln();
                Some(*acc)
            })
            .collect();
        let post_scl: Vec<_> = {
            let mut post_scl: Vec<_> = memory
                .post_scl
                .iter()
                .rev()
                .scan(0f64, |acc, scl| {
                    *acc += scl.ln();
                    Some(*acc)
                })
                .collect();
            post_scl.reverse();
            post_scl
        };
        let mut mat_probs: [LogSumExp; 16] = [LogSumExp::new(); 16];
        let mut ins_probs = [LogSumExp::new(); 20];
        let mut mat_to_mat = LogSumExp::new();
        let mut mat_to_ins = LogSumExp::new();
        let mut mat_to_del = LogSumExp::new();
        let mut ins_to_mat = LogSumExp::new();
        let mut ins_to_ins = LogSumExp::new();
        let mut ins_to_del = LogSumExp::new();
        let mut del_to_mat = LogSumExp::new();
        let mut del_to_ins = LogSumExp::new();
        let mut del_to_del = LogSumExp::new();
        let len = rs.len() + qs.len();
        let mp = std::f64::MIN_POSITIVE;
        for (ad, &(offset, start, end)) in memory.fr.offsets.iter().enumerate().take(len) {
            let scale = pre_scl[ad / BLOCK_SIZE] + post_scl[ad / BLOCK_SIZE];
            for q_idx in start..end.min(qs.len()) {
                let (pre_mat, pre_ins, _) = memory.pre_dp[offset + q_idx - start];
                let (post_mat, post_ins, _) = memory.post_dp[offset + q_idx - start];
                let r_idx = ad - q_idx;
                // We miss the (q_idx, r_idx) = (0,_), (_, 0)'s transition moves. But who cares?
                if q_idx == 0 || r_idx == 0 {
                    continue;
                }
                let mat_lk = (pre_mat * post_mat).max(std::f64::MIN_POSITIVE).ln() + scale;
                let ins_lk = (pre_ins * post_ins).max(std::f64::MIN_POSITIVE).ln() + scale;
                let prev = match 1 < q_idx {
                    true => BASE_TABLE[qs[q_idx - 2] as usize],
                    false => 4,
                };
                let (q, r) = (qs[q_idx - 1], rs[r_idx - 1]);
                let mat_slot = (BASE_TABLE[r as usize] << 2) | BASE_TABLE[q as usize];
                let ins_slot = (prev << 2) | BASE_TABLE[q as usize];
                mat_probs[mat_slot] += mat_lk;
                ins_probs[ins_slot] += ins_lk;

                // Transition probs
                if q_idx == qs.len() || r_idx == rs.len() {
                    continue;
                }
                let indel_scale = pre_scl[ad / BLOCK_SIZE] + post_scl[(ad + 1) / BLOCK_SIZE];
                let mat_scale = match post_scl.get((ad + 2) / BLOCK_SIZE) {
                    Some(post) => pre_scl[ad / BLOCK_SIZE] + post,
                    None => 0f64,
                };
                let ins_prob = self.ins(qs[q_idx], (0 < q_idx).then(|| qs[q_idx - 1]));
                let mat_prob = self.obs(rs[r_idx], qs[q_idx]);
                let del_prob = self.del(rs[r_idx]);
                let (from_mat, from_ins, from_del) = memory.pre_dp[offset + q_idx - start];
                let fb = (0f64, 0f64, 0f64);
                let (after_mat, _, _) = memory.fr.get(ad + 2, q_idx + 1, &memory.post_dp, fb);
                let after_ins = memory.fr.get(ad + 1, q_idx + 1, &memory.post_dp, fb).1;
                let after_del = memory.fr.get(ad + 1, q_idx, &memory.post_dp, fb).2;
                // Mat, If possible.
                if ad + 2 < rs.len() + qs.len() + 1 {
                    let prob = from_mat * self.mat_mat * mat_prob * after_mat;
                    mat_to_mat += prob.max(mp).ln() + mat_scale;
                    let prob = from_del * self.del_mat * mat_prob * after_mat;
                    del_to_mat += prob.max(mp).ln() + mat_scale;
                    let prob = from_ins * self.ins_mat * mat_prob * after_mat;
                    ins_to_mat += prob.max(mp).ln() + mat_scale;
                }
                // InDel, No condition.
                let prob = from_mat * self.mat_ins * ins_prob * after_ins;
                mat_to_ins += prob.max(mp).ln() + indel_scale;
                let prob = from_ins * self.ins_ins * ins_prob * after_ins;
                ins_to_ins += prob.max(mp).ln() + indel_scale;
                let prob = from_del * self.del_ins * ins_prob * after_ins;
                del_to_ins += prob.max(mp).ln() + indel_scale;

                let prob = from_mat * self.mat_del * del_prob * after_del;
                mat_to_del += prob.max(mp).ln() + indel_scale;
                let prob = from_ins * self.ins_del * del_prob * after_del;
                ins_to_del += prob.max(mp).ln() + indel_scale;
                let prob = from_del * self.del_del * del_prob * after_del;
                del_to_del += prob.max(mp).ln() + indel_scale;
            }
        }
        assert_eq!(next.mat_emit.len(), mat_probs.len());
        let mat_lks: Vec<f64> = mat_probs.iter().map(|&x| x.into()).collect();
        let mat_total = crate::logsumexp(&mat_lks);
        for (next, mat) in next.mat_emit.iter_mut().zip(mat_lks) {
            *next += (mat - mat_total).exp();
        }
        assert_eq!(next.ins_emit.len(), ins_probs.len());
        let ins_lks: Vec<f64> = ins_probs.iter().map(|&x| x.into()).collect();
        let ins_total = crate::logsumexp(&ins_lks);
        for (next, ins) in next.ins_emit.iter_mut().zip(ins_lks) {
            *next += (ins - ins_total).exp();
        }
        let mat_to_mat: f64 = mat_to_mat.into();
        let mat_to_ins: f64 = mat_to_ins.into();
        let mat_to_del: f64 = mat_to_del.into();
        let ins_to_mat: f64 = ins_to_mat.into();
        let ins_to_ins: f64 = ins_to_ins.into();
        let ins_to_del: f64 = ins_to_del.into();
        let del_to_mat: f64 = del_to_mat.into();
        let del_to_ins: f64 = del_to_ins.into();
        let del_to_del: f64 = del_to_del.into();
        let mat = crate::logsumexp(&[mat_to_mat, mat_to_ins, mat_to_del]);
        let ins = crate::logsumexp(&[ins_to_mat, ins_to_ins, ins_to_del]);
        let del = crate::logsumexp(&[del_to_mat, del_to_ins, del_to_del]);
        next.mat_mat = (mat_to_mat - mat).exp();
        next.mat_ins = (mat_to_ins - mat).exp();
        next.mat_del = (mat_to_del - mat).exp();
        next.ins_mat = (ins_to_mat - ins).exp();
        next.ins_ins = (ins_to_ins - ins).exp();
        next.ins_del = (ins_to_del - ins).exp();
        next.del_mat = (del_to_mat - del).exp();
        next.del_ins = (del_to_ins - del).exp();
        next.del_del = (del_to_del - del).exp();
    }
}

fn div_range(ad: usize, sum: f64, fr: &FillingRegions, dptable: &mut [(f64, f64, f64)]) {
    let bucket = ad / BLOCK_SIZE;
    let b_start = bucket * BLOCK_SIZE;
    let b_end = ((bucket + 1) * BLOCK_SIZE).min(fr.offsets.len());
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
const MIN_RADIUS: usize = 10;
const RAD_SCALE: usize = 2;
use crate::op::Edit;
// TODO: Has a bug? Too small MIN_RADIUS?
fn update_radius(
    ops: &[Op],
    changed_pos: &[(usize, Edit)],
    radius: &mut Vec<(usize, usize)>,
    default: usize,
    qlen: usize,
    rlen: usize,
) {
    let orig_len = radius.len();
    let (mut qpos, mut rpos) = (0, 0);
    let mut offset: isize = 0;
    let mut changed_pos = changed_pos.iter().peekable();
    let mut current_radius = default;
    for &op in ops {
        match changed_pos.peek() {
            Some(&&(pos, ed)) if pos == rpos => {
                match ed {
                    Edit::Subst => {}
                    Edit::Insertion => offset -= 1,
                    Edit::Copy(l) => offset -= l as isize,
                    Edit::Deletion(l) => offset += l as isize,
                }
                changed_pos.next();
                current_radius = default;
            }
            _ => {
                let diag = ((qpos + rpos) as isize + offset).max(0) as usize;
                current_radius = (current_radius - 1)
                    .max(radius[diag].1 / RAD_SCALE)
                    .max(MIN_RADIUS);
            }
        };
        // let new_r = default;
        let new_r = current_radius;
        match op {
            Op::Mismatch | Op::Match => {
                radius.push((qpos, new_r));
                rpos += 1;
                radius.push((qpos, new_r));
                qpos += 1;
            }
            Op::Ins => {
                radius.push((qpos, new_r));
                qpos += 1
            }
            Op::Del => {
                radius.push((qpos, new_r));
                rpos += 1;
            }
        }
    }
    let diag = ((qpos + rpos) as isize + offset) as usize;
    let new_r = (radius[diag].1 / RAD_SCALE).max(MIN_RADIUS);
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
    fn with_capacity(rlen: usize, qlen: usize, radius: usize) -> Self {
        let len = 3 * (rlen + qlen) * radius / 2;
        Self {
            total_cells: 0,
            offsets: Vec::with_capacity(len),
        }
    }
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
    fn get<T: Copy>(&self, ad: usize, qpos: usize, dp: &[T], fallback: T) -> T {
        match self.offsets.get(ad) {
            Some(&(offset, start, end)) if (start..end).contains(&qpos) => {
                dp[offset + qpos - start]
            }
            _ => fallback,
        }
    }
    fn set<T: Copy>(&self, ad: usize, qpos: usize, dp: &mut [T], val: T) {
        match self.offsets.get(ad) {
            Some(&(offset, start, end)) if (start..end).contains(&qpos) => {
                dp[offset + qpos - start] = val;
            }
            _ => {}
        }
    }

    fn idx(&self, ad: usize, qpos: usize) -> Option<usize> {
        match self.offsets.get(ad) {
            Some(&(offset, start, end)) if (start..end).contains(&qpos) => {
                Some(offset + qpos - start)
            }
            _ => None,
        }
    }
    fn update_by_radius(&mut self, radius: &[(usize, usize)], rlen: usize, qlen: usize) {
        assert_eq!(radius.len(), rlen + qlen + 1);
        self.total_cells = 0;
        self.offsets.clear();
        let regions = radius
            .iter()
            .enumerate()
            .map(|(ad, &(qpos, radius))| cap(ad, qpos, radius, rlen, qlen));
        for (s, e) in regions {
            self.offsets.push((self.total_cells, s, e));
            self.total_cells += e - s;
        }
    }
}

impl PairHiddenMarkovModelOnStrands {
    pub fn modification_table_antidiagonal(
        &self,
        rs: &[u8],
        qs: &[u8],
        ops: &[Op],
        radius: usize,
        is_forward: bool,
    ) -> (Vec<f64>, f64) {
        match is_forward {
            true => self
                .forward()
                .modification_table_antidiagonal(rs, qs, ops, radius),
            false => self
                .reverse()
                .modification_table_antidiagonal(rs, qs, ops, radius),
        }
    }
    pub fn align_antidiagonal(
        &self,
        rs: &[u8],
        qs: &[u8],
        ops: &[Op],
        radius: usize,
        is_forward: bool,
    ) -> (f64, Vec<Op>) {
        match is_forward {
            true => self.forward().align_antidiagonal(rs, qs, ops, radius),
            false => self.reverse().align_antidiagonal(rs, qs, ops, radius),
        }
    }
    fn baum_welch_antidiagonal<'a, T, O>(
        &self,
        datapack: &TrainingDataPack<'a, T, O>,
        radius: usize,
    ) -> Vec<(Memory, &'a [u8], bool)>
    where
        T: std::borrow::Borrow<[u8]> + Sync + Send,
        O: std::borrow::Borrow<[Op]> + Sync + Send,
    {
        use rayon::prelude::*;
        let qlen = datapack
            .sequences
            .iter()
            .map(|x| x.borrow().len())
            .max()
            .unwrap();
        let ops = datapack.operations.par_iter();
        let seqs = datapack.sequences.par_iter();
        let direction = datapack.directions.par_iter();
        let rs = datapack.consensus;
        ops.zip(seqs)
            .zip(direction)
            .filter_map(|((ops, seq), direction)| {
                let qs = seq.borrow();
                let ops = ops.borrow();
                let radius_range: Vec<_> = center_line(ops)
                    .iter()
                    .map(|&qpos| (qpos, radius))
                    .collect();
                let mut memory = Memory::new(rs.len(), qlen, radius);
                let target = match direction {
                    true => self.forward(),
                    false => self.reverse(),
                };
                let lk = target.dp_memory(rs, qs, &radius_range, &mut memory);
                let lk_2 =
                    memory.post_dp[0].0.ln() + memory.post_scl.iter().map(|x| x.ln()).sum::<f64>();
                if lk.is_finite() && lk_2.is_finite() {
                    Some((memory, qs, *direction))
                } else {
                    None
                }
            })
            .collect()
    }
    pub fn fit_antidiagonal_par_multiple<'a, T, O>(
        &mut self,
        training_datapack: &[TrainingDataPack<'a, T, O>],
        radius: usize,
    ) where
        T: std::borrow::Borrow<[u8]> + Sync + Send,
        O: std::borrow::Borrow<[Op]> + Sync + Send,
    {
        use rayon::prelude::*;
        const INIT_WEIGHT: f64 = 0.0005;
        fn init_model() -> PairHiddenMarkovModelOnStrands {
            let forward = PairHiddenMarkovModel::zeros();
            let reverse = PairHiddenMarkovModel::zeros();
            PairHiddenMarkovModelOnStrands::new(forward, reverse)
        }
        let mut next = training_datapack
            .par_iter()
            .flat_map(|datapack| {
                assert_eq!(datapack.sequences.len(), datapack.operations.len());
                self.baum_welch_antidiagonal(datapack, radius)
                    .into_par_iter()
                    .map(move |(memory, qs, direction)| (memory, qs, direction, datapack.consensus))
            })
            .fold(init_model, |mut model, (memory, qs, direction, rs)| {
                match direction {
                    true => {
                        self.forward()
                            .register_antidiagonal(&memory, rs, qs, &mut model.forward)
                    }
                    false => {
                        self.reverse()
                            .register_antidiagonal(&memory, rs, qs, &mut model.reverse)
                    }
                };
                model
            })
            .reduce(init_model, |mut next, part| {
                next.forward.merge(&part.forward);
                next.reverse.merge(&part.reverse);
                next
            });
        next.forward
            .merge(&PairHiddenMarkovModel::uniform(INIT_WEIGHT));
        next.reverse
            .merge(&PairHiddenMarkovModel::uniform(INIT_WEIGHT));
        next.forward.normalize();
        next.reverse.normalize();
        *self = next;
    }
    /// With bootstrap operations and taking numbers.
    /// The returned operations would be consistent with the returned sequence.
    /// Only the *first* `take_num` seuqneces would be used to polish,
    /// but the operations of the rest are also updated accordingly.
    /// This is (slightly) faster than the usual/full polishing...
    /// strands: `true` if forward.
    pub fn polish_until_converge_antidiagonal<T, O>(
        &self,
        draft: &[u8],
        xss: &[T],
        opss: &mut [O],
        strands: &[bool],
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
        let qmax = xss
            .iter()
            .map(|x| x.borrow().len())
            .max()
            .expect("Reads empty.");
        let mut memory = Memory::new(draft.len(), qmax, radius);
        for t in 0..100 {
            let inactive = INACTIVE_TIME + (t * INACTIVE_TIME) % rs.len();
            let mut modif_table = vec![0f64; (rs.len() + 1) * NUM_ROW];
            let mut lk = 0f64;
            let stream = xss.iter().zip(&radius_reads).zip(strands).take(take_num);
            for ((seq, radius), is_forward) in stream {
                let hmm = match is_forward {
                    true => self.forward(),
                    false => self.reverse(),
                };
                assert_eq!(radius.len(), seq.borrow().len() + rs.len() + 1);
                let m_lk = hmm.dp_memory(&rs, seq.borrow(), radius, &mut memory);
                let mt = memory.modif_table.as_slice();
                if m_lk.is_finite() && mt.iter().all(|x| x.is_finite()) {
                    lk += m_lk;
                    modif_table.iter_mut().zip(mt).for_each(|(x, y)| *x += y);
                }
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
            assert!(modif_table.iter().all(|x| x.is_finite()));
            // eprintln!("CHANGES\t{t}\t{changed_pos:?}");
            let changed_pos: Vec<_> = changed_pos
                .iter()
                .map(|&(pos, op)| (pos, super::usize_to_edit_op(op)))
                .collect();
            let stream = radius_reads
                .iter_mut()
                .zip(xss)
                .zip(opss.iter_mut())
                .zip(strands);
            for (((radius_per, xs), ops), is_forward) in stream {
                let (ops, xs) = (ops.borrow_mut(), xs.borrow());
                let (qlen, rlen) = (xs.len(), rs.len());
                let changed_iter = changed_pos.iter().copied();
                crate::op::fix_alignment_path(ops, changed_iter, qlen, rlen);
                update_radius(ops, &changed_pos, radius_per, default_radius, qlen, rlen);
                assert_eq!(radius_per.len(), xs.len() + rs.len() + 1);
                memory.fr.update_by_radius(radius_per, rs.len(), xs.len());
                memory.initialize_aln();
                let hmm = match is_forward {
                    true => self.forward(),
                    false => self.reverse(),
                };
                let (_, new_ops) =
                    hmm.align_antidiagonal_filling(&rs, xs, &memory.fr, &mut memory.pre_dp);
                *ops = new_ops;
            }
            if changed_pos.is_empty() {
                break;
            }
        }
        rs
    }
}

type FBSlice<'a> = (&'a [(f64, f64, f64)], &'a [f64]);
type FBTable = (Vec<(f64, f64, f64)>, Vec<f64>);
#[derive(Debug, Clone)]
struct Memory {
    modif_table: Vec<f64>,
    fr: FillingRegions,
    post_dp: Vec<(f64, f64, f64)>,
    post_scl: Vec<f64>,
    pre_dp: Vec<(f64, f64, f64)>,
    pre_scl: Vec<f64>,
}

impl Memory {
    fn new(rlen: usize, qlen: usize, radius: usize) -> Self {
        let max_rlen = 3 * rlen / 2;
        let fr = FillingRegions::with_capacity(rlen, qlen, radius);
        let dp_size = (max_rlen * qlen) * radius * 2;
        let ad_len = max_rlen + qlen;
        Self {
            modif_table: Vec::with_capacity(max_rlen),
            fr,
            post_dp: Vec::with_capacity(dp_size),
            post_scl: Vec::with_capacity(ad_len),
            pre_dp: Vec::with_capacity(dp_size),
            pre_scl: Vec::with_capacity(ad_len),
        }
    }
    fn initialize(&mut self, rlen: usize) {
        self.post_scl.clear();
        self.pre_scl.clear();
        self.modif_table.iter_mut().for_each(|x| *x = 0f64);
        let total_cells = self.fr.total_cells;
        fit_vector(&mut self.modif_table, 0f64, (rlen + 1) * NUM_ROW);
        fit_vector(&mut self.pre_dp, (0f64, 0f64, 0f64), total_cells);
        fit_vector(&mut self.post_dp, (0f64, 0f64, 0f64), total_cells);
    }
    fn initialize_aln(&mut self) {
        fit_vector(&mut self.pre_dp, (EP, EP, EP), self.fr.total_cells)
    }
}

fn fit_vector<T: Copy>(xs: &mut Vec<T>, elm: T, len: usize) {
    match xs.len().cmp(&len) {
        std::cmp::Ordering::Less => xs.extend(std::iter::repeat(elm).take(len - xs.len())),
        std::cmp::Ordering::Equal => {}
        std::cmp::Ordering::Greater => xs.truncate(len),
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
        for i in 0..100 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let template = gen_seq::generate_seq(&mut rng, 300);
            let profile = gen_seq::PROFILE;
            let hmm = PairHiddenMarkovModel::default();
            let radius = 50;
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let ops = crate::op::bootstrap_ops(template.len(), seq.len());
            let (dptable, scale) = hmm.post(&template, &seq, &ops, radius);
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
            let filling_regions = filling_region(&ops, radius, rs.len(), qs.len());
            let mut scaling = Vec::with_capacity(filling_regions.len() / BLOCK_SIZE + 1);
            let fr = FillingRegions::new(&filling_regions);
            let mut post_dp = vec![(0f64, 0f64, 0f64); fr.total_cells];
            hmm.post_fill(&rs, &qs, &fr, &mut post_dp, &mut scaling);

            let answer = hmm.backward(&rs, &qs);
            let block_num = match (rs.len() + qs.len()) % BLOCK_SIZE == 0 {
                true => (rs.len() + qs.len()) / BLOCK_SIZE + 1,
                false => (rs.len() + qs.len()) / BLOCK_SIZE + 1,
            };
            assert_eq!(scaling.len(), block_num);
            for (ad, &(ofs, start, end)) in fr.offsets.iter().enumerate().rev() {
                let block = ad / BLOCK_SIZE;
                for i in (start..end).rev() {
                    let j = ad - i;
                    if (j.max(i) - j.min(i)) > radius / 2 {
                        continue;
                    }
                    let scale: f64 = scaling.iter().skip(block).map(|x| x.ln()).sum();
                    let (mat, ins, del) = post_dp[ofs + i - start];
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
            let filling_regions = filling_region(&ops, radius, rs.len(), qs.len());
            let fr = FillingRegions::new(&filling_regions);
            let mut scaling = Vec::with_capacity(filling_regions.len() / BLOCK_SIZE + 3);
            let mut pre_dp = vec![(0f64, 0f64, 0f64); fr.total_cells];
            hmm.pre_fill(&rs, &qs, &fr, &mut pre_dp, &mut scaling);
            let answer = hmm.forward(&rs, &qs);
            let block_num = (qs.len() + rs.len()) / BLOCK_SIZE + 1;
            assert_eq!(block_num, scaling.len());
            for (ad, &(ofs, start, end)) in fr.offsets.iter().enumerate().rev() {
                let block = ad / BLOCK_SIZE + 1;
                for i in start..end {
                    let j = ad - i;
                    if (j.max(i) - j.min(i)) > radius / 2 {
                        continue;
                    }
                    let scale: f64 = scaling.iter().take(block).map(|x| x.ln()).sum();
                    let (mat, ins, del) = pre_dp[ofs + i - start];
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
            let modif_table = hmm
                .modification_table_antidiagonal(&template, &seq, &ops, radius)
                .0;
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

    #[test]
    fn fit_test() {
        const LEN: usize = 2_000;
        const COVERAGE: usize = 20;
        const RADIUS: usize = 50;
        const PACK: usize = 4;
        const SEED: u64 = 309482;
        use crate::{gen_seq::Generate, hmm::TrainingDataPack};
        use rand::{Rng, SeedableRng};
        use rand_xoshiro::Xoroshiro128StarStar;

        let hmm = crate::hmm::PairHiddenMarkovModelOnStrands::default();
        let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(SEED);
        let profile = crate::gen_seq::Profile::new(0.03, 0.03, 0.03);
        let templates: Vec<_> = (0..PACK)
            .map(|_| crate::gen_seq::generate_seq(&mut rng, LEN))
            .collect();
        let reads: Vec<Vec<_>> = templates
            .iter()
            .map(|rs| (0..COVERAGE).map(|_| profile.gen(rs, &mut rng)).collect())
            .collect();
        let strands: Vec<Vec<_>> = (0..PACK)
            .map(|_| (0..COVERAGE).map(|_| rng.gen_bool(0.5)).collect())
            .collect();
        let opss: Vec<Vec<_>> = std::iter::zip(&templates, &reads)
            .map(|(rs, qss)| {
                qss.iter()
                    .map(|qs| hmm.forward().align_antidiagonal_bootstrap(rs, qs, RADIUS).1)
                    .collect()
            })
            .collect();
        let training_data: Vec<_> = std::iter::zip(&templates, &reads)
            .zip(strands.iter())
            .zip(opss.iter())
            .map(|(((rs, qss), strands), ops)| TrainingDataPack::new(rs, strands, qss, ops))
            .collect();
        let mut hmm_a = hmm;
        hmm_a.fit_antidiagonal_par_multiple(&training_data, RADIUS / 2);
        println!("{hmm_a}");
    }
}
