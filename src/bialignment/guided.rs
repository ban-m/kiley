use crate::dptable::DPTable;
use crate::op::Op;

// Return vector of (start,end) D' = D + E, where
// D = { (i,j) | 0 <= i <= qlen , s[i] <= j < e[i] } is equal to
// { (i,j) | there exists (a,b) in the alignment path such that |a-i| + |b-j| <= radius.
// And E is {(s[i],rlen+1)}.
fn convert_to_fill_range(
    qlen: usize,
    rlen: usize,
    ops: &[Op],
    radius: usize,
) -> Vec<(usize, usize)> {
    let mut ranges = vec![(rlen + 1, 0); qlen + 1];
    re_fill_fill_range(qlen, rlen, ops, radius, &mut ranges);
    ranges
}

fn re_fill_fill_range(
    qlen: usize,
    rlen: usize,
    ops: &[Op],
    radius: usize,
    ranges: &mut [(usize, usize)],
) {
    // Alignment path.
    let (mut qpos, mut rpos) = (0, 0);
    update_range(ranges, qpos, rpos, qlen, rlen, radius);
    for op in ops.iter() {
        match op {
            Op::Del => rpos += 1,
            Op::Ins => qpos += 1,
            Op::Match | Op::Mismatch => {
                qpos += 1;
                rpos += 1;
            }
        }
        update_range(ranges, qpos, rpos, qlen, rlen, radius);
        if qpos == qlen || rpos == rlen {
            break;
        }
    }
    // Follow through.
    let map_coef = rlen as f64 / qlen as f64;
    while qpos < qlen && rpos < rlen {
        let corresp_rpos = (qpos as f64 * map_coef).round() as usize;
        match corresp_rpos.cmp(&rpos) {
            std::cmp::Ordering::Less => qpos += 1,
            std::cmp::Ordering::Equal => {
                qpos += 1;
                rpos += 1;
            }
            std::cmp::Ordering::Greater => rpos += 1,
        }
        update_range(ranges, qpos, rpos, qlen, rlen, radius);
    }
    while qpos < qlen {
        qpos += 1;
        update_range(ranges, qpos, rpos, qlen, rlen, radius);
    }
    while rpos < rlen {
        rpos += 1;
        update_range(ranges, qpos, rpos, qlen, rlen, radius);
    }
    assert_eq!(qpos, qlen);
    assert_eq!(rpos, rlen);
    ranges[qlen].1 = rlen + 1;
}

fn update_range(
    ranges: &mut [(usize, usize)],
    qpos: usize,
    rpos: usize,
    qlen: usize,
    rlen: usize,
    radius: usize,
) {
    let qstart = qpos.saturating_sub(radius);
    let qend = (qpos + radius + 1).min(qlen + 1);
    for i in qstart..qend {
        let v_dist = i.max(qpos) - i.min(qpos);
        assert!(v_dist <= radius);
        let rem = radius - v_dist;
        let rstart = rpos.saturating_sub(rem);
        let rend = (rpos + rem + 1).min(rlen + 1);
        let range = ranges.get_mut(i).unwrap();
        range.0 = range.0.min(rstart);
        range.1 = range.1.max(rend);
    }
}

pub fn edit_dist_guided(
    reference: &[u8],
    query: &[u8],
    ops: &[Op],
    radius: usize,
) -> (u32, Vec<Op>) {
    // -----------------> Reference
    // |
    // V
    // Query
    let (qs, rs) = (query, reference);
    let qlen = ops.iter().filter(|&&op| op != Op::Del).count();
    assert_eq!(qlen, query.len());
    let fill_range = convert_to_fill_range(qs.len(), rs.len(), ops, radius);
    let upperbound = (qs.len() + rs.len() + 3) as u32;
    let mut dp = DPTable::new(&fill_range, upperbound);
    // 1. Initialization. It is OK to consume O(xs.len() + ys.len()) time.
    for i in 0..qs.len() + 1 {
        dp.set(i, 0, i as u32);
    }
    for j in 0..rs.len() + 1 {
        dp.set(0, j, j as u32);
    }
    // Skipping the first element in both i and j.
    for (i, &(start, end)) in fill_range.iter().enumerate().skip(1) {
        let q = qs[i - 1];
        for j in start.max(1)..end {
            let r = rs[j - 1];
            let mat = if q == r { 0 } else { 1 };
            let dist = (dp.get(i - 1, j) + 1)
                .min(dp.get(i, j - 1) + 1)
                .min(dp.get(i - 1, j - 1) + mat);
            dp.set(i, j, dist);
        }
    }
    // Traceback
    let mut qpos = qs.len();
    let mut rpos = fill_range.last().unwrap().1 - 1;
    assert_eq!(rpos, rs.len());
    // Init with appropriate number of deletion
    let mut ops = Vec::with_capacity(qs.len() + rs.len());
    let score = dp.get(qpos, rpos);
    while 0 < qpos && 0 < rpos {
        let current = dp.get(qpos, rpos);
        if current == dp.get(qpos - 1, rpos) + 1 {
            ops.push(Op::Ins);
            qpos -= 1;
        } else if current == dp.get(qpos, rpos - 1) + 1 {
            ops.push(Op::Del);
            rpos -= 1;
        } else {
            let mat = if qs[qpos - 1] == rs[rpos - 1] { 0 } else { 1 };
            assert_eq!(mat + dp.get(qpos - 1, rpos - 1), current);
            qpos -= 1;
            rpos -= 1;
            if mat == 0 {
                ops.push(Op::Match);
            } else {
                ops.push(Op::Mismatch);
            }
        }
    }
    ops.reverse();
    (score, ops)
}

pub fn global_guided(
    reference: &[u8],
    query: &[u8],
    ops: &[Op],
    radius: usize,
    (match_score, mism, open, ext): (i32, i32, i32, i32),
) -> (i32, Vec<Op>) {
    let (qs, rs) = (query, reference);
    let fill_range = convert_to_fill_range(qs.len(), rs.len(), ops, radius);
    let lower = (reference.len() + query.len()) as i32 * open;
    let mut mat = DPTable::new(&fill_range, lower);
    let mut del = DPTable::new(&fill_range, lower);
    let mut ins = DPTable::new(&fill_range, lower);
    // 1. Initialization
    for i in 1..qs.len() + 1 {
        mat.set(i, 0, lower);
        ins.set(i, 0, open + i.saturating_sub(1) as i32 * ext);
        del.set(i, 0, lower);
    }
    for j in 1..rs.len() + 1 {
        mat.set(0, j, lower);
        ins.set(0, j, lower);
        del.set(0, j, open + j.saturating_sub(1) as i32 * ext);
    }
    mat.set(0, 0, 0);
    ins.set(0, 0, lower);
    del.set(0, 0, lower);
    // 2. Recur.
    for (i, &(start, end)) in fill_range.iter().enumerate().skip(1) {
        let q = qs[i - 1];
        for j in start.max(1)..end {
            let r = rs[j - 1];
            let aln = if q == r { match_score } else { mism };
            let mat_next = mat
                .get(i - 1, j - 1)
                .max(del.get(i - 1, j - 1))
                .max(ins.get(i - 1, j - 1))
                + aln;
            mat.set(i, j, mat_next);
            let ins_next = (mat.get(i - 1, j) + open).max(ins.get(i - 1, j) + ext);
            ins.set(i, j, ins_next);
            let del_next = (mat.get(i, j - 1) + open)
                .max(del.get(i, j - 1) + ext)
                .max(ins.get(i, j - 1) + open);
            del.set(i, j, del_next);
        }
    }
    // Traceback.
    let mut qpos = qs.len();
    let mut rpos = fill_range.last().unwrap().1 - 1;
    assert_eq!(rpos, rs.len());
    let (mut state, score) = {
        let mat_to_fin = mat.get(qpos, rpos);
        let ins_to_fin = ins.get(qpos, rpos);
        let del_to_fin = del.get(qpos, rpos);
        if ins_to_fin <= mat_to_fin && del_to_fin <= mat_to_fin {
            (0, mat_to_fin)
        } else if mat_to_fin <= ins_to_fin && del_to_fin <= ins_to_fin {
            (1, ins_to_fin)
        } else {
            assert!(mat_to_fin <= del_to_fin && ins_to_fin <= del_to_fin,);
            (2, del_to_fin)
        }
    };
    let mut ops = Vec::with_capacity(qs.len() + rs.len());
    while 0 < qpos && 0 < rpos {
        if state == 0 {
            let is_mat = qs[qpos - 1] == rs[rpos - 1];
            let aln = if is_mat { match_score } else { mism };
            let current = mat.get(qpos, rpos) - aln;
            if current == mat.get(qpos - 1, rpos - 1) {
                state = 0;
            } else if current == ins.get(qpos - 1, rpos - 1) {
                state = 1;
            } else {
                assert_eq!(current, del.get(qpos - 1, rpos - 1));
                state = 2;
            }
            qpos -= 1;
            rpos -= 1;
            if is_mat {
                ops.push(Op::Match)
            } else {
                ops.push(Op::Mismatch)
            };
        } else if state == 1 {
            let current = ins.get(qpos, rpos);
            if current == mat.get(qpos - 1, rpos) + open {
                state = 0;
            } else {
                assert_eq!(current, ins.get(qpos - 1, rpos) + ext);
                state = 1;
            }
            qpos -= 1;
            ops.push(Op::Ins);
        } else {
            assert_eq!(state, 2);
            let current = del.get(qpos, rpos);
            if current == mat.get(qpos, rpos - 1) + open {
                state = 0;
            } else if current == ins.get(qpos, rpos - 1) + open {
                state = 1;
            } else {
                assert_eq!(current, del.get(qpos, rpos - 1) + ext);
                state = 2;
            }
            rpos -= 1;
            ops.push(Op::Del);
        }
    }
    ops.extend(std::iter::repeat(Op::Del).take(rpos));
    ops.extend(std::iter::repeat(Op::Ins).take(qpos));
    (score, ops)
}

const COPY_SIZE: usize = 3;
const DEL_SIZE: usize = 3;
// for each position, four type of mutation,
// four type of insertion,
// 1bp, 2bp, 3bp copy,
// 1bp, 2bp, 3bp deletion,
// so, in total, there are 4 + 4 + 3 + 3 = 14 patterns of modifications.
const NUM_ROW: usize = 8 + COPY_SIZE + DEL_SIZE;
// After introducing mutation, we would take INACTIVE_TIME bases just as-is.
const INACTIVE_TIME: usize = 5;

fn fill_mod_table(
    mod_table: &mut Vec<u32>,
    rs: &[u8],
    qs: &[u8],
    pre: &DPTable<u32>,
    post: &DPTable<u32>,
    fill_range: &[(usize, usize)],
) {
    assert_eq!(fill_range.len(), qs.len() + 1);
    let total_len = NUM_ROW * (rs.len() + 1);
    mod_table.truncate(total_len);
    mod_table.iter_mut().for_each(|x| *x = pre.upperbound());
    if mod_table.len() < total_len {
        mod_table.extend(std::iter::repeat(pre.upperbound()).take(total_len));
    }
    for (i, &(start, end)) in fill_range.iter().enumerate().take(qs.len()) {
        let q = qs[i];
        for j in start..end.min(rs.len() + 1) {
            let pre_mat = pre.get(i, j);
            let (post_del, post_ins) = (post.get(i, j + 1), post.get(i + 1, j));
            let (post_mat, post_sta) = (post.get(i + 1, j + 1), post.get(i, j));
            let pre_copy_line = pre.get_line(i, j + 1, (rs.len() - j).min(COPY_SIZE));
            let post_del_line = post.get_line(i, j + 1, (rs.len() - j).min(DEL_SIZE));
            let row_start = NUM_ROW * j;
            // change the j-th base into ...
            let mat_pattern = pre_mat + post_mat;
            let del_pattern = pre_mat + post_del + 1;
            mod_table
                .iter_mut()
                .skip(row_start)
                .zip(b"ACGT")
                .for_each(|(x, &base)| {
                    *x = (*x).min(mat_pattern + (q != base) as u32).min(del_pattern);
                });
            // insert before the j-th base...
            let mat_pattern = pre_mat + post_ins;
            let del_pattern = pre_mat + post_sta + 1;
            mod_table
                .iter_mut()
                .skip(row_start + 4)
                .zip(b"ACGT")
                .for_each(|(x, &base)| {
                    *x = (*x).min(mat_pattern + (base != q) as u32).min(del_pattern);
                });
            // Copying the j..j+c bases..
            mod_table
                .iter_mut()
                .skip(row_start + 8)
                .zip(pre_copy_line)
                .for_each(|(x, pre_c)| {
                    *x = (*x).min(pre_c + post_sta);
                });
            // Deleting the j..j+d bases...
            mod_table
                .iter_mut()
                .skip(row_start + 8 + COPY_SIZE)
                .zip(post_del_line)
                .for_each(|(x, post_d)| {
                    *x = (*x).min(pre_mat + post_d);
                });
        }
    }

    // Insertion at hte last position.
    if let Some(&(start, end)) = fill_range.last() {
        let i = fill_range.len() - 1;
        for j in start..end.min(rs.len() + 1) {
            let pre_mat = pre.get(i, j);
            let post_sta = post.get(i, j);
            let post_del = post.get(i, j + 1);
            let row_start = NUM_ROW * j;
            // change the j-th base into ...
            mod_table
                .iter_mut()
                .skip(row_start)
                .take(4)
                .for_each(|x| *x = (*x).min(pre_mat + post_del + 1));
            // insert before the j-th base...
            mod_table
                .iter_mut()
                .skip(row_start + 4)
                .take(4)
                .for_each(|x| *x = (*x).min(pre_mat + post_sta + 1));
            // Copying the j..j+c bases..
            mod_table
                .iter_mut()
                .skip(row_start + 8)
                .take(COPY_SIZE)
                .enumerate()
                .filter(|(c, _)| j + c + 1 <= rs.len())
                .for_each(|(len, x)| *x = (*x).min(pre.get(i, j + len + 1) + post_sta));
            // Deleting the j..j+d bases...
            mod_table
                .iter_mut()
                .skip(row_start + 8 + COPY_SIZE)
                .take(DEL_SIZE)
                .enumerate()
                .filter(|(d, _)| j + d + 1 <= rs.len())
                .for_each(|(len, x)| *x = (*x).min(pre_mat + post.get(i, j + len + 1)));
        }
    }
}

struct Aligner {
    pre_dp: DPTable<u32>,
    post_dp: DPTable<u32>,
    fill_ranges: Vec<(usize, usize)>,
}

impl Aligner {
    fn with_capacity(qlen: usize, radius: usize) -> Self {
        let pre_dp = DPTable::with_capacity(qlen, radius, 3 * qlen as u32);
        let post_dp = DPTable::with_capacity(qlen, radius, 3 * qlen as u32);
        let fill_ranges = Vec::with_capacity(qlen * 3);
        Self {
            pre_dp,
            post_dp,
            fill_ranges,
        }
    }
    fn align(
        &mut self,
        rs: &[u8],
        qs: &[u8],
        radius: usize,
        ops: &mut Vec<Op>,
        table: &mut Vec<u32>,
    ) -> u32 {
        self.fill_ranges.clear();
        self.fill_ranges
            .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
        re_fill_fill_range(qs.len(), rs.len(), &ops, radius, &mut self.fill_ranges);
        let upperbound = (rs.len() + qs.len() + 3) as u32;
        self.pre_dp.initialize(upperbound, &self.fill_ranges);
        self.post_dp.initialize(upperbound, &self.fill_ranges);
        Self::fill_pre_dp(&mut self.pre_dp, &self.fill_ranges, rs, qs, ops);
        Self::fill_post_dp(&mut self.post_dp, &self.fill_ranges, rs, qs);
        let fr = &self.fill_ranges;
        fill_mod_table(table, rs, qs, &self.pre_dp, &self.post_dp, fr);
        self.post_dp.get(0, 0)
    }
    fn fill_pre_dp(
        dp: &mut DPTable<u32>,
        fill_range: &[(usize, usize)],
        rs: &[u8],
        qs: &[u8],
        ops: &mut Vec<Op>,
    ) {
        // 1. Initialization. It is OK to consume O(xs.len() + ys.len()) time.
        for i in 0..qs.len() + 1 {
            dp.set(i, 0, i as u32);
        }
        for j in 0..rs.len() + 1 {
            dp.set(0, j, j as u32);
        }
        // Skipping the first element in both i and j.
        for (i, &(start, end)) in fill_range.iter().enumerate().skip(1) {
            let q = qs[i - 1];
            for j in start.max(1)..end {
                let r = rs[j - 1];
                let mat = if q == r { 0 } else { 1 };
                let dist = (dp.get(i - 1, j) + 1)
                    .min(dp.get(i, j - 1) + 1)
                    .min(dp.get(i - 1, j - 1) + mat);
                dp.set(i, j, dist);
            }
        }
        // Traceback
        let mut qpos = qs.len();
        let mut rpos = fill_range.last().unwrap().1 - 1;
        assert_eq!(rpos, rs.len());
        // Init with appropriate number of deletion
        //let mut ops = Vec::with_capacity(rs.len() + qs.len());
        ops.clear();
        while 0 < qpos && 0 < rpos {
            let current = dp.get(qpos, rpos);
            if current == dp.get(qpos - 1, rpos) + 1 {
                ops.push(Op::Ins);
                qpos -= 1;
            } else if current == dp.get(qpos, rpos - 1) + 1 {
                ops.push(Op::Del);
                rpos -= 1;
            } else {
                let mat = if qs[qpos - 1] == rs[rpos - 1] { 0 } else { 1 };
                assert_eq!(mat + dp.get(qpos - 1, rpos - 1), current);
                qpos -= 1;
                rpos -= 1;
                if mat == 0 {
                    ops.push(Op::Match);
                } else {
                    ops.push(Op::Mismatch);
                }
            }
        }
        ops.reverse();
    }
    fn fill_post_dp(dp: &mut DPTable<u32>, fill_range: &[(usize, usize)], rs: &[u8], qs: &[u8]) {
        // 1. Initialization. It is OK to consume O(xs.len() + ys.len()) time.
        for i in 0..qs.len() + 1 {
            dp.set(i, rs.len(), (qs.len() - i) as u32);
        }
        for j in 0..rs.len() + 1 {
            dp.set(qs.len(), j, (rs.len() - j) as u32);
        }
        // Skipping the first element in both i and j.
        for (i, &(start, end)) in fill_range.iter().enumerate().rev().skip(1) {
            let q = qs[i];
            for j in (start..end.min(rs.len())).rev() {
                let r = rs[j];
                let mat = if q == r { 0 } else { 1 };
                let dist = (dp.get(i + 1, j) + 1)
                    .min(dp.get(i, j + 1) + 1)
                    .min(dp.get(i + 1, j + 1) + mat);
                dp.set(i, j, dist);
            }
        }
    }
}

fn bootstrap_ops(rlen: usize, qlen: usize) -> Vec<Op> {
    let (mut qpos, mut rpos) = (0, 0);
    let mut ops = Vec::with_capacity(rlen + qlen);
    let map_coef = rlen as f64 / qlen as f64;
    while qpos < qlen && rpos < rlen {
        let corresp_rpos = (qpos as f64 * map_coef).round() as usize;
        if corresp_rpos < rpos {
            // rpos is too fast. Move qpos only.
            qpos += 1;
            ops.push(Op::Ins);
        } else if corresp_rpos == rpos {
            qpos += 1;
            rpos += 1;
            ops.push(Op::Match);
        } else {
            rpos += 1;
            ops.push(Op::Del);
        }
    }
    ops.extend(std::iter::repeat(Op::Ins).take(qlen - qpos));
    ops.extend(std::iter::repeat(Op::Del).take(rlen - rpos));
    ops
}

// struct AlignInfo<'a> {
//     seq: &'a [u8],
//     ops: Vec<Op>,
//     // pre_dp: DPTable<u32>,
//     // post_dp: DPTable<u32>,
//     // fill_range: Vec<(usize, usize)>,
//     mod_table: Vec<u32>,
//     dist: u32,
// }

// impl<'a> AlignInfo<'a> {
//     fn new(template: &[u8], seq: &'a [u8], radius: usize) -> Self {
//         let (mut qpos, mut rpos) = (0, 0);
//         let mut ops = Vec::with_capacity(template.len() + seq.len());
//         let map_coef = template.len() as f64 / seq.len() as f64;
//         while qpos < seq.len() && rpos < template.len() {
//             let corresp_rpos = (qpos as f64 * map_coef).round() as usize;
//             if corresp_rpos < rpos {
//                 // rpos is too fast. Move qpos only.
//                 qpos += 1;
//                 ops.push(Op::Ins);
//             } else if corresp_rpos == rpos {
//                 qpos += 1;
//                 rpos += 1;
//                 ops.push(Op::Match);
//             } else {
//                 rpos += 1;
//                 ops.push(Op::Del);
//             }
//         }
//         ops.extend(std::iter::repeat(Op::Ins).take(seq.len() - qpos));
//         ops.extend(std::iter::repeat(Op::Del).take(template.len() - rpos));
//         AlignInfo::from_aln_path(template, seq, &ops, radius)
//     }
//     fn fill_pre_dp(
//         dp: &mut DPTable<u32>,
//         fill_range: &[(usize, usize)],
//         rs: &[u8],
//         qs: &[u8],
//         ops: &mut Vec<Op>,
//     ) {
//         // 1. Initialization. It is OK to consume O(xs.len() + ys.len()) time.
//         for i in 0..qs.len() + 1 {
//             dp.set(i, 0, i as u32);
//         }
//         for j in 0..rs.len() + 1 {
//             dp.set(0, j, j as u32);
//         }
//         // Skipping the first element in both i and j.
//         for (i, &(start, end)) in fill_range.iter().enumerate().skip(1) {
//             let q = qs[i - 1];
//             for j in start.max(1)..end {
//                 let r = rs[j - 1];
//                 let mat = if q == r { 0 } else { 1 };
//                 let dist = (dp.get(i - 1, j) + 1)
//                     .min(dp.get(i, j - 1) + 1)
//                     .min(dp.get(i - 1, j - 1) + mat);
//                 dp.set(i, j, dist);
//             }
//         }
//         // Traceback
//         let mut qpos = qs.len();
//         let mut rpos = fill_range.last().unwrap().1 - 1;
//         assert_eq!(rpos, rs.len());
//         // Init with appropriate number of deletion
//         //let mut ops = Vec::with_capacity(rs.len() + qs.len());
//         ops.clear();
//         while 0 < qpos && 0 < rpos {
//             let current = dp.get(qpos, rpos);
//             if current == dp.get(qpos - 1, rpos) + 1 {
//                 ops.push(Op::Ins);
//                 qpos -= 1;
//             } else if current == dp.get(qpos, rpos - 1) + 1 {
//                 ops.push(Op::Del);
//                 rpos -= 1;
//             } else {
//                 let mat = if qs[qpos - 1] == rs[rpos - 1] { 0 } else { 1 };
//                 assert_eq!(mat + dp.get(qpos - 1, rpos - 1), current);
//                 qpos -= 1;
//                 rpos -= 1;
//                 if mat == 0 {
//                     ops.push(Op::Match);
//                 } else {
//                     ops.push(Op::Mismatch);
//                 }
//             }
//         }
//         ops.reverse();
//     }
//     fn fill_post_dp(dp: &mut DPTable<u32>, fill_range: &[(usize, usize)], rs: &[u8], qs: &[u8]) {
//         // 1. Initialization. It is OK to consume O(xs.len() + ys.len()) time.
//         for i in 0..qs.len() + 1 {
//             dp.set(i, rs.len(), (qs.len() - i) as u32);
//         }
//         for j in 0..rs.len() + 1 {
//             dp.set(qs.len(), j, (rs.len() - j) as u32);
//         }
//         // Skipping the first element in both i and j.
//         for (i, &(start, end)) in fill_range.iter().enumerate().rev().skip(1) {
//             let q = qs[i];
//             for j in (start..end.min(rs.len())).rev() {
//                 let r = rs[j];
//                 let mat = if q == r { 0 } else { 1 };
//                 let dist = (dp.get(i + 1, j) + 1)
//                     .min(dp.get(i, j + 1) + 1)
//                     .min(dp.get(i + 1, j + 1) + mat);
//                 dp.set(i, j, dist);
//             }
//         }
//     }
// fn from_aln_path(rs: &[u8], qs: &'a [u8], ops: &[Op], radius: usize) -> Self {
//     let fill_range = convert_to_fill_range(qs.len(), rs.len(), ops, radius);
//     let upperbound = (qs.len() + rs.len() + 3) as u32;
//     let (ops, pre_dp) = {
//         let mut dp = DPTable::new(&fill_range, upperbound);
//         let mut ops = vec![];
//         Self::fill_pre_dp(&mut dp, &fill_range, rs, qs, &mut ops);
//         (ops, dp)
//     };
//     let post_dp = {
//         let mut dp = DPTable::new(&fill_range, upperbound);
//         Self::fill_post_dp(&mut dp, &fill_range, rs, qs);
//         dp
//     };
//     let dist = post_dp.get(0, 0);
//     let dist2 = pre_dp.get(qs.len(), rs.len());
//     assert_eq!(dist, dist2);
//     let mut mod_table = Vec::with_capacity((rs.len() + 1) * NUM_ROW);
//     fill_mod_table(&mut mod_table, &rs, &qs, &pre_dp, &post_dp, &fill_range);
//     AlignInfo {
//         seq: qs,
//         ops,
//         // pre_dp,
//         // post_dp,
//         // fill_range,
//         dist,
//         mod_table,
//     }
// }
// fn update(&mut self, rs: &[u8], radius: usize) {
//     let upperbound = (self.seq.len() + rs.len() + 3) as u32;
//     // Clear the old information.
//     self.fill_range = convert_to_fill_range(self.seq.len(), rs.len(), &self.ops, radius);
//     self.pre_dp.clear();
//     self.pre_dp.initialize(upperbound, &self.fill_range);
//     self.post_dp.clear();
//     self.post_dp.initialize(upperbound, &self.fill_range);
//     self.mod_table.clear();
//     // Fill new values.
//     self.ops = Self::fill_pre_dp(&mut self.pre_dp, &self.fill_range, rs, self.seq);
//     Self::fill_post_dp(&mut self.post_dp, &self.fill_range, rs, self.seq);
//     self.dist = self.post_dp.get(0, 0);
//     let dist2 = self.pre_dp.get(self.seq.len(), rs.len());
//     assert_eq!(self.dist, dist2);
//     fill_mod_table(
//         &mut self.mod_table,
//         &rs,
//         &self.seq,
//         &self.pre_dp,
//         &self.post_dp,
//         &self.fill_range,
//     );
// }
// }

fn polish_guided(
    template: &[u8],
    modif_table: &[u32],
    current_dist: u32,
    inactive: usize,
) -> Option<Vec<u8>> {
    let mut improved = Vec::with_capacity(template.len() * 11 / 10);
    let mut modif_table = modif_table.chunks_exact(NUM_ROW);
    let mut pos = 0;
    while let Some(row) = modif_table.next() {
        let (op, &dist) = row.iter().enumerate().min_by_key(|x| x.1).unwrap();
        if dist < current_dist && pos < template.len() {
            if op < 4 {
                // Mutateion.
                improved.push(b"ACGT"[op]);
                pos += 1;
            } else if op < 8 {
                // Insertion before this base.
                improved.push(b"ACGT"[op - 4]);
                improved.push(template[pos]);
                pos += 1;
            } else if op < 8 + COPY_SIZE {
                // copy from here to here + 1 + (op - 8) base
                let len = (op - 8) + 1;
                improved.extend(&template[pos..pos + len]);
                improved.push(template[pos]);
                pos += 1;
            } else if op < NUM_ROW {
                // Delete from here to here + 1 + (op - 8 - COPY_SIZE).
                let del_size = op - 8 - COPY_SIZE + 1;
                pos += del_size;
                (0..del_size - 1).filter_map(|_| modif_table.next()).count();
            } else {
                unreachable!()
            }
            improved.extend(template.iter().skip(pos).take(inactive));
            pos = (pos + inactive).min(template.len());
            (0..inactive).filter_map(|_| modif_table.next()).count();
        } else if pos < template.len() {
            improved.push(template[pos]);
            pos += 1;
        } else if dist < current_dist && 4 <= op && op < 8 {
            // Here, we need to consider the last insertion...
            improved.push(b"ACGT"[op - 4]);
        }
    }
    assert_eq!(pos, template.len());
    (improved != template).then(|| improved)
}

pub fn polish_until_converge<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    xs: &[T],
    radius: usize,
) -> Vec<u8> {
    let mut template = template.to_vec();
    let len = template.len().min(21);
    let mut ops: Vec<_> = xs
        .iter()
        .map(|seq| bootstrap_ops(template.len(), seq.borrow().len()))
        .collect();
    let mut modif_table = Vec::new();
    let mut temp_table = Vec::new();
    let mut aligner = Aligner::with_capacity(template.len(), radius);
    for inactive in (0..100).map(|x| INACTIVE_TIME + (x * INACTIVE_TIME) % len) {
        modif_table.clear();
        let dist: u32 = ops
            .iter_mut()
            .zip(xs.iter())
            .map(|(ops, seq)| {
                let dist = aligner.align(&template, seq.borrow(), radius, ops, &mut temp_table);
                match modif_table.is_empty() {
                    true => modif_table.extend_from_slice(&temp_table),
                    false => {
                        modif_table
                            .iter_mut()
                            .zip(temp_table.iter())
                            .for_each(|(x, y)| *x += y);
                    }
                }
                dist
            })
            .sum();
        match polish_guided(&template, &modif_table, dist, inactive) {
            Some(next) => template = next,
            None => break,
        }
    }
    template
}

#[cfg(test)]
pub mod test {
    use super::*;
    fn edit_dist_base_ops(xs: &[u8], ys: &[u8]) -> (u32, Vec<Op>) {
        let mut dp = vec![vec![0; ys.len() + 1]; xs.len() + 1];
        for i in 0..xs.len() + 1 {
            dp[i][0] = i as u32;
        }
        for j in 0..ys.len() + 1 {
            dp[0][j] = j as u32;
        }
        for (i, x) in xs.iter().enumerate().map(|(i, x)| (i + 1, x)) {
            for (j, y) in ys.iter().enumerate().map(|(j, y)| (j + 1, y)) {
                let mat = if x == y { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + mat);
            }
        }
        let (mut i, mut j) = (xs.len(), ys.len());
        let mut ops = vec![];
        while 0 < i && 0 < j {
            let current = dp[i][j];
            if current == dp[i - 1][j] + 1 {
                i -= 1;
                ops.push(Op::Del);
            } else if current == dp[i][j - 1] + 1 {
                j -= 1;
                ops.push(Op::Ins);
            } else {
                let mat = if xs[i - 1] == ys[j - 1] { 0 } else { 1 };
                assert_eq!(dp[i - 1][j - 1] + mat, current);
                if mat == 0 {
                    ops.push(Op::Match);
                } else {
                    ops.push(Op::Mismatch);
                }
                i -= 1;
                j -= 1;
            }
        }
        ops.extend(std::iter::repeat(Op::Del).take(i));
        ops.extend(std::iter::repeat(Op::Ins).take(j));
        ops.reverse();
        (dp[xs.len()][ys.len()], ops)
    }
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    const SEED: u64 = 1293890;
    #[test]
    fn edit_dist_check_iden() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        for _ in 0..100 {
            // let xslen = rng.gen::<usize>() % 100 + 20;
            let xslen = 50;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let (score, ops) = edit_dist_base_ops(&xs, &xs);
            let (score2, _ops2) = edit_dist_guided(&xs, &xs, &ops, 5);
            assert_eq!(score, score2);
        }
    }
    #[test]
    fn edit_dist_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 100 + 20;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let prof = crate::gen_seq::PROFILE;
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let (score, ops) = edit_dist_base_ops(&xs, &ys);
            let (score2, _ops2) = edit_dist_guided(&xs, &ys, &ops, 5);
            assert_eq!(score, score2);
        }
    }
    #[test]
    fn global_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let (mat, mism, open, ext) = (3, -6, -10, -2);
        let param = (mat, mism, open, ext);
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 100 + 20;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let prof = crate::gen_seq::PROFILE;
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let (score, ops) = crate::bialignment::global(&xs, &ys, mat, mism, open, ext);
            let (score2, _ops2) = global_guided(&xs, &ys, &ops, 5, param);
            assert_eq!(score, score2);
        }
    }
    #[test]
    fn modification_table_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        for i in 0..100 {
            println!("{}", i);
            let radius = 20;
            let xslen = rng.gen::<usize>() % 100 + 20;
            let mut xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let prof = crate::gen_seq::PROFILE;
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let (_dist, ops) = edit_dist_base_ops(&xs, &ys);
            let mut mod_table = Vec::new();
            let mut aligner = Aligner::with_capacity(xs.len(), radius);
            {
                let mut ops = ops.clone();
                aligner.align(&xs, &ys, radius, &mut ops, &mut mod_table);
            }
            // Mutation.
            for j in 0..xs.len() {
                let seek = j * NUM_ROW;
                let orig = xs[j];
                for (pos, &base) in b"ACGT".iter().enumerate() {
                    xs[j] = base;
                    let (dist, _) = edit_dist_guided(&xs, &ys, &ops, radius);
                    assert_eq!(dist, mod_table[seek + pos],);
                    xs[j] = orig;
                }
            }
            // Insertion
            for j in 0..xs.len() {
                let seek = j * NUM_ROW + 4;
                for (pos, &base) in b"ACGT".iter().enumerate() {
                    xs.insert(j, base);
                    let (dist, _) = edit_dist_guided(&xs, &ys, &ops, radius);
                    assert_eq!(dist, mod_table[seek + pos]);
                    xs.remove(j);
                }
            }
            for (pos, &base) in b"ACGT".iter().enumerate() {
                let seek = xs.len() * NUM_ROW + 4;
                xs.push(base);
                let (dist, _) = edit_dist_guided(&xs, &ys, &ops, radius);
                assert_eq!(dist, mod_table[seek + pos]);
                xs.pop();
            }
            // Copy region.
            for j in 0..xs.len() {
                let seek = j * NUM_ROW + 8;
                for len in (0..COPY_SIZE).filter(|c| j + c + 1 <= xs.len()) {
                    let xs: Vec<_> = xs[..j + len + 1].iter().chain(&xs[j..]).copied().collect();
                    let (dist, _) = edit_dist_guided(&xs, &ys, &ops, radius);
                    assert_eq!(dist, mod_table[seek + len], "{},{},{}", xs.len(), j, len);
                }
            }
            // Delete region.
            for j in 0..xs.len() {
                let seek = j * NUM_ROW + 8 + COPY_SIZE;
                for len in (0..DEL_SIZE).filter(|d| j + d + 1 <= xs.len()) {
                    let xs: Vec<_> = xs[..j].iter().chain(&xs[j + len + 1..]).copied().collect();
                    let (dist, _) = edit_dist_guided(&xs, &ys, &ops, radius);
                    assert_eq!(dist, mod_table[seek + len]);
                }
            }
        }
    }
    #[test]
    fn polish_test() {
        for i in 0..100 {
            println!("ST\t{}", i);
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED + i);
            let radius = 20;
            let xslen = rng.gen::<usize>() % 100 + 20;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let prof = crate::gen_seq::PROFILE;
            let yss: Vec<_> = (0..30)
                .map(|_| crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof))
                .collect();
            let template = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let consed = polish_until_converge(&template, &yss, radius);
            let (dist, _) = edit_dist_base_ops(&xs, &consed);
            let (prev, _) = edit_dist_base_ops(&xs, &template);
            assert_eq!(dist, 0, "{},{},{}", i, prev, dist);
        }
    }
}
