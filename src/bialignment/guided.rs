use crate::dptable::DPTable;
use crate::op::bootstrap_ops;
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

pub(crate) fn re_fill_fill_range(
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
        let v_dist = if i < qpos { qpos - i } else { i - qpos };
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
            let mat = (q != r) as u32;
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
            let mat = (qs[qpos - 1] != rs[rpos - 1]) as u32;
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
    ops.extend(std::iter::repeat(Op::Del).take(rpos));
    ops.extend(std::iter::repeat(Op::Ins).take(qpos));
    ops.reverse();
    (score, ops)
}

/// Global alignment, guided by the `ops`.
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
    let mut dp = DPTable::new(&fill_range, (lower, lower, lower));
    // 1. Initialization
    dp.set(0, 0, (0, lower, lower));
    for j in 1..rs.len() + 1 {
        dp.set(0, j, (lower, lower, open + (j - 1) as i32 * ext));
    }
    for i in 1..qs.len() + 1 {
        dp.set(i, 0, (lower, open + (i - 1) as i32 * ext, lower));
    }
    // 2. Recur.
    let rows = fill_range.iter().enumerate().skip(1).zip(qs.iter());
    let param = (match_score, mism, open, ext);
    for ((i, &(start, end)), &q) in rows {
        fill_row((i, q), (start, end, rs), param, &mut dp);
    }
    // Traceback.
    let mut qpos = qs.len();
    let mut rpos = fill_range.last().unwrap().1 - 1;
    assert_eq!(rpos, rs.len());
    let (mut state, score) = get_max_pair(qpos, rpos, &dp);
    let mut ops = Vec::with_capacity(qs.len() + rs.len());
    while 0 < qpos && 0 < rpos {
        let (new_state, op) = track_back_one_op(state, &dp, (qpos, qs), (rpos, rs), param);
        match state {
            0 => {
                qpos -= 1;
                rpos -= 1;
            }
            1 => qpos -= 1,
            _ => rpos -= 1,
        }
        state = new_state;
        ops.push(op);
    }
    ops.extend(std::iter::repeat(Op::Del).take(rpos));
    ops.extend(std::iter::repeat(Op::Ins).take(qpos));
    ops.reverse();
    (score, ops)
}

/// Infix alignment, guided by the `ops`.
/// The both end of the reference is not penalized.
pub fn infix_guided(
    reference: &[u8],
    query: &[u8],
    ops: &[Op],
    radius: usize,
    (match_score, mism, open, ext): (i32, i32, i32, i32),
) -> (i32, Vec<Op>) {
    let (qs, rs) = (query, reference);
    let fill_range = convert_to_fill_range(qs.len(), rs.len(), ops, radius);
    let lower = (reference.len() + query.len()) as i32 * open;
    let mut dp = DPTable::new(&fill_range, (lower, lower, lower));
    // 1. Initialization
    dp.set(0, 0, (0, lower, lower));
    // Deletion is not penalized.
    for j in 1..rs.len() + 1 {
        dp.set(0, j, (lower, lower, 0));
    }
    for i in 1..qs.len() + 1 {
        dp.set(i, 0, (lower, open + (i - 1) as i32 * ext, lower));
    }
    // 2. Recur.
    let rows = fill_range.iter().enumerate().skip(1).zip(qs.iter());
    let param = (match_score, mism, open, ext);
    for ((i, &(start, end)), &q) in rows {
        fill_row((i, q), (start, end, rs), param, &mut dp);
    }
    let last_row = fill_range.iter().enumerate().skip(1).zip(qs.iter()).last();
    if let Some(((i, &(start, end)), &q)) = last_row {
        for j in start.max(1)..end {
            let r = rs[j - 1];
            let aln = if q == r { match_score } else { mism };
            let mat_next = {
                let (mat, ins, del) = dp.get(i - 1, j - 1);
                mat.max(ins).max(del) + aln
            };
            let ins_next = {
                let (mat, ins, _) = dp.get(i - 1, j);
                (mat + open).max(ins + ext)
            };
            let del_next = {
                let (mat, ins, del) = dp.get(i, j - 1);
                (mat).max(del).max(ins)
            };
            dp.set(i, j, (mat_next, ins_next, del_next));
        }
    }
    // Traceback.
    let mut qpos = qs.len();
    let mut rpos = rs.len();
    let (mut state, score) = get_max_pair(qpos, rpos, &dp);
    let mut ops = Vec::with_capacity(qs.len() + rs.len());
    while qpos == qs.len() && 0 < qpos && 0 < rpos {
        let (mat, ins, del) = dp.get(qpos, rpos);
        if state == 0 {
            let is_mat = qs[qpos - 1] == rs[rpos - 1];
            match is_mat {
                true => ops.push(Op::Match),
                false => ops.push(Op::Mismatch),
            };
            let aln = if is_mat { match_score } else { mism };
            let current = mat - aln;
            let (m_prev, i_prev, d_prev) = dp.get(qpos - 1, rpos - 1);
            if current == m_prev {
                state = 0;
            } else if current == i_prev {
                state = 1;
            } else {
                assert_eq!(current, d_prev);
                state = 2;
            }
            qpos -= 1;
            rpos -= 1;
        } else if state == 1 {
            let (m_prev, i_prev, _) = dp.get(qpos - 1, rpos);
            ops.push(Op::Ins);
            if ins == m_prev + open {
                state = 0;
            } else {
                assert_eq!(ins, i_prev + ext);
                state = 1;
            }
            qpos -= 1;
        } else {
            assert_eq!(state, 2);
            ops.push(Op::Del);
            let current = del;
            let (m_prev, i_prev, d_prev) = dp.get(qpos, rpos - 1);
            if current == m_prev {
                state = 0;
            } else if current == i_prev {
                state = 1;
            } else {
                assert_eq!(current, d_prev);
                state = 2;
            }
            rpos -= 1;
        }
    }
    while 0 < qpos && 0 < rpos {
        let (new_state, op) = track_back_one_op(state, &dp, (qpos, qs), (rpos, rs), param);
        match state {
            0 => {
                qpos -= 1;
                rpos -= 1;
            }
            1 => qpos -= 1,
            _ => rpos -= 1,
        }
        state = new_state;
        ops.push(op);
    }
    ops.extend(std::iter::repeat(Op::Del).take(rpos));
    ops.extend(std::iter::repeat(Op::Ins).take(qpos));
    ops.reverse();
    (score, ops)
}

/// Overlap alignment, guided by the `ops`.
/// The beggining of the `from` and the trailing sequence of `to` would not be penalized.
pub fn overlap_guided(
    from: &[u8],
    to: &[u8],
    ops: &[Op],
    radius: usize,
    (match_score, mism, open, ext): (i32, i32, i32, i32),
) -> (i32, Vec<Op>) {
    let (qs, rs) = (to, from);
    let fill_range = convert_to_fill_range(qs.len(), rs.len(), ops, radius);
    let lower = (from.len() + to.len()) as i32 * open;
    let mut dp = DPTable::new(&fill_range, (lower, lower, lower));
    // 1. Initialization
    dp.set(0, 0, (0, lower, lower));
    // Deletion is not penalized.
    for j in 1..rs.len() + 1 {
        dp.set(0, j, (lower, lower, 0));
    }
    for i in 1..qs.len() + 1 {
        dp.set(i, 0, (lower, open + (i - 1) as i32 * ext, lower));
    }
    // 2. Recur.
    let rows = fill_range.iter().enumerate().skip(1).zip(qs.iter());
    let param = (match_score, mism, open, ext);
    for ((i, &(start, end)), &q) in rows {
        fill_row((i, q), (start, end, rs), param, &mut dp);
        // The last position, the insertion does not give any penalty.
        if end == rs.len() + 1 {
            let j = end - 1;
            let ins_next = {
                let (mat, ins, _) = dp.get(i - 1, j);
                (mat).max(ins)
            };
            if let Some((_, i, _)) = dp.get_mut(i, j) {
                *i = ins_next;
            }
        }
    }
    // Traceback.
    let mut qpos = qs.len();
    let mut rpos = rs.len();
    assert_eq!(rpos, rs.len());
    let (mut state, score) = get_max_pair(qpos, rpos, &dp);
    let mut ops = Vec::with_capacity(qs.len() + rs.len());
    while 0 < qpos && 0 < rpos && rpos == rs.len() {
        let (mat, ins, del) = dp.get(qpos, rpos);
        if state == 0 {
            let is_mat = qs[qpos - 1] == rs[rpos - 1];
            if is_mat {
                ops.push(Op::Match)
            } else {
                ops.push(Op::Mismatch)
            };
            let aln = if is_mat { match_score } else { mism };
            let current = mat - aln;
            let (m_prev, i_prev, d_prev) = dp.get(qpos - 1, rpos - 1);
            if current == m_prev {
                state = 0;
            } else if current == i_prev {
                state = 1;
            } else {
                assert_eq!(current, d_prev);
                state = 2;
            }
            qpos -= 1;
            rpos -= 1;
        } else if state == 1 {
            ops.push(Op::Ins);
            let current = ins;
            let (m_prev, i_prev, _) = dp.get(qpos - 1, rpos);
            if current == m_prev {
                state = 0;
            } else {
                assert_eq!(current, i_prev, "{}", score);
                state = 1;
            }
            qpos -= 1;
        } else {
            ops.push(Op::Del);
            assert_eq!(state, 2);
            let current = del;
            let (m_prev, i_prev, d_prev) = dp.get(qpos, rpos - 1);
            if current == m_prev + open {
                state = 0;
            } else if current == i_prev + open {
                state = 1;
            } else {
                assert_eq!(current, d_prev + ext);
                state = 2;
            }
            rpos -= 1;
        }
    }
    while 0 < qpos && 0 < rpos {
        let (new_state, op) = track_back_one_op(state, &dp, (qpos, qs), (rpos, rs), param);
        match state {
            0 => {
                qpos -= 1;
                rpos -= 1;
            }
            1 => qpos -= 1,
            _ => rpos -= 1,
        }
        state = new_state;
        ops.push(op);
    }
    ops.extend(std::iter::repeat(Op::Del).take(rpos));
    ops.extend(std::iter::repeat(Op::Ins).take(qpos));
    ops.reverse();
    (score, ops)
}

fn fill_row(
    (i, q): (usize, u8),
    (start, end, rs): (usize, usize, &[u8]),
    (match_score, mism, open, ext): (i32, i32, i32, i32),
    dp: &mut DPTable<(i32, i32, i32)>,
) {
    for j in start.max(1)..end {
        let r = rs[j - 1];
        let aln = if q == r { match_score } else { mism };
        let mat_next = {
            let (mat, ins, del) = dp.get(i - 1, j - 1);
            mat.max(ins).max(del) + aln
        };
        let ins_next = {
            let (mat, ins, _) = dp.get(i - 1, j);
            (mat + open).max(ins + ext)
        };
        let del_next = {
            let (mat, ins, del) = dp.get(i, j - 1);
            (mat + open).max(del + ext).max(ins + open)
        };
        dp.set(i, j, (mat_next, ins_next, del_next));
    }
}

fn get_max_pair(qpos: usize, rpos: usize, dp: &DPTable<(i32, i32, i32)>) -> (u8, i32) {
    let (mat, ins, del) = dp.get(qpos, rpos);
    if ins <= mat && del <= mat {
        (0, mat)
    } else if mat <= ins && del <= ins {
        (1, ins)
    } else {
        assert!(mat <= del && ins <= del);
        (2, del)
    }
}

fn track_back_one_op(
    state: u8,
    dp: &DPTable<(i32, i32, i32)>,
    (qpos, qs): (usize, &[u8]),
    (rpos, rs): (usize, &[u8]),
    (match_score, mism, open, ext): (i32, i32, i32, i32),
) -> (u8, Op) {
    let (mat, ins, del) = dp.get(qpos, rpos);
    if state == 0 {
        let is_mat = qs[qpos - 1] == rs[rpos - 1];
        let op = match is_mat {
            true => Op::Match,
            false => Op::Mismatch,
        };
        let aln = if is_mat { match_score } else { mism };
        let current = mat - aln;
        let (m_prev, i_prev, d_prev) = dp.get(qpos - 1, rpos - 1);
        if current == m_prev {
            (0, op)
        } else if current == i_prev {
            (1, op)
        } else {
            assert_eq!(current, d_prev);
            (2, op)
        }
    } else if state == 1 {
        let (m_prev, i_prev, _) = dp.get(qpos - 1, rpos);
        if ins == m_prev + open {
            (0, Op::Ins)
        } else {
            assert_eq!(ins, i_prev + ext);
            (state, Op::Ins)
        }
    } else {
        assert_eq!(state, 2);
        let current = del;
        let (m_prev, i_prev, d_prev) = dp.get(qpos, rpos - 1);
        if current == m_prev + open {
            (0, Op::Del)
        } else if current == i_prev + open {
            (1, Op::Del)
        } else {
            assert_eq!(current, d_prev + ext);
            (2, Op::Del)
        }
    }
}

const COPY_SIZE: usize = 4;
const DEL_SIZE: usize = 4;
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
        let len = total_len - mod_table.len();
        mod_table.extend(std::iter::repeat(pre.upperbound()).take(len));
    }
    let fills = fill_range.iter().enumerate().zip(qs.iter());
    for ((i, &(start, end)), &q) in fills {
        for (j, mod_table) in mod_table
            .chunks_exact_mut(NUM_ROW)
            .enumerate()
            .take(end)
            .skip(start)
        {
            // for j in start..end {
            let pre_mat = pre.get(i, j);
            let (post_del, post_ins) = (post.get(i, j + 1), post.get(i + 1, j));
            let (post_mat, post_sta) = (post.get(i + 1, j + 1), post.get(i, j));
            let pre_copy_line = pre.get_line(i, j + 1, (rs.len() - j).min(COPY_SIZE));
            let post_del_line = post.get_line(i, j + 1, (rs.len() - j).min(DEL_SIZE));
            // change the j-th base into ...
            let mat_pattern = pre_mat + post_mat;
            let del_pattern = pre_mat + post_del + 1;
            mod_table.iter_mut().zip(b"ACGT").for_each(|(x, &base)| {
                *x = (*x).min(mat_pattern + (q != base) as u32).min(del_pattern);
            });
            // insert before the j-th base...
            let mat_pattern = pre_mat + post_ins;
            let del_pattern = pre_mat + post_sta + 1;
            mod_table
                .iter_mut()
                .skip(4)
                .zip(b"ACGT")
                .for_each(|(x, &base)| {
                    *x = (*x).min(mat_pattern + (base != q) as u32).min(del_pattern);
                });
            // Copying the j..j+c bases..
            mod_table
                .iter_mut()
                .skip(8)
                .zip(pre_copy_line)
                .for_each(|(x, pre_c)| {
                    *x = (*x).min(pre_c + post_sta);
                });
            // Deleting the j..j+d bases...
            mod_table
                .iter_mut()
                .skip(8 + COPY_SIZE)
                .zip(post_del_line)
                .for_each(|(x, post_d)| {
                    *x = (*x).min(pre_mat + post_d);
                });
        }
    }
    // The last position (no base to be aligned!)
    if let Some(&(start, end)) = fill_range.last() {
        let i = fill_range.len() - 1;
        for (j, mod_table) in mod_table
            .chunks_exact_mut(NUM_ROW)
            .enumerate()
            .take(end)
            .skip(start)
        {
            let pre_mat = pre.get(i, j);
            let post_sta = post.get(i, j);
            let post_del = post.get(i, j + 1);
            // change the j-th base into ...
            mod_table
                .iter_mut()
                .take(4)
                .for_each(|x| *x = (*x).min(pre_mat + post_del + 1));
            // insert before the j-th base...
            mod_table
                .iter_mut()
                .skip(4)
                .take(4)
                .for_each(|x| *x = (*x).min(pre_mat + post_sta + 1));
            // Copying the j..j+c bases..
            mod_table
                .iter_mut()
                .skip(8)
                .take(COPY_SIZE)
                .enumerate()
                .filter(|(c, _)| j + c < rs.len())
                .for_each(|(len, x)| *x = (*x).min(pre.get(i, j + len + 1) + post_sta));
            // Deleting the j..j+d bases...
            mod_table
                .iter_mut()
                .skip(8 + COPY_SIZE)
                .take(DEL_SIZE)
                .enumerate()
                .filter(|(d, _)| j + d < rs.len())
                .for_each(|(len, x)| *x = (*x).min(pre_mat + post.get(i, j + len + 1)));
        }
    }
}

struct Aligner {
    pre_dp: DPTable<u32>,
    post_dp: DPTable<u32>,
    default_radius: usize,
    radius: Vec<usize>,
    fill_ranges: Vec<(usize, usize)>,
    mod_table: Vec<u32>,
}

impl Aligner {
    const MIN_RADIUS: usize = 4;
    fn with_capacity(qlen: usize, radius: usize) -> Self {
        Self {
            pre_dp: DPTable::with_capacity(qlen, radius, 3 * qlen as u32),
            post_dp: DPTable::with_capacity(qlen, radius, 3 * qlen as u32),
            fill_ranges: Vec::with_capacity(qlen * 3),
            mod_table: Vec::with_capacity(qlen * 3),
            radius: Vec::with_capacity(qlen * 3),
            default_radius: radius,
        }
    }
    fn re_define_fill_range(&mut self, qlen: usize, rlen: usize, ops: &[Op]) {
        let (mut qpos, mut rpos) = (0, 0);
        let radius = self.radius[rpos];
        let ranges = &mut self.fill_ranges;
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
            let radius = self.radius[rpos];
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
            let radius = self.radius[rpos];
            update_range(ranges, qpos, rpos, qlen, rlen, radius);
        }
        while qpos < qlen {
            qpos += 1;
            let radius = self.radius[rpos];
            update_range(ranges, qpos, rpos, qlen, rlen, radius);
        }
        while rpos < rlen {
            rpos += 1;
            let radius = self.radius[rpos];
            update_range(ranges, qpos, rpos, qlen, rlen, radius);
        }
        assert_eq!(qpos, qlen);
        assert_eq!(rpos, rlen);
        ranges[qlen].1 = rlen + 1;
    }
    fn set_fill_ranges(&mut self, rlen: usize, qlen: usize, ops: &[Op]) {
        self.fill_ranges.clear();
        self.fill_ranges
            .extend(std::iter::repeat((rlen + 1, 0)).take(qlen + 1));
        let fill_len = (rlen + 1).saturating_sub(self.radius.len());
        self.radius
            .extend(std::iter::repeat(self.default_radius).take(fill_len));
        self.re_define_fill_range(qlen, rlen, ops);
    }
    fn align(&mut self, rs: &[u8], qs: &[u8], ops: &mut Vec<Op>) -> u32 {
        self.set_fill_ranges(rs.len(), qs.len(), ops);
        let upperbound = (rs.len() + qs.len() + 3) as u32;
        self.pre_dp.initialize(upperbound, &self.fill_ranges);
        self.post_dp.initialize(upperbound, &self.fill_ranges);
        Self::fill_pre_dp(&mut self.pre_dp, &self.fill_ranges, rs, qs, ops);
        Self::fill_post_dp(&mut self.post_dp, &self.fill_ranges, rs, qs);
        let fr = &self.fill_ranges;
        fill_mod_table(&mut self.mod_table, rs, qs, &self.pre_dp, &self.post_dp, fr);
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
        let &(s, e) = fill_range.first().unwrap();
        for j in s..e {
            dp.set(0, j, j as u32);
        }
        // Skipping the first element in both i and j.
        for ((i, &(start, end)), &q) in fill_range.iter().enumerate().skip(1).zip(qs.iter()) {
            if start == 0 {
                dp.set(i, 0, i as u32);
            }
            for (j, &r) in rs.iter().enumerate().take(end - 1).skip(start.max(1) - 1) {
                let j = j + 1;
                let mat = (q != r) as u32;
                let dist = (dp.get(i - 1, j - 1) + mat)
                    .min(dp.get(i, j - 1) + 1)
                    .min(dp.get(i - 1, j) + 1);
                dp.set(i, j, dist);
            }
        }
        // Traceback
        let mut qpos = qs.len();
        let mut rpos = fill_range.last().unwrap().1 - 1;
        assert_eq!(rpos, rs.len());
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
                let mat = (qs[qpos - 1] != rs[rpos - 1]) as u32;
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
        ops.extend(std::iter::repeat(Op::Del).take(rpos));
        ops.extend(std::iter::repeat(Op::Ins).take(qpos));
        ops.reverse();
    }
    fn fill_post_dp(dp: &mut DPTable<u32>, fill_range: &[(usize, usize)], rs: &[u8], qs: &[u8]) {
        // 1. Initialization. It is OK to consume O(xs.len() + ys.len()) time.
        let &(s, e) = fill_range.last().unwrap();
        for j in s..e {
            dp.set(qs.len(), j, (rs.len() - j) as u32);
        }
        // Skipping the first element in both i and j.
        let fills = fill_range.iter().enumerate().zip(qs.iter()).rev();
        for ((i, &(start, end)), &q) in fills {
            if end == rs.len() + 1 {
                dp.set(i, rs.len(), (qs.len() - i) as u32);
            }
            for (j, &r) in rs.iter().enumerate().take(end).skip(start).rev() {
                let mat = (q != r) as u32;
                let dist = (dp.get(i + 1, j) + 1)
                    .min(dp.get(i, j + 1) + 1)
                    .min(dp.get(i + 1, j + 1) + mat);
                dp.set(i, j, dist);
            }
        }
    }
    fn update_radius(&mut self, updated_position: &[(usize, usize)], len: usize) {
        let orig_len = self.radius.len();
        let mut prev_pos = 0;
        for &(position, op) in updated_position.iter() {
            for pos in prev_pos..position {
                self.radius
                    .push((self.radius[pos] / 2).max(Self::MIN_RADIUS));
            }
            prev_pos = if op < 4 {
                // Mism
                self.radius.push(self.default_radius);
                position + 1
            } else if op < 8 {
                // Insertion
                self.radius.push(self.default_radius);
                self.radius.push(self.default_radius);
                position + 1
            } else if op < 8 + COPY_SIZE {
                // Copying.
                let len = op - 8 + 2;
                self.radius
                    .extend(std::iter::repeat(self.default_radius).take(len));
                position + 1
            } else {
                // Deletion
                position + op - 8 - COPY_SIZE + 1
            };
        }
        // Finally, prev_pos -> prev_len, exact matches.
        for pos in prev_pos..orig_len {
            self.radius
                .push((self.radius[pos] / 2).max(Self::MIN_RADIUS));
        }
        let mut idx = 0;
        self.radius.retain(|_| {
            idx += 1;
            orig_len < idx
        });
        assert_eq!(self.radius.len(), len + 1);
    }
}

fn polish_guided(
    template: &mut Vec<u8>,
    changed_positions: &mut Vec<(usize, usize)>,
    modif_table: &[u32],
    current_dist: u32,
    inactive: usize,
) {
    changed_positions.clear();
    let orig_len = template.len();
    let mut modif_table = modif_table.chunks_exact(NUM_ROW);
    let mut pos = 0;
    while let Some(row) = modif_table.next() {
        let (op, &dist) = row.iter().enumerate().min_by_key(|x| x.1).unwrap();
        if dist < current_dist && pos < orig_len {
            changed_positions.push((pos, op));
            if op < 4 {
                // Mutateion.
                template.push(b"ACGT"[op]);
                pos += 1;
            } else if op < 8 {
                // Insertion before this base.
                template.push(b"ACGT"[op - 4]);
                template.push(template[pos]);
                pos += 1;
            } else if op < 8 + COPY_SIZE {
                // copy from here to here + 1 + (op - 8) base
                let len = (op - 8) + 1;
                for i in pos..(pos + len).min(orig_len) {
                    template.push(template[i]);
                }
                template.push(template[pos]);
                pos += 1;
            } else if op < NUM_ROW {
                // Delete from here to here + 1 + (op - 8 - COPY_SIZE).
                let del_size = op - 8 - COPY_SIZE + 1;
                pos += del_size;
                (0..del_size - 1).filter_map(|_| modif_table.next()).count();
            } else {
                unreachable!()
            }
            for i in pos..(pos + inactive).min(orig_len) {
                template.push(template[i]);
            }
            pos = (pos + inactive).min(orig_len);
            (0..inactive).filter_map(|_| modif_table.next()).count();
        } else if pos < orig_len {
            template.push(template[pos]);
            pos += 1;
        } else if dist < current_dist && (4..8).contains(&op) {
            // Here, we need to consider the last insertion...
            changed_positions.push((pos, op));
            template.push(b"ACGT"[op - 4]);
        }
    }
    let mut idx = 0;
    template.retain(|_| {
        idx += 1;
        orig_len < idx
    });
}

pub fn polish_until_converge_with_take<T, O>(
    template: &[u8],
    xs: &[T],
    ops: &mut [O],
    radius: usize,
    take: usize,
) -> Vec<u8>
where
    T: std::borrow::Borrow<[u8]>,
    O: std::borrow::BorrowMut<Vec<Op>> + std::hash::Hash,
{
    let take = take.min(xs.len());
    let mut template = template.to_vec();
    let len = (template.len() / 2).max(20);
    let mut modif_table = Vec::new();
    let mut changed_pos = Vec::new();
    let mut aligner = Aligner::with_capacity(template.len(), radius);
    let first_dist: Vec<_> = ops
        .iter()
        .map(|ops| ops.borrow().iter().filter(|&&op| op != Op::Match).count())
        .collect();
    for t in 0..100 {
        let inactive = INACTIVE_TIME + (INACTIVE_TIME * t) % len;
        modif_table.clear();
        let dist: u32 = ops
            .iter_mut()
            .zip(xs.iter())
            .enumerate()
            .map(|(i, (ops, seq))| {
                let dist = aligner.align(&template, seq.borrow(), ops.borrow_mut());
                if take <= i {
                    return 0;
                }
                match modif_table.is_empty() {
                    true => modif_table.extend_from_slice(&aligner.mod_table),
                    false => {
                        modif_table
                            .iter_mut()
                            .zip(aligner.mod_table.iter())
                            .for_each(|(x, y)| *x += y);
                    }
                }
                dist
            })
            .sum();
        assert_eq!(template.len() + 1, aligner.radius.len());
        let (temp, cpos) = (&mut template, &mut changed_pos);
        polish_guided(temp, cpos, &modif_table, dist, inactive);
        let edit_path = cpos.iter().map(|&(pos, op)| {
            if op < 4 {
                (pos, crate::op::Edit::Subst)
            } else if op < 8 {
                (pos, crate::op::Edit::Insertion)
            } else if op < 8 + COPY_SIZE {
                (pos, crate::op::Edit::Copy(op - 8 + 1))
            } else {
                (pos, crate::op::Edit::Deletion(op - 8 - COPY_SIZE + 1))
            }
        });
        for ((ops, seq), first_dist) in ops.iter_mut().zip(xs.iter()).zip(first_dist.iter()) {
            let seq = seq.borrow();
            let ops = ops.borrow_mut();
            let dist = ops.iter().filter(|&&op| op != Op::Match).count();
            if dist < 2 * first_dist {
                crate::op::fix_alignment_path(ops, edit_path.clone(), seq.len(), temp.len());
            } else {
                // If the alignment is too diverged, fallback to the default method.
                *ops = bootstrap_ops(temp.len(), seq.len());
                *ops = edit_dist_guided(temp, seq, ops, 3 * radius).1;
            }
        }
        aligner.update_radius(cpos, temp.len());
        if changed_pos.is_empty() {
            break;
        }
    }
    template
}

pub fn polish_until_converge_with<T>(
    template: &[u8],
    xs: &[T],
    ops: &mut [Vec<Op>],
    radius: usize,
) -> Vec<u8>
where
    T: std::borrow::Borrow<[u8]>,
{
    polish_until_converge_with_take(template, xs, ops, radius, xs.len())
}

pub fn polish_until_converge<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    xs: &[T],
    radius: usize,
) -> Vec<u8> {
    let mut ops: Vec<_> = xs
        .iter()
        .map(|seq| bootstrap_ops(template.len(), seq.borrow().len()))
        .collect();
    polish_until_converge_with(template, xs, &mut ops, radius)
}

// #[derive(Debug, Clone)]
// struct Pileup {
//     coverage: usize,
//     ins: [usize; 4],
//     del: usize,
//     subst: [usize; 4],
//     ref_base: Option<u8>,
// }
// impl Pileup {
//     fn new() -> Self {
//         Self {
//             coverage: 0,
//             ins: [0; 4],
//             del: 0,
//             subst: [0; 4],
//             ref_base: None,
//         }
//     }
//     fn consensus_cov(&self, coverage: usize) -> Option<(crate::op::Edit, u8)> {
//         let thr = coverage / 2;
//         let (ins_arg, &ins) = self.ins.iter().enumerate().max_by_key(|x| x.1).unwrap();
//         let (sub_arg, &sub) = self.subst.iter().enumerate().max_by_key(|x| x.1).unwrap();
//         if thr < ins {
//             Some((crate::op::Edit::Insertion, b"ACGT"[ins_arg]))
//         } else if thr < self.del {
//             Some((crate::op::Edit::Deletion(1), b'-'))
//         } else if thr < sub {
//             Some((crate::op::Edit::Subst, b"ACGT"[sub_arg]))
//         } else {
//             Some((crate::op::Edit::Subst, self.ref_base?))
//         }
//     }
//     #[allow(dead_code)]
//     fn consensus(&self) -> Option<(crate::op::Edit, u8)> {
//         let thr = self.coverage / 2;
//         let (ins_arg, &ins) = self.ins.iter().enumerate().max_by_key(|x| x.1).unwrap();
//         let (sub_arg, &sub) = self.subst.iter().enumerate().max_by_key(|x| x.1).unwrap();
//         if thr < ins {
//             Some((crate::op::Edit::Insertion, b"ACGT"[ins_arg]))
//         } else if thr < self.del {
//             Some((crate::op::Edit::Deletion(1), b'-'))
//         } else if thr < sub {
//             Some((crate::op::Edit::Subst, b"ACGT"[sub_arg]))
//         } else {
//             Some((crate::op::Edit::Subst, self.ref_base?))
//         }
//     }
// }

// /// Consensus by pileup. Polished seq & number of base changed.
// pub fn polish_by_pileup<T: std::borrow::Borrow<[u8]>>(
//     draft: &[u8],
//     seqs: &[T],
//     ops: &mut [Vec<Op>],
// ) -> (Vec<u8>, usize) {
//     const COOLDOWN: usize = 5;
//     let mut pileup = vec![Pileup::new(); draft.len() + 1];
//     for (pu, refseq) in pileup.iter_mut().zip(draft.iter()) {
//         pu.ref_base = Some(*refseq);
//     }
//     let coverage = seqs.len();
//     // Register
//     for (seq, ops) in seqs.iter().zip(ops.iter()) {
//         let seq = seq.borrow();
//         let (mut rpos, mut qpos) = (0, 0);
//         for op in ops.iter() {
//             match op {
//                 Op::Del => {
//                     pileup[rpos].del += 1;
//                     rpos += 1;
//                 }
//                 Op::Ins => {
//                     pileup[rpos].ins[LOOKUP_TABLE[seq[qpos] as usize] as usize] += 1;
//                     qpos += 1;
//                 }
//                 _ => {
//                     pileup[rpos].coverage += 1;
//                     pileup[rpos].subst[LOOKUP_TABLE[seq[qpos] as usize] as usize] += 1;
//                     rpos += 1;
//                     qpos += 1;
//                 }
//             }
//         }
//     }
//     pileup.last_mut().unwrap().coverage = seqs.len();
//     // Polish
//     let mut changed_position = vec![];
//     let mut pileup_iter = pileup.iter().zip(draft.iter()).enumerate();
//     let mut consensus = Vec::with_capacity(draft.len());
//     while let Some((pos, (pu, &refseq))) = pileup_iter.next() {
//         let (op, base) = pu.consensus_cov(coverage).unwrap();
//         use crate::op::Edit;
//         match op {
//             Edit::Subst if base == refseq => {
//                 consensus.push(base);
//             }
//             Edit::Subst => {
//                 consensus.push(base);
//                 changed_position.push((pos, op));
//             }
//             Edit::Insertion => {
//                 consensus.push(base);
//                 consensus.push(refseq);
//                 changed_position.push((pos, op));
//             }
//             Edit::Deletion(1) => {
//                 changed_position.push((pos, op));
//             }
//             _ => unreachable!(),
//         }
//         if op != Edit::Subst || base != refseq {
//             for _ in 0..COOLDOWN {
//                 if let Some((_, (_, &refseq))) = pileup_iter.next() {
//                     consensus.push(refseq);
//                 }
//             }
//         }
//     }
//     if let Some((op, base)) = pileup.last().unwrap().consensus_cov(coverage) {
//         if op == crate::op::Edit::Insertion {
//             changed_position.push((draft.len(), op));
//             consensus.push(base);
//         }
//     }
//     // Fix alignments.
//     let rlen = consensus.len();
//     for (ops, seq) in ops.iter_mut().zip(seqs.iter()) {
//         let qlen = seq.borrow().len();
//         let cpos = changed_position.iter().copied();
//         crate::op::fix_alignment_path(ops, cpos, qlen, rlen)
//     }
//     (consensus, changed_position.len())
// }

// /// Consensus by pileup. Polished seq & number of base changed.
// pub fn polish_by_pileup_until<T: std::borrow::Borrow<[u8]>>(
//     draft: &[u8],
//     seqs: &[T],
//     ops: &mut [Vec<Op>],
//     radius: usize,
//     loop_limit: usize,
// ) -> Vec<u8> {
//     let mut consed = draft.to_vec();
//     for _ in 0..loop_limit {
//         let (new, num) = polish_by_pileup(&consed, seqs, ops);
//         for (ops, ys) in ops.iter_mut().zip(seqs.iter()) {
//             *ops = edit_dist_guided(&new, ys.borrow(), ops, radius).1;
//         }
//         consed = new;
//         if num == 0 {
//             break;
//         }
//     }
//     consed
// }

#[cfg(test)]
pub mod test {
    use super::*;
    fn edit_dist_base_ops(xs: &[u8], ys: &[u8]) -> (u32, Vec<Op>) {
        let mut dp = vec![vec![0; ys.len() + 1]; xs.len() + 1];
        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i as u32;
        }
        for j in 0..ys.len() + 1 {
            dp[0][j] = j as u32;
        }
        for (i, x) in xs.iter().enumerate().map(|(i, x)| (i + 1, x)) {
            for (j, y) in ys.iter().enumerate().map(|(j, y)| (j + 1, y)) {
                let mat = (x != y) as u32;
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
                let mat = (xs[i - 1] != ys[j - 1]) as u32;
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
    fn infix_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let (mat, mism, open, ext) = (3, -6, -10, -2);
        let param = (mat, mism, open, ext);
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 100 + 20;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let pad = crate::gen_seq::generate_seq(&mut rng, 10);
            let prof = crate::gen_seq::PROFILE;
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let xs = vec![pad.clone(), xs, pad].concat();
            let (score, ops) = crate::bialignment::global(&xs, &ys, mat, mism, open, ext);
            let (score2, ops2) = infix_guided(&xs, &ys, &ops, 10, param);
            let (xr, ar, yr) = crate::op::recover(&xs, &ys, &ops2);
            eprintln!("{}", std::str::from_utf8(&xr).unwrap());
            eprintln!("{}", std::str::from_utf8(&ar).unwrap());
            eprintln!("{}\n", std::str::from_utf8(&yr).unwrap());
            assert!(ops2.iter().take(7).all(|&x| x == Op::Del), "{:?}", ops2);
            assert!(
                ops2.iter().rev().take(7).all(|&x| x == Op::Del),
                "{:?}",
                ops2
            );
            assert!(score <= score2, "{},{}", score, score2);
        }
    }
    #[test]
    fn overlap_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let (mat, mism, open, ext) = (3, -6, -10, -2);
        let param = (mat, mism, open, ext);
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 100 + 20;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let pad = crate::gen_seq::generate_seq(&mut rng, 10);
            let prof = crate::gen_seq::PROFILE;
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let xs = vec![pad.clone(), xs].concat();
            let ys = vec![ys, pad.clone()].concat();
            let (score, ops) = crate::bialignment::global(&xs, &ys, mat, mism, open, ext);
            let (score2, ops2) = overlap_guided(&xs, &ys, &ops, 10, param);
            let (xr, ar, yr) = crate::op::recover(&xs, &ys, &ops2);
            eprintln!("{}", std::str::from_utf8(&xr).unwrap());
            eprintln!("{}", std::str::from_utf8(&ar).unwrap());
            eprintln!("{}\n", std::str::from_utf8(&yr).unwrap());
            assert!(ops2.iter().take(7).all(|&x| x == Op::Del), "{:?}", ops2);
            assert!(
                ops2.iter().rev().take(7).all(|&x| x == Op::Ins),
                "{:?}",
                ops2
            );
            assert!(score <= score2, "{},{}", score, score2);
        }
    }

    #[test]
    fn modification_table_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        for i in 0..50 {
            println!("{}", i);
            let radius = 20;
            let xslen = rng.gen::<usize>() % 100 + 20;
            let mut xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let prof = crate::gen_seq::PROFILE;
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let (_dist, ops) = edit_dist_base_ops(&xs, &ys);
            let mut al_ops = ops.clone();
            let mut aligner = Aligner::with_capacity(xs.len(), radius);
            for _ in 0..4 {
                aligner.align(&xs, &ys, &mut al_ops);
                let mod_table = &aligner.mod_table;
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
                    for len in (0..COPY_SIZE).filter(|c| j + c < xs.len()) {
                        let xs: Vec<_> =
                            xs[..j + len + 1].iter().chain(&xs[j..]).copied().collect();
                        let (dist, _) = edit_dist_guided(&xs, &ys, &ops, radius);
                        assert_eq!(dist, mod_table[seek + len], "{},{},{}", xs.len(), j, len);
                    }
                }
                // Delete region.
                for j in 0..xs.len() {
                    let seek = j * NUM_ROW + 8 + COPY_SIZE;
                    for len in (0..DEL_SIZE).filter(|d| j + d < xs.len()) {
                        let xs: Vec<_> =
                            xs[..j].iter().chain(&xs[j + len + 1..]).copied().collect();
                        let (dist, _) = edit_dist_guided(&xs, &ys, &ops, radius);
                        assert_eq!(dist, mod_table[seek + len]);
                    }
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
    // #[test]
    // fn polish_test_2() {
    //     for i in 0..100 {
    //         println!("ST\t{}", i);
    //         let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED + i);
    //         let xslen = rng.gen::<usize>() % 100 + 20;
    //         let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
    //         let prof = crate::gen_seq::Profile {
    //             sub: 0.01,
    //             del: 0.01,
    //             ins: 0.01,
    //         };
    //         let yss: Vec<_> = (0..30)
    //             .map(|_| crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof))
    //             .collect();
    //         let template = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
    //         let mut consed = template.clone();
    //         let mut ops: Vec<_> = yss
    //             .iter()
    //             .map(|seq| crate::bialignment::edit_dist_ops(&consed, seq).1)
    //             .collect();
    //         for _ in 0..10 {
    //             let (new, num) = polish_by_pileup(&consed, &yss, &mut ops);
    //             ops = ops
    //                 .iter()
    //                 .zip(yss.iter())
    //                 .map(|(ops, ys)| super::edit_dist_guided(&new, ys, ops, 20).1)
    //                 .collect();
    //             consed = new;
    //             if num == 0 {
    //                 break;
    //             }
    //         }
    //         let (dist, _) = edit_dist_base_ops(&xs, &consed);
    //         let (prev, _) = edit_dist_base_ops(&xs, &template);
    //         if dist != 0 {
    //             eprintln!("{}", std::str::from_utf8(&xs).unwrap());
    //             eprintln!("{}", std::str::from_utf8(&consed).unwrap());
    //         }
    //         assert_eq!(dist, 0, "{},{},{}", i, prev, dist);
    //     }
    // }
}
