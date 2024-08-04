pub mod banded;
pub mod guided;
use crate::op::Op;
use crate::padseq::PadSeq;

const fn match_mat() -> [u32; 64] {
    let mut scores = [1_000_000_000; 64];
    let mut x = 0;
    while x < 4 {
        let mut y = 0;
        while y < 4 {
            let mat_score = (x != y) as u32;
            scores[x << 3 | y] = mat_score;
            y += 1;
        }
        x += 1;
    }
    scores
}

// Match matrix.
const MATMAT: [u32; 64] = match_mat();

// Upper Bound for edit distance.
const UPPER_BOUND: u32 = 1_000_000_000;

/// Edit distance by diff. Runs O(kn).
/// The input should be shorter than 2^30
pub fn edit_dist(xs: &[u8], ys: &[u8]) -> u32 {
    // dp[d][k] = "the far-reaching position in the diagonal k with edit distance is d"
    // here, we mean diagonal k by offset with x.len()
    // In other words, to specify j = i + k diagonal, we index this diagonal as k + x.len()
    // Search far reaching point from diagonal k and reaching point t.
    fn search_far_reaching(xs: &[u8], ys: &[u8], diag: usize, reach: usize) -> usize {
        let i = reach;
        let j = reach + diag - xs.len();
        xs.iter()
            .skip(i)
            .zip(ys.iter().skip(j))
            .take_while(|(x, y)| x == y)
            .count()
    }
    // This is the filled flag.
    let filled_flag = 0b1 << 31;
    let mut dp: Vec<Vec<u32>> = vec![vec![0; xs.len() + 3]];
    let reach = search_far_reaching(xs, ys, xs.len(), 0);
    if reach == xs.len() && reach == ys.len() {
        return 0;
    }
    dp[0][xs.len()] = reach as u32 ^ filled_flag;
    for d in 1.. {
        let prev = dp.last().unwrap();
        let start = xs.len() - d.min(xs.len());
        let end = xs.len() + d.min(ys.len());
        let mut row = vec![0; xs.len() + d + 3];
        if start == 0 {
            let diagonal = start;
            row[diagonal] = (dp[d - 1][diagonal] + 1).max(dp[d - 1][diagonal + 1] + 1);
        }
        for (diagonal, reach) in row.iter_mut().enumerate().take(end + 1).skip(start.max(1)) {
            *reach = (prev[diagonal - 1])
                .max(prev[diagonal] + 1)
                .max(prev[diagonal + 1] + 1);
        }
        for (diagonal, reach) in row.iter_mut().enumerate().take(end + 1).skip(start) {
            let mut reach_wo_flag =
                ((*reach ^ filled_flag) as usize).min(ys.len() + xs.len() - diagonal);
            reach_wo_flag += search_far_reaching(xs, ys, diagonal, reach_wo_flag);
            let (i, j) = (reach_wo_flag, reach_wo_flag + diagonal - xs.len());
            if i == xs.len() && j == ys.len() {
                return d as u32;
            }
            *reach = reach_wo_flag as u32 ^ filled_flag;
        }
        dp.push(row);
    }
    unreachable!()
}

/// Edit distance and its operations.
pub fn edit_dist_ops(xs: &[u8], ys: &[u8]) -> (u32, Vec<Op>) {
    let dp = edit_dist_dp_pre(xs, ys);
    let (mut i, mut j) = (xs.len(), ys.len());
    let mut ops = vec![];
    while 0 < i && 0 < j {
        let dist = dp[i][j];
        if dist == dp[i - 1][j] + 1 {
            ops.push(Op::Del);
            i -= 1;
        } else if dist == dp[i][j - 1] + 1 {
            ops.push(Op::Ins);
            j -= 1;
        } else {
            let mat_pen = (xs[i - 1] != ys[j - 1]) as u32;
            assert_eq!(dist, dp[i - 1][j - 1] + mat_pen);
            ops.push(if mat_pen == 0 {
                Op::Match
            } else {
                Op::Mismatch
            });
            i -= 1;
            j -= 1;
        }
    }
    ops.extend(std::iter::repeat(Op::Del).take(i));
    ops.extend(std::iter::repeat(Op::Ins).take(j));
    ops.reverse();
    let dist = dp[xs.len()][ys.len()];
    (dist, ops)
}

/// Exact algorithm.
pub fn polish_until_converge<T: std::borrow::Borrow<[u8]>>(template: &[u8], xs: &[T]) -> Vec<u8> {
    let mut polished = template.to_vec();
    let mut current_dist = std::u32::MAX;
    while let Some((imp, dist)) = polish_by_flip(&polished, xs) {
        assert!(dist < current_dist);
        current_dist = dist;
        polished = imp;
    }
    polished
}

fn polish_by_flip<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    xs: &[T],
) -> Option<(Vec<u8>, u32)> {
    let mut current_edit_distance = 0;
    let profile_with_diff = xs
        .iter()
        .map(|query| {
            let query = query.borrow();
            let (dist, prf) = get_modification_table(template, query);
            current_edit_distance += dist;
            prf
        })
        .reduce(|mut x, y| {
            x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
            x
        })
        .unwrap();
    profile_with_diff
        .chunks_exact(9)
        .enumerate()
        .find_map(|(pos, with_diff)| {
            // diff = [A,C,G,T,A,C,G,T,-], first four element is for mutation,
            // second four element is for insertion.
            with_diff
                .iter()
                .enumerate()
                .filter(|&(_, &dist)| dist < current_edit_distance)
                .min_by_key(|x| x.1)
                .map(|(op, dist)| {
                    let (op, base) = (op / 4, b"ACGT"[op % 4]);
                    let op = [Op::Match, Op::Ins, Op::Del][op];
                    (pos, op, base, dist)
                })
        })
        .map(|(pos, op, base, &dist)| {
            let mut template = template.to_vec();
            match op {
                Op::Match | Op::Mismatch => template[pos] = base,
                Op::Del => {
                    template.remove(pos);
                }
                Op::Ins => {
                    template.insert(pos, base);
                }
            }
            (template, dist)
        })
}

fn get_modification_table(xs: &[u8], ys: &[u8]) -> (u32, Vec<u32>) {
    let pre_dp = edit_dist_dp_pre(xs, ys);
    let post_dp = edit_dist_dp_post(xs, ys);
    let dist = pre_dp[xs.len()][ys.len()];
    let mut dists = vec![UPPER_BOUND; 9 * (xs.len() + 1)];
    for (pos, slots) in dists.chunks_exact_mut(9).enumerate().take(xs.len()) {
        let pre_row = &pre_dp[pos];
        // slots.iter_mut().for_each(|x| *x = 0);
        // Mutation for 4 slots, insertion for 4 slots,
        // and deletion for the last slot.
        slots
            .iter_mut()
            .take(4)
            .for_each(|x| *x = pre_row[ys.len()] + 1 + post_dp[pos + 1][ys.len()]);
        slots
            .iter_mut()
            .skip(4)
            .for_each(|x| *x = pre_row[ys.len()] + 1 + post_dp[pos][ys.len()]);
        slots[8] = pre_row[ys.len()] + post_dp[pos + 1][ys.len()];
        for (j, &y) in ys.iter().enumerate() {
            let pre = pre_row[j];
            for base_idx in 0..4 {
                let mat_pen = (b"ACGT"[base_idx] != y) as u32;
                let mat = post_dp[pos][j + 1] + mat_pen;
                let del = post_dp[pos][j] + 1;
                slots[base_idx + 4] = slots[base_idx + 4].min(mat + pre).min(del + pre);
                // Match
                let mat = post_dp[pos + 1][j + 1] + mat_pen;
                let del = post_dp[pos + 1][j] + 1;
                slots[base_idx] = slots[base_idx].min(pre + mat).min(pre + del);
            }
            slots[8] = slots[8].min(pre_row[j] + post_dp[pos + 1][j]);
        }
    }
    // The last insertion.
    if let Some((pos, slots)) = dists.chunks_exact_mut(9).enumerate().last() {
        let pre_row = &pre_dp[pos];
        for (base_idx, &base) in b"ACGT".iter().enumerate() {
            let mut min_ins = pre_row[ys.len()] + 1 + post_dp[pos][ys.len()];
            for (j, &y) in ys.iter().enumerate() {
                let pre = pre_row[j];
                let mat_pen = (y != base) as u32;
                let mat = post_dp[pos][j + 1] + mat_pen;
                let del = post_dp[pos][j] + 1;
                min_ins = min_ins.min(mat + pre).min(del + pre);
            }
            slots[4 + base_idx] = min_ins;
        }
    }
    (dist, dists)
}

fn edit_dist_dp_pre(xs: &[u8], ys: &[u8]) -> Vec<Vec<u32>> {
    let mut dp = vec![vec![0; ys.len() + 1]; xs.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i as u32;
    }
    for (j, x) in dp[0].iter_mut().enumerate() {
        *x = j as u32;
    }
    for (i, x) in xs.iter().enumerate().map(|(i, &x)| (i + 1, x)) {
        for (j, y) in ys.iter().enumerate().map(|(j, &y)| (j + 1, y)) {
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + (x != y) as u32);
        }
    }
    dp
}

fn edit_dist_dp_post(xs: &[u8], ys: &[u8]) -> Vec<Vec<u32>> {
    let mut dp = vec![vec![0; ys.len() + 1]; xs.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[ys.len()] = (xs.len() - i) as u32;
    }
    for (j, x) in dp[xs.len()].iter_mut().enumerate() {
        *x = (ys.len() - j) as u32;
    }
    for (i, x) in xs.iter().enumerate().rev() {
        for (j, y) in ys.iter().enumerate().rev() {
            let mat = (x != y) as u32 + dp[i + 1][j + 1];
            let del = 1 + dp[i][j + 1];
            let ins = 1 + dp[i + 1][j];
            dp[i][j] = mat.min(ins).min(del);
        }
    }
    dp
}

// Polish each windows, concat the reuslt, return the result.
fn polish_in_windows<T: std::borrow::Borrow<PadSeq>>(
    template: &PadSeq,
    queries: &[T],
    alignments: &[Vec<Op>],
    windows: &[(usize, usize)],
) -> PadSeq {
    let template: Vec<u8> = template.clone().into();
    let mut current_position = 0;
    let mut buffer = vec![];
    for &(start, end) in windows {
        buffer.extend_from_slice(&template[current_position..start]);
        // Clip [start..end) region.
        let queries: Vec<_> = queries
            .iter()
            .zip(alignments.iter())
            .map(|(q, aln)| crop_region(q.borrow(), aln, start, end))
            .collect();
        let focal_region = &template[start..end];
        let polished = polish_until_converge(focal_region, &queries);
        buffer.extend(polished);
        current_position = end;
    }
    buffer.extend_from_slice(&template[current_position..]);
    PadSeq::from(buffer)
}

fn crop_region(query: &PadSeq, aln: &[Op], start: usize, end: usize) -> Vec<u8> {
    let (mut rpos, mut qpos) = (0, 0);
    let (mut qstart, mut qend) = (None, None);
    for op in aln.iter() {
        match op {
            Op::Match | Op::Mismatch => {
                if rpos == start {
                    assert!(qstart.is_none());
                    qstart = Some(qpos);
                } else if rpos == end {
                    assert!(qstart.is_some());
                    qend = Some(qpos);
                    break;
                }
                rpos += 1;
                qpos += 1;
            }
            Op::Del => {
                if rpos == start {
                    assert!(qstart.is_none());
                    qstart = Some(qpos);
                } else if rpos == end {
                    assert!(qstart.is_some());
                    qend = Some(qpos);
                    break;
                }
                rpos += 1;
            }
            Op::Ins => {
                qpos += 1;
            }
        }
    }
    if qend.is_none() && rpos == end {
        assert!(qstart.is_some());
        qend = Some(qpos);
    }
    assert!(qend.is_some());
    assert!(qstart.is_some());
    query
        .iter()
        .take(qend.unwrap())
        .skip(qstart.unwrap())
        .map(|&x| b"ACGT"[x as usize])
        .collect()
}

/// Global alignment with affine gap score. xs is the reference and ys is the query.
pub fn global(xs: &[u8], ys: &[u8], mat: i32, mism: i32, open: i32, ext: i32) -> (i32, Vec<Op>) {
    let min = (xs.len() + ys.len()) as i32 * open.min(mism);
    // 0->Match, 1-> Del, 2-> Ins phase.
    let mut dp = vec![vec![vec![min; ys.len() + 1]; xs.len() + 1]; 3];
    // Initialize.
    dp[0][0][0] = 0;
    for i in 1..xs.len() + 1 {
        dp[1][i][0] = open + ext * (i - 1) as i32;
    }
    for j in 1..ys.len() + 1 {
        dp[2][0][j] = open + ext * (j - 1) as i32;
    }
    // Recur.
    for (i, x) in xs.iter().enumerate() {
        let i = i + 1;
        for (j, y) in ys.iter().enumerate() {
            let j = j + 1;
            let mat_score = if x == y { mat } else { mism };
            dp[0][i][j] = (dp[0][i - 1][j - 1] + mat_score)
                .max(dp[1][i - 1][j - 1] + mat_score)
                .max(dp[2][i - 1][j - 1] + mat_score);
            dp[1][i][j] = (dp[0][i - 1][j] + open)
                .max(dp[1][i - 1][j] + ext)
                .max(dp[2][i - 1][j] + open);
            dp[2][i][j] = (dp[0][i][j - 1] + open).max(dp[2][i][j - 1] + ext);
        }
    }
    let (mut xpos, mut ypos) = (xs.len(), ys.len());
    let (mat_end, del_end, ins_end) = (dp[0][xpos][ypos], dp[1][xpos][ypos], dp[2][xpos][ypos]);
    let score = mat_end.max(del_end).max(ins_end);
    let mut state = match score {
        x if x == mat_end => 0,
        x if x == del_end => 1,
        x if x == ins_end => 2,
        _ => panic!(),
    };
    let mut ops = vec![];
    while 0 < xpos && 0 < ypos {
        let current = dp[state][xpos][ypos];
        let (x, y) = (xs[xpos - 1], ys[ypos - 1]);
        let mat_score = if x == y { mat } else { mism };
        let (op, new_state) = if state == 0 {
            let mat = dp[0][xpos - 1][ypos - 1] + mat_score;
            let del = dp[1][xpos - 1][ypos - 1] + mat_score;
            let ins = dp[2][xpos - 1][ypos - 1] + mat_score;
            let mat_op = if x == y { Op::Match } else { Op::Mismatch };
            if current == mat {
                (mat_op, 0)
            } else if current == del {
                (mat_op, 1)
            } else {
                assert_eq!(current, ins);
                (mat_op, 2)
            }
        } else if state == 1 {
            if current == dp[0][xpos - 1][ypos] + open {
                (Op::Del, 0)
            } else if current == dp[1][xpos - 1][ypos] + ext {
                (Op::Del, 1)
            } else {
                assert_eq!(current, dp[2][xpos - 1][ypos] + open);
                (Op::Del, 2)
            }
        } else {
            assert_eq!(state, 2);
            if current == dp[0][xpos][ypos - 1] + open {
                (Op::Ins, 0)
            } else {
                assert_eq!(current, dp[2][xpos][ypos - 1] + ext);
                (Op::Ins, 2)
            }
        };
        match op {
            Op::Del => xpos -= 1,
            Op::Ins => ypos -= 1,
            Op::Mismatch | Op::Match => {
                xpos -= 1;
                ypos -= 1;
            }
        }
        ops.push(op);
        state = new_state;
    }
    ops.extend(std::iter::repeat(Op::Del).take(xpos));
    ops.extend(std::iter::repeat(Op::Ins).take(ypos));
    ops.reverse();
    (score, ops)
}

/// Local alignment with affine gap score. xs is the reference and ys is the query.
/// The returned value is (score, xstart, xend, ystart, yend, operations)
type LocalResult = (i32, usize, usize, usize, usize, Vec<Op>);
pub fn local(xs: &[u8], ys: &[u8], mat: i32, mism: i32, open: i32, ext: i32) -> LocalResult {
    // 0->Match, 1-> Del, 2-> Ins phase.
    // Initialize.
    let mut dp = vec![vec![vec![0; ys.len() + 1]; xs.len() + 1]; 3];
    // current max.
    let mut max = 0;
    let mut argmax = (0, 0, 0);
    // Recur.
    for (i, x) in xs.iter().enumerate() {
        let i = i + 1;
        for (j, y) in ys.iter().enumerate() {
            let j = j + 1;
            let mat_score = if x == y { mat } else { mism };
            dp[0][i][j] = (dp[0][i - 1][j - 1] + mat_score)
                .max(dp[1][i - 1][j - 1] + mat_score)
                .max(dp[2][i - 1][j - 1] + mat_score)
                .max(0);
            dp[1][i][j] = (dp[0][i - 1][j] + open)
                .max(dp[1][i - 1][j] + ext)
                .max(dp[2][i - 1][j] + open)
                .max(0);
            dp[2][i][j] = (dp[0][i][j - 1] + open).max(dp[2][i][j - 1] + ext).max(0);
            for (s, inner_dp) in dp.iter().enumerate() {
                if max < inner_dp[i][j] {
                    max = inner_dp[i][j];
                    argmax = (s, i, j);
                }
            }
        }
    }
    let (mut state, xend, yend) = argmax;
    let (mut xpos, mut ypos) = (xend, yend);
    let score = max;
    let mut ops = vec![];
    while dp[state][xpos][ypos] != 0 {
        let current = dp[state][xpos][ypos];
        let (x, y) = (xs[xpos - 1], ys[ypos - 1]);
        let mat_score = if x == y { mat } else { mism };
        let (op, new_state) = if state == 0 {
            let mat = dp[0][xpos - 1][ypos - 1] + mat_score;
            let del = dp[1][xpos - 1][ypos - 1] + mat_score;
            let ins = dp[2][xpos - 1][ypos - 1] + mat_score;
            let mat_op = if x == y { Op::Match } else { Op::Mismatch };
            if current == mat {
                (mat_op, 0)
            } else if current == del {
                (mat_op, 1)
            } else {
                assert_eq!(current, ins);
                (mat_op, 2)
            }
        } else if state == 1 {
            if current == dp[0][xpos - 1][ypos] + open {
                (Op::Del, 0)
            } else if current == dp[1][xpos - 1][ypos] + ext {
                (Op::Del, 1)
            } else {
                assert_eq!(current, dp[2][xpos - 1][ypos] + open);
                (Op::Del, 2)
            }
        } else {
            assert_eq!(state, 2);
            if current == dp[0][xpos][ypos - 1] + open {
                (Op::Ins, 0)
            } else {
                assert_eq!(current, dp[2][xpos][ypos - 1] + ext);
                (Op::Ins, 2)
            }
        };
        match op {
            Op::Del => xpos -= 1,
            Op::Ins => ypos -= 1,
            Op::Mismatch | Op::Match => {
                xpos -= 1;
                ypos -= 1;
            }
        }
        ops.push(op);
        state = new_state;
    }
    ops.reverse();
    (score, xpos, xend, ypos, yend, ops)
}

#[cfg(test)]
mod test {
    // Edist distance, exact.
    fn edit_dist_base(xs: &[u8], ys: &[u8]) -> u32 {
        edit_dist_base_ops(xs, ys).0
    }
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
        ops.extend(std::iter::repeat(Op::Ins).take(i));
        ops.extend(std::iter::repeat(Op::Del).take(j));
        ops.reverse();
        (dp[xs.len()][ys.len()], ops)
    }
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    const SEED: u64 = 1293890;
    use super::*;
    #[test]
    fn edit_dist_check() {
        let xs = b"AAGTT";
        let ys = b"AAGT";
        assert_eq!(edit_dist_base(xs, xs), 0);
        assert_eq!(edit_dist_base(xs, ys), 1);
        assert_eq!(edit_dist_base(xs, b""), 5);
        assert_eq!(edit_dist_base(b"AAAA", b"A"), 3);
        assert_eq!(edit_dist_base(b"AGCT", b"CAT"), 3);
    }
    #[test]
    fn edit_dist_ops_check() {
        let xs = b"AAGTCA";
        let ys = b"AAGCA";
        let (dist, ops) = edit_dist_base_ops(xs, xs);
        assert_eq!(dist, 0);
        assert_eq!(ops, vec![Op::Match; xs.len()]);
        let (dist, ops) = edit_dist_base_ops(xs, ys);
        assert_eq!(dist, 1);
        assert_eq!(
            ops,
            vec![
                Op::Match,
                Op::Match,
                Op::Match,
                Op::Del,
                Op::Match,
                Op::Match
            ]
        );
        let ys = b"ATGCA";
        let xs = b"AAGTCA";
        let (dist, ops) = edit_dist_base_ops(ys, xs);
        assert_eq!(dist, 2);
        use Op::*;
        let answer = vec![Match, Mismatch, Match, Ins, Match, Match];
        assert_eq!(ops, answer);
    }
    #[test]
    fn edit_dist_fast_check_short() {
        let xs = b"TC";
        let ys = b"CAT";
        assert_eq!(edit_dist_base(xs, ys), edit_dist(xs, ys));
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        for _ in 0..1000 {
            let xslen = rng.gen::<usize>() % 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let yslen = rng.gen::<usize>() % 10;
            let ys = crate::gen_seq::generate_seq(&mut rng, yslen);
            let score = edit_dist_base(&xs, &ys);
            let fastscore = edit_dist(&xs, &ys);
            let xs = String::from_utf8_lossy(&xs);
            let ys = String::from_utf8_lossy(&ys);
            assert_eq!(score, fastscore, "{},{}", xs, ys);
        }
    }
    #[test]
    fn edit_dist_fast_check() {
        let xs = b"AAGTT";
        let ys = b"AAGT";
        assert_eq!(edit_dist(xs, xs), 0);
        assert_eq!(edit_dist(xs, ys), 1);
        assert_eq!(edit_dist(xs, b""), 5);
        assert_eq!(edit_dist(b"AAAA", b"A"), 3);
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        for _ in 0..1000 {
            let xslen = rng.gen::<usize>() % 1000;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let yslen = rng.gen::<usize>() % 1000;
            let ys = crate::gen_seq::generate_seq(&mut rng, yslen);
            let score = edit_dist_base(&xs, &ys);
            let fastscore = edit_dist(&xs, &ys);
            assert_eq!(score, fastscore);
        }
    }
    #[test]
    fn edit_dist_fast_check_sim() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..1000 {
            let xslen = rng.gen::<usize>() % 1000;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let score = edit_dist_base(&xs, &ys);
            let fastscore = edit_dist(&xs, &ys);
            assert_eq!(score, fastscore);
        }
    }
    #[test]
    fn naive_edit_dist_modification_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        // let prof = crate::gen_seq::PROFILE;
        for _ in 0..500 {
            let xslen = rng.gen::<usize>() % 10 + 5;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let mut xs_mut = xs.clone();
            let ys = crate::gen_seq::generate_seq(&mut rng, xslen);
            eprintln!("{}", String::from_utf8_lossy(&xs));
            eprintln!("{}", String::from_utf8_lossy(&ys));
            let (_, profiles) = get_modification_table(&xs, &ys);
            for (pos, diffs) in profiles.chunks_exact(9).enumerate().take(xs.len()) {
                let original = xs_mut[pos];
                for (idx, &base) in b"ACGT".iter().enumerate() {
                    xs_mut[pos] = base;
                    assert_eq!(edit_dist(&xs_mut, &ys), diffs[idx], "{},{}", pos, xs.len());
                    xs_mut[pos] = original;
                }
                for (idx, &base) in b"ACGT".iter().enumerate() {
                    xs_mut.insert(pos, base);
                    assert_eq!(edit_dist(&xs_mut, &ys), diffs[idx + 4]);
                    xs_mut.remove(pos);
                }
                xs_mut.remove(pos);
                let exact_dist = edit_dist(&xs_mut, &ys);
                let dist = diffs[8];
                assert_eq!(dist, exact_dist, "{},{}", pos, xs.len());
                xs_mut.insert(pos, original);
            }
        }
    }
    #[test]
    fn edit_dist_post_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..1000 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let dp = edit_dist_dp_pre(&xs, &ys);
            let bdp = edit_dist_dp_post(&xs, &ys);
            assert_eq!(dp[xs.len()][ys.len()], bdp[0][0]);
        }
    }

    #[test]
    fn global_affine_check() {
        let xs = b"AAAA";
        let ys = b"AAAA";
        let (score, ops) = global(xs, ys, 1, -1, -3, -1);
        assert_eq!(score, 4);
        assert_eq!(ops, vec![Op::Match; 4]);
        let (score, ops) = global(b"", b"", 1, -1, -3, -1);
        assert_eq!(score, 0);
        assert_eq!(ops, vec![]);
        let xs = b"ACGT";
        let ys = b"ACCT";
        let (score, ops) = global(xs, ys, 1, -1, -3, -1);
        assert_eq!(score, 2);
        assert_eq!(ops, vec![Op::Match, Op::Match, Op::Mismatch, Op::Match]);
        let xs = b"ACTGT";
        let ys = b"ACGT";
        let (score, ops) = global(xs, ys, 1, -1, -3, -1);
        assert_eq!(score, 1);
        assert_eq!(
            ops,
            vec![Op::Match, Op::Match, Op::Del, Op::Match, Op::Match,]
        );
        let xs = b"ACGT";
        let ys = b"ACTGT";
        let (score, ops) = global(xs, ys, 1, -1, -3, -1);
        assert_eq!(score, 1);
        assert_eq!(
            ops,
            vec![Op::Match, Op::Match, Op::Ins, Op::Match, Op::Match,]
        );
        let xs = b"ACTTTGT";
        let ys = b"ACGT";
        let (score, ops) = global(xs, ys, 1, -1, -3, -1);
        assert_eq!(score, -1);
        assert_eq!(
            ops,
            vec![
                Op::Match,
                Op::Match,
                Op::Del,
                Op::Del,
                Op::Del,
                Op::Match,
                Op::Match,
            ]
        );
    }
}
