use crate::padseq::PadSeq;
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Op {
    Del,
    Ins,
    Mat,
}

impl Op {
    /// Convert this operation by flipping the reference and the query.
    pub fn rev(&self) -> Self {
        match *self {
            Op::Del => Op::Ins,
            Op::Ins => Op::Del,
            Op::Mat => Op::Mat,
        }
    }
}

const fn match_mat() -> [u32; 64] {
    let mut scores = [1_000_000_000; 64];
    let mut x = 0;
    while x < 4 {
        let mut y = 0;
        while y < 4 {
            let mat_score = if x == y { 0 } else { 1 };
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

pub fn edit_dist_slow(xs: &[u8], ys: &[u8]) -> u32 {
    edit_dist_dp_pre(xs, ys)[xs.len()][ys.len()]
}

pub fn edit_dist_dp_pre(xs: &[u8], ys: &[u8]) -> Vec<Vec<u32>> {
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

pub fn edit_dist_dp_post(xs: &[u8], ys: &[u8]) -> Vec<Vec<u32>> {
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

/// Slow edit distance calculation, allowing removing any prefix and any suffix from **ys**.
/// In other words, it allows us to comsume leading bases and drop trailing bases as much as we like.
/// Note that even thought this function is semi-global and so the first argument of the return value is,
/// the returned alignment opration is global. In other words, we have prenty of deletion at the
/// end and start of the returned vector.
pub fn edit_dist_slow_ops_semiglobal(xs: &[u8], ys: &[u8]) -> (u32, Vec<Op>) {
    let mut dp = vec![vec![0; ys.len() + 1]; xs.len() + 1];
    for (i, x) in xs.iter().enumerate() {
        for (j, y) in ys.iter().enumerate() {
            dp[i + 1][j + 1] = (dp[i][j] + (x != y) as u32)
                .min(dp[i][j + 1] + 1)
                .min(dp[i + 1][j] + 1);
        }
    }
    let (mut j, dist) = dp
        .last()
        .unwrap()
        .iter()
        .enumerate()
        .min_by_key(|x| x.1)
        .unwrap();
    let mut i = xs.len();
    let mut ops = vec![Op::Ins; ys.len() - j];
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
            ops.push(Op::Mat);
            i -= 1;
            j -= 1;
        }
    }
    ops.extend(std::iter::repeat(Op::Del).take(i));
    ops.extend(std::iter::repeat(Op::Ins).take(j));
    ops.reverse();
    (*dist, ops)
}

/// Edit distance and its operations.
pub fn edit_dist_slow_ops(xs: &[u8], ys: &[u8]) -> (u32, Vec<Op>) {
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
            ops.push(Op::Mat);
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

// Return edit distance by modifying the j-th position of the `xs` inot `base`.
pub fn edit_dist_with_mutation(
    ys: &[u8],
    pre_dp: &[Vec<u32>],
    post_dp: &[Vec<u32>],
    position: usize,
    base: u8,
) -> u32 {
    let mut modified_row = vec![UPPER_BOUND; ys.len() + 1];
    // Fill the last element.
    modified_row[ys.len()] = post_dp
        .get(position + 1)
        .map(|row| row[ys.len()] + 1)
        .unwrap_or(UPPER_BOUND);
    for (j, &y) in ys.iter().enumerate().rev() {
        let ins = modified_row[j + 1] + 1;
        let del = post_dp[position + 1][j] + 1;
        let mat = post_dp[position + 1][j + 1] + (y != base) as u32;
        modified_row[j] = ins.min(del).min(mat);
    }
    modified_row
        .iter()
        .zip(pre_dp[position].iter())
        .map(|(x, y)| x + y)
        .min()
        .unwrap()
}
// Return edit distance by inserting `base` to the position *before* the j-th character.
pub fn edit_dist_with_insertion(
    ys: &[u8],
    pre_dp: &[Vec<u32>],
    post_dp: &[Vec<u32>],
    position: usize,
    base: u8,
) -> u32 {
    let mut modified_row = vec![UPPER_BOUND; ys.len() + 1];
    modified_row[ys.len()] = 1 + post_dp
        .get(position)
        .and_then(|row| row.get(ys.len()))
        .copied()
        .unwrap_or(UPPER_BOUND);
    for (j, &y) in ys.iter().enumerate().rev() {
        let mat = (y != base) as u32 + post_dp[position][j + 1];
        let del = 1 + post_dp[position][j];
        let ins = 1 + modified_row[j + 1];
        modified_row[j] = mat.min(del).min(ins);
    }
    modified_row
        .iter()
        .zip(pre_dp[position].iter())
        .map(|(x, y)| x + y)
        .min()
        .unwrap()
}

// Return edit distance by deleting the `position`-th base.
pub fn edit_dist_with_deletion(pre_dp: &[Vec<u32>], post_dp: &[Vec<u32>], position: usize) -> u32 {
    post_dp[position + 1]
        .iter()
        .zip(pre_dp[position].iter())
        .map(|(x, y)| x + y)
        .min()
        .unwrap()
}

pub fn polish_until_converge<T: std::borrow::Borrow<[u8]>>(template: &[u8], xs: &[T]) -> Vec<u8> {
    let mut polished = template.to_vec();
    let mut start_position = 0;
    while let Some((imp, pos)) = polish_by_flip(&polished, xs, start_position) {
        polished = imp;
        start_position = pos;
    }
    polished
}

pub fn polish_by_flip<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    xs: &[T],
    start_position: usize,
) -> Option<(Vec<u8>, usize)> {
    let dps: Vec<_> = xs
        .iter()
        .map(|x| {
            let pre_dp = edit_dist_dp_pre(template, x.borrow());
            let post_dp = edit_dist_dp_post(template, x.borrow());
            (x.borrow(), pre_dp, post_dp)
        })
        .collect();
    let current_edit_distance = dps.iter().map(|(_, _, dp)| dp[0][0]).sum::<u32>();
    let mut improved = template.to_vec();
    for pos in 0..template.len() {
        let pos = (start_position + pos) % template.len();
        // Check mutation.
        for &base in b"ACGT" {
            let edit_dist = dps
                .iter()
                .map(|(x, pre, post)| edit_dist_with_mutation(x, pre, post, pos, base))
                .sum::<u32>();
            if edit_dist < current_edit_distance {
                improved[pos] = base;
                return Some((improved, pos));
            }
        }
        // Insertion
        for &base in b"ACGT" {
            let edit_dist = dps
                .iter()
                .map(|(x, pre, post)| edit_dist_with_insertion(x, pre, post, pos, base))
                .sum::<u32>();
            if edit_dist < current_edit_distance {
                improved.insert(pos, base);
                return Some((improved, pos));
            }
        }
        // Deletion
        let edit_dist = dps
            .iter()
            .map(|(_, pre, post)| edit_dist_with_deletion(pre, post, pos))
            .sum::<u32>();
        if edit_dist < current_edit_distance {
            improved.remove(pos);
            return Some((improved, pos));
        }
    }
    None
}

#[derive(Debug, Clone)]
struct DPTable {
    data: Vec<u32>,
    column: usize,
    row: usize,
}

impl DPTable {
    const OFFSET: usize = 3;
    fn new(row: usize, column: usize, init: u32) -> Self {
        Self {
            data: vec![init; (row + 2 * Self::OFFSET) * (column + 2 * Self::OFFSET)],
            column,
            row,
        }
    }
    fn get_location(&self, i: isize, j: isize) -> usize {
        let r = (i + Self::OFFSET as isize) as usize;
        let c = (j + Self::OFFSET as isize) as usize;
        r * (self.column + 2 * Self::OFFSET) + c
    }
    fn get(&self, i: isize, j: isize) -> u32 {
        unsafe { *self.data.get_unchecked(self.get_location(i, j)) }
        // let location = self.get_location(i, j);
        // assert!(location < self.data.len());
        // self.data[self.get_location(i, j)]
    }
    fn get_check(&self, i: isize, j: isize) -> Option<u32> {
        self.data.get(self.get_location(i, j)).copied()
    }
    fn get_mut(&mut self, i: isize, j: isize) -> &mut u32 {
        let location = self.get_location(i, j);
        self.data.get_mut(location).unwrap()
    }
    // Return the i-th row. Exact.
    fn get_row(&self, i: isize) -> &[u32] {
        let row = (i + Self::OFFSET as isize) as usize;
        let collen = self.column + 2 * Self::OFFSET;
        let start = row * collen + Self::OFFSET;
        let end = (row + 1) * collen - Self::OFFSET;
        &self.data[start..end]
    }
    // Return the i-th row. Conaining trailing offsets.
    // fn get_row_pad(&self, i: isize) -> &[u32] {
    //     let row = (i + Self::OFFSET as isize) as usize;
    //     let collen = self.column + 2 * Self::OFFSET;
    //     let start = row * collen + Self::OFFSET;
    //     let end = (row + 1) * collen;
    //     &self.data[start..end]
    // }
}

fn edit_dist_banded_dp_pre(xs: &PadSeq, ys: &PadSeq, radius: usize) -> (Vec<isize>, DPTable) {
    let ub = UPPER_BOUND;
    let mut dp = DPTable::new(xs.len() + ys.len() + 1, 2 * radius + 1, ub);
    let radius = radius as isize;
    // Fill the first diagonal.
    *dp.get_mut(0, radius) = 0;
    // Fill the second diagnal.
    *dp.get_mut(1, radius) = 1;
    *dp.get_mut(1, radius + 1) = 1;
    // Fill diagonals.
    let mut centers = vec![0, 0, 1];
    let (mut center, mut matdiff, mut gapdiff) = (1, 1, 1);
    for k in 2..(xs.len() + ys.len() + 1) as isize {
        let (start, end) = {
            let start = (radius - center).max(k - center + radius - ys.len() as isize);
            let end = (xs.len() as isize + 1 - center + radius).min(k + 1 - center + radius);
            (start.max(0), end.min(2 * radius + 1))
        };
        let (mut min_dist, mut min_pos) = (ub, end);
        for pos in start..end {
            let u = pos + center - radius;
            let y = ys[k - u - 1];
            let x = xs[u - 1];
            let prev_mat = pos as isize + matdiff;
            let prev_gap = pos as isize + gapdiff;
            let mat_pen = MATMAT[(x << 3 | y) as usize];
            let mat = dp.get(k - 2, prev_mat - 1) + mat_pen;
            let del = dp.get(k - 1, prev_gap - 1) + 1;
            let ins = dp.get(k - 1, prev_gap) + 1;
            let dist = mat.min(del).min(ins);
            if dist < min_dist {
                min_dist = dist;
                min_pos = pos;
            }
            *dp.get_mut(k, pos) = dist;
        }
        let min_u = min_pos as isize + center - radius;
        let diff = (center < min_u) as isize;
        center += diff;
        matdiff = gapdiff + diff;
        gapdiff = diff;
        centers.push(center);
    }
    (centers, dp)
}

// Return edit distance by modifying the `postion`-th position of the `xs` inot `base`.
fn edit_dist_banded_with_mutation(
    ys: &PadSeq,
    pre_dp: &DPTable,
    post_dp: &DPTable,
    ranges: &[(isize, isize)],
    position: isize,
    base: u8,
) -> u32 {
    let (start, end) = ranges[position as usize];
    let (prev_start, _) = ranges[position as usize + 1];
    let mut modified_row = vec![UPPER_BOUND; (end - start) as usize];
    let mut prev = UPPER_BOUND;
    let matmat = &MATMAT[(base << 3) as usize..(base << 3 | 0b111) as usize];
    for ((j, &y), x) in ys
        .get_range(start, end)
        .iter()
        .enumerate()
        .zip(modified_row.iter_mut())
        .rev()
    {
        let j = j as isize;
        let next_j = j + start - prev_start;
        let mat_pen = matmat[y as usize];
        let mat = post_dp.get(position + 1, next_j + 1) + mat_pen;
        let del = post_dp.get(position + 1, next_j) + 1;
        let current = mat.min(del).min(prev + 1);
        *x = current;
        prev = current;
    }
    pre_dp
        .get_row(position)
        .iter()
        .zip(modified_row.iter())
        .map(|(x, y)| x + y)
        .min()
        .unwrap_or(UPPER_BOUND)
}

fn edit_dist_banded_with_insertion(
    ys: &PadSeq,
    pre_dp: &DPTable,
    post_dp: &DPTable,
    ranges: &[(isize, isize)],
    position: isize,
    base: u8,
) -> u32 {
    let (start, end) = ranges[position as usize];
    let mut modified_row = vec![UPPER_BOUND; (end - start) as usize];
    let mut prev = UPPER_BOUND;
    let matmat = &MATMAT[(base << 3) as usize..(base << 3 | 6) as usize];
    for ((j, &y), x) in ys
        .get_range(start, end)
        .iter()
        .enumerate()
        .zip(modified_row.iter_mut())
        .rev()
    {
        let j = j as isize;
        let matpen = matmat[y as usize];
        let mat = post_dp.get(position, j + 1) + matpen;
        let del = post_dp.get(position, j) + 1;
        let current = mat.min(del).min(prev + 1);
        *x = current;
        prev = current;
    }
    pre_dp
        .get_row(position)
        .iter()
        .zip(modified_row.iter())
        .map(|(x, y)| x + y)
        .min()
        .unwrap_or(UPPER_BOUND)
}

fn edit_dist_banded_with_deletion(
    pre_dp: &DPTable,
    post_dp: &DPTable,
    ranges: &[(isize, isize)],
    position: isize,
) -> u32 {
    let (start, _) = ranges[position as usize];
    let (next_start, _) = ranges[position as usize + 1];
    pre_dp
        .get_row(position)
        .iter()
        .enumerate()
        .map(|(j, dist)| dist + post_dp.get(position + 1, j as isize + start - next_start))
        .min()
        .unwrap_or(UPPER_BOUND)
}

// Convert diagonl DP into usual orthogonal DP.
// O(L*K*logK)
fn convert_diagonal_to_orthogonal(
    xs: &PadSeq,
    ys: &PadSeq,
    pre_dp: &DPTable,
    post_dp: &DPTable,
    centers: &[isize],
    radius: isize,
) -> (DPTable, DPTable, Vec<(isize, isize)>) {
    // Registered the filled cells.
    let loop_ranges: Vec<_> = centers
        .iter()
        .enumerate()
        .map(|(k, center)| {
            let k = k as isize;
            let start = (radius - center).max(k - center + radius - ys.len() as isize);
            let end = (xs.len() as isize + 1 - center + radius).min(k + 1 - center + radius);
            (k, center, start.max(0), end.min(2 * radius + 1))
        })
        .collect();
    // (longest range so far, current expanding range)
    let mut max_ranges: Vec<_> = vec![((0, 0), (0, 0)); xs.len() + 1];
    for &(k, center, start, end) in loop_ranges.iter() {
        for pos in start..end {
            let x_pos = (pos + center - radius) as usize;
            let y_pos = k - x_pos as isize;
            let ((m_start, m_end), (c_start, c_end)) = max_ranges.get_mut(x_pos).unwrap();
            if *c_end == y_pos {
                // Expanding current range.
                *c_end = y_pos + 1;
            } else if *m_end - *m_start <= *c_end - *c_start {
                // Update if needed.
                *m_start = *c_start;
                *m_end = *c_end;
                *c_start = y_pos;
                *c_end = y_pos + 1;
            }
        }
    }
    let ranges: Vec<_> = max_ranges
        .iter()
        .map(|&((m_start, m_end), (c_start, c_end))| {
            if m_end - m_start <= c_end - c_start {
                (c_start, c_end)
            } else {
                (m_start, m_end)
            }
        })
        .collect();
    // Note that the length of the ranges can be any length.
    let max_range = ranges.iter().map(|(s, e)| (e - s)).max().unwrap() as usize;
    // Maybe we can make it faster here.
    let mut pre_dp_orth = DPTable::new(xs.len() + 1, max_range, UPPER_BOUND);
    let mut post_dp_orth = DPTable::new(xs.len() + 1, max_range, UPPER_BOUND);
    for (k, center, start, end) in loop_ranges {
        let pre_dp = pre_dp.get_row(k);
        let post_dp = post_dp.get_row(k);
        let (start, end) = (start as usize, end as usize);
        for (position, (&pre, &post)) in
            pre_dp.iter().zip(post_dp).enumerate().take(end).skip(start)
        {
            let position = position as isize;
            let x_pos = position + center - radius;
            let y_pos = k - x_pos;
            let (orth_start, orth_end) = ranges[x_pos as usize];
            if orth_start <= y_pos && y_pos < orth_end {
                *pre_dp_orth.get_mut(x_pos, y_pos - orth_start) = pre;
                *post_dp_orth.get_mut(x_pos, y_pos - orth_start) = post;
            }
        }
    }
    (pre_dp_orth, post_dp_orth, ranges)
}

// Input: The position of the filled cell in y coordinate.
// Output: The start/end position of the logest run:[start,end).
// It consumes ypos, so that we can certain that the retrn value is not the index of ypos,
// but the index of the original DP table.
// Note that ypos should not be an emtpry array.
// fn get_range(ypos: Vec<isize>) -> (isize, isize) {
//     assert!(!ypos.is_empty());
//     assert!(ypos.is_sorted());
//     // ypos.sort_unstable();
//     let mut pointer = 0;
//     let (mut max_pointer, mut max_run_length) = (0, 1);
//     while pointer < ypos.len() {
//         let mut reach = pointer + 1;
//         while reach < ypos.len() && ypos[reach - 1] + 1 == ypos[reach] {
//             reach += 1;
//         }
//         if max_run_length < reach - pointer {
//             max_run_length = reach - pointer;
//             max_pointer = pointer;
//         }
//         pointer = reach;
//     }
//     let start = ypos[max_pointer];
//     let end = ypos[max_pointer + max_run_length - 1] + 1;
//     (start, end)
// }

// return "Post" DP.
fn edit_dist_banded_dp_post(xs: &PadSeq, ys: &PadSeq, radius: usize, centers: &[isize]) -> DPTable {
    let ub = UPPER_BOUND;
    let mut dp = DPTable::new(xs.len() + ys.len() + 1, 2 * radius + 1, ub);
    let radius = radius as isize;
    // Fill the last diagonal
    {
        let k = (xs.len() + ys.len()) as isize;
        let u = xs.len() as isize - centers[k as usize] + radius;
        *dp.get_mut(k, u) = 0;
    }
    // Fill the second last diagonal
    {
        let k = (xs.len() + ys.len()) as isize - 1;
        let center = centers[k as usize];
        let (start, end) = {
            let start = (radius - center).max(k - center + radius - ys.len() as isize);
            let end = (xs.len() as isize + 1 - center + radius).min(k + 1 - center + radius);
            (start.max(0), end.min(2 * radius + 1))
        };
        // It should be 1.
        for pos in start..end {
            *dp.get_mut(k, pos) = 1;
        }
    }
    for k in (0..(xs.len() + ys.len() - 1) as isize).rev() {
        // for (k, centers) in centers.windows(3).enumerate().rev() {
        let center = centers[k as usize];
        let matdiff = center - centers[k as usize + 2];
        let gapdiff = center - centers[k as usize + 1];
        let (start, end) = {
            let start = (radius - center).max(k - center + radius - ys.len() as isize);
            let end = (xs.len() as isize + 1 - center + radius).min(k + 1 - center + radius);
            (start.max(0), end.min(2 * radius + 1))
        };
        for pos in start..end {
            let u = pos + center - radius;
            let (x, y) = (xs[u], ys[k - u]);
            let prev_mat = pos as isize + matdiff;
            let prev_gap = pos as isize + gapdiff;
            let mat_pen = MATMAT[(x << 3 | y) as usize];
            let mat = dp.get(k + 2, prev_mat + 1) + mat_pen;
            let ins = dp.get(k + 1, prev_gap + 1) + 1;
            let del = dp.get(k + 1, prev_gap) + 1;
            let dist = mat.min(del).min(ins);
            *dp.get_mut(k, pos) = dist;
        }
    }
    dp
}

pub fn polish_until_converge_banded<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    xs: &[T],
    radius: usize,
) -> Option<Vec<u8>> {
    let xs: Vec<_> = xs.iter().map(|x| PadSeq::new(x.borrow())).collect();
    let mut polished = PadSeq::new(template);
    let mut start_position = 0;
    let mut total_edit_dist = UPPER_BOUND;
    while let Some((imp, pos, dist)) = polish_by_flip_banded(&polished, &xs, start_position, radius)
    {
        start_position = pos;
        polished = imp;
        if total_edit_dist <= dist {
            return None;
        } else {
            total_edit_dist = dist;
        }
    }
    Some(polished.into())
}

// /// Find a candidate x such that
// /// the sum of edit distance from x to queries would
// /// possibly smaller thant that from template.
// /// If the edit distance from x to template is 1,
// /// then that condition is comfirmed.
// /// Also, if this function returns None,
// /// there is no such sequence, for sure.
// /// I recommend you to use this function for a few time,
// /// then polish up by `polish_by_flip_banded`.
// /// This is because there is no theoritically support for halting,
// /// and in some case iterative polishing by this function never stops.
// /// DONOT USE THIS FUNCTION.
// #[allow(dead_code)]
// fn polish_by_batch_banded<T: std::borrow::Borrow<PadSeq>>(
//     template: &PadSeq,
//     queries: &[T],
//     radius: usize,
// ) -> Option<PadSeq> {
//     eprintln!("DO NOT USE THIS FUNCTION");
//     let mut current_edit_distance = 0;
//     let dps: Vec<_> = queries
//         .iter()
//         .filter_map(|query| {
//             let q = query.borrow();
//             let (centers, pre_dp) = edit_dist_banded_dp_pre(template, q, radius);
//             let post_dp = edit_dist_banded_dp_post(template, q, radius, &centers);
//             let radius = radius as isize;
//             let dist = {
//                 let k = (template.len() + q.len()) as isize;
//                 let u = template.len() as isize;
//                 let u_in_dp = u - centers[k as usize] + radius as isize;
//                 pre_dp.get_check(k, u_in_dp)?
//             };
//             current_edit_distance += dist;
//             let (pre_dp, post_dp, ranges) =
//                 convert_diagonal_to_orthogonal(&template, q, &pre_dp, &post_dp, &centers, radius);
//             Some((q, ranges, pre_dp, post_dp))
//         })
//         .collect();
//     let mut improved_sequence = vec![];
//     use crate::padseq;
//     'outer: for (pos, &refr) in template.as_ref().iter().enumerate() {
//         let pos = pos as isize;
//         // Insertion
//         let insertion = b"ACGT".iter().find(|b| {
//             let base = padseq::convert_to_twobit(b);
//             let edit_dist = dps
//                 .iter()
//                 .map(|(x, ranges, pre, post)| {
//                     edit_dist_banded_with_insertion(x, pre, post, ranges, pos, base)
//                 })
//                 .sum::<u32>();
//             edit_dist < current_edit_distance
//         });
//         if let Some(base) = insertion {
//             eprintln!("I:{}", pos);
//             improved_sequence.push(*base);
//         }
//         // Check deletion
//         let edit_dist = dps
//             .iter()
//             .map(|(_, ranges, pre, post)| edit_dist_banded_with_deletion(pre, post, ranges, pos))
//             .sum::<u32>();
//         if edit_dist < current_edit_distance {
//             eprintln!("D:{}", pos);
//             continue 'outer;
//         }
//         // Check mutations
//         let mutation = b"ACGT".iter().find(|b| {
//             let base = padseq::convert_to_twobit(b);
//             let edit_dist = dps
//                 .iter()
//                 .map(|(x, ranges, pre, post)| {
//                     edit_dist_banded_with_mutation(x, pre, post, ranges, pos, base)
//                 })
//                 .sum::<u32>();
//             edit_dist < current_edit_distance
//         });
//         if let Some(base) = mutation {
//             improved_sequence.push(*base);
//             eprintln!("M:{}", pos);
//             continue 'outer;
//         }
//         // Here, we arrive at "match" state.
//         improved_sequence.push(b"ACGT"[refr as usize]);
//     }
//     // Check the last insertion.
//     let insertion = b"ACGT".iter().find(|b| {
//         let pos = template.len() as isize;
//         let base = padseq::convert_to_twobit(b);
//         let edit_dist = dps
//             .iter()
//             .map(|(x, ranges, pre, post)| {
//                 edit_dist_banded_with_insertion(x, pre, post, ranges, pos, base)
//             })
//             .sum::<u32>();
//         edit_dist < current_edit_distance
//     });
//     if let Some(base) = insertion {
//         improved_sequence.push(*base);
//     }
//     let improved_sequence = PadSeq::new(improved_sequence.as_slice());
//     if improved_sequence.as_ref() == template.as_ref() {
//         None
//     } else {
//         Some(improved_sequence)
//     }
// }

/// Find if there is a sequence x such that
/// the edit distance from x to template is 1,
/// and the sum of edit distance from x to queries is
/// smaller than that from template.
pub fn polish_by_flip_banded<T: std::borrow::Borrow<PadSeq>>(
    template: &PadSeq,
    queries: &[T],
    start_position: usize,
    radius: usize,
) -> Option<(PadSeq, usize, u32)> {
    let mut current_edit_distance = 0;
    let dps: Vec<_> = queries
        .iter()
        .filter_map(|query| {
            let q = query.borrow();
            let (centers, pre_dp) = edit_dist_banded_dp_pre(template, q, radius);
            let post_dp = edit_dist_banded_dp_post(template, q, radius, &centers);
            let radius = radius as isize;
            let lk = {
                let k = (template.len() + q.len()) as isize;
                let u = template.len() as isize;
                let u_in_dp = u - centers[k as usize] + radius as isize;
                pre_dp.get_check(k, u_in_dp)?
            };
            current_edit_distance += lk;
            let (pre_dp, post_dp, ranges) =
                convert_diagonal_to_orthogonal(&template, q, &pre_dp, &post_dp, &centers, radius);
            Some((q, ranges, pre_dp, post_dp))
        })
        .collect();
    let mut improved = template.clone();
    use crate::padseq;
    for pos in 0..template.len() {
        let pos = ((start_position + pos) % template.len()) as isize;
        let deletion = dps
            .iter()
            .map(|(_, ranges, pre, post)| edit_dist_banded_with_deletion(pre, post, ranges, pos))
            .sum::<u32>();
        let (mutation, mut_base) = b"ACGT"
            .iter()
            .map(padseq::convert_to_twobit)
            .map(|base| {
                let edit_dist = dps
                    .iter()
                    .map(|(x, ranges, pre, post)| {
                        edit_dist_banded_with_mutation(x, pre, post, ranges, pos, base)
                    })
                    .sum::<u32>();
                (edit_dist, base)
            })
            .min_by_key(|x| x.0)
            .unwrap();
        let (insertion, ins_base) = b"ACGT"
            .iter()
            .map(padseq::convert_to_twobit)
            .map(|base| {
                let edit_dist = dps
                    .iter()
                    .map(|(x, ranges, pre, post)| {
                        edit_dist_banded_with_insertion(x, pre, post, ranges, pos, base)
                    })
                    .sum::<u32>();
                (edit_dist, base)
            })
            .min_by_key(|x| x.0)
            .unwrap();
        let minimum = deletion.min(insertion).min(mutation);
        if minimum < current_edit_distance {
            if minimum == insertion {
                improved.insert(pos, ins_base);
            } else if minimum == deletion {
                improved.remove(pos);
            } else {
                assert_eq!(minimum, mutation);
                improved[pos] = mut_base;
            }
            return Some((improved, pos as usize, minimum));
        }
    }
    let pos = template.len() as isize;
    let insertion = b"ACGT"
        .iter()
        .map(padseq::convert_to_twobit)
        .filter_map(|base| {
            let edit_dist = dps
                .iter()
                .map(|(x, ranges, pre, post)| {
                    edit_dist_banded_with_insertion(x, pre, post, ranges, pos, base)
                })
                .sum::<u32>();
            if edit_dist < current_edit_distance {
                Some((edit_dist, base))
            } else {
                None
            }
        })
        .min_by_key(|x| x.0);
    if let Some((dist, base)) = insertion {
        improved.insert(pos, base);
        return Some((improved, pos as usize, dist));
    }
    None
}

pub fn edit_dist_banded(xs: &[u8], ys: &[u8], radius: usize) -> Option<(u32, Vec<Op>)> {
    assert!(!xs.is_empty());
    assert!(!ys.is_empty());
    let radius = (xs.len() / 2).min(ys.len() / 2).min(radius);
    let xs = PadSeq::new(xs);
    let ys = PadSeq::new(ys);
    // Upper bound.
    let ub = UPPER_BOUND;
    let (centers, dp) = edit_dist_banded_dp_pre(&xs, &ys, radius);
    let radius = radius as isize;
    let edit_dist = {
        let (k, u) = ((xs.len() + ys.len()) as isize, xs.len() as isize);
        let u_in_dp = u + radius - centers[k as usize];
        dp.get_check(k, u_in_dp)?
    };
    let (mut k, mut u) = ((xs.len() + ys.len()) as isize, xs.len() as isize);
    let mut ops = vec![];
    loop {
        let (i, j) = (u, k - u);
        if i == 0 || j == 0 {
            break;
        }
        let center = centers[k as usize];
        let matdiff = center - centers[k as usize - 2];
        let gapdiff = center - centers[k as usize - 1];
        let u_in_dp = u + radius - center;
        let prev_mat = u_in_dp + matdiff;
        let prev_gap = u_in_dp + gapdiff;
        let mat_pen = MATMAT[(xs[i - 1] << 3 | ys[j - 1]) as usize];
        let mat = dp.get_check(k - 2, prev_mat - 1).unwrap_or(ub) + mat_pen;
        let del = dp.get_check(k - 1, prev_gap - 1).unwrap_or(ub) + 1;
        let ins = dp.get_check(k - 1, prev_gap).unwrap_or(ub) + 1;
        let dist = dp.get(k, u_in_dp);
        if dist == mat {
            k -= 2;
            u -= 1;
            ops.push(Op::Mat);
        } else if dist == del {
            k -= 1;
            u -= 1;
            ops.push(Op::Del);
        } else {
            assert_eq!(dist, ins);
            k -= 1;
            ops.push(Op::Ins);
        }
    }
    let (i, j) = (u as usize, (k - u) as usize);
    ops.extend(std::iter::repeat(Op::Del).take(i as usize));
    ops.extend(std::iter::repeat(Op::Ins).take(j as usize));
    ops.reverse();
    Some((edit_dist, ops))
}

#[cfg(test)]
mod test {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    const SEED: u64 = 1293890;
    use super::*;
    #[test]
    fn edist_dist_check() {
        let xs = b"AAGTT";
        let ys = b"AAGT";
        assert_eq!(edit_dist_slow(xs, xs), 0);
        assert_eq!(edit_dist_slow(xs, ys), 1);
        assert_eq!(edit_dist_slow(xs, b""), 5);
        assert_eq!(edit_dist_slow(b"AAAA", b"A"), 3);
        assert_eq!(edit_dist_slow(b"AGCT", b"CAT"), 3);
    }
    #[test]
    fn edit_dist_ops_check() {
        let xs = b"AAGTCA";
        let ys = b"AAGCA";
        let (dist, ops) = edit_dist_slow_ops(xs, xs);
        assert_eq!(dist, 0);
        assert_eq!(ops, vec![Op::Mat; xs.len()]);
        let (dist, ops) = edit_dist_slow_ops(xs, ys);
        assert_eq!(dist, 1);
        assert_eq!(
            ops,
            vec![Op::Mat, Op::Mat, Op::Mat, Op::Del, Op::Mat, Op::Mat]
        );
        let ys = b"ATGCA";
        let xs = b"AAGTCA";
        let (dist, ops) = edit_dist_slow_ops(ys, xs);
        assert_eq!(dist, 2);
        use Op::*;
        let answer = vec![Mat, Mat, Mat, Ins, Mat, Mat];
        assert_eq!(ops, answer);
    }
    #[test]
    fn edit_dist_fast_check_short() {
        let xs = b"TC";
        let ys = b"CAT";
        assert_eq!(edit_dist_slow(xs, ys), edit_dist(xs, ys));
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        for _ in 0..1000 {
            let xslen = rng.gen::<usize>() % 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let yslen = rng.gen::<usize>() % 10;
            let ys = crate::gen_seq::generate_seq(&mut rng, yslen);
            let score = edit_dist_slow(&xs, &ys);
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
            let score = edit_dist_slow(&xs, &ys);
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
            let score = edit_dist_slow(&xs, &ys);
            let fastscore = edit_dist(&xs, &ys);
            assert_eq!(score, fastscore);
        }
    }
    #[test]
    fn edit_dist_banded_check() {
        let xs = b"AAGTT";
        let ys = b"AAGT";
        assert_eq!(edit_dist_banded(xs, xs, 2).unwrap().0, 0);
        let (dist, ops) = edit_dist_banded(xs, ys, 2).unwrap();
        assert_eq!(dist, 1);
        assert_eq!(
            ops,
            vec![vec![Op::Mat; 3], vec![Op::Del], vec![Op::Mat]].concat()
        );
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..1000 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let (score, _ops) = edit_dist_slow_ops(&xs, &ys);
            let (bscore, _bops) = edit_dist_banded(&xs, &ys, 20).unwrap();
            assert_eq!(score, bscore);
            // assert_eq!(ops, bops);
        }
    }
    #[test]
    fn edit_dist_banded_backward_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        let radius = 20;
        for _ in 0..1000 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let xs = PadSeq::new(xs.as_slice());
            let ys = PadSeq::new(ys.as_slice());
            let (centers, dp) = edit_dist_banded_dp_pre(&xs, &ys, radius);
            let bdp = edit_dist_banded_dp_post(&xs, &ys, radius, &centers);
            let score = {
                let k = (xs.len() + ys.len()) as isize;
                let u = xs.len() as isize - centers[k as usize] + radius as isize;
                dp.get(k, u)
            };
            let bscore = {
                let k = 0 as isize;
                let u = 0 - centers[k as usize] + radius as isize;
                bdp.get(k, u)
            };
            assert_eq!(score, bscore);
        }
    }
    #[test]
    fn edit_dist_mutation_banded_diff_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let xs = PadSeq::new(xs.as_slice());
            let ys = PadSeq::new(ys.as_slice());
            let mut xs_mut = xs.clone();
            let (centers, pre_dp) = edit_dist_banded_dp_pre(&xs, &ys, radius);
            let post_dp = edit_dist_banded_dp_post(&xs, &ys, radius, &centers);
            let radius = radius as isize;
            let (pre_dp, post_dp, ranges) =
                convert_diagonal_to_orthogonal(&xs, &ys, &pre_dp, &post_dp, &centers, radius);
            for pos in 0..xs.len() as isize {
                for base in b"ACGT".iter().map(crate::padseq::convert_to_twobit) {
                    let dist =
                        edit_dist_banded_with_mutation(&ys, &pre_dp, &post_dp, &ranges, pos, base);
                    xs_mut[pos] = base;
                    let exact_dist = edit_dist(xs_mut.as_ref(), ys.as_ref());
                    assert_eq!(dist, exact_dist);
                }
                xs_mut[pos] = xs[pos];
            }
        }
    }
    #[test]
    fn edit_dist_insertion_banded_diff_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let xs = PadSeq::new(xs.as_slice());
            let ys = PadSeq::new(ys.as_slice());
            let mut xs_mut = xs.as_ref().to_vec();
            let (centers, pre_dp) = edit_dist_banded_dp_pre(&xs, &ys, radius);
            let post_dp = edit_dist_banded_dp_post(&xs, &ys, radius, &centers);
            let radius = radius as isize;
            let (pre_dp, post_dp, ranges) =
                convert_diagonal_to_orthogonal(&xs, &ys, &pre_dp, &post_dp, &centers, radius);
            for pos in 0..xs.len() as isize + 1 {
                xs_mut.insert(pos as usize, b'A');
                for base in b"ACGT".iter().map(crate::padseq::convert_to_twobit) {
                    let dist =
                        edit_dist_banded_with_insertion(&ys, &pre_dp, &post_dp, &ranges, pos, base);
                    xs_mut[pos as usize] = base;
                    let exact_dist = edit_dist(&xs_mut, ys.as_ref());
                    assert_eq!(dist, exact_dist);
                }
                xs_mut.remove(pos as usize);
            }
        }
    }
    #[test]
    fn edit_dist_deletion_banded_diff_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        let radius = 30;
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let xs = PadSeq::new(xs.as_slice());
            let ys = PadSeq::new(ys.as_slice());
            let mut xs_mut = xs.as_ref().to_vec();
            let (centers, pre_dp) = edit_dist_banded_dp_pre(&xs, &ys, radius);
            let post_dp = edit_dist_banded_dp_post(&xs, &ys, radius, &centers);
            let radius = radius as isize;
            let (pre_dp, post_dp, ranges) =
                convert_diagonal_to_orthogonal(&xs, &ys, &pre_dp, &post_dp, &centers, radius);
            for pos in 0..xs.len() as isize {
                xs_mut.remove(pos as usize);
                let dist = edit_dist_banded_with_deletion(&pre_dp, &post_dp, &ranges, pos);
                let exact_dist = edit_dist(&xs_mut, ys.as_ref());
                assert_eq!(dist, exact_dist);
                xs_mut.insert(pos as usize, xs[pos]);
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
    fn edit_dist_mutation_diff_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let pre_dp = edit_dist_dp_pre(&xs, &ys);
            let post_dp = edit_dist_dp_post(&xs, &ys);
            let mut xs_mut = xs.clone();
            for pos in 0..xs.len() {
                for &base in b"ACGT" {
                    let dist = edit_dist_with_mutation(&ys, &pre_dp, &post_dp, pos, base);
                    xs_mut[pos] = base;
                    let exact_dist = edit_dist(&xs_mut, &ys);
                    assert_eq!(dist, exact_dist);
                }
                xs_mut[pos] = xs[pos];
            }
        }
    }
    #[test]
    fn edit_dist_insertion_diff_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let pre_dp = edit_dist_dp_pre(&xs, &ys);
            let post_dp = edit_dist_dp_post(&xs, &ys);
            let mut xs_mut = xs.clone();
            for pos in 0..xs.len() + 1 {
                xs_mut.insert(pos, b'A');
                for &base in b"ACGT" {
                    let dist = edit_dist_with_insertion(&ys, &pre_dp, &post_dp, pos, base);
                    xs_mut[pos] = base;
                    let exact_dist = edit_dist(&xs_mut, &ys);
                    assert_eq!(dist, exact_dist);
                }
                xs_mut.remove(pos);
            }
        }
    }
    #[test]
    fn edit_dist_deletion_diff_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..100 {
            let xslen = rng.gen::<usize>() % 1000 + 10;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let pre_dp = edit_dist_dp_pre(&xs, &ys);
            let post_dp = edit_dist_dp_post(&xs, &ys);
            for pos in 0..xs.len() {
                let dist = edit_dist_with_deletion(&pre_dp, &post_dp, pos);
                let mut xs = xs.clone();
                xs.remove(pos);
                let exact_dist = edit_dist(&xs, &ys);
                assert_eq!(dist, exact_dist);
            }
        }
    }
    // #[test]
    // fn get_range_test() {
    //     let ypos = vec![0, 1, 2, 3, 4, 5];
    //     let range = get_range(ypos);
    //     assert_eq!(range, (0, 6));
    // }
}
