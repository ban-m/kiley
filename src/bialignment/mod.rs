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

pub fn recover(xs: &[u8], ys: &[u8], ops: &[Op]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut i, mut j) = (0, 0);
    let (mut xr, mut opr, mut yr) = (vec![], vec![], vec![]);
    for op in ops {
        match op {
            Op::Del => {
                xr.push(xs[i]);
                opr.push(b' ');
                yr.push(b' ');
                i += 1;
            }
            Op::Ins => {
                xr.push(b' ');
                opr.push(b' ');
                yr.push(ys[j]);
                j += 1;
            }
            Op::Mat => {
                xr.push(xs[i]);
                yr.push(ys[j]);
                opr.push(if xs[i] == ys[j] { b'|' } else { b'X' });
                i += 1;
                j += 1;
            }
        }
    }
    assert_eq!(i, xs.len());
    assert_eq!(j, ys.len());
    (xr, opr, yr)
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

pub fn get_modification_table_naive(xs: &[u8], ys: &[u8]) -> (u32, Vec<u32>) {
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
            // Deletion.
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

pub fn polish_by_flip<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    xs: &[T],
) -> Option<(Vec<u8>, u32)> {
    let mut current_edit_distance = 0;
    let profile_with_diff = xs
        .iter()
        .map(|query| {
            let query = query.borrow();
            let (dist, prf) = get_modification_table_naive(&template, &query);
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
                    let op = [Op::Mat, Op::Ins, Op::Del][op];
                    (pos, op, base, dist)
                })
        })
        .map(|(pos, op, base, &dist)| {
            let mut template = template.to_vec();
            match op {
                Op::Mat => template[pos] = base,
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
    // Create an empty table.
    // fn with_capacity(row: usize, column: usize, init: u32) -> Self {
    //     let mut data = Vec::with_capacity((row + 2 * Self::OFFSET) * (column + 2 * Self::OFFSET));
    //     let len = column + 2 * Self::OFFSET;
    //     for _ in 0..len * Self::OFFSET {
    //         data.push(init);
    //     }
    //     // data.extend(std::iter::repeat(init).take(len * Self::OFFSET));
    //     Self { data, column, row }
    // }
    // // // Push new row with init value.
    // fn push_row(&mut self, init: u32) {
    //     let len = self.column + 2 * Self::OFFSET;
    //     for _ in 0..len {
    //         self.data.push(init);
    //     }
    //     // self.data.extend(std::iter::repeat(init).take(len));
    // }
    fn get_location(&self, i: isize, j: isize) -> usize {
        let r = (i + Self::OFFSET as isize) as usize;
        let c = (j + Self::OFFSET as isize) as usize;
        r * (self.column + 2 * Self::OFFSET) + c
    }
    fn get(&self, i: isize, j: isize) -> u32 {
        // unsafe { *self.data.get_unchecked(self.get_location(i, j)) }
        *self.data.get(self.get_location(i, j)).unwrap()
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
}

impl std::ops::Index<(isize, isize)> for DPTable {
    type Output = u32;
    fn index(&self, (i, j): (isize, isize)) -> &Self::Output {
        let loc = self.get_location(i, j);
        &self.data[loc]
    }
}
impl std::ops::IndexMut<(isize, isize)> for DPTable {
    fn index_mut(&mut self, (i, j): (isize, isize)) -> &mut Self::Output {
        let loc = self.get_location(i, j);
        &mut self.data[loc]
    }
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

fn convert_dp_to_modification_table(
    xs: &PadSeq,
    ys: &PadSeq,
    radius: usize,
    pre: &DPTable,
    post: &DPTable,
    centers: &[isize],
) -> (u32, Vec<u32>) {
    let radius = radius as isize;
    let dist = post[(0, radius)];
    let mut dists = vec![UPPER_BOUND; 9 * (xs.len() + 1)];
    for (pos, slots) in dists.chunks_exact_mut(9).enumerate().take(xs.len()) {
        let current_pre_row = pre.get_row(pos as isize);
        let center = centers[pos];
        // By adding this offset, you can access to the next position of j.
        let offset = center - centers[pos + 1];
        let pos = pos as isize;
        for j in get_range(radius, ys.len() as isize, center) {
            let pre_score = current_pre_row[j as usize];
            let j_orig = j + center - radius;
            let y_base = ys[j_orig] as usize;
            let mat = &MATMAT[(y_base << 3)..(y_base << 3 | 0b111)];
            let current_mat = post.get(pos, j + 1);
            let current_del = post.get(pos, j);
            for base in 0..4usize {
                let ins_min = (1 + current_del).min(mat[base] + current_mat) + pre_score;
                slots[base + 4] = slots[base + 4].min(ins_min);
            }
            let next_mat = post.get(pos + 1, j + offset + 1);
            let next_del = post.get(pos + 1, j + offset);
            for base in 0..4usize {
                let mat_min = (1 + next_del).min(mat[base] + next_mat) + pre_score;
                slots[base] = slots[base].min(mat_min);
            }
            slots[8] = slots[8].min(pre_score + next_del);
        }
    }
    // The last insertion.
    if let Some((pos, slots)) = dists.chunks_exact_mut(9).enumerate().last() {
        let current_pre_row = pre.get_row(pos as isize);
        let center = centers[pos];
        for j in get_range(radius, ys.len() as isize, center) {
            for base in 0..4usize {
                let j_orig = j + center - radius;
                let y_base = ys[j_orig] as usize;
                let mat = MATMAT[y_base << 3 | base];
                let pre_score = current_pre_row[j as usize];
                slots[4 + base] = slots[4 + base]
                    .min(pre_score + 1 + post.get(pos as isize, j))
                    .min(pre_score + mat + post.get(pos as isize, j + 1));
            }
        }
    }
    (dist, dists)
}

pub fn get_modification_table(xs: &PadSeq, ys: &PadSeq, radius: usize) -> (u32, Vec<u32>, Vec<Op>) {
    // let now = || std::time::Instant::now();
    let (centers, pre) = naive_banded_dp_pre(xs, ys, radius);
    let post = naive_banded_dp_post(xs, ys, radius, &centers);
    let (dist, prf) = convert_dp_to_modification_table(xs, ys, radius, &pre, &post, &centers);
    let aln = convert_dp_to_alignment(xs, ys, &pre, &centers, radius);
    (dist, prf, aln)
}

// Return edit distance by modifying the `postion`-th position of the `xs` inot `base`.
#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
#[allow(dead_code)]
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

// return "Post" DP.
#[allow(dead_code)]
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

/// Polish template sequence by reads, with alignment by radius banded alignment.
pub fn polish_until_converge_banded<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    reads: &[T],
    radius: usize,
) -> Vec<u8> {
    let consensus = PadSeq::new(template);
    let seqs: Vec<_> = reads.iter().map(|x| PadSeq::new(x.borrow())).collect();
    let (mut cons, _) = match polish_by_focused_banded(&consensus, &seqs, radius, 20, 25) {
        Some(res) => res,
        None => return template.to_vec(),
    };
    while let Some((improved, _)) = polish_by_batch_banded(&cons, &seqs, radius, 20) {
        cons = improved;
    }
    cons.into()
}

fn get_range(radius: isize, ylen: isize, center: isize) -> std::ops::Range<isize> {
    let start = radius - center;
    let end = ylen + radius - center;
    start.max(0)..(end.min(2 * radius) + 1)
}

fn naive_banded_dp_pre(xs: &PadSeq, ys: &PadSeq, radius: usize) -> (Vec<isize>, DPTable) {
    let mut dp = DPTable::new(xs.len() + 1, 2 * radius + 1, UPPER_BOUND);
    // The location where the radius-th element is in the original DP table.
    // In other words, if you want to convert the j-th element in the banded DP table into the original coordinate,
    // j + centers[i] - radius would be oK.
    // Inverse convertion is the sam, j_orig + radius - centers[i] would be OK.
    let mut centers = vec![0, 0];
    // Initialize.
    let radius = radius as isize;
    dp[(0, radius)] = 0;
    for j in radius + 1..2 * radius + 1 {
        let j_orig = j - radius - 1;
        if !(0..ys.len() as isize).contains(&j_orig) {
            continue;
        }
        dp[(0, j)] = (j - radius) as u32;
    }
    for (i, &x) in xs.iter().enumerate().map(|(pos, x)| (pos + 1, x)) {
        let (center, prev) = (centers[i], centers[i - 1]);
        for j in get_range(radius, ys.len() as isize, center) {
            let y = ys[j + center - radius - 1];
            let prev_j = j + center - prev;
            let i = i as isize;
            dp[(i, j)] = (dp[(i, j - 1)] + 1)
                .min(dp[(i - 1, prev_j)] + 1)
                .min(dp[(i - 1, prev_j - 1)] + (x != y) as u32);
        }
        centers.push((i * ys.len() / xs.len()) as isize);
    }
    (centers, dp)
}

fn convert_dp_to_alignment(
    xs: &PadSeq,
    ys: &PadSeq,
    dp: &DPTable,
    centers: &[isize],
    radius: usize,
) -> Vec<Op> {
    let mut ops = vec![];
    // Get the last DP cell.
    let radius = radius as isize;
    let mut i = xs.len() as isize;
    let mut j_orig = ys.len() as isize;
    while 0 < i && 0 < j_orig {
        let (center, prev) = (centers[i as usize], centers[i as usize - 1]);
        let j = j_orig + radius - center;
        let j_prev = j + center - prev;
        let current_score = dp[(i, j)];
        if current_score == dp[(i, j - 1)] + 1 {
            ops.push(Op::Ins);
            j_orig -= 1;
        } else if current_score == dp[(i - 1, j_prev)] + 1 {
            ops.push(Op::Del);
            i -= 1;
        } else {
            let x = xs[i - 1];
            let y = ys[j_orig - 1];
            let score = dp[(i - 1, j_prev - 1)] + (x != y) as u32;
            assert_eq!(score, current_score);
            ops.push(Op::Mat);
            i -= 1;
            j_orig -= 1;
        }
    }
    ops.extend(std::iter::repeat(Op::Del).take(i as usize));
    ops.extend(std::iter::repeat(Op::Ins).take(j_orig as usize));
    ops.reverse();
    ops
}

fn naive_banded_dp_post(xs: &PadSeq, ys: &PadSeq, radius: usize, centers: &[isize]) -> DPTable {
    let mut dp = DPTable::new(xs.len() + 1, 2 * radius + 1, UPPER_BOUND);
    // Fill the last element.
    let radius = radius as isize;
    let (xslen, yslen) = (xs.len() as isize, ys.len() as isize);
    dp[(xslen, yslen - centers[xs.len()] + radius)] = 0;
    for j in get_range(radius, yslen, centers[xs.len()]).rev() {
        let j_orig = j + centers[xs.len()] - radius;
        dp[(xslen, j)] = (yslen - j_orig) as u32;
    }
    for (i, &x) in xs.iter().enumerate().rev() {
        let (center, next) = (centers[i], centers[i + 1]);
        for j in get_range(radius, yslen, centers[i]).rev() {
            let j_orig = j + center - radius;
            let j_next = j + center - next;
            let y = ys[j_orig];
            let i = i as isize;
            dp[(i, j)] = (dp[(i, j + 1)] + 1)
                .min(dp[(i + 1, j_next)] + 1)
                .min(dp[(i + 1, j_next + 1)] + (x != y) as u32);
        }
    }
    dp
}

/// Search if there is a sequence x such that the edit distance
/// from x to `queries` is possibly smaller than that of from `template`.
/// Also, it returns the total edit distance from `template` to `queries`.
/// CAUTION: the 2nd argument of the return value is the distance from `template`to `queries`.
/// Not the polished one.
/// Also, it is not garanteed, when the returned value is Some(x), that the total edit
/// distance decreases.
/// In contrast, when the return value is None, it IS garanteed that there is no sequence
/// within distance 1 from `template`, such that the total edit ditance would be smaller.
/// In this function, first we search locations to be flipped,
/// then we perform an exact algorithm in the `windows_size` window around these positions.
/// Note that, if windows are overlapping, they would be merged as long as it
/// is smaller than window_size * 4 length.
pub fn polish_by_focused_banded<T: std::borrow::Borrow<PadSeq>>(
    template: &PadSeq,
    queries: &[T],
    radius: usize,
    skip_size: usize,
    window_size: usize,
) -> Option<(PadSeq, u32)> {
    let mut current_edit_distance = 0;
    let mut alignments = vec![];
    let profile_with_diff = queries
        .iter()
        .map(|query| {
            let query = query.borrow();
            let (dist, prf, aln) = get_modification_table(template, query, radius);
            current_edit_distance += dist;
            alignments.push(aln);
            prf
        })
        .reduce(|mut x, y| {
            x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
            x
        })
        .unwrap();
    // Locate the changed positions.
    let mut profile_with_diff = profile_with_diff.chunks_exact(9).enumerate();
    let mut changed_pos = vec![];
    while let Some((pos, with_diff)) = profile_with_diff.next() {
        if let Some(_dist) = with_diff.iter().find(|&&dist| dist < current_edit_distance) {
            changed_pos.push(pos);
            // Seek `skip_size` so that each variants would be separated.
            for _ in 0..skip_size {
                profile_with_diff.next();
            }
        }
    }
    (!changed_pos.is_empty()).then(|| {
        // debug!("Changed Pos:{:?}", changed_pos);
        let first_pos = changed_pos[0];
        let mut start = first_pos.saturating_sub(window_size);
        let mut end = (first_pos + window_size).min(template.len());
        let mut windows = vec![];
        for &pos in changed_pos.iter().skip(1) {
            if pos.saturating_sub(window_size) < end {
                // Proceed window.
                end = (pos + window_size).min(template.len());
            } else {
                // Go to next window.
                windows.push((start, end));
                start = pos.saturating_sub(window_size);
                end = (pos + window_size).min(template.len());
            }
        }
        windows.push((start, end));
        // debug!("Window:{:?}", windows);
        let windows: Vec<_> = windows
            .iter()
            .flat_map(|&(s, e): &(usize, usize)| {
                if e - s < 4 * window_size {
                    vec![(s, e)]
                } else {
                    // Split them into 2 * window_size window.
                    let start_positions: Vec<_> = (0..)
                        .map(|i| s + i * 2 * window_size)
                        .take_while(|pos| pos + 2 * window_size < e)
                        .collect();
                    let mut splits: Vec<(usize, usize)> =
                        start_positions.windows(2).map(|w| (w[0], w[1])).collect();
                    // Push the last one.
                    let last = *start_positions.last().unwrap();
                    splits.push((last, e));
                    splits
                }
            })
            .collect();
        // debug!("Window:{:?}", windows);
        let polished = polish_in_windows(template, queries, &alignments, &windows);
        // let end2 = std::time::Instant::now();
        // debug!("Windowed:{}", (end2 - profiled).as_millis());
        (polished, current_edit_distance)
    })
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
            Op::Mat => {
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

/// Search if there is a sequence x such that the edit distance
/// from x to `queries` is possibly smaller than that of from `template`.
/// Also, it returns the total edit distance from `template` to `queries`.
/// CAUTION: the 2nd argument of the return value is the distance from `template`to `queries`.
/// Not the polished one.
/// Also, it is not garanteed, when the returned value is Some(x), that the total edit
/// distance decreases.
/// In contrast, when the return value is None, it IS garanteed that there is no sequence
/// within distance 1 from `template`, such that the total edit ditance would be smaller.
/// It run by flippin bases in templates so that edit distance would descrease.
/// It does not perform any flipping when there is flipped base in`skip_size` bp.
pub fn polish_by_batch_banded<T: std::borrow::Borrow<PadSeq>>(
    template: &PadSeq,
    queries: &[T],
    radius: usize,
    skip_size: usize,
) -> Option<(PadSeq, u32)> {
    let mut current_edit_distance = 0;
    let profile_with_diff = queries
        .iter()
        .map(|query| {
            let (dist, prf, _) = get_modification_table(&template, query.borrow(), radius);
            current_edit_distance += dist;
            prf
        })
        .reduce(|mut x, y| {
            x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
            x
        })
        .unwrap();
    let mut improved = template.clone();
    let mut profiles = profile_with_diff.chunks_exact(9).enumerate();
    // # of insertion - # of deletion. You can locate the focal position
    // BY adding this offset to pos.
    let mut offset: isize = 0;
    let mut is_diffed = false;
    let mut changed_pos = vec![];
    while let Some((pos, with_diff)) = profiles.next() {
        // diff = [A,C,G,T,A,C,G,T,-], first four element is for mutation,
        // second four element is for insertion.
        let min_value = with_diff.iter().enumerate().min_by_key(|(_, &dist)| dist);
        let (op, &dist) = min_value.unwrap();
        if dist < current_edit_distance {
            changed_pos.push(pos);
            let pos = pos as isize + offset;
            let (op, base) = (op / 4, op % 4);
            let op = [Op::Mat, Op::Ins, Op::Del][op];
            let base = base as u8;
            is_diffed = true;
            match op {
                Op::Mat => {
                    assert_ne!(base, improved[pos]);
                    improved[pos] = base;
                }
                Op::Del => {
                    offset -= 1;
                    improved.remove(pos);
                }
                Op::Ins => {
                    offset += 1;
                    improved.insert(pos, base);
                }
            }
            for _ in 0..skip_size {
                profiles.next();
            }
        }
    }
    // debug!("Batch:{:?}", changed_pos);
    is_diffed.then(|| (improved, current_edit_distance))
}

/// Find if there is a sequence x such that
/// the edit distance from x to template is 1,
/// and the sum of edit distance from x to queries is
/// smaller than that from template.
pub fn polish_by_flip_banded<T: std::borrow::Borrow<PadSeq>>(
    template: &PadSeq,
    queries: &[T],
    _start_position: usize,
    radius: usize,
) -> Option<(PadSeq, usize, u32)> {
    let mut current_edit_distance = 0;
    let profile_with_diff = queries
        .iter()
        .map(|query| {
            let (dist, prf, _) = get_modification_table(&template, query.borrow(), radius);
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
                    let (op, base) = (op / 4, op % 4);
                    let op = [Op::Mat, Op::Ins, Op::Del][op];
                    (pos, op, base as u8, dist)
                })
        })
        .map(|(pos, op, base, &dist)| {
            let mut template = template.clone();
            match op {
                Op::Mat => template[pos as isize] = base,
                Op::Del => {
                    template.remove(pos as isize);
                }
                Op::Ins => {
                    template.insert(pos as isize, base);
                }
            }
            (template, pos + 1, dist)
        })
}

pub fn edit_dist_banded(xs: &[u8], ys: &[u8], radius: usize) -> Option<(u32, Vec<Op>)> {
    assert!(radius > 0);
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
            assert_eq!(
                dist,
                ins,
                "{}\n{}\n{}",
                radius,
                String::from_utf8(xs.into()).unwrap(),
                String::from_utf8(ys.into()).unwrap(),
            );
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
            let (_, profiles) = get_modification_table_naive(&xs, &ys);
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
    #[test]
    fn edit_dist_modification_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..50 {
            let xslen = rng.gen::<usize>() % 1000 + 50;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let mut xs_mut = xs.clone();
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let ys_normal = ys.clone();
            let (xs, ys) = (PadSeq::new(xs.as_slice()), PadSeq::new(ys.as_slice()));
            let (_, profiles, _) = get_modification_table(&xs, &ys, 50);
            for (pos, diffs) in profiles.chunks_exact(9).enumerate().take(xs.len()) {
                let original = xs_mut[pos];
                for &base in b"ACGT" {
                    xs_mut[pos] = base;
                    let exact_dist = edit_dist(&xs_mut, &ys_normal);
                    let dist = diffs[crate::padseq::convert_to_twobit(&base) as usize];
                    assert_eq!(dist, exact_dist);
                    xs_mut[pos] = original;
                }
                for &base in b"ACGT" {
                    xs_mut.insert(pos, base);
                    let exact_dist = edit_dist(&xs_mut, &ys_normal);
                    let dist = diffs[crate::padseq::convert_to_twobit(&base) as usize + 4];
                    assert_eq!(dist, exact_dist);
                    xs_mut.remove(pos);
                }
                xs_mut.remove(pos);
                let exact_dist = edit_dist(&xs_mut, &ys_normal);
                let dist = diffs[8];
                assert_eq!(dist, exact_dist);
                xs_mut.insert(pos, original);
            }
        }
    }
    #[test]
    fn banded_naive_check() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..50 {
            let xslen = rng.gen::<usize>() % 1000 + 50;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            let exact_dist = edit_dist(&xs, &ys);
            let (xs, ys) = (PadSeq::new(xs.as_slice()), PadSeq::new(ys.as_slice()));
            let radius = 50;
            let (centers, dp) = naive_banded_dp_pre(&xs, &ys, radius);
            let (xslen, yslen) = (xs.len() as isize, ys.len() as isize);
            let j = yslen - centers[xs.len()] + radius as isize;
            let dist = dp[(xslen, j)];
            assert_eq!(dist, exact_dist, "{:?}", dp.get_row(xslen));
            let dp = naive_banded_dp_post(&xs, &ys, radius, &centers);
            let dist = dp[(0, radius as isize)];
            assert_eq!(dist, exact_dist, "{:?}", dp.get_row(0 as isize));
        }
    }
}
