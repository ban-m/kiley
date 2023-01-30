use super::MATMAT;
use crate::op::Op;
use crate::padseq::PadSeq;

#[derive(Debug, Clone)]
struct DPTable {
    data: Vec<u32>,
    column: usize,
    #[allow(dead_code)]
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
        *self.data.get(self.get_location(i, j)).unwrap()
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

fn convert_dp_to_modification_table(
    xs: &PadSeq,
    ys: &PadSeq,
    radius: usize,
    pre: &DPTable,
    post: &DPTable,
    centers: &[isize],
) -> (u32, Vec<u32>) {
    let radius = radius as isize;
    let upper_bound = (xs.len() + ys.len() + 100) as u32;
    let dist = post[(0, radius)];
    let mut dists = vec![upper_bound; 9 * (xs.len() + 1)];
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

fn get_modification_table(xs: &PadSeq, ys: &PadSeq, radius: usize) -> (u32, Vec<u32>, Vec<Op>) {
    let (centers, pre) = dp_pre(xs, ys, radius);
    let post = dp_post(xs, ys, radius, &centers);
    let (dist, prf) = convert_dp_to_modification_table(xs, ys, radius, &pre, &post, &centers);
    let aln = convert_dp_to_alignment(xs, ys, &pre, &centers, radius);
    (dist, prf, aln)
}

/// Polish template sequence by reads, with alignment by radius banded alignment.
pub fn polish_until_converge<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    reads: &[T],
    radius: usize,
) -> Vec<u8> {
    let consensus = PadSeq::new(template);
    let seqs: Vec<_> = reads.iter().map(|x| PadSeq::new(x.borrow())).collect();
    let (mut cons, _) = match polish_by_focused(&consensus, &seqs, radius, 20, 25) {
        Some(res) => res,
        None => return template.to_vec(),
    };
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut skip_size = 7;
    let mut loop_count = 0;
    let mut hash_values_sofar = vec![];
    while let Some((improved, _)) = polish_with_interval(&cons, &seqs, radius, skip_size) {
        let mut hasher = DefaultHasher::new();
        improved.as_ref().hash(&mut hasher);
        let hv = hasher.finish();
        if hash_values_sofar.contains(&hv) || 1_000 < loop_count {
            break;
        } else {
            skip_size += 3;
            loop_count += 1;
            cons = improved;
            hash_values_sofar.push(hv);
        }
    }
    cons.into()
}

fn get_range(radius: isize, ylen: isize, center: isize) -> std::ops::Range<isize> {
    let start = radius - center;
    let end = ylen + radius - center;
    start.max(0)..(end.min(2 * radius) + 1)
}

fn dp_pre(xs: &PadSeq, ys: &PadSeq, radius: usize) -> (Vec<isize>, DPTable) {
    let upper_bound = (xs.len() + ys.len() + 1000) as u32;
    let mut dp = DPTable::new(xs.len() + 1, 2 * radius + 1, upper_bound);
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
            ops.push(if x == y { Op::Match } else { Op::Mismatch });
            i -= 1;
            j_orig -= 1;
        }
    }
    ops.extend(std::iter::repeat(Op::Del).take(i as usize));
    ops.extend(std::iter::repeat(Op::Ins).take(j_orig as usize));
    ops.reverse();
    ops
}

fn dp_post(xs: &PadSeq, ys: &PadSeq, radius: usize, centers: &[isize]) -> DPTable {
    let upper_bound = (xs.len() + ys.len() + 100) as u32;
    let mut dp = DPTable::new(xs.len() + 1, 2 * radius + 1, upper_bound);
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
pub(crate) fn polish_by_focused<T: std::borrow::Borrow<PadSeq>>(
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
        use super::polish_in_windows;
        let polished = polish_in_windows(template, queries, &alignments, &windows);
        (polished, current_edit_distance)
    })
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
pub(crate) fn polish_with_interval<T: std::borrow::Borrow<PadSeq>>(
    template: &PadSeq,
    queries: &[T],
    radius: usize,
    skip_size: usize,
) -> Option<(PadSeq, u32)> {
    let mut current_edit_distance = 0;
    let profile_with_diff = queries
        .iter()
        .map(|query| {
            let (dist, prf, _) = get_modification_table(template, query.borrow(), radius);
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
            let op = [Op::Match, Op::Ins, Op::Del][op];
            let base = base as u8;
            is_diffed = true;
            match op {
                Op::Match | Op::Mismatch => {
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
    is_diffed.then_some((improved, current_edit_distance))
}

/// Global alignment with banded alignment. In this algorithm, the band is "fixed", not adaptive(currently).
pub fn global_banded(
    reference: &[u8],
    query: &[u8],
    mat: i32,
    mism: i32,
    open: i32,
    ext: i32,
    band: usize,
) -> (i32, Vec<Op>) {
    let (xs, ys) = (reference, query);
    match (xs.is_empty(), ys.is_empty()) {
        (true, true) => return (0, vec![]),
        (true, false) => return (open + (ys.len() - 1) as i32 * ext, vec![Op::Ins; ys.len()]),
        (false, true) => return (open + (xs.len() - 1) as i32 * ext, vec![Op::Del; xs.len()]),
        _ => {}
    }
    // We fill the [center-band..center+band](2*band) cells for each reference base,
    // resulting to 2*band*xs.len()*3 cells in total.
    // here, the center is i * ys.len()/xs.len() in the i-th row.
    let min = (xs.len() + ys.len()) as i32 * open.min(mism);
    let row_offset = 2 * band * 3;
    let mut dp = vec![min; 2 * band * (xs.len() + 1) * 3];
    // To access the [s][i][j] location in the "full" DP,
    // (2 * band * 3) * i + 3 * (j + band - center) + s
    macro_rules! get {
        ($i:expr, $j:expr, $s:expr, $center:expr) => {
            row_offset * $i + 3 * ($j + band - $center) + $s
        };
    }
    // Initialization(Match).
    dp[get!(0, 0, 0, 0)] = 0;
    // Initialization(Ins).
    for j in 1..band {
        dp[get!(0, j, 2, 0)] = open + (j - 1) as i32 * ext;
    }
    // Recur.
    // Start and end index of the previous row.
    let (mut start, mut end) = (0, band);
    for (i, &x) in xs.iter().enumerate() {
        let i = i + 1;
        let center = i * ys.len() / xs.len();
        for j in center.max(band) - band..(center + band).min(ys.len() + 1) {
            // Initialization(Lazy). We should `continue` loop, as the index would be invalid in i-1,j-1.
            if j == 0 {
                dp[get!(i, j, 1, center)] = open + (i - 1) as i32 * ext;
                continue;
            }
            // Recur(Usual).
            let y = ys[j - 1];
            let mat_score = if x == y { mat } else { mism };
            let prev_center = (i - 1) * ys.len() / xs.len();
            let fill_pos = get!(i, j, 0, center);
            dp[fill_pos] = if (start..end).contains(&(j - 1)) {
                let mat_pos = get!(i - 1, j - 1, 0, prev_center);
                dp[mat_pos].max(dp[mat_pos + 1]).max(dp[mat_pos + 2]) + mat_score
            } else {
                min
            };
            dp[fill_pos + 1] = if (start..end).contains(&j) {
                let del_pos = get!(i - 1, j, 0, prev_center);
                (dp[del_pos] + open)
                    .max(dp[del_pos + 2] + open)
                    .max(dp[del_pos + 1] + ext)
            } else {
                min
            };
            dp[fill_pos + 2] = if j != center.max(band) - band {
                let ins_pos = get!(i, j - 1, 0, center);
                (dp[ins_pos] + open).max(dp[ins_pos + 2] + ext)
            } else {
                min
            };
        }
        start = center.max(band) - band;
        end = center + band;
    }
    // Trackback.
    let (mut xpos, mut ypos) = (xs.len(), ys.len());
    let end_pos = get!(xpos, ypos, 0, xpos * ys.len() / xs.len());
    let (mut state, score) = dp[end_pos..end_pos + 3]
        .iter()
        .enumerate()
        .max_by_key(|x| x.1)
        .unwrap();
    let mut ops = vec![];
    while 0 < xpos && 0 < ypos {
        let center = xpos * ys.len() / xs.len();
        let prev_center = (xpos - 1) * ys.len() / xs.len();
        let current = dp[get!(xpos, ypos, state, center)];
        let (x, y) = (xs[xpos - 1], ys[ypos - 1]);
        let mat_score = if x == y { mat } else { mism };
        let (op, new_state) = if state == 0 {
            let mat_pos = get!(xpos - 1, ypos - 1, 0, prev_center);
            let mat_op = if x == y { Op::Match } else { Op::Mismatch };
            if dp[mat_pos] + mat_score == current {
                (mat_op, 0)
            } else if dp[mat_pos + 1] + mat_score == current {
                (mat_op, 1)
            } else {
                assert_eq!(dp[mat_pos + 2] + mat_score, current);
                (mat_op, 2)
            }
        } else if state == 1 {
            let del_pos = get!(xpos - 1, ypos, 0, prev_center);
            if dp[del_pos] + open == current {
                (Op::Del, 0)
            } else if dp[del_pos + 1] + ext == current {
                (Op::Del, 1)
            } else {
                assert_eq!(dp[del_pos + 2] + open, current);
                (Op::Del, 2)
            }
        } else {
            assert_eq!(state, 2);
            let ins_pos = get!(xpos, ypos - 1, 0, center);
            if dp[ins_pos] + open == current {
                (Op::Ins, 0)
            } else {
                assert_eq!(dp[ins_pos + 2] + ext, current);
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
    (*score, ops)
}

#[cfg(test)]
mod tests {
    const SEED: u64 = 394820;
    use super::*;
    use crate::bialignment::edit_dist;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
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
    fn global_affine_banded_check() {
        let xs = b"AAAA";
        let ys = b"AAAA";
        let (score, ops) = global_banded(xs, ys, 1, -1, -3, -1, 2);
        assert_eq!(score, 4);
        assert_eq!(ops, vec![Op::Match; 4]);
        let (score, ops) = global_banded(b"", b"", 1, -1, -3, -1, 1);
        assert_eq!(score, 0);
        assert_eq!(ops, vec![]);
        let xs = b"ACGT";
        let ys = b"ACCT";
        let (score, ops) = global_banded(xs, ys, 1, -1, -3, -1, 2);
        assert_eq!(score, 2);
        assert_eq!(ops, vec![Op::Match, Op::Match, Op::Mismatch, Op::Match,]);
        let xs = b"ACTGT";
        let ys = b"ACGT";
        let (score, ops) = global_banded(xs, ys, 1, -1, -3, -1, 2);
        assert_eq!(score, 1);
        assert_eq!(
            ops,
            vec![Op::Match, Op::Match, Op::Del, Op::Match, Op::Match,]
        );
        let xs = b"ACGT";
        let ys = b"ACTGT";
        let (score, ops) = global_banded(xs, ys, 1, -1, -3, -1, 2);
        assert_eq!(score, 1);
        assert_eq!(
            ops,
            vec![Op::Match, Op::Match, Op::Ins, Op::Match, Op::Match,]
        );
        let xs = b"ACTTTGT";
        let ys = b"ACGT";
        let (score, ops) = global_banded(xs, ys, 1, -1, -3, -1, 3);
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
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
        let prof = crate::gen_seq::PROFILE;
        for _ in 0..10 {
            let xs = crate::gen_seq::generate_seq(&mut rng, 1000);
            let ys = crate::gen_seq::introduce_randomness(&xs, &mut rng, &prof);
            use crate::bialignment::global;
            let (score, ops) = global(&xs, &ys, 1, -1, -3, -1);
            let (score_b, ops_b) = global_banded(&xs, &ys, 1, -1, -3, -1, 200);
            assert_eq!(score, score_b);
            assert_eq!(ops, ops_b);
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
            let (centers, dp) = dp_pre(&xs, &ys, radius);
            let (xslen, yslen) = (xs.len() as isize, ys.len() as isize);
            let j = yslen - centers[xs.len()] + radius as isize;
            let dist = dp[(xslen, j)];
            assert_eq!(dist, exact_dist, "{:?}", dp.get_row(xslen));
            let dp = dp_post(&xs, &ys, radius, &centers);
            let dist = dp[(0, radius as isize)];
            assert_eq!(dist, exact_dist, "{:?}", dp.get_row(0));
        }
    }
}
