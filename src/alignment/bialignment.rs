const fn match_mat() -> [u16; 64] {
    let mut scores = [1_000; 64];
    let mut x = 0;
    let mut y = 0;
    while x < 4 {
        while y < 4 {
            let mat_score = if x == y { 0 } else { 1 };
            scores[x << 3 | y] = mat_score;
            y += 1;
        }
        x += 1;
    }
    scores[0b111111] = 0;
    scores
}

const MATMAT: [u16; 64] = match_mat();

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
    dp[xs.len()][ys.len()]
}

pub fn fast_align(xs: &[u8], ys: &[u8]) -> u16 {
    let xs: Vec<_> = std::iter::once(0b111)
        .chain(xs.iter().map(crate::alignment::convert_to_twobit))
        .collect();
    let ys: Vec<_> = std::iter::once(0b111)
        .chain(ys.iter().map(crate::alignment::convert_to_twobit))
        .collect();
    let mut dp = vec![1_000; (ys.len() + 1) * (xs.len() + 1)];
    dp[0] = 0;
    let colnum = ys.len() + 1;
    for i in 1..xs.len() + 1 {
        for j in 1..ys.len() + 1 {
            let score = get_match(&dp, xs[i - 1], ys[j - 1], i, j, colnum);
            dp[i * colnum + j] = score;
        }
    }
    *dp.last().unwrap()
}

fn get_match(dp: &[u16], x: u8, y: u8, i: usize, j: usize, colnum: usize) -> u16 {
    let mat = MATMAT[(x << 3 | y) as usize];
    (dp[(i - 1) * colnum + j - 1] + mat)
        .min(dp[(i - 1) * colnum + j] + 1)
        .min(dp[i * colnum + j - 1] + 1)
}

pub fn naive_align(xs: &[u8], ys: &[u8]) -> u16 {
    let xs: Vec<_> = xs.iter().map(crate::alignment::convert_to_twobit).collect();
    let ys: Vec<_> = ys.iter().map(crate::alignment::convert_to_twobit).collect();
    let mut dp = vec![0; (ys.len() + 1) * (xs.len() + 1)];
    let colnum = ys.len() + 1;
    for i in 0..xs.len() + 1 {
        for j in 0..ys.len() + 1 {
            dp[i * colnum + j] = if i == 0 && j == 0 {
                0
            } else if j == 0 {
                i as u16
            } else if i == 0 {
                j as u16
            } else {
                let mat = MATMAT[(xs[i - 1] << 3 | ys[j - 1]) as usize];
                (dp[(i - 1) * colnum + j - 1] + mat)
                    .min(dp[(i - 1) * colnum + j] + 1)
                    .min(dp[i * colnum + j - 1] + 1)
            };
        }
    }
    *dp.last().unwrap()
}

#[cfg(test)]
mod test {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    const SEED: u64 = 1293890;
    use super::*;
    #[test]
    fn alignment_check() {
        for i in 0..100u64 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED + i);
            let xslen = rng.gen::<usize>() % 100;
            let xs = crate::gen_seq::generate_seq(&mut rng, xslen);
            let yslen = rng.gen::<usize>() % 100;
            let ys = crate::gen_seq::generate_seq(&mut rng, yslen);
            let score = naive_align(&xs, &ys);
            let fastscore = fast_align(&xs, &ys);
            eprintln!("S:{}, F:{}", score, fastscore);
            assert_eq!(score, fastscore);
        }
    }
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
}
