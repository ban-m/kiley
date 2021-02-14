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
            let mat = MATMAT[(xs[i - 1] << 3 | ys[j - 1]) as usize];
            dp[i * colnum + j] = (dp[(i - 1) * colnum + j - 1] + mat)
                .min(dp[(i - 1) * colnum + j] + 1)
                .min(dp[i * colnum + j - 1] + 1);
        }
    }
    *dp.last().unwrap()
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
}
