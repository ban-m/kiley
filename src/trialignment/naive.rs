use super::Op;
use super::MA32;
use crate::padseq::*;
/// Compute the exact edit distance among `xs`, `ys`, and `zs`, and return the distance and edit operations.
pub fn alignment(xs: &[u8], ys: &[u8], zs: &[u8]) -> (u32, Vec<Op>) {
    let xs: Vec<_> = xs.iter().map(convert_to_twobit).collect();
    let ys: Vec<_> = ys.iter().map(convert_to_twobit).collect();
    let zs: Vec<_> = zs.iter().map(convert_to_twobit).collect();
    // Filled x == y == z == 0 case.
    let mut dp = vec![vec![vec![std::u32::MIN; zs.len() + 1]; ys.len() + 1]; xs.len() + 1];
    // Fill y == 0, x == 0 case.
    for z in 0..=zs.len() {
        dp[0][0][z] = z as u32;
    }
    // Fill x == 0, z ==0 case
    for y in 0..=ys.len() {
        dp[0][y][0] = y as u32;
    }
    // Fill y == z == 0 case
    for (x, yz) in dp.iter_mut().enumerate().take(xs.len() + 1) {
        // for x in 0..=xs.len() {
        // dp[x][0][0] = x as u32;
        yz[0][0] = x as u32;
    }
    // Fill X == 0, y > 0, z > 0 case.
    for y in 1..=ys.len() {
        for z in 1..=zs.len() {
            // (-,-,zs[z-1]), causing 1 penalty.
            let del = dp[0][y][z - 1] + 1;
            // (-,ys[y-1],-), causing 1 penalty.
            let ins = dp[0][y - 1][z] + 1;
            // (-, ys[y-1], zs[z-1]), causing 1 or 2 penalty.
            //let mat = dp[0][y - 1][z - 1] + 1 + (ys[y - 1] != zs[z - 1]) as u32;
            let mat_score = MA32[(0b100 << 6) | ((ys[y - 1] << 3) | zs[z - 1]) as usize];
            dp[0][y][z] = (dp[0][y - 1][z - 1] + mat_score).min(del).min(ins);
        }
    }
    // Fill y == 0 , x > 0, z > 0 case.
    for x in 1..=xs.len() {
        for z in 1..=zs.len() {
            let del = dp[x][0][z - 1] + 1;
            let ins = dp[x - 1][0][z] + 1;
            let mat_score = MA32[0b1_0000_0000 | ((zs[z - 1] << 3) | xs[x - 1]) as usize];
            dp[x][0][z] = del.min(ins).min(dp[x - 1][0][z - 1] + mat_score);
        }
    }
    // Fill z == 0, x > 0, y > 0 case
    for x in 1..=xs.len() {
        for y in 1..=ys.len() {
            let del = dp[x][y - 1][0] + 1;
            let ins = dp[x - 1][y][0] + 1;
            let mat_score = MA32[0b1_0000_0000 | ((xs[x - 1] << 3) | ys[y - 1]) as usize];
            dp[x][y][0] = del.min(ins).min(dp[x - 1][y - 1][0] + mat_score);
        }
    }
    // Fill x > 0, y > 0, z > 0 case.
    for x in 1..=xs.len() {
        for y in 1..=ys.len() {
            for z in 1..=zs.len() {
                // (xs[x-1], -,-), causing 1 penalty.
                let x_ins = dp[x - 1][y][z] + 1;
                let y_ins = dp[x][y - 1][z] + 1;
                let z_ins = dp[x][y][z - 1] + 1;
                let gap = 0b1_0000_0000;
                // (-, ys[y-1], zs[z-1]), causing 1 or 2 penalty.
                let x_del = dp[x][y - 1][z - 1] + MA32[gap | (ys[y - 1] << 3 | zs[z - 1]) as usize];
                let y_del = dp[x - 1][y][z - 1] + MA32[gap | (xs[x - 1] << 3 | zs[z - 1]) as usize];
                let z_del = dp[x - 1][y - 1][z] + MA32[gap | (ys[y - 1] << 3 | xs[x - 1]) as usize];
                // (xs[x-1], ys[y-1], zs[z-1]), causing 0 or 1 or 2 penalty.
                let mat_score =
                    MA32[(xs[x - 1] as usize) << 6 | (ys[y - 1] << 3 | zs[z - 1]) as usize];
                let mat = dp[x - 1][y - 1][z - 1] + mat_score;
                dp[x][y][z] = x_ins
                    .min(y_ins)
                    .min(z_ins)
                    .min(x_del)
                    .min(y_del)
                    .min(z_del)
                    .min(mat);
            }
        }
    }
    // Traceback.
    let min_score = dp[xs.len()][ys.len()][zs.len()];
    let (mut x, mut y, mut z) = (xs.len(), ys.len(), zs.len());
    let mut ops = vec![];
    while 0 < x || 0 < y || 0 < z {
        let current_score = dp[x][y][z];
        let op = if x == 0 && y == 0 {
            Op::ZInsertion
        } else if z == 0 && y == 0 {
            Op::XInsertion
        } else if z == 0 && x == 0 {
            Op::YInsertion
        } else if x == 0 {
            let z_ins = dp[x][y][z - 1] + 1;
            let y_ins = dp[x][y - 1][z] + 1;
            let x_del = dp[x][y - 1][z - 1] + 1 + (ys[y - 1] != zs[z - 1]) as u32;
            match current_score {
                s if s == z_ins => Op::ZInsertion,
                s if s == y_ins => Op::YInsertion,
                _ => {
                    assert_eq!(current_score, x_del);
                    Op::XDeletion
                }
            }
        } else if y == 0 {
            let z_ins = dp[x][y][z - 1] + 1;
            let x_ins = dp[x - 1][y][z] + 1;
            let y_del = dp[x - 1][y][z - 1] + 1 + (xs[x - 1] != zs[z - 1]) as u32;
            match current_score {
                s if s == z_ins => Op::ZInsertion,
                s if s == x_ins => Op::XInsertion,
                _ => {
                    assert_eq!(current_score, y_del);
                    Op::YDeletion
                }
            }
        } else if z == 0 {
            let x_ins = dp[x - 1][y][z] + 1;
            let y_ins = dp[x][y - 1][z] + 1;
            let z_del = dp[x - 1][y - 1][z] + 1 + (xs[x - 1] != ys[y - 1]) as u32;
            match current_score {
                x if x == x_ins => Op::XInsertion,
                x if x == y_ins => Op::YInsertion,
                _ => {
                    assert_eq!(current_score, z_del);
                    Op::ZDeletion
                }
            }
        } else {
            let x_ins = dp[x - 1][y][z] + 1;
            let y_ins = dp[x][y - 1][z] + 1;
            let z_ins = dp[x][y][z - 1] + 1;
            let x_del = dp[x][y - 1][z - 1] + 1 + (ys[y - 1] != zs[z - 1]) as u32;
            let y_del = dp[x - 1][y][z - 1] + 1 + (xs[x - 1] != zs[z - 1]) as u32;
            let z_del = dp[x - 1][y - 1][z] + 1 + (ys[y - 1] != xs[x - 1]) as u32;
            match current_score {
                s if s == x_ins => Op::XInsertion,
                s if s == y_ins => Op::YInsertion,
                s if s == z_ins => Op::ZInsertion,
                s if s == x_del => Op::XDeletion,
                s if s == y_del => Op::YDeletion,
                s if s == z_del => Op::ZDeletion,
                _ => Op::Match,
            }
        };
        match op {
            Op::XInsertion => x = x.max(1) - 1,
            Op::YInsertion => y = y.max(1) - 1,
            Op::ZInsertion => z = z.max(1) - 1,
            Op::XDeletion => {
                y = y.max(1) - 1;
                z = z.max(1) - 1;
            }
            Op::YDeletion => {
                x = x.max(1) - 1;
                z = z.max(1) - 1;
            }
            Op::ZDeletion => {
                x = x.max(1) - 1;
                y = y.max(1) - 1;
            }
            Op::Match => {
                x = x.max(1) - 1;
                y = y.max(1) - 1;
                z = z.max(1) - 1;
            }
        }
        ops.push(op);
    }
    ops.reverse();
    (min_score, ops)
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoroshiro128PlusPlus;
    #[test]
    fn it_works() {}
    #[test]
    fn call_test() {
        let xs = vec![0];
        let ys = vec![1];
        let zs = vec![4];
        alignment(&xs, &ys, &zs);
    }
    #[test]
    fn one_op() {
        let xs = b"A";
        let ys = b"A";
        let zs = b"A";
        let (score, ops) = alignment(xs, ys, zs);
        assert_eq!(score, 0);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let xs = b"C";
        let ys = b"A";
        let zs = b"A";
        let (score, ops) = alignment(xs, ys, zs);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let (score, ops) = alignment(ys, xs, zs);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let (score, ops) = alignment(ys, zs, xs);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let xs = b"C";
        let ys = b"A";
        let zs = b"T";
        let (score, ops) = alignment(xs, ys, zs);
        assert_eq!(score, 2);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
    }
    #[test]
    fn short_test() {
        let xs = b"AAATGGGG";
        let ys = b"AAAGGGG";
        let zs = b"AAAGGGG";
        let (score, ops) = alignment(xs, ys, zs);
        assert_eq!(score, 1);
        let op_ans = vec![
            Op::Match,
            Op::Match,
            Op::Match,
            Op::XInsertion,
            Op::Match,
            Op::Match,
            Op::Match,
            Op::Match,
        ];
        assert_eq!(ops, op_ans);
        let xs = b"AAATGGGG";
        let ys = b"AAATGGG";
        let zs = b"AATGGGG";
        let (score, ops) = alignment(xs, ys, zs);
        assert_eq!(score, 2);
        let op_ans = vec![
            Op::Match,
            Op::Match,
            Op::ZDeletion,
            Op::Match,
            Op::Match,
            Op::Match,
            Op::Match,
            Op::YDeletion,
        ];
        assert_eq!(ops, op_ans);
    }
    #[test]
    fn random_seq() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(423430);
        for _ in 0..10 {
            let xs: Vec<u8> = (0..rng.gen::<usize>() % 100)
                .map(|_| rng.gen::<u8>())
                .collect();
            let ys: Vec<u8> = (0..rng.gen::<usize>() % 100)
                .map(|_| rng.gen::<u8>())
                .collect();
            let zs: Vec<u8> = (0..rng.gen::<usize>() % 100)
                .map(|_| rng.gen::<u8>())
                .collect();
            alignment(&xs, &ys, &zs);
        }
    }
    #[test]
    fn random_similar_seq() {
        let length = 200;
        let p = crate::gen_seq::Profile {
            sub: 0.01,
            ins: 0.01,
            del: 0.01,
        };
        let possible_score = (length as f64 * 0.15).ceil() as u32;
        for i in 0..10u64 {
            let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let template = crate::gen_seq::generate_seq(&mut rng, length);
            let x = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
            let y = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
            let z = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
            let (score, _) = super::alignment(&x, &y, &z);
            assert!(score < possible_score, "{}\t{}", i, score);
        }
    }
}
