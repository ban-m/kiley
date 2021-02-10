use super::Op;
// First we implement a rotate DP, then move to banded DP.
// Maybe we can convert u32 to u16....
struct DPTableu32<'a> {
    dp: Vec<Vec<Vec<u32>>>,
    xlen: usize,
    ylen: usize,
    zlen: usize,
    xs: &'a [u8],
    ys: &'a [u8],
    zs: &'a [u8],
}

impl<'a> DPTableu32<'a> {
    fn new(xs: &'a [u8], ys: &'a [u8], zs: &'a [u8]) -> Self {
        let xlen = xs.len();
        let ylen = ys.len();
        let zlen = zs.len();
        Self {
            xlen,
            ylen,
            zlen,
            dp: vec![vec![vec![0; zlen + 1]; ylen + 1]; xlen + 1],
            xs,
            ys,
            zs,
        }
    }
    fn get(&self, i: usize, j: usize, k: usize) -> u32 {
        self.dp[i][j][k]
    }
    fn get_mut(&mut self, i: usize, j: usize, k: usize) -> Option<&mut u32> {
        self.dp.get_mut(i)?.get_mut(j)?.get_mut(k)
    }
    fn get_x_ins(&self, i: usize, j: usize, k: usize) -> u32 {
        self.get(i - 1, j, k) + 1
    }
    fn get_y_ins(&self, i: usize, j: usize, k: usize) -> u32 {
        self.get(i, j - 1, k) + 1
    }
    fn get_z_ins(&self, i: usize, j: usize, k: usize) -> u32 {
        self.get(i, j, k - 1) + 1
    }
    fn get_x_del(&self, i: usize, j: usize, k: usize) -> u32 {
        self.get(i, j - 1, k - 1) + 1 + (self.ys[j - 1] != self.zs[k - 1]) as u32
    }
    fn get_y_del(&self, i: usize, j: usize, k: usize) -> u32 {
        self.get(i - 1, j, k - 1) + 1 + (self.xs[i - 1] != self.zs[k - 1]) as u32
    }
    fn get_z_del(&self, i: usize, j: usize, k: usize) -> u32 {
        self.get(i - 1, j - 1, k) + 1 + (self.xs[i - 1] != self.ys[j - 1]) as u32
    }
    fn get_mat(&self, i: usize, j: usize, k: usize) -> u32 {
        let mat_score = match (
            self.xs[i - 1] == self.ys[j - 1],
            self.ys[j - 1] == self.zs[k - 1],
            self.zs[k - 1] == self.xs[i - 1],
        ) {
            (true, true, true) => 0,
            (true, false, false) => 1,
            (false, true, false) => 1,
            (false, false, true) => 1,
            (false, false, false) => 2,
            _ => panic!("{}\t{}\t{}", self.xs[i - 1], self.ys[j - 1], self.zs[k - 1]),
        };
        self.get(i - 1, j - 1, k - 1) + mat_score
    }
    fn get_min_score(&self) -> u32 {
        self.get(self.xlen, self.ylen, self.zlen)
    }
    fn update(&mut self, s: usize, t: usize, u: usize) {
        let (i, j, k) = (s - t, t - u, u);
        self.naive_update(i, j, k);
    }
    fn naive_update(&mut self, i: usize, j: usize, k: usize) {
        let next_score = if i == 0 && j == 0 {
            k as u32
        } else if i == 0 && k == 0 {
            j as u32
        } else if j == 0 && k == 0 {
            i as u32
        } else if i == 0 {
            // convert y to -, incuring 1
            self.get_y_ins(i, j, k)
                .min(self.get_z_ins(i, j, k))
                .min(self.get_x_del(i, j, k))
        } else if j == 0 {
            self.get_x_ins(i, j, k)
                .min(self.get_z_ins(i, j, k))
                .min(self.get_y_del(i, j, k))
        } else if k == 0 {
            self.get_y_ins(i, j, k)
                .min(self.get_x_ins(i, j, k))
                .min(self.get_z_del(i, j, k))
        } else {
            self.get_x_ins(i, j, k)
                .min(self.get_y_ins(i, j, k))
                .min(self.get_z_ins(i, j, k))
                .min(self.get_x_del(i, j, k))
                .min(self.get_y_del(i, j, k))
                .min(self.get_z_del(i, j, k))
                .min(self.get_mat(i, j, k))
        };
        *self.get_mut(i, j, k).unwrap() = next_score;
    }
    fn alignment(xs: &'a [u8], ys: &'a [u8], zs: &'a [u8], _r: usize) -> (u32, Vec<Op>) {
        let mut tab = Self::new(xs, ys, zs);
        let (xlen, ylen, zlen) = (xs.len(), ys.len(), zs.len());
        for s in 0..(xlen + ylen + zlen + 1) {
            let t_range = s.saturating_sub(xlen)..(s + 1).min(ylen + zlen + 1);
            for t in t_range {
                let u_range = t.saturating_sub(ylen)..(t + 1).min(zlen + 1);
                for u in u_range {
                    assert!(s - t <= xlen && t - u <= ylen && u <= zlen);
                    tab.update(s, t, u);
                }
            }
        }
        let min_score = tab.get_min_score();
        // Traceback.
        let traceprobe = TraceProbe::new(&tab);
        let mut ops: Vec<_> = traceprobe.collect();
        ops.reverse();
        (min_score, ops)
    }
}

struct TraceProbe<'a, 'b> {
    xpos: usize,
    ypos: usize,
    zpos: usize,
    dp: &'a DPTableu32<'b>,
}

impl<'a, 'b> TraceProbe<'a, 'b> {
    fn new(dp: &'a DPTableu32<'b>) -> Self {
        Self {
            xpos: dp.xlen,
            ypos: dp.ylen,
            zpos: dp.zlen,
            dp,
        }
    }
}

impl<'a, 'b> std::iter::Iterator for TraceProbe<'a, 'b> {
    type Item = Op;
    fn next(&mut self) -> Option<Self::Item> {
        let current_score = self.dp.get(self.xpos, self.ypos, self.zpos);
        if self.xpos == 0 && self.ypos == 0 && self.zpos == 0 {
            return None;
        }
        let op = if self.xpos == 0 && self.ypos == 0 {
            Op::ZInsertion
        } else if self.zpos == 0 && self.ypos == 0 {
            Op::XInsertion
        } else if self.zpos == 0 && self.xpos == 0 {
            Op::YInsertion
        } else if self.xpos == 0 {
            let z_ins = self.dp.get_z_ins(self.xpos, self.ypos, self.zpos);
            let y_ins = self.dp.get_y_ins(self.xpos, self.ypos, self.zpos);
            let x_del = self.dp.get_x_del(self.xpos, self.ypos, self.zpos);
            match current_score {
                s if s == z_ins => Op::ZInsertion,
                s if s == y_ins => Op::YInsertion,
                _ => {
                    assert_eq!(current_score, x_del);
                    Op::XDeletion
                }
            }
        } else if self.ypos == 0 {
            let z_ins = self.dp.get_z_ins(self.xpos, self.ypos, self.zpos);
            let x_ins = self.dp.get_x_ins(self.xpos, self.ypos, self.zpos);
            let y_del = self.dp.get_y_del(self.xpos, self.ypos, self.zpos);
            match current_score {
                s if s == z_ins => Op::ZInsertion,
                s if s == x_ins => Op::XInsertion,
                _ => {
                    assert_eq!(current_score, y_del);
                    Op::YDeletion
                }
            }
        } else if self.zpos == 0 {
            let x_ins = self.dp.get_x_ins(self.xpos, self.ypos, self.zpos);
            let y_ins = self.dp.get_y_ins(self.xpos, self.ypos, self.zpos);
            let z_del = self.dp.get_z_del(self.xpos, self.ypos, self.zpos);
            match current_score {
                x if x == x_ins => Op::XInsertion,
                x if x == y_ins => Op::YInsertion,
                _ => {
                    assert_eq!(current_score, z_del);
                    Op::ZDeletion
                }
            }
        } else {
            let x_ins = self.dp.get_x_ins(self.xpos, self.ypos, self.zpos);
            let y_ins = self.dp.get_y_ins(self.xpos, self.ypos, self.zpos);
            let z_ins = self.dp.get_z_ins(self.xpos, self.ypos, self.zpos);
            let x_del = self.dp.get_x_del(self.xpos, self.ypos, self.zpos);
            let y_del = self.dp.get_y_del(self.xpos, self.ypos, self.zpos);
            let z_del = self.dp.get_z_del(self.xpos, self.ypos, self.zpos);
            let mat = self.dp.get_mat(self.xpos, self.ypos, self.zpos);
            match current_score {
                s if s == x_ins => Op::XInsertion,
                s if s == y_ins => Op::YInsertion,
                s if s == z_ins => Op::ZInsertion,
                s if s == x_del => Op::XDeletion,
                s if s == y_del => Op::YDeletion,
                s if s == z_del => Op::ZDeletion,
                _ => {
                    assert_eq!(current_score, mat);
                    Op::Match
                }
            }
        };
        match op {
            Op::XInsertion => self.xpos = self.xpos.max(1) - 1,
            Op::YInsertion => self.ypos = self.ypos.max(1) - 1,
            Op::ZInsertion => self.zpos = self.zpos.max(1) - 1,
            Op::XDeletion => {
                self.ypos = self.ypos.max(1) - 1;
                self.zpos = self.zpos.max(1) - 1;
            }
            Op::YDeletion => {
                self.xpos = self.xpos.max(1) - 1;
                self.zpos = self.zpos.max(1) - 1;
            }
            Op::ZDeletion => {
                self.xpos = self.xpos.max(1) - 1;
                self.ypos = self.ypos.max(1) - 1;
            }
            Op::Match => {
                self.xpos -= 1;
                self.ypos -= 1;
                self.zpos -= 1;
            }
        }
        Some(op)
    }
}

/// Compute the banded edit distance among `xs`, `ys`, and `zs`, and return the distance and edit operations.
pub fn alignment(xs: &[u8], ys: &[u8], zs: &[u8], r: usize) -> (u32, Vec<Op>) {
    if (xs.len() as u16) < std::u16::MAX
        && (ys.len() as u16) < std::u16::MAX
        && (zs.len() as u16) < std::u16::MAX
    {
        DPTableu32::alignment(xs, ys, zs, r)
    } else {
        assert!(
            (xs.len() as u32) < std::u32::MAX
                && (ys.len() as u32) < std::u32::MAX
                && (zs.len() as u32) < std::u32::MAX
        );
        DPTableu32::alignment(xs, ys, zs, r)
    }
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
        alignment(&xs, &ys, &zs, 2);
    }
    #[test]
    fn one_op() {
        let xs = b"A";
        let ys = b"A";
        let zs = b"A";
        let (score, ops) = alignment(xs, ys, zs, 2);
        assert_eq!(score, 0);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let xs = b"C";
        let ys = b"A";
        let zs = b"A";
        let (score, ops) = alignment(xs, ys, zs, 2);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let (score, ops) = alignment(ys, xs, zs, 2);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let (score, ops) = alignment(ys, zs, xs, 2);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let xs = b"C";
        let ys = b"A";
        let zs = b"T";
        let (score, ops) = alignment(xs, ys, zs, 2);
        assert_eq!(score, 2);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
    }
    #[test]
    fn short_test() {
        let xs = b"AAATGGGG";
        let ys = b"AAAGGGG";
        let zs = b"AAAGGGG";
        let (score, ops) = alignment(xs, ys, zs, 2);
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
        let (score, ops) = alignment(xs, ys, zs, 2);
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
            alignment(&xs, &ys, &zs, 20);
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
            let (score, _) = super::alignment(&x, &y, &z, 20);
            assert!(score < possible_score, "{}\t{}", i, score);
        }
    }
}
