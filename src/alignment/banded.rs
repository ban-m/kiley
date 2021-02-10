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
        let (xlen, ylen, zlen) = (xs.len(), ys.len(), zs.len());
        let dp: Vec<Vec<_>> = (0..(xlen + ylen + zlen + 1))
            .map(|s| {
                (0..(s + 1).min(ylen + zlen + 1))
                    .map(|t| vec![0; (t + 1).min(zlen + 1)])
                    .collect()
            })
            .collect();
        Self {
            xlen,
            ylen,
            zlen,
            dp,
            xs,
            ys,
            zs,
        }
    }
    fn get(&self, s: usize, t: usize, u: usize) -> u32 {
        self.dp[s][t][u]
    }
    fn get_mut(&mut self, s: usize, t: usize, u: usize) -> Option<&mut u32> {
        self.dp.get_mut(s)?.get_mut(t)?.get_mut(u)
    }
    fn get_x_ins(&self, s: usize, t: usize, u: usize) -> u32 {
        self.get(s - 1, t, u) + 1
    }
    fn get_y_ins(&self, s: usize, t: usize, u: usize) -> u32 {
        self.get(s - 1, t - 1, u) + 1
    }
    fn get_z_ins(&self, s: usize, t: usize, u: usize) -> u32 {
        self.get(s - 1, t - 1, u - 1) + 1
    }
    fn get_x_del(&self, s: usize, t: usize, u: usize) -> u32 {
        self.get(s - 2, t - 2, u - 1) + 1 + (self.ys[t - u - 1] != self.zs[u - 1]) as u32
    }
    fn get_y_del(&self, s: usize, t: usize, u: usize) -> u32 {
        self.get(s - 2, t - 1, u - 1) + 1 + (self.xs[s - t - 1] != self.zs[u - 1]) as u32
    }
    fn get_z_del(&self, s: usize, t: usize, u: usize) -> u32 {
        self.get(s - 2, t - 1, u) + 1 + (self.xs[s - t - 1] != self.ys[t - u - 1]) as u32
    }
    fn get_mat(&self, s: usize, t: usize, u: usize) -> u32 {
        let mat_score = match (
            self.xs[s - t - 1] == self.ys[t - u - 1],
            self.ys[t - u - 1] == self.zs[u - 1],
            self.zs[u - 1] == self.xs[s - t - 1],
        ) {
            (true, true, true) => 0,
            (true, false, false) => 1,
            (false, true, false) => 1,
            (false, false, true) => 1,
            (false, false, false) => 2,
            _ => panic!(
                "{}\t{}\t{}",
                self.xs[s - t - 1],
                self.ys[t - u - 1],
                self.zs[u - 1]
            ),
        };
        self.get(s - 3, t - 2, u - 1) + mat_score
    }
    fn get_min_score(&self) -> u32 {
        self.get(
            self.xlen + self.ylen + self.zlen,
            self.ylen + self.zlen,
            self.zlen,
        )
    }
    fn update(&mut self, s: usize, t: usize, u: usize) {
        self.update_diag(s, t, u);
    }
    fn update_diag(&mut self, s: usize, t: usize, u: usize) {
        let next_score = if s == t && t == u {
            u as u32
        } else if s == t && u == 0 {
            (t - u) as u32
        } else if t == u && u == 0 {
            (s - t) as u32
        } else if s == t {
            self.get_y_ins(s, t, u)
                .min(self.get_z_ins(s, t, u))
                .min(self.get_x_del(s, t, u))
        } else if t == u {
            self.get_x_ins(s, t, u)
                .min(self.get_z_ins(s, t, u))
                .min(self.get_y_del(s, t, u))
        } else if u == 0 {
            self.get_y_ins(s, t, u)
                .min(self.get_x_ins(s, t, u))
                .min(self.get_z_del(s, t, u))
        } else {
            self.get_x_ins(s, t, u)
                .min(self.get_y_ins(s, t, u))
                .min(self.get_z_ins(s, t, u))
                .min(self.get_x_del(s, t, u))
                .min(self.get_y_del(s, t, u))
                .min(self.get_z_del(s, t, u))
                .min(self.get_mat(s, t, u))
        };
        *self.get_mut(s, t, u).unwrap() = next_score;
    }
    fn alignment(xs: &'a [u8], ys: &'a [u8], zs: &'a [u8], _r: usize) -> (u32, Vec<Op>) {
        let mut tab = Self::new(xs, ys, zs);
        let (xlen, ylen, zlen) = (xs.len(), ys.len(), zs.len());
        for s in 0..(xlen + ylen + zlen + 1) {
            let t_range = s.saturating_sub(xlen)..(s + 1).min(ylen + zlen + 1);
            for t in t_range {
                let u_range = t.saturating_sub(ylen)..(t + 1).min(zlen + 1);
                for u in u_range {
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
    spos: usize,
    tpos: usize,
    upos: usize,
    dp: &'a DPTableu32<'b>,
}

impl<'a, 'b> TraceProbe<'a, 'b> {
    fn new(dp: &'a DPTableu32<'b>) -> Self {
        Self {
            spos: dp.xlen + dp.ylen + dp.zlen,
            tpos: dp.ylen + dp.zlen,
            upos: dp.zlen,
            dp,
        }
    }
}

impl<'a, 'b> std::iter::Iterator for TraceProbe<'a, 'b> {
    type Item = Op;
    fn next(&mut self) -> Option<Self::Item> {
        let current_score = self.dp.get(self.spos, self.tpos, self.upos);
        if self.spos == 0 && self.tpos == 0 && self.upos == 0 {
            return None;
        }
        let op = if (self.spos - self.tpos) == 0 && (self.tpos - self.upos) == 0 {
            Op::ZInsertion
        } else if (self.tpos - self.upos) == 0 && self.upos == 0 {
            Op::XInsertion
        } else if self.upos == 0 && (self.spos - self.tpos) == 0 {
            Op::YInsertion
        } else if (self.spos - self.tpos) == 0 {
            if current_score == self.dp.get_z_ins(self.spos, self.tpos, self.upos) {
                Op::ZInsertion
            } else if current_score == self.dp.get_y_ins(self.spos, self.tpos, self.upos) {
                Op::YInsertion
            } else {
                let xdel = self.dp.get_x_del(self.spos, self.tpos, self.upos);
                assert_eq!(current_score, xdel);
                Op::XDeletion
            }
        } else if (self.tpos - self.upos) == 0 {
            if current_score == self.dp.get_z_ins(self.spos, self.tpos, self.upos) {
                Op::ZInsertion
            } else if current_score == self.dp.get_x_ins(self.spos, self.tpos, self.upos) {
                Op::XInsertion
            } else {
                let y_del = self.dp.get_y_del(self.spos, self.tpos, self.upos);
                assert_eq!(current_score, y_del);
                Op::YDeletion
            }
        } else if self.upos == 0 {
            if current_score == self.dp.get_x_ins(self.spos, self.tpos, self.upos) {
                Op::XInsertion
            } else if current_score == self.dp.get_y_ins(self.spos, self.tpos, self.upos) {
                Op::YInsertion
            } else {
                let z_del = self.dp.get_z_del(self.spos, self.tpos, self.upos);
                assert_eq!(current_score, z_del);
                Op::ZDeletion
            }
        } else {
            if current_score == self.dp.get_x_ins(self.spos, self.tpos, self.upos) {
                Op::XInsertion
            } else if current_score == self.dp.get_y_ins(self.spos, self.tpos, self.upos) {
                Op::YInsertion
            } else if current_score == self.dp.get_z_ins(self.spos, self.tpos, self.upos) {
                Op::ZInsertion
            } else if current_score == self.dp.get_x_del(self.spos, self.tpos, self.upos) {
                Op::XDeletion
            } else if current_score == self.dp.get_y_del(self.spos, self.tpos, self.upos) {
                Op::YDeletion
            } else if current_score == self.dp.get_z_del(self.spos, self.tpos, self.upos) {
                Op::ZDeletion
            } else {
                let mat = self.dp.get_mat(self.spos, self.tpos, self.upos);
                assert_eq!(current_score, mat);
                Op::Match
            }
        };
        let op_bits: u8 = op.into();
        self.spos = self.spos.saturating_sub(op_bits.count_ones() as usize);
        self.tpos = self
            .tpos
            .saturating_sub((op_bits >> 1).count_ones() as usize);
        self.upos = self
            .upos
            .saturating_sub((op_bits >> 2).count_ones() as usize);
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
        for i in 0..10 {
            eprintln!("Start {}", i);
            let len = 50 + rng.gen::<usize>() % 40;
            let xs: Vec<u8> = crate::gen_seq::generate_seq(&mut rng, len);
            let len = 50 + rng.gen::<usize>() % 40;
            let ys: Vec<u8> = crate::gen_seq::generate_seq(&mut rng, len);
            let len = 50 + rng.gen::<usize>() % 40;
            let zs: Vec<u8> = crate::gen_seq::generate_seq(&mut rng, len);
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
