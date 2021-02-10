use super::Op;
// First we implement a rotate DP, then move to banded DP.
// Maybe we can convert u32 to u16....
struct DPTableu32<'a> {
    // The location of the minimum value on each diagonal slice.
    // In other words, if the centor[s]= (t,u), then,
    // the minimum "usual" DP value of DP[i][j][k] satisfying i + j + k = s is DP[s+t+u][t+u][u].
    centors: Vec<(usize, usize)>,
    dp: Vec<Vec<Vec<u32>>>,
    rad: usize,
    xlen: usize,
    ylen: usize,
    zlen: usize,
    xs: &'a [u8],
    ys: &'a [u8],
    zs: &'a [u8],
}

impl<'a> DPTableu32<'a> {
    fn new(xs: &'a [u8], ys: &'a [u8], zs: &'a [u8], rad: usize) -> Self {
        let tot = (xs.len() + ys.len() + zs.len()) as u32;
        let (xlen, ylen, zlen) = (xs.len(), ys.len(), zs.len());
        let dp: Vec<Vec<Vec<_>>> = (0..(xlen + ylen + zlen + 1))
            .map(|_| vec![vec![tot; 2 * rad + 1]; 2 * rad + 1])
            .collect();
        Self {
            xlen,
            ylen,
            zlen,
            dp,
            rad,
            xs,
            ys,
            zs,
            centors: Vec::with_capacity(xlen + ylen + zlen + 1),
        }
    }
    fn get(&self, s: usize, t: usize, u: usize) -> u32 {
        // Convert (s,t,u) -> (s,t-tcentor+rad,t-ucentor+rad).
        // If t or u is outside of the vector, return some large value.
        let (t, u) = (t + self.rad, u + self.rad);
        let (tcentor, ucentor) = self.centors[s];
        if tcentor <= t
            && t < tcentor + 2 * self.rad + 1
            && ucentor <= u
            && u < ucentor + 2 * self.rad + 1
        {
            self.dp[s][t - tcentor][u - ucentor]
        } else {
            1_000_000
        }
    }
    fn get_mut(&mut self, s: usize, t: usize, u: usize) -> Option<&mut u32> {
        // Convert (s,t,u) -> (s,t-tcentor+rad,t-ucentor+rad).
        // If t or u is outside of the vector, return some large value.
        let (t, u) = (t + self.rad, u + self.rad);
        let (tcentor, ucentor) = self.centors[s];
        if tcentor <= t
            && t < tcentor + 2 * self.rad + 1
            && ucentor <= u
            && u < ucentor + 2 * self.rad + 1
        {
            self.dp
                .get_mut(s)?
                .get_mut(t - tcentor)?
                .get_mut(u - ucentor)
        } else {
            None
        }
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
    fn update(&mut self, s: usize, t: usize, u: usize) -> u32 {
        self.update_diag(s, t, u);
        self.get(s, t, u)
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
    fn alignment(xs: &'a [u8], ys: &'a [u8], zs: &'a [u8], rad: usize) -> (u32, Vec<Op>) {
        let mut tab = Self::new(xs, ys, zs, rad);
        let (xlen, ylen, zlen) = (xs.len(), ys.len(), zs.len());
        let (mut tcentor, mut ucentor): (usize, usize) = (0, 0);
        tab.centors.push((tcentor, ucentor));
        for s in 0..(xlen + ylen + zlen + 1) {
            eprintln!("Centor:({},{},{})", s, tcentor, ucentor);
            let t_start = tcentor.saturating_sub(rad).max(s.saturating_sub(xlen));
            let t_end = (tcentor + rad + 1).min(ylen + zlen + 1).min(s + 1);
            let mut min_distance = std::u32::MAX;
            let (mut min_t, mut min_u) = (0, 0);
            for t in t_start..t_end {
                let t_diff = abs_diff(t, tcentor);
                let residual = rad - t_diff;
                let u_start = ucentor.saturating_sub(residual).max(t.saturating_sub(ylen));
                let u_end = (ucentor + rad + 1).min(zlen + 1).min(t + 1);
                for u in u_start..u_end {
                    let ed_dist = tab.update(s, t, u);
                    eprintln!("{},{},{}=>{}", s, t, u, ed_dist);
                    if ed_dist < min_distance {
                        min_t = t;
                        min_u = u;
                        min_distance = ed_dist;
                    }
                }
            }
            tcentor = min_t;
            ucentor = min_u;
            tab.centors.push((tcentor, ucentor));
        }
        eprintln!("OK");
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

fn abs_diff(x: usize, y: usize) -> usize {
    if x < y {
        y - x
    } else {
        x - y
    }
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
    if xs.len() + ys.len() + zs.len() < std::u16::MAX as usize {
        DPTableu32::alignment(xs, ys, zs, r)
    } else {
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
        let (score, ops) = alignment(xs, ys, zs, 4);
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
        let (score, ops) = alignment(xs, ys, zs, 4);
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
