//! Banded Trialignment.
use super::Op;
use super::MA32;
use crate::padseq::*;
// TODO: maybe we can use macros to implement u32 bit version of code and u16 version of code.
// Or, maybe we can use default implementation of trait for each type.
// The DP table of the ternary alignment for long sequence.
// Note that it fill the DP cell not usual three x,y, and z for loop, but
// fill the cells by anti-diagonal lines.
const OFFSET: isize = 3;

// Get the (s,t,u) index. Here, the len is the diameter of the DP, 2 * rad + 1 + OFFSET.
macro_rules! get {
    ($s:expr, $t:expr, $u:expr, $len:expr) => {
        ($len * $len * $s + $len * ($t + OFFSET) + ($u + OFFSET)) as usize
    };
}

// min_swap(x,y) is equivalent to x = x.min(y), but is more efficient.
macro_rules! min_swap {
    ($x:expr,$y:expr) => {
        if $y < $x {
            $x = $y;
        }
    };
}

#[derive(Debug, Clone)]
pub struct Aligner {
    memory: Vec<u32>,
}

impl Aligner {
    pub fn new(xslen: usize, yslen: usize, zslen: usize, radius: usize) -> Self {
        let total_length = xslen + yslen + zslen + 3;
        let len = (2 * radius + 1) + OFFSET as usize;
        let dp = vec![total_length as u32; len * len * (total_length + 1)];
        Self { memory: dp }
    }
    pub fn align(&mut self, xs: &[u8], ys: &[u8], zs: &[u8], radius: usize) -> (u32, Vec<Op>) {
        // Tune memories.
        let total_length = xs.len() + ys.len() + zs.len() + 3;
        let len = (2 * radius + 1) + OFFSET as usize;
        let extend_size = (len * len * (total_length + 1)).saturating_sub(self.memory.len());
        self.memory
            .extend(std::iter::repeat(total_length as u32).take(extend_size));
        self.align_inner(xs, ys, zs, radius)
    }
    fn align_inner(&mut self, xs: &[u8], ys: &[u8], zs: &[u8], rad: usize) -> (u32, Vec<Op>) {
        // Convert A, C, G, and T into two bit encoding.
        // Also, we put a "sentinel" value at the beggining of each input sequence,
        // so that we would not get invalid access to any xs[i-1] value.
        // For example, if xs = b"ACACA", the converted array would be [NULL, 0b000, 0b001, 0b000, 0b001,0b000];
        // let start = std::time::Instant::now();
        let xs: Vec<_> = std::iter::once(NULL)
            .chain(xs.iter().map(convert_to_twobit))
            .collect();
        let ys: Vec<_> = std::iter::once(NULL)
            .chain(ys.iter().map(convert_to_twobit))
            .collect();
        let zs: Vec<_> = std::iter::once(NULL)
            .chain(zs.iter().map(convert_to_twobit))
            .collect();
        let rad = rad as isize;
        // The first base should be a mock value.
        assert_eq!(xs[0], NULL);
        assert_eq!(ys[0], NULL);
        assert_eq!(zs[0], NULL);
        let (xlen, ylen, zlen) = (xs.len(), ys.len(), zs.len());
        let total_length = (xlen + ylen + zlen) as u32;
        let len = (2 * rad + 1) + OFFSET;
        let dp = self.memory.as_mut_slice();
        // We do not need this.
        dp.iter_mut().for_each(|x| *x = total_length);
        // let alloc = std::time::Instant::now();
        let len = len as isize;
        // Initialization.
        dp[get!(0, rad, rad, len)] = 0;
        let mut centers: Vec<(isize, isize)> = vec![(0, 0), (0, 0), (0, 0), (2, 1)];
        for s in 3..(total_length + 1) as isize {
            let (t_center, u_center) = *centers.last().unwrap();
            // We can derive the condition under which
            // we are certain that the index would never violate vector boundary.
            let t_start = (s - xlen as isize + rad - t_center).max(1);
            let t_end = (2 * rad + 1).min(s + rad - t_center);
            let diffs = {
                let (t_c, u_c) = (t_center, u_center);
                match &centers[s as usize - 3..s as usize] {
                    [(x3, y3), (x2, y2), (x1, y1)] => {
                        [t_c - x1, t_c - x2, t_c - x3, u_c - y1, u_c - y2, u_c - y3]
                    }
                    _ => panic!(),
                }
            };
            let (filled_dp, filling_dp): (&[u32], &mut [u32]) = {
                let (x, y): (&mut [u32], &mut [u32]) = dp.split_at_mut(get!(s, 0, 0, len));
                (x as &[u32], y)
            };
            let (mut min, mut min_t, mut min_u) = (total_length, 0, 0);
            for t in t_start..t_end {
                let t_orig = t + t_center - rad;
                let x_axis = s - t_orig;
                let u_start = (1 + rad - u_center)
                    .max(t + t_center - u_center - ylen as isize)
                    .max(1);
                let u_end = (2 * rad + 1)
                    .min(zlen as isize + rad - u_center + 1)
                    .min(t + t_center - u_center);
                for u in u_start..u_end {
                    let u_orig = u + u_center - rad;
                    let y_axis = t_orig - u_orig;
                    let x_base = xs[(x_axis - 1) as usize];
                    let y_base = ys[(y_axis - 1) as usize];
                    let z_base = zs[(u_orig - 1) as usize];
                    let score = get_next_score(
                        &filled_dp,
                        len,
                        (s, t, u),
                        &diffs,
                        (x_base, y_base, z_base),
                    );
                    filling_dp[(t * len + u) as usize] = score;
                    if score < min {
                        min = score;
                        min_t = t;
                        min_u = u;
                    }
                }
            }
            let next_t = min_t + t_center - rad;
            let next_u = min_u + u_center - rad;
            // Find the nearest cell to the (next_t and next_u).
            let next_center = vec![
                (t_center, u_center),
                (t_center + 1, u_center),
                (t_center + 1, u_center + 1),
            ]
            .into_iter()
            .min_by_key(|&(t, u)| (t - next_t).abs() + (u - next_u).abs())
            .unwrap();
            centers.push(next_center);
        }
        // let filled = std::time::Instant::now();
        // Traceback.
        let (mut s, mut t, mut u, min, mut ops) = {
            // Search the last element filled.
            let (idx, &min) = dp
                .iter()
                .enumerate()
                .rev()
                .find(|&(_, &score)| score < total_length)
                .unwrap();
            let idx = idx as isize;
            let s = idx / (len * len);
            let t = (idx - (s * len * len)) / len - OFFSET;
            let u = (idx - (s * len * len)) % len - OFFSET;
            let mut ops = vec![];
            let (t_center, u_center) = centers[s as usize];
            let t_orig = t + t_center - rad;
            let u_orig = u + u_center - rad;
            let x_axis = (s - t_orig) as usize;
            let y_axis = (t_orig - u_orig) as usize;
            let z_axis = u_orig as usize;
            // If x_axis, y_axis, z_axis do not reache the end of each sequence,
            // the alignment, or the filled pattern, is skewed toward the other sides of the DP cells.
            ops.extend(std::iter::repeat(Op::XDeletion).take(xs.len().saturating_sub(x_axis)));
            ops.extend(std::iter::repeat(Op::YDeletion).take(ys.len().saturating_sub(y_axis)));
            ops.extend(std::iter::repeat(Op::ZDeletion).take(zs.len().saturating_sub(z_axis)));
            (s, t, u, min, ops)
        };
        while 3 < s || 2 + rad < t || 1 + rad < u {
            let (t_center, u_center) = centers[s as usize];
            let diffs = {
                let (t_c, u_c) = (t_center, u_center);
                match &centers[s as usize - 3..s as usize] {
                    [(x3, y3), (x2, y2), (x1, y1)] => {
                        [t_c - x1, t_c - x2, t_c - x3, u_c - y1, u_c - y2, u_c - y3]
                    }
                    _ => panic!(),
                }
            };
            let t_orig = t + t_center - rad;
            let u_orig = u + u_center - rad;
            let x_axis = s - t_orig;
            let y_axis = t_orig - u_orig;
            let z_axis = u_orig;
            let x_base = xs[(x_axis - 1) as usize];
            let y_base = ys[(y_axis - 1) as usize];
            let z_base = zs[(z_axis - 1) as usize];
            let (op, next_position) =
                get_next_position(&dp, len, (s, t, u), &diffs, (x_base, y_base, z_base));
            s = next_position.0;
            t = next_position.1;
            u = next_position.2;
            ops.push(op);
        }
        ops.reverse();
        // let traced = std::time::Instant::now();
        // debug!(
        //     "ALN\t{}\t{}\t{}\t{}",
        //     dp.len(),
        //     (alloc - start).as_millis(),
        //     (filled - alloc).as_millis(),
        //     (traced - filled).as_millis()
        // );
        (min, ops)
    }
    pub fn consensus(&mut self, xs: &[u8], ys: &[u8], zs: &[u8], radius: usize) -> (u32, Vec<u8>) {
        let (dist, aln) = self.align(xs, ys, zs, radius);
        (dist, correct_by_alignment(xs, ys, zs, &aln))
    }
}

pub fn correct_by_alignment(xs: &[u8], ys: &[u8], zs: &[u8], aln: &[Op]) -> Vec<u8> {
    let (mut x, mut y, mut z) = (0, 0, 0);
    let mut buffer = vec![];
    for &op in aln {
        match op {
            Op::XInsertion => {
                x += 1;
            }
            Op::YInsertion => {
                y += 1;
            }
            Op::ZInsertion => {
                z += 1;
            }
            Op::XDeletion => {
                // '-' or ys[y], or zs[z].
                // actually, it is hard to determine...
                if ys[y] == zs[z] {
                    buffer.push(ys[y]);
                } else {
                    match buffer.len() % 3 {
                        0 => buffer.push(ys[y]),
                        1 => buffer.push(zs[z]),
                        _ => {}
                    }
                }
                y += 1;
                z += 1;
            }
            Op::YDeletion => {
                if xs[x] == zs[z] {
                    buffer.push(xs[x]);
                } else {
                    match buffer.len() % 3 {
                        0 => buffer.push(xs[x]),
                        1 => buffer.push(zs[z]),
                        _ => {}
                    }
                }
                x += 1;
                z += 1;
            }
            Op::ZDeletion => {
                if ys[y] == xs[x] {
                    buffer.push(ys[y]);
                } else {
                    match buffer.len() % 3 {
                        0 => buffer.push(xs[x]),
                        1 => buffer.push(ys[y]),
                        _ => {}
                    }
                }
                x += 1;
                y += 1;
            }
            Op::Match => {
                if xs[x] == ys[y] || xs[x] == zs[z] {
                    buffer.push(xs[x]);
                } else if ys[y] == zs[z] {
                    buffer.push(zs[z])
                } else {
                    match buffer.len() % 3 {
                        0 => buffer.push(xs[x]),
                        1 => buffer.push(ys[y]),
                        _ => buffer.push(zs[z]),
                    }
                }
                x += 1;
                y += 1;
                z += 1;
            }
        }
    }
    buffer
}

fn get_next_position(
    dp: &[u32],
    len: isize,
    (s, t, u): (isize, isize, isize),
    diffs: &[isize; 6],
    (x_base, y_base, z_base): (u8, u8, u8),
) -> (Op, (isize, isize, isize)) {
    let (t_d1, t_d2, t_d3) = (t + diffs[0], t + diffs[1], t + diffs[2]);
    let (u_d1, u_d2, u_d3) = (u + diffs[3], u + diffs[4], u + diffs[5]);
    let mut dp_scores = unsafe {
        [
            *dp.get_unchecked(get!(s - 1, t_d1, u_d1, len)),
            *dp.get_unchecked(get!(s - 1, t_d1 - 1, u_d1, len)),
            *dp.get_unchecked(get!(s - 1, t_d1 - 1, u_d1 - 1, len)),
            *dp.get_unchecked(get!(s - 2, t_d2 - 2, u_d2 - 1, len)),
            *dp.get_unchecked(get!(s - 2, t_d2 - 1, u_d2 - 1, len)),
            *dp.get_unchecked(get!(s - 2, t_d2 - 1, u_d2, len)),
            *dp.get_unchecked(get!(s - 3, t_d3 - 2, u_d3 - 1, len)),
        ]
    };
    dp_scores.iter_mut().take(6).for_each(|x| *x += 1);
    dp_scores[3] += (y_base != z_base) as u32;
    dp_scores[4] += (x_base != z_base) as u32;
    dp_scores[5] += (x_base != y_base) as u32;
    dp_scores[6] += MA32[(x_base as usize) << 6 | (y_base << 3 | z_base) as usize];
    let score = dp[get!(s, t, u, len)];
    if score == dp_scores[0] {
        (Op::XInsertion, (s - 1, t_d1, u_d1))
    } else if score == dp_scores[1] {
        (Op::YInsertion, (s - 1, t_d1 - 1, u_d1))
    } else if score == dp_scores[2] {
        (Op::ZInsertion, (s - 1, t_d1 - 1, u_d1 - 1))
    } else if score == dp_scores[3] {
        (Op::XDeletion, (s - 2, t_d2 - 2, u_d2 - 1))
    } else if score == dp_scores[4] {
        (Op::YDeletion, (s - 2, t_d2 - 1, u_d2 - 1))
    } else if score == dp_scores[5] {
        (Op::ZDeletion, (s - 2, t_d2 - 1, u_d2))
    } else {
        assert_eq!(score, dp_scores[6]);
        (Op::Match, (s - 3, t_d3 - 2, u_d3 - 1))
    }
}

/// Compute the banded edit distance among `xs`, `ys`, and `zs`, and return the distance and edit operations.
/// We only fill the dynamic programming cells in a banded region, and`r` specifies the radius of the band.
/// Note that `r` is the radius, so that the number of allocated/filled DP cells is around `r*r*(xs.len+ys.len+zs.len)
pub fn alignment(xs: &[u8], ys: &[u8], zs: &[u8], rad: usize) -> (u32, Vec<Op>) {
    Aligner::new(xs.len(), ys.len(), zs.len(), rad).align(xs, ys, zs, rad)
}

#[inline]
fn get_next_score(
    dp: &[u32],
    len: isize,
    (s, t, u): (isize, isize, isize),
    diffs: &[isize; 6],
    (x_base, y_base, z_base): (u8, u8, u8),
) -> u32 {
    let (t_d1, t_d2, t_d3) = (t + diffs[0], t + diffs[1], t + diffs[2]);
    let (u_d1, u_d2, u_d3) = (u + diffs[3], u + diffs[4], u + diffs[5]);
    unsafe {
        // Insertion scores.
        let mut min_ins = *dp.get_unchecked(get!(s - 1, t_d1, u_d1, len));
        let score_ins = *dp.get_unchecked(get!(s - 1, t_d1 - 1, u_d1, len));
        min_swap!(min_ins, score_ins);
        let score_ins = *dp.get_unchecked(get!(s - 1, t_d1 - 1, u_d1 - 1, len));
        min_swap!(min_ins, score_ins);
        // Deletion scores.
        let mut min_del =
            *dp.get_unchecked(get!(s - 2, t_d2 - 2, u_d2 - 1, len)) + (y_base != z_base) as u32;
        let score_del =
            *dp.get_unchecked(get!(s - 2, t_d2 - 1, u_d2 - 1, len)) + (x_base != z_base) as u32;
        min_swap!(min_del, score_del);
        let score_del =
            *dp.get_unchecked(get!(s - 2, t_d2 - 1, u_d2, len)) + (x_base != y_base) as u32;
        min_swap!(min_del, score_del);
        let score = *dp.get_unchecked(get!(s - 3, t_d3 - 2, u_d3 - 1, len))
            + MA32[(x_base as usize) << 6 | (y_base << 3 | z_base) as usize];
        min_swap!(min_del, min_ins);
        min_del += 1;
        min_swap!(min_del, score);
        min_del
    }
}

// #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[allow(dead_code)]
fn min_of_array(xs: &[u32; 7]) -> u32 {
    *xs.iter().min().unwrap()
}

// #[allow(dead_code)]
// #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
// #[target_feature(enable = "avx2")]
// unsafe fn min_of_array(xs: &[u32; 8]) -> u32 {
//     use std::arch::x86_64;
//     use x86_64::{
//         __m256i, _mm256_alignr_epi8, _mm256_extract_epi32, _mm256_loadu_si256, _mm256_min_epi32,
//         _mm256_permute2x128_si256,
//     };
//     // panic!("SIMD");
//     let mut xs_min = _mm256_loadu_si256(xs.as_ptr() as *const __m256i);
//     xs_min = _mm256_min_epi32(xs_min, _mm256_alignr_epi8(xs_min, xs_min, 4));
//     xs_min = _mm256_min_epi32(xs_min, _mm256_alignr_epi8(xs_min, xs_min, 8));
//     xs_min = _mm256_min_epi32(xs_min, _mm256_permute2x128_si256(xs_min, xs_min, 1));
//     _mm256_extract_epi32(xs_min, 0) as u32
// }

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
        let (score, ops) = alignment(xs, ys, zs, 2);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let (score, ops) = alignment(ys, xs, zs, 2);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        let (score, ops) = alignment(ys, xs, zs, 2);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
        println!("OK");
        let (score, ops) = alignment(ys, zs, xs, 2);
        assert_eq!(score, 1);
        assert_eq!(ops, vec![Op::Match]);
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
        eprintln!("OK");
        let xs = b"ATG";
        let ys = b"TG";
        let zs = b"ATG";
        let op_ans = vec![Op::YDeletion, Op::Match, Op::Match];
        let (score, ops) = alignment(xs, ys, zs, 4);
        assert_eq!(score, 1);
        assert_eq!(ops, op_ans);
        let (score, ops) = alignment(xs, ys, zs, 4);
        assert_eq!(score, 1);
        assert_eq!(ops, op_ans);

        let xs = b"ATG";
        let ys = b"ATG";
        let zs = b"TG";
        let op_ans = vec![Op::ZDeletion, Op::Match, Op::Match];
        let (score, ops) = alignment(xs, ys, zs, 4);
        assert_eq!(score, 1);
        assert_eq!(ops, op_ans);
        let (score, ops) = alignment(xs, ys, zs, 4);
        assert_eq!(score, 1);
        assert_eq!(ops, op_ans);

        let xs = b"AAATGGGG";
        let ys = b"AAATGGG";
        let zs = b"AATGGGG";
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
        let (score, ops) = alignment(xs, ys, zs, 4);
        assert_eq!(score, 2);
        assert_eq!(ops, op_ans);
        let (score, ops) = alignment(xs, ys, zs, 4);
        assert_eq!(score, 2);
        assert_eq!(ops, op_ans);
    }
    #[test]
    fn random_seq() {
        let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(423430);
        for _ in 0..10 {
            let len = 50 + rng.gen::<usize>() % 40;
            let xs: Vec<u8> = crate::gen_seq::generate_seq(&mut rng, len);
            let len = 50 + rng.gen::<usize>() % 40;
            let ys: Vec<u8> = crate::gen_seq::generate_seq(&mut rng, len);
            let len = 50 + rng.gen::<usize>() % 40;
            let zs: Vec<u8> = crate::gen_seq::generate_seq(&mut rng, len);
            alignment(&xs, &ys, &zs, 20);
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
            let (score, _) = alignment(&x, &y, &z, 20);
            assert!(score < possible_score, "{}\t{}", i, score);
            let (score, _) = alignment(&x, &y, &z, 20);
            assert!(score < possible_score, "{}\t{}", i, score);
        }
    }
    #[test]
    fn random_similar_seq_many() {
        let length = 2000;
        let p = crate::gen_seq::Profile {
            sub: 0.1,
            ins: 0.1,
            del: 0.1,
        };
        let i = 15;
        //for i in 0..1000u64 {
        let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
        let template = crate::gen_seq::generate_seq(&mut rng, length);
        let x = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
        let y = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
        let z = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
        eprintln!("{}", i);
        let rad = rng.gen::<usize>() % 10 + 3;
        alignment(&x, &y, &z, rad);
        // }
    }
    #[test]
    fn multiple_alignment() {
        let len = 150;
        let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(3);
        let p = crate::gen_seq::Profile {
            sub: 0.1,
            ins: 0.1,
            del: 0.1,
        };
        let seqs: Vec<_> = (0..100)
            .map(|_| {
                let len = len + rng.gen_range(0..400);
                let template = crate::gen_seq::generate_seq(&mut rng, len);
                let x = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
                let y = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
                let z = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
                (x, y, z)
            })
            .collect();
        let radius = 30;
        let mut aligner = Aligner::new(len, len, len, radius);
        for (x, y, z) in seqs.iter() {
            let (dist, _) = aligner.align(x, y, z, radius);
            let (dist2, _) = alignment(x, y, z, radius);
            assert_eq!(dist, dist2);
        }
    }
    // #[test]
    // #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // fn min_score_test() {
    //     let xs = [0, 0, 1, 2, 1, 1, 1];
    //     assert_eq!(unsafe { min_of_array(&xs) }, 0);
    //     let xs = [2, 2, 3, 1_0000, 1_000_000, 231, 23120];
    //     assert_eq!(unsafe { min_of_array(&xs) }, 2);
    //     let xs = [1_0000, 1_000_000, 231, 2323412, 23498, 23, 23180293];
    //     assert_eq!(unsafe { min_of_array(&xs) }, 23);
    // }
}
