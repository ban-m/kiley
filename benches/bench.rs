#![feature(test)]
extern crate test;
use rand::SeedableRng;
//use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256StarStar;
const SEED: u64 = 1293890;
const SHORT_LEN: usize = 200;

#[bench]
fn naive_aln(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let zs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        kiley::alignment::naive::alignment(&xs, &ys, &zs)
    });
}

#[bench]
fn banded_aln(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let zs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        kiley::alignment::banded::alignment_u32(&xs, &ys, &zs, 10)
    });
}

#[bench]
fn banded_mock(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let zs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        mock_alignment(&xs, &ys, &zs, 10)
    });
}

fn new(rng: &mut Xoshiro256StarStar) -> [u32; 7] {
    use rand::Rng;
    [
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
        rng.gen(),
    ]
}

#[bench]
fn min_array_simd(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    b.iter(|| {
        for _ in 0..100 {
            test::black_box(unsafe { min_of_array(&new(&mut rng)) });
        }
    })
}
#[bench]
fn min_array(b: &mut test::Bencher) {
    fn min(xs: &[u32; 7]) -> u32 {
        *xs.iter().min().unwrap()
    }
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    b.iter(|| {
        for _ in 0..100 {
            test::black_box(min(&new(&mut rng)));
        }
    })
}

#[bench]
fn new_array(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    b.iter(|| {
        for _ in 0..100 {
            test::black_box(&new(&mut rng));
        }
    })
}

const OFFSET: isize = 3;
macro_rules! get {
    ($s:expr, $t:expr, $u:expr, $len:expr) => {
        ($len * $len * $s + $len * ($t + OFFSET) + ($u + OFFSET)) as usize
    };
}

use kiley::alignment::convert_to_twobit;
const NULL: u8 = 0b101;
use kiley::alignment::Op;
use kiley::alignment::MA32;
pub fn mock_alignment(xs: &[u8], ys: &[u8], zs: &[u8], rad: usize) -> (u32, Vec<Op>) {
    // Convert A, C, G, and T into two bit encoding.
    // Also, we put a "sentinel" value at the beggining of each input sequence,
    // so that we would not get invalid access to any xs[i-1] value.
    // For example, if xs = b"ACACA", the converted array would be [NULL, 0b000, 0b001, 0b000, 0b001,0b000];
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
    let (xlen, ylen, zlen) = (xs.len(), ys.len(), zs.len());
    let total_length = (xlen + ylen + zlen) as u32;
    let len = (2 * rad + 1) + OFFSET;
    let mut dp = vec![total_length; (len * len * (total_length + 1) as isize) as usize];
    let len = len as isize;
    // Initialization.
    dp[get!(0, rad, rad, len)] = 0;
    // Centers are (0,0,0), (1,0,0), (2,0,0), (3,2,1),
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
                &[(x3, y3), (x2, y2), (x1, y1)] => {
                    [t_c - x1, t_c - x2, t_c - x3, u_c - y1, u_c - y2, u_c - y3]
                }
                _ => panic!(),
            }
        };
        // We only need to keep the last three sheets of the dp cells.
        let (filled_dp, filling_dp): (&[u32], &mut [u32]) = {
            let (x, y): (&mut [u32], &mut [u32]) = dp.split_at_mut(get!(s, 0, 0, len));
            (x as &[u32], y)
        };
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
                let z_axis = u_orig;
                let x_base = xs[(x_axis - 1) as usize];
                let y_base = ys[(y_axis - 1) as usize];
                let z_base = zs[(z_axis - 1) as usize];
                // This is the hottest loop.
                let score =
                    get_next_score(&filled_dp, len, (s, t, u), &diffs, (x_base, y_base, z_base));
                filling_dp[(t * len + u) as usize] = score;
            }
        }
        let start = ((len * len * s) + len * 3 + 3) as usize;
        let end = ((len * len * s) + len * (2 * rad + 3) + (2 * rad + 3)) as usize;
        let (mut min, mut min_t, mut min_u) = (total_length, 0, 0);
        for (idx, &score) in dp[start..end + 1].iter().enumerate() {
            if score < min {
                let location = idx as isize % (len * len);
                let t = location / len - 3;
                let u = location % len - 3;
                min = score;
                min_t = t;
                min_u = u;
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
    let last = *dp.last().unwrap();
    (last, vec![])
}

macro_rules! min_swap {
    ($x:expr,$y:expr) => {
        if $y < $x {
            $x = $y;
        }
    };
}

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
        let mut min = *dp.get_unchecked(get!(s - 1, t_d1, u_d1, len));
        min_swap!(min, *dp.get_unchecked(get!(s - 1, t_d1 - 1, u_d1, len)));
        let score = *dp.get_unchecked(get!(s - 1, t_d1 - 1, u_d1 - 1, len));
        min_swap!(min, score);
        let score = *dp.get_unchecked(get!(s - 2, t_d2 - 1, u_d2, len)) + (x_base != y_base) as u32;
        min_swap!(min, score);
        let score =
            *dp.get_unchecked(get!(s - 2, t_d2 - 2, u_d2 - 1, len)) + (y_base != z_base) as u32;
        min_swap!(min, score);
        let score =
            *dp.get_unchecked(get!(s - 2, t_d2 - 1, u_d2 - 1, len)) + (x_base != z_base) as u32;
        min_swap!(min, score);
        min += 1;
        let score = *dp.get_unchecked(get!(s - 3, t_d3 - 2, u_d3 - 1, len))
            + MA32[(x_base as usize) << 6 | (y_base << 3 | z_base) as usize];
        min_swap!(min, score);
        min
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn min_of_array(xs: &[u32; 7]) -> u32 {
    *xs.iter().min().unwrap()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn min_of_array(xs: &[u32; 7]) -> u32 {
    use std::arch::x86_64;
    use x86_64::{
        __m256i, _mm256_alignr_epi8, _mm256_extract_epi32, _mm256_loadu_si256, _mm256_min_epi32,
        _mm256_permute2x128_si256,
    };
    let mut xs_min = _mm256_loadu_si256(xs.as_ptr() as *const __m256i);
    xs_min = _mm256_min_epi32(xs_min, _mm256_alignr_epi8(xs_min, xs_min, 4));
    xs_min = _mm256_min_epi32(xs_min, _mm256_alignr_epi8(xs_min, xs_min, 8));
    xs_min = _mm256_min_epi32(xs_min, _mm256_permute2x128_si256(xs_min, xs_min, 1));
    _mm256_extract_epi32(xs_min, 0) as u32
}

// #[bench]
// fn bi_aln_naive(b: &mut test::Bencher) {
//     let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
//     let template = kiley::gen_seq::generate_seq(&mut rng, 2_000);
//     let prof = &kiley::gen_seq::PROFILE;
//     b.iter(|| {
//         let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
//         let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
//         kiley::alignment::bialignment::naive_align(&xs, &ys)
//     });
// }

// #[bench]
// fn bi_aln_fast(b: &mut test::Bencher) {
//     let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
//     let template = kiley::gen_seq::generate_seq(&mut rng, 2_000);
//     let prof = &kiley::gen_seq::PROFILE;
//     b.iter(|| {
//         let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
//         let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
//         kiley::alignment::bialignment::fast_align(&xs, &ys)
//     });
// }

// const VECTOR_LEN: u64 = 1_000_000;

// #[bench]
// fn sum_by_u64(b: &mut test::Bencher) {
//     let vectors: Vec<u64> = (0..VECTOR_LEN).collect();
//     b.iter(|| {
//         let mut sum = 0;
//         for &x in vectors.iter() {
//             sum += x;
//         }
//         test::black_box(sum);
//     });
// }

// #[bench]
// fn sum_by_i64(b: &mut test::Bencher) {
//     let vectors: Vec<u64> = (0..VECTOR_LEN).collect();
//     b.iter(|| {
//         let mut sum = 0;
//         for &x in vectors.iter() {
//             sum += x as i64;
//         }
//         test::black_box(sum);
//     });
// }

// #[bench]
// fn access_in_random(b: &mut test::Bencher) {
//     let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(492380);
//     let vectors: Vec<u64> = (0..VECTOR_LEN).collect();
//     let mut access_order: Vec<_> = vectors.iter().enumerate().map(|x| x.0).collect();
//     access_order.shuffle(&mut rng);
//     b.iter(|| test::black_box(access_order.iter().map(|&i| vectors[i]).sum::<u64>()));
// }

// #[bench]
// fn access_two_in_row(b: &mut test::Bencher) {
//     let vector1: Vec<u64> = (0..VECTOR_LEN).collect();
//     let vector2: Vec<u64> = (0..VECTOR_LEN).collect();
//     let access_order: Vec<_> = vector1.iter().enumerate().map(|x| x.0).collect();
//     b.iter(|| {
//         let mut sum1 = 0;
//         let mut sum2 = 0;
//         for &i in access_order.iter() {
//             sum1 += vector1[i];
//         }
//         for &i in access_order.iter() {
//             sum2 += vector2[i];
//         }
//         test::black_box(sum1 + sum2);
//     });
// }

// #[bench]
// fn access_two_in_row_int(b: &mut test::Bencher) {
//     let vector1: Vec<u64> = (0..VECTOR_LEN).collect();
//     let vector2: Vec<u64> = (0..VECTOR_LEN).collect();
//     let access_order: Vec<_> = vector1.iter().enumerate().map(|x| x.0).collect();
//     b.iter(|| {
//         let (mut sum1, mut sum2) = (0, 0);
//         for &i in access_order.iter() {
//             sum1 += vector1[i];
//             sum2 += vector2[i];
//         }
//         test::black_box(sum1 + sum2);
//     });
// }
