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
fn edit_dist_naive(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 2_000);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        kiley::alignment::bialignment::edit_dist_slow(&xs, &ys)
    });
}

#[bench]
fn edit_dist(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 2_000);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        kiley::alignment::bialignment::edit_dist(&xs, &ys)
    });
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
