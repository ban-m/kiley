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
        kiley::trialignment::naive::alignment(&xs, &ys, &zs)
    });
}

#[bench]
fn gphmm_banded_ops(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let band = 20;
    use kiley::gphmm::*;
    let phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 2_000);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        phmm.likelihood_banded(&xs, &ys, band)
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
        kiley::bialignment::edit_dist(&xs, &ys)
    });
}

const HMMLEN: usize = 500;

#[bench]
fn hmm_naive(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let profile = kiley::gen_seq::Profile {
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };
    let phmm = kiley::hmm::PHMM::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, HMMLEN);
        let seq = kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile);
        test::black_box(phmm.likelihood(&template, &seq))
    });
}

#[bench]
fn hmm_banded(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let profile = kiley::gen_seq::Profile {
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };
    let phmm = kiley::hmm::PHMM::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, HMMLEN);
        let seq = kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile);
        test::black_box(phmm.likelihood_banded(&template, &seq, 20))
    });
}

#[bench]
fn hmm_forward(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let profile = kiley::gen_seq::Profile {
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };
    let phmm = kiley::hmm::PHMM::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, HMMLEN);
        let seq = kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile);
        test::black_box(phmm.forward(&template, &seq))
    });
}

#[bench]
fn hmm_forward_banded(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let profile = kiley::gen_seq::Profile {
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };
    let phmm = kiley::hmm::PHMM::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, HMMLEN);
        let seq = kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile);
        test::black_box(phmm.forward_banded(&template, &seq, 20))
    });
}

#[bench]
fn hmm_forward_backward(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let profile = kiley::gen_seq::Profile {
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };
    let phmm = kiley::hmm::PHMM::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, HMMLEN);
        let seq = kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile);
        test::black_box(phmm.get_profile(&template, &seq))
    });
}

#[bench]
fn hmm_forward_backward_banded(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let profile = kiley::gen_seq::Profile {
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };
    let phmm = kiley::hmm::PHMM::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, HMMLEN);
        let seq = kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile);
        test::black_box(phmm.get_profile_banded(&template, &seq, 20))
    });
}
