#![feature(test)]
extern crate test;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use test::bench::black_box;
const SEED: u64 = 1293890;
const SHORT_LEN: usize = 200;

const GLOBAL_BAND: usize = 50;

#[bench]
fn edit_dist_guided(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
    let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
    let ops = kiley::bialignment::edit_dist_ops(&xs, &ys).1;
    b.iter(|| {
        black_box(kiley::bialignment::guided::edit_dist_guided(
            &xs,
            &ys,
            &ops,
            GLOBAL_BAND,
        ))
    });
}

#[bench]
fn edit_dist(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
    let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
    b.iter(|| black_box(kiley::bialignment::edit_dist(&xs, &ys)));
}

#[bench]
fn global_banded(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
    let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
    b.iter(|| {
        let score = kiley::bialignment::banded::global_banded(&xs, &ys, 2, -6, -5, -1, GLOBAL_BAND);
        black_box(score)
    });
}

#[bench]
fn global_guided(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
    let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
    let ops = kiley::bialignment::edit_dist_ops(&xs, &ys).1;
    b.iter(|| {
        use kiley::bialignment::guided::*;
        let score = global_guided(&xs, &ys, &ops, GLOBAL_BAND, (2, -6, -5, -1));
        black_box(score)
    });
}

#[bench]
fn viterbi_guided_hmm_ops(b: &mut test::Bencher) {
    let band = 20;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let hmm = kiley::hmm::PairHiddenMarkovModel::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 500);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        hmm.align_guided(&xs, &ys, band)
    });
}

#[bench]
fn likelihood_guided_hmm_ops(b: &mut test::Bencher) {
    let band = 20;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let hmm = kiley::hmm::PairHiddenMarkovModel::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 500);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        hmm.likelihood_bootstrap(&xs, &ys, band)
    });
}

const DRAFT: kiley::gen_seq::Profile = kiley::gen_seq::Profile {
    sub: 0.007,
    del: 0.007,
    ins: 0.007,
};

const CONS_COV: usize = 10;
const CONS_RAD: usize = 10;
const CONS_LEN: usize = 200;

#[bench]
fn polish_hmm(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let hmm = kiley::hmm::PairHiddenMarkovModel::default();
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, CONS_LEN);
        let draft = kiley::gen_seq::introduce_randomness(&template, &mut rng, &DRAFT);
        let xss: Vec<_> = (0..CONS_COV)
            .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, prof))
            .collect();
        hmm.polish_until_converge(&draft, &xss, CONS_RAD)
    });
}

#[bench]
fn polish_edit_dist_p_guided(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, CONS_LEN);
        let draft = kiley::gen_seq::introduce_randomness(&template, &mut rng, &DRAFT);
        let xss: Vec<_> = (0..CONS_COV)
            .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, prof))
            .collect();
        kiley::bialignment::guided::polish_until_converge(&draft, &xss, CONS_RAD)
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
