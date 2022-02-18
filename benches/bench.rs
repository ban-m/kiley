#![feature(test)]
extern crate test;
use rand::SeedableRng;
//use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256StarStar;
const SEED: u64 = 1293890;
const SHORT_LEN: usize = 200;

const GLOBAL_BAND: usize = 50;

#[bench]
fn edit_dist_guided(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ops = kiley::bialignment::guided::bootstrap_ops(xs.len(), ys.len());
        kiley::bialignment::guided::edit_dist_guided(&xs, &ys, &ops, GLOBAL_BAND)
    });
}

#[bench]
fn edit_dist(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        kiley::bialignment::edit_dist(&xs, &ys)
    });
}

#[bench]
fn global_banded(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        kiley::bialignment::global_banded(&xs, &ys, 2, -6, -5, -1, GLOBAL_BAND)
    });
}

#[bench]
fn global_guided(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, SHORT_LEN);
    let prof = &kiley::gen_seq::PROFILE;
    b.iter(|| {
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        use kiley::bialignment::guided::*;
        let ops = bootstrap_ops(xs.len(), ys.len());
        global_guided(&xs, &ys, &ops, GLOBAL_BAND, (2, -6, -5, -1))
    });
}

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
fn likelihood_gphmm_banded_ops(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let band = 20;
    use kiley::gphmm::*;
    let phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 500);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        phmm.likelihood_banded(&xs, &ys, band)
    });
}

#[bench]
fn viterbi_guided_hmm_ops(b: &mut test::Bencher) {
    let band = 20;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let mat = (0.8, 0.1, 0.1);
    let ins = (0.8, 0.15, 0.05);
    let del = (0.85, 0.15);
    let mut emission = [0.05 / 3f64; 16];
    for i in 0..4 {
        emission[i * 4 + i] = 0.95;
    }
    let hmm = kiley::hmm::guided::PairHiddenMarkovModel::new(mat, ins, del, &emission);
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 500);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        hmm.align(&xs, &ys, band)
    });
}

#[bench]
fn likelihood_guided_hmm_ops(b: &mut test::Bencher) {
    let band = 20;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let mat = (0.8, 0.1, 0.1);
    let ins = (0.8, 0.15, 0.05);
    let del = (0.85, 0.15);
    let mut emission = [0.05 / 3f64; 16];
    for i in 0..4 {
        emission[i * 4 + i] = 0.95;
    }
    let hmm = kiley::hmm::guided::PairHiddenMarkovModel::new(mat, ins, del, &emission);
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 500);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        hmm.likelihood(&xs, &ys, band)
    });
}

#[bench]
fn likelihood_gphmm_ops(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    use kiley::gphmm::*;
    let phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
    b.iter(|| {
        let template = kiley::gen_seq::generate_seq(&mut rng, 500);
        let xs = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        let ys = kiley::gen_seq::introduce_randomness(&template, &mut rng, prof);
        phmm.likelihood(&xs, &ys)
    });
}

const DRAFT: kiley::gen_seq::Profile = kiley::gen_seq::Profile {
    sub: 0.005,
    del: 0.005,
    ins: 0.005,
};

const CONS_COV: usize = 10;
const CONS_RAD: usize = 10;
const CONS_LEN: usize = 100;

#[bench]
fn polish_gphmm(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    use kiley::gphmm::*;
    let phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
    let template = kiley::gen_seq::generate_seq(&mut rng, CONS_LEN);
    let draft = kiley::gen_seq::introduce_randomness(&template, &mut rng, &DRAFT);
    let xss: Vec<_> = (0..CONS_COV)
        .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, prof))
        .collect();
    b.iter(|| phmm.polish_banded_batch(&draft, &xss, CONS_RAD, 3));
}

#[bench]
fn polish_hmm(b: &mut test::Bencher) {
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let mat = (0.8, 0.1, 0.1);
    let ins = (0.8, 0.15, 0.05);
    let del = (0.85, 0.15);
    let mut emission = [0.05 / 3f64; 16];
    for i in 0..4 {
        emission[i * 4 + i] = 0.95;
    }
    let hmm = kiley::hmm::guided::PairHiddenMarkovModel::new(mat, ins, del, &emission);
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
