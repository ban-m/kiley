use std::time::Duration;

use kiley::{gen_seq::Generate, hmm::HMMPolishConfig};
macro_rules! elapsed {
    ($a:expr) => {{
        let start = std::time::Instant::now();
        let return_value = $a;
        let end = std::time::Instant::now();
        (return_value, (end - start))
    }};
}

fn main() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    let profile = kiley::gen_seq::Profile {
        sub: 0.03,
        del: 0.03,
        ins: 0.03,
    };
    let draft = kiley::gen_seq::Profile {
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };

    let len = 1_000;
    let seed = 2;
    let is_anti = std::env::args().nth(1) == Some("anti".to_string());
    let mut anti = 0;
    let mut diff = 0;
    let coverage = 20;
    for seed in 0..seed {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
        let template = kiley::gen_seq::generate_seq(&mut rng, len);
        let d_seq = draft.gen(&template, &mut rng);
        let reads: Vec<_> = (0..coverage)
            .map(|_| profile.gen(&template, &mut rng))
            .collect();
        // let query = profile.gen(&template, &mut rng);
        let (seq, a) = bench(&d_seq, &reads, is_anti);
        diff += kiley::bialignment::edit_dist(&seq, &template);
        anti += a.as_micros();
    }
    println!("{anti}\t{diff}");
}
const RADIUS: usize = 100;

fn bench(draft: &[u8], xss: &[Vec<u8>], is_anti: bool) -> (Vec<u8>, Duration) {
    let hmm = kiley::hmm::PairHiddenMarkovModel::default();
    let mut opss: Vec<_> = xss
        .iter()
        .map(|xs| hmm.align_antidiagonal_bootstrap(draft, xs, RADIUS).1)
        .collect();
    let opss = &mut opss;
    let config = HMMPolishConfig::new(RADIUS, xss.len(), 0);
    let a_config = HMMPolishConfig::new(RADIUS / 2, xss.len(), 0);
    match is_anti {
        true => elapsed!(hmm.polish_until_converge_antidiagonal(draft, xss, opss, &a_config)),
        false => elapsed!(hmm.polish_until_converge_guided(draft, xss, opss, &config)),
    }
}
// fn bench(template: &[u8], query: &[Vec<u8>], is_anti: bool) -> Duration {
//     let hmm = kiley::hmm::PairHiddenMarkovModel::default();
// let ops = hmm.align_antidiagonal_bootstrap(template, query, RADIUS).1;
// match is_anti {
//     true => elapsed!(hmm.likelihood_antidiagonal(template, query, &ops, RADIUS)).1,
//     false => elapsed!(hmm.likelihood_guided(template, query, &ops, RADIUS)).1,
// }
// match is_anti {
//     true => elapsed!(hmm.modification_table_antidiagonal(template, query, &ops, RADIUS / 2)).1,
//     false => elapsed!(hmm.modification_table_guided(template, query, &ops, RADIUS)).1,
// }
// }
