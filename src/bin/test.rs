use std::time::Duration;

use kiley::gen_seq::Generate;
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
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };
    let len = 1_000;
    let seed = 20;
    let is_anti = std::env::args().nth(1) == Some("anti".to_string());
    let mut anti = 0;
    for seed in 0..seed {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
        let template = kiley::gen_seq::generate_seq(&mut rng, len);
        let query = profile.gen(&template, &mut rng);
        let a = bench(&template, &query, is_anti);
        anti += a.as_micros();
    }
    println!("{anti}");
}

const RADIUS: usize = 100;
fn bench(template: &[u8], query: &[u8], is_anti: bool) -> Duration {
    let hmm = kiley::hmm::PairHiddenMarkovModel::default();
    let ops = hmm.align_antidiagonal_bootstrap(template, query, RADIUS).1;
    // match is_anti {
    //     true => elapsed!(hmm.likelihood_antidiagonal(template, query, &ops, RADIUS)).1,
    //     false => elapsed!(hmm.likelihood_guided(template, query, &ops, RADIUS)).1,
    // }
    match is_anti {
        true => elapsed!(hmm.modification_table_antidiagonal(template, query, &ops, RADIUS)).1,
        false => elapsed!(hmm.modification_table_guided(template, query, &ops, RADIUS)).1,
    }
}
