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
    let len = 2_000;
    let seed = 100;
    for seed in 0..seed {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
        let template = kiley::gen_seq::generate_seq(&mut rng, len);
        let query = profile.gen(&template, &mut rng);
        bench(&template, &query, seed);
    }
}

const RADIUS: usize = 100;
fn bench(template: &[u8], query: &[u8], seed: u64) {
    let hmm = kiley::hmm::PairHiddenMarkovModel::default();
    let ((lk_g, _), guided) = elapsed!(hmm.align_guided_bootstrap(template, query, RADIUS));
    let ((lk_f, op_f), full) = elapsed!(hmm.align(template, query));
    let ((lk_a, op_a), anti) = elapsed!(hmm.align_antidiagonal_bootstrap(template, query, RADIUS));
    assert!((lk_g - lk_f).abs() < 0.001);
    assert!(
        (lk_a - lk_f).abs() < 0.001,
        "{},{}\n{:?}\n{:?}",
        lk_f,
        lk_a,
        op_f,
        op_a
    );
    println!(
        "{seed}\t{}\t{}\t{}",
        full.as_micros(),
        guided.as_micros(),
        anti.as_micros()
    );
}
