const SEED: u64 = 4320948;
const HMMLEN: usize = 100;
fn main() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let profile = kiley::gen_seq::Profile {
        sub: 0.01,
        del: 0.01,
        ins: 0.01,
    };
    let mut time = std::time::Duration::new(0, 0);
    let phmm = kiley::hmm::PHMM::default();
    for _ in 0..100 {
        let template = kiley::gen_seq::generate_seq(&mut rng, HMMLEN);
        let seq = kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile);
        let start = std::time::Instant::now();
        phmm.modification_table_full(&template, &seq);
        time += std::time::Instant::now() - start;
    }
    println!("{}", time.as_millis());
}
