use rand::SeedableRng;
fn main() -> std::io::Result<()> {
    let len: usize = 500;
    let seed: u64 = 12;
    let coverage: usize = 20;
    let error_rate: f64 = 0.15;
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    use kiley::gen_seq;
    let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    let prof = gen_seq::PROFILE.norm().mul(error_rate);
    let seqs: Vec<_> = (0..coverage)
        .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
        .collect();
    // Kiley
    {
        let start = std::time::Instant::now();
        let consensus = kiley::consensus(&seqs, 24, 3, 20).unwrap();
        let end = std::time::Instant::now();
        let time = (end - start).as_millis();
        let dist = kiley::bialignment::edit_dist(&template, &consensus);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\tKiley",
            len, seed, coverage, error_rate, time, dist,
        );
    }
    // Old
    {
        let start = std::time::Instant::now();
        let consensus = kiley::consensus_bounded(&seqs, 24, 3, 20, 100).unwrap();
        let end = std::time::Instant::now();
        let time = (end - start).as_millis();
        let dist = kiley::bialignment::edit_dist(&template, &consensus);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\tOld",
            len, seed, coverage, error_rate, time, dist,
        );
    }
    Ok(())
}
