use rand::SeedableRng;
fn main() {
    env_logger::init();
    let args: Vec<_> = std::env::args().collect();
    let len: usize = args[1].parse().unwrap();
    let seed: u64 = args[2].parse().unwrap();
    let coverage: usize = args[3].parse().unwrap();
    let error_rate: f64 = args[4].parse().unwrap();
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    use kiley::gen_seq;
    let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    let prof = gen_seq::PROFILE.norm().mul(error_rate);
    let seqs: Vec<_> = (0..coverage)
        .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
        .collect();
    {
        ///// Kiley
        let start = std::time::Instant::now();
        let consensus = kiley::consensus(&seqs, seed, 3, 15);
        let end = std::time::Instant::now();
        let kiley_time = (end - start).as_millis();
        let kiley_dist = edit_dist(&template, &consensus);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\tKiley",
            len, seed, coverage, error_rate, kiley_time, kiley_dist
        );
    }
    //// Ternary
    {
        let start = std::time::Instant::now();
        // let consensus = kiley::consensus_poa(&seqs, seed, 10, 10, "CLR");
        let consensus = kiley::ternary_consensus_by_chunk(&seqs, 100);
        let end = std::time::Instant::now();
        let poa_time = (end - start).as_millis();
        let poa_dist = edit_dist(&template, &consensus);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\tTern",
            len, seed, coverage, error_rate, poa_time, poa_dist
        );
    }
    //// SWG
    {
        let start = std::time::Instant::now();
        let radius = len / 10;
        let params = (2, -4, -4, -2);
        let consensus = kiley::consensus_by_pileup_affine(&seqs, params, radius, 10);
        let end = std::time::Instant::now();
        let poa_time = (end - start).as_millis();
        let poa_dist = edit_dist(&template, &consensus);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\tSWG",
            len, seed, coverage, error_rate, poa_time, poa_dist
        );
    }
}

fn edit_dist(x1: &[u8], x2: &[u8]) -> u32 {
    let mut dp = vec![vec![0; x2.len() + 1]; x1.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i as u32;
    }
    for j in 0..=x2.len() {
        dp[0][j] = j as u32;
    }
    for (i, x1_b) in x1.iter().enumerate() {
        for (j, x2_b) in x2.iter().enumerate() {
            let m = if x1_b == x2_b { 0 } else { 1 };
            dp[i + 1][j + 1] = (dp[i][j + 1] + 1).min(dp[i + 1][j] + 1).min(dp[i][j] + m);
        }
    }
    dp[x1.len()][x2.len()]
}
