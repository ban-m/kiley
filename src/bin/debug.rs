use rand::SeedableRng;
const SEED: u64 = 1293890;

fn main() {
    use kiley::gen_seq;
    let len = 500;
    let cov = 30;
    for seed in 0..10 {
        let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(SEED + seed);
        let template = gen_seq::generate_seq(&mut rng, len);
        let seqs: Vec<_> = (0..cov)
            .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, &gen_seq::PROFILE))
            .collect();
        let consensus = kiley::consensus_kiley(&seqs, SEED + seed, 10, 20);
        eprintln!("{}", edit_dist(&consensus, &template));
        let consensus = kiley::consensus_poa(&seqs, SEED + seed, 10, 10, "CLR");
        eprintln!("{}", edit_dist(&consensus, &template));
        eprintln!();
    }
}

#[allow(dead_code)]
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
