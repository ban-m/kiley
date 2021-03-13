use rand::SeedableRng;
fn main() {
    let len: usize = 2000;
    let coverage: usize = 20;
    let error_rate: f64 = 0.15;
    for seed in 20..30 {
        // let seed: u64 = 8;
        let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
        use kiley::gen_seq;
        let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
        let prof = gen_seq::PROFILE.norm().mul(error_rate);
        let seqs: Vec<_> = (0..coverage)
            .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
            .collect();
        let draft = kiley::consensus_poa(&seqs, seed, 10, 10, "CLR");
        let start = std::time::Instant::now();
        use kiley::alignment::bialignment;
        // let corrected = bialignment::polish_until_converge(&draft, &seqs);
        let corrected = bialignment::polish_until_converge_banded(&draft, &seqs, 20);
        let end = std::time::Instant::now();
        let dist = kiley::alignment::bialignment::edit_dist(&template, &corrected);
        let bdist = kiley::alignment::bialignment::edit_dist(&template, &draft);
        let time = (end - start).as_millis();
        println!("{}\t{}\t{}\t{}", seed, bdist, dist, time);
    }
}
