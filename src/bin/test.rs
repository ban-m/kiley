use rand::SeedableRng;

fn main() {
    use kiley::gen_seq;
    let coverage = 10;
    let rad = 20;
    let prof = gen_seq::ProfileWithContext::default();
    let seed = 36;
    let len = 1000;
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    let seqs: Vec<_> = (0..coverage)
        .map(|_| gen_seq::introduce_randomness_with_context(&template, &mut rng, &prof))
        .collect();
    use kiley::bialignment::edit_dist;
    let (gtime, gdist, after) = {
        let start = std::time::Instant::now();
        let draft = kiley::ternary_consensus_by_chunk(&seqs, rad);
        let consensus = kiley::bialignment::guided::polish_until_converge(&draft, &seqs, rad);
        let end = std::time::Instant::now();
        let before = edit_dist(&template, &draft);
        let after = edit_dist(&template, &consensus);
        ((end - start).as_millis(), before, after)
    };
    println!("{}\t{}\t{}", gtime, gdist, after);
}
