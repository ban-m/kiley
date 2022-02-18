use rand::SeedableRng;

fn main() {
    use kiley::gen_seq;
    let error_rate = 0.1;
    let coverage = 10;
    let rad = 20;
    let prof = gen_seq::PROFILE.norm().mul(error_rate);
    let seed = 36;
    let len = 1000;
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    let seqs: Vec<_> = (0..coverage)
        .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
        .collect();
    let err = prof.norm().mul(0.03);
    use kiley::bialignment::edit_dist;
    let draft = kiley::gen_seq::introduce_randomness(&template, &mut rng, &err);
    let before = edit_dist(&template, &draft);
    let (gtime, gdist) = {
        let start = std::time::Instant::now();
        let consensus = kiley::bialignment::guided::polish_until_converge(&draft, &seqs, rad);
        let end = std::time::Instant::now();
        ((end - start).as_millis(), edit_dist(&template, &consensus))
    };
    eprintln!("{}\t{}\t{}", gtime, gdist, before);
}
