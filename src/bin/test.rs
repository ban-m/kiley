use rand::SeedableRng;
fn main() {
    let phmm = kiley::hmm::PHMM::default();
    let len: usize = 2000;
    let seed: u64 = 8;
    let coverage: usize = 20;
    let error_rate: f64 = 0.15;
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    use kiley::gen_seq;
    let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    let prof = gen_seq::PROFILE.norm().mul(error_rate);
    let seqs: Vec<_> = (0..coverage)
        .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
        .collect();
    let draft = kiley::consensus_poa(&seqs, seed, 10, 10, "CLR");
    let (corrected, lks) = phmm.correct_flip_banded(&draft, &seqs, &mut rng, 20, 5);
    for i in 0..corrected.len() {
        let mat = lks.match_prob[i];
        let ins = lks.insertion_prob[i];
        let ins_base = lks.insertion_bases[i];
        let del = lks.deletion_prob[i];
        let mism_base = lks.match_bases[i];
        println!(
            "{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:?}\t{:?}",
            i, corrected[i] as char, mat, del, ins, mism_base, ins_base,
        );
    }
    println!();
    for (x, y) in corrected.chunks(200).zip(template.chunks(200)) {
        println!("{}", String::from_utf8_lossy(x));
        println!("{}", String::from_utf8_lossy(y));
        println!();
    }
    println!("{}", lks.total_likelihood);
    let true_lk = seqs
        .iter()
        .map(|x| phmm.likelihood_banded(&template, x, 20).unwrap().1)
        .sum::<f64>();
    println!("{}", true_lk);
    println!(
        "{}",
        kiley::alignment::bialignment::edit_dist(&template, &corrected)
    );
}
