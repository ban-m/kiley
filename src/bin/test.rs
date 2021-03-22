fn main() -> std::io::Result<()> {
    // use std::io::*;
    // let args: Vec<_> = std::env::args().collect();
    // let seqs: Vec<Vec<u8>> = std::fs::File::open(&args[1])
    //     .map(BufReader::new)?
    //     .lines()
    //     .filter_map(|line| line.ok())
    //     .map(|line| line.into_bytes())
    //     .collect();
    // for s in seqs.iter() {
    //     println!("{}", s.len());
    // }
    // let start = std::time::Instant::now();
    // let corrected = kiley::consensus(&seqs, 30294, 3, 30).unwrap();
    // let end = std::time::Instant::now();
    // let time = (end - start).as_millis();
    // println!("{}\t{}", corrected.len(), time);
    let len: usize = 2000;
    let coverage: usize = 20;
    let error_rate: f64 = 0.15;
    use kiley::gen_seq;
    let dif = gen_seq::Profile {
        sub: 0.001,
        ins: 0.001,
        del: 0.001,
    };
    use rand::SeedableRng;
    let phmm = kiley::hmm::PHMM::default();
    for seed in 0..100 {
        let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
        let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
        let prof = gen_seq::PROFILE.norm().mul(error_rate);
        let seqs: Vec<_> = (0..coverage)
            .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
            .collect();
        let draft = gen_seq::introduce_randomness(&template, &mut rng, &dif);
        let start = std::time::Instant::now();
        // use kiley::alignment::bialignment;
        // let corrected = bialignment::polish_until_converge_banded(&draft, &seqs, 20).unwrap();
        let corrected = phmm.correct_flip_banded(&draft, &seqs, &mut rng, 10, 30).0;
        let end = std::time::Instant::now();
        let dist = kiley::alignment::bialignment::edit_dist(&template, &corrected);
        // let bdist = kiley::alignment::bialignment::edit_dist(&template, &draft);
        let time = (end - start).as_millis();
        println!("{}\t{}\t{}", seed, dist, time);
        let corrected = kiley::consensus(&seqs, 10, 3, 20).unwrap();
        let dist = kiley::alignment::bialignment::edit_dist(&template, &corrected);
        println!("{}\t{}\t{}\tP", seed, dist, time);
        // println!("{}\t{}\t{}\t{}", seed, bdist, dist, time);
    }
    Ok(())
}
