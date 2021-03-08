use rand::SeedableRng;
use rand_xoshiro::Xoroshiro128PlusPlus;
fn main() {
    let phmm = kiley::hmm::PHMM::default();
    let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(29814);
    let coverage = 20;
    // let profile = kiley::gen_seq::Profile {
    //     sub: 0.02,
    //     del: 0.02,
    //     ins: 0.02,
    // };
    // length * 2 * radius * 3
    // use rand::Rng;
    // let lks: Vec<_> = (0..2 * 2000 * 40 * 3)
    //     .map(|_| rng.gen::<f64>().abs())
    //     .collect();
    // let start = std::time::Instant::now();
    // let tot = lks.iter().map(|x| x.exp().ln()).sum::<f64>();
    // let end = std::time::Instant::now();
    // println!("{},{:.1}", (end - start).as_millis(), tot);
    let profile = kiley::gen_seq::PROFILE;
    let template = kiley::gen_seq::generate_seq(&mut rng, 2000);
    let draft = kiley::gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
    let xs: Vec<_> = (0..coverage)
        .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile))
        .collect();
    let (corrected, lks) = phmm.correct_flip_banded(&draft, &xs, &mut rng, 20, 5);
    // for i in 0..corrected.len() {
    //     let mat = lks.match_prob[i];
    //     let ins = lks.insertion_prob[i];
    //     let ins_base = lks.insertion_bases[i];
    //     let del = lks.deletion_prob[i];
    //     let mism_base = lks.match_bases[i];
    //     println!(
    //         "{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:?}\t{:?}",
    //         i, corrected[i] as char, mat, del, ins, mism_base, ins_base,
    //     );
    // }
    println!();
    println!("{}", String::from_utf8_lossy(&corrected));
    println!("{}", String::from_utf8_lossy(&template));
    println!("{}", lks.total_likelihood);
    let true_lk = xs
        .iter()
        .map(|x| phmm.likelihood_banded(&template, x, 20).unwrap().1)
        .sum::<f64>();
    println!("{}", true_lk);
    println!(
        "{}",
        kiley::alignment::bialignment::edit_dist(&template, &corrected)
    );
}
