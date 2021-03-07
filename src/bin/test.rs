use rand::SeedableRng;
use rand_xoshiro::Xoroshiro128PlusPlus;
fn main() {
    let phmm = kiley::hmm::PHMM::default();
    let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(29814);
    let coverage = 30;
    // let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(2914);
    let profile = kiley::gen_seq::Profile {
        sub: 0.02,
        del: 0.02,
        ins: 0.02,
    };
    let template = kiley::gen_seq::generate_seq(&mut rng, 200);
    let draft = kiley::gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
    let xs: Vec<_> = (0..coverage)
        .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, &profile))
        .collect();
    for x in xs.iter() {
        phmm.forward_banded(&draft, x, 20);
    }
    //let (corrected, lks) = phmm.correct_flip(&draft, &xs, &mut rng, 20);
    // let (corrected, lks) = phmm.correct_flip_banded(&draft, &xs, &mut rng, 20, 20);
    // for i in 0..corrected.len() {
    //     let mat = lks.match_prob[i];
    //     let ins = lks.insertion_prob[i];
    //     let ins_base = lks.insertion_bases[i];
    //     let del = lks.deletion_prob[i];
    //     let mism = (lks.likelihood_trajectry[i + 1] - lks.likelihood_trajectry[i]).exp();
    //     let mism_base = lks.match_bases[i];
    //     println!(
    //         "{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:?}\t{:?}",
    //         i, corrected[i] as char, mat, del, ins, mism, mism_base, ins_base,
    //     );
    // }
    // println!();
    // println!("{}", String::from_utf8_lossy(&corrected));
    // println!("{}", String::from_utf8_lossy(&template));
    // println!("{}", lks.total_likelihood);
    // let true_lk = xs
    //     .iter()
    //     .map(|x| phmm.likelihood(&template, x).1)
    //     .sum::<f64>();
    // println!("{}", true_lk);
    // println!(
    //     "{}",
    //     kiley::alignment::bialignment::edit_dist(&template, &corrected)
    // );
}
