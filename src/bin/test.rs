use kiley::hmm::generalized_pair_hidden_markov_model::GPHMM;
use rand::SeedableRng;
fn main() -> std::io::Result<()> {
    // let phmm = GPHMM::new_conditional_single_state(0.5, 0.3);
    let phmm = GPHMM::new_conditional_three_state(0.9, 0.05, 0.05, 0.958);
    let len: usize = 1000;
    for seed in 20..40u64 {
        // let seed: u64 = 12;
        let coverage: usize = 20;
        let error_rate: f64 = 0.15;
        let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
        use kiley::gen_seq;
        let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
        let prof = gen_seq::PROFILE.norm().mul(error_rate);
        let seqs: Vec<_> = (0..coverage)
            .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
            .collect();
        ///// Kiley
        let draft = kiley::ternary_consensus(&seqs, 231, 4, 30);
        // let draft = gen_seq::introduce_errors(&template, &mut rng, 1, 1, 1);
        let consensus = kiley::polish_until_converge_banded(&draft, &seqs, 30).unwrap();
        let dist = kiley::bialignment::edit_dist(&draft, &template);
        let p_dist = kiley::bialignment::edit_dist(&consensus, &template);
        let start = std::time::Instant::now();
        let consensus = phmm.correct_until_convergence(&draft, &seqs);
        let end = std::time::Instant::now();
        let kiley_time = (end - start).as_millis();
        let kiley_dist = kiley::bialignment::edit_dist(&template, &consensus);
        let lk_old = seqs.iter().map(|q| phmm.likelihood(&draft, q)).sum::<f64>();
        let lk_new: f64 = seqs.iter().map(|q| phmm.likelihood(&consensus, q)).sum();
        let lk_opt: f64 = seqs.iter().map(|q| phmm.likelihood(&template, q)).sum();
        eprintln!("OPT:{:.2} NEW:{:.2} OLD:{:.2}", lk_opt, lk_new, lk_old);
        // println!("{}", String::from_utf8_lossy(&template));
        // println!("{}", String::from_utf8_lossy(&consensus));
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tTernary",
            len, seed, coverage, error_rate, kiley_time, dist, kiley_dist, p_dist
        );
    }
    Ok(())
}

// fn dump(xs: &[u8], ys: &[u8], model: &GPHMM) {
//     let (lk, ops, _states) = model.align(xs, ys);
//     let (xr, opr, yr) = kiley::hmm::recover(xs, ys, &ops);
//     println!("{:.2}", lk);
//     for ((xr, opr), yr) in xr.chunks(200).zip(opr.chunks(200)).zip(yr.chunks(200)) {
//         println!("{}", String::from_utf8_lossy(xr));
//         println!("{}", String::from_utf8_lossy(opr));
//         println!("{}\n", String::from_utf8_lossy(yr));
//     }
// }
