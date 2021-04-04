use kiley::hmm::generalized_pair_hidden_markov_model::GPHMM;
use kiley::hmm::generalized_pair_hidden_markov_model::*;
use rand::SeedableRng;
fn main() -> std::io::Result<()> {
    let len: usize = 500;
    let seed: u64 = 12;
    let coverage: usize = 20;
    let error_rate: f64 = 0.15;
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    use kiley::gen_seq;
    let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    let prof = gen_seq::PROFILE.norm().mul(error_rate);
    let seqs: Vec<_> = (0..coverage)
        .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
        .collect();
    let draft = kiley::ternary_consensus(&seqs, 231, 4, 30);
    let mut phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.958);
    let mut lk: f64 = seqs.iter().map(|q| phmm.likelihood(&draft, q)).sum();
    let start = std::time::Instant::now();
    let mut consensus = phmm.correct_until_convergence(&draft, &seqs);
    eprintln!("{}", phmm);
    loop {
        eprintln!("{}", String::from_utf8_lossy(&consensus));
        let new_cons = loop {
            let new_phmm = phmm.fit(&consensus, &seqs);
            let new_lk: f64 = seqs
                .iter()
                .map(|q| new_phmm.likelihood(&consensus, q))
                .sum();
            eprintln!("LK:{:.4}, {:.4}", lk, new_lk);
            if new_lk - lk < 0.1f64 {
                eprintln!("Break");
                break phmm.correct_until_convergence(&consensus, &seqs);
            } else {
                lk = new_lk;
                phmm = new_phmm;
            }
        };
        if consensus == new_cons {
            break;
        } else {
            consensus = new_cons;
        }
    }
    eprintln!("{}\nLK:{:.4}", phmm, lk);
    let end = std::time::Instant::now();
    let kiley_time = (end - start).as_millis();
    let kiley_dist = kiley::bialignment::edit_dist(&template, &consensus);
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\tTernary",
        len, seed, coverage, error_rate, kiley_time, kiley_dist,
    );
    // for q in seqs.iter() {
    //     dump(&consensus, q, &phmm);
    // }
    Ok(())
}

// fn dump<T: HMMType>(xs: &[u8], ys: &[u8], model: &GPHMM<T>) {
//     let (lk, ops, _states) = model.align(xs, ys);
//     let (xr, opr, yr) = kiley::hmm::recover(xs, ys, &ops);
//     println!("{:.2}", lk);
//     for ((xr, opr), yr) in xr.chunks(200).zip(opr.chunks(200)).zip(yr.chunks(200)) {
//         println!("{}", String::from_utf8_lossy(xr));
//         println!("{}", String::from_utf8_lossy(opr));
//         println!("{}\n", String::from_utf8_lossy(yr));
//     }
// }
