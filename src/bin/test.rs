use kiley::gen_seq;
use kiley::gphmm::*;
use rand::SeedableRng;
// use rayon::prelude::*;
fn main() -> std::io::Result<()> {
    env_logger::init();
    let coverage: usize = 20;
    let error_rate: f64 = 0.15;
    // let prof = gen_seq::PROFILE.norm().mul(error_rate);
    let prof = gen_seq::Profile {
        sub: 0.05,
        del: 0.10,
        ins: 0.00,
    };
    let len: usize = 100;
    let radius = 50;
    // let model = {
    //     let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(320);
    //     let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    //     let seqs: Vec<_> = (0..coverage)
    //         .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
    //         .collect();
    //     // Kiley

    //     // let consensus = kiley::consensus_bounded(&seqs, 24, 3, 20, 100).unwrap();
    //     // let config = kiley::PolishConfig::new(100, 0, radius);
    //     let phmm = kiley::gphmm::GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.1, 0.95);
    //     phmm.fit(&template, &seqs)
    //         .fit(&template, &seqs)
    //         .fit(&template, &seqs)
    //         .fit(&template, &seqs)
    // };
    // let _result = (20u64..30u64)
    //     .map(|seed| {
    let seed = 3422;
    let model = kiley::gphmm::GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.10, 0.95);
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    let seqs: Vec<_> = (0..coverage)
        .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
        .collect();
    let consensus = kiley::consensus_bounded(&seqs, 24, 3, 20, radius).unwrap();
    // Kiley
    {
        println!("{}", model);
        let config = kiley::PolishConfig::with_model(radius, 0, 1, 0, 0, model);
        let model = kiley::fit_model(&template, &seqs, &config);
        println!("{}", model);
        let start = std::time::Instant::now();
        let consensus = model
            .correct_until_convergence_banded(&consensus, &seqs, 100)
            .unwrap();
        let end = std::time::Instant::now();
        let time = (end - start).as_millis();
        let dist = kiley::bialignment::edit_dist(&template, &consensus);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\tKiley",
            len, seed, coverage, error_rate, time, dist,
        );
    }
    // let start = std::time::Instant::now();
    // let consensus =
    //     kiley::bialignment::polish_until_converge_banded(&consensus, &seqs, radius)
    //         .unwrap();
    // let end = std::time::Instant::now();
    // let time = (end - start).as_millis();
    // let dist = kiley::bialignment::edit_dist(&template, &consensus);
    // println!(
    //     "{}\t{}\t{}\t{}\t{}\t{}\tOld",
    //     len, seed, coverage, error_rate, time, dist,
    // );
    // })
    // .count();
    Ok(())
}
