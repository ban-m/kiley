use kiley::gen_seq;
use rand::SeedableRng;
// use rayon::prelude::*;
fn main() -> std::io::Result<()> {
    env_logger::init();
    let coverage: usize = 20;
    let error_rate: f64 = 0.15;
    let prof = gen_seq::PROFILE.norm().mul(error_rate);
    let len: usize = 2000;
    let radius = 50;
    let model = {
        let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(320);
        let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
        let seqs: Vec<_> = (0..coverage)
            .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
            .collect();
        // Kiley
        let consensus = kiley::consensus_bounded(&seqs, 24, 3, 20, 100).unwrap();
        let config = kiley::PolishConfig::new(100, 0, 20);
        kiley::fit_model(&consensus, &seqs, &config)
    };
    let _result = (20u64..30u64)
        .map(|seed| {
            let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
            let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
            let seqs: Vec<_> = (0..coverage)
                .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
                .collect();
            let consensus = kiley::consensus_bounded(&seqs, 24, 3, 20, radius).unwrap();
            // Kiley
            {
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
            // })
            // .collect();
            // Old
            // let odist = {
            let start = std::time::Instant::now();
            let consensus =
                kiley::bialignment::polish_until_converge_banded(&consensus, &seqs, radius)
                    .unwrap();
            let end = std::time::Instant::now();
            let time = (end - start).as_millis();
            let dist = kiley::bialignment::edit_dist(&template, &consensus);
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\tOld",
                len, seed, coverage, error_rate, time, dist,
            );
            //     dist
            // };
            // POA
            // {
            //     let start = std::time::Instant::now();
            //     let consensus = kiley::consensus_poa(&seqs, seed, 10, 10, "CLR");
            //     let end = std::time::Instant::now();
            //     let time = (end - start).as_millis();
            //     let dist = kiley::bialignment::edit_dist(&template, &consensus);
            //     println!(
            //         "{}\t{}\t{}\t{}\t{}\t{}\tPOA",
            //         len, seed, coverage, error_rate, time, dist,
            //     );
            // }
            // kdist <= odist
        })
        .count();
    // println!("{}", result);
    Ok(())
}
