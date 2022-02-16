use kiley::bialignment::edit_dist;
use rand::SeedableRng;
use rayon::prelude::*;
fn main() {
    env_logger::init();
    let args: Vec<_> = std::env::args().collect();
    let len: usize = args[1].parse().unwrap();
    let seed: u64 = args[2].parse().unwrap();
    let coverage: usize = args[3].parse().unwrap();
    let error_rates = vec![0.15, 0.10, 0.05, 0.02, 0.01];
    let radius = vec![10, 20, 30, 40, 50];
    let mut parameters = vec![];
    for &er in error_rates.iter() {
        for &rad in radius.iter() {
            parameters.push((er, rad));
        }
    }
    let result: Vec<_> = parameters
        .par_iter()
        .map(|&(er, rad)| run_bench(len, coverage, er, rad, seed))
        .collect();
    println!("ErrorRate\tRadius\tBandedTime\tBandedDist\tGuidedTime\tGuidedDist");
    for (er, rad, result) in result.iter() {
        for (btime, bdist, gtime, gdist) in result {
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}",
                er, rad, btime, bdist, gtime, gdist
            );
        }
    }
}
type BenchResult = (f64, usize, Vec<(u128, u32, u128, u32)>);
fn run_bench(len: usize, coverage: usize, error_rate: f64, rad: usize, seed: u64) -> BenchResult {
    let seed = seed + (rad + coverage) as u64;
    use kiley::gen_seq;
    let prof = gen_seq::PROFILE.norm().mul(error_rate);
    use kiley::gphmm::*;
    let gphmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
    let mat = (0.8, 0.1, 0.1);
    let ins = (0.8, 0.15, 0.05);
    let del = (0.85, 0.15);
    let mut emission = [0.05 / 3f64; 16];
    for i in 0..4 {
        emission[i * 4 + i] = 0.95;
    }
    let phmm = kiley::hmm::guided::PairHiddenMarkovModel::new(mat, ins, del, &emission);
    let times: Vec<_> = (0..50)
        .into_par_iter()
        .map(|i| {
            let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed + i);
            let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
            let seqs: Vec<_> = (0..coverage)
                .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
                .collect();
            let err = prof.norm().mul(0.001);
            let draft = kiley::gen_seq::introduce_randomness(&template, &mut rng, &err);
            let (btime, bdist) = {
                let start = std::time::Instant::now();
                // let consensus =
                //     kiley::bialignment::polish_until_converge_banded(&draft, &seqs, rad);
                let consensus = gphmm.polish_banded_batch(&draft, &seqs, rad, 3);
                let end = std::time::Instant::now();
                ((end - start).as_millis(), edit_dist(&template, &consensus))
            };
            let (gtime, gdist) = {
                let start = std::time::Instant::now();
                // let consensus =
                //     kiley::bialignment::guided::polish_until_converge(&draft, &seqs, rad);
                let consensus = phmm.polish_until_converge(&draft, &seqs, rad);
                let end = std::time::Instant::now();
                ((end - start).as_millis(), edit_dist(&template, &consensus))
            };
            (btime, bdist, gtime, gdist)
        })
        .collect();
    (error_rate, rad, times)
}
