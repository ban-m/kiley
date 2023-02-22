const LEN: usize = 2_000;
const COVERAGE: usize = 20;
const RADIUS: usize = 50;
const PACK: usize = 4;
const SEED: u64 = 309482;
use kiley::{gen_seq::Generate, hmm::TrainingDataPack};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro128StarStar;

macro_rules! elapsed {
    ($a:expr) => {{
        let start = std::time::Instant::now();
        let return_value = $a;
        let end = std::time::Instant::now();
        (return_value, (end - start))
    }};
}

fn main() {
    let hmm = kiley::hmm::PairHiddenMarkovModelOnStrands::default();
    let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(SEED);
    let profile = kiley::gen_seq::Profile::new(0.03, 0.03, 0.03);
    let templates: Vec<_> = (0..PACK)
        .map(|_| kiley::gen_seq::generate_seq(&mut rng, LEN))
        .collect();
    let reads: Vec<Vec<_>> = templates
        .iter()
        .map(|rs| (0..COVERAGE).map(|_| profile.gen(rs, &mut rng)).collect())
        .collect();
    let strands: Vec<Vec<_>> = (0..PACK)
        .map(|_| (0..COVERAGE).map(|_| rng.gen_bool(0.5)).collect())
        .collect();
    let opss: Vec<Vec<_>> = std::iter::zip(&templates, &reads)
        .map(|(rs, qss)| {
            qss.iter()
                .map(|qs| hmm.forward().align_antidiagonal_bootstrap(rs, qs, RADIUS).1)
                .collect()
        })
        .collect();
    let training_data: Vec<_> = std::iter::zip(&templates, &reads)
        .zip(strands.iter())
        .zip(opss.iter())
        .map(|(((rs, qss), strands), ops)| TrainingDataPack::new(rs, strands, qss, ops))
        .collect();
    let mut hmm_g = hmm.clone();
    let g = elapsed!(hmm_g.fit_guided_par_multiple(&training_data, RADIUS)).1;
    let mut hmm_a = hmm;
    let a = elapsed!(hmm_a.fit_antidiagonal_par_multiple(&training_data, RADIUS / 2)).1;
    println!("{hmm_g}");
    println!("{hmm_a}");
    println!("{}\t{}", g.as_micros(), a.as_micros());
}
