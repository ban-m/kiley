const SEED: u64 = 1293890;
const CONS_COV: usize = 10;
const CONS_RAD: usize = 10;
const CONS_LEN: usize = 100;
fn main() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let template = kiley::gen_seq::generate_seq(&mut rng, CONS_LEN);
    let xss: Vec<_> = (0..CONS_COV)
        .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, prof))
        .collect();
    // Usual.
    {
        use kiley::gphmm::*;
        let mut phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        for t in 0..10 {
            let s = std::time::Instant::now();
            phmm = phmm.fit(&template, &xss);
            let e = std::time::Instant::now();
            let lk: f64 = xss
                .iter()
                .map(|x| phmm.likelihood_banded(&template, x, CONS_RAD).unwrap())
                .sum();
            let time = (e - s).as_millis();
            println!("Exac\t{}\t{}\t{}", t, time, lk);
        }
    }
    // Banded
    {
        use kiley::gphmm::*;
        let mut phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        for t in 0..10 {
            let s = std::time::Instant::now();
            phmm = phmm.fit_banded(&template, &xss, CONS_RAD);
            let e = std::time::Instant::now();
            let lk: f64 = xss
                .iter()
                .map(|x| phmm.likelihood_banded(&template, x, CONS_RAD).unwrap())
                .sum();
            let time = (e - s).as_millis();
            println!("Banded\t{}\t{}\t{}", t, time, lk);
        }
    }
    {
        let mat = (0.8, 0.1, 0.1);
        let ins = (0.8, 0.15, 0.05);
        let del = (0.85, 0.15);
        let mut emission = [0.05 / 3f64; 16];
        for i in 0..4 {
            emission[i * 4 + i] = 0.95;
        }
        let mut hmm = kiley::hmm::guided::PairHiddenMarkovModel::new(mat, ins, del, &emission);
        let ops: Vec<_> = xss
            .iter()
            .map(|xs| kiley::bialignment::guided::bootstrap_ops(template.len(), xs.len()))
            .collect();
        for t in 0..10 {
            let s = std::time::Instant::now();
            hmm.fit_guided(&template, &xss, &ops, CONS_RAD);
            let e = std::time::Instant::now();
            let lk: f64 = xss
                .iter()
                .map(|x| hmm.likelihood(&template, x, CONS_RAD))
                .sum();
            let time = (e - s).as_millis();
            println!("Banded\t{}\t{}\t{}", t, time, lk);
        }
    }
}
