const SEED: u64 = 129004923;
const CONS_RAD: usize = 2;
const CONS_LEN: usize = 10;
const TIME: usize = 2;
fn main() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, CONS_LEN);
    // let prof = &kiley::gen_seq::PROFILE;
    // let xss: Vec<_> = (0..CONS_COV)
    //     .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, prof))
    //     .collect();
    // let xss: Vec<_> = (0..CONS_COV)
    //     .map(|_| kiley::gen_seq::introduce_errors(&template, &mut rng, 0, 1, 1))
    //     .collect();
    let mut xs = template.clone();
    // xs.remove(2);
    xs.insert(7, b'T');
    let xss = vec![xs];
    // Usual.
    // {
    //     use kiley::gphmm::*;
    //     let mut phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
    //     for t in 0..10 {
    //         let s = std::time::Instant::now();
    //         phmm = phmm.fit(&template, &xss);
    //         let e = std::time::Instant::now();
    //         let lk: f64 = xss
    //             .iter()
    //             .map(|x| phmm.likelihood_banded(&template, x, CONS_RAD).unwrap())
    //             .sum();
    //         let time = (e - s).as_millis();
    //         println!("Exac\t{}\t{}\t{}", t, time, lk);
    //     }
    // }
    // // Banded
    // {
    //     use kiley::gphmm::*;
    //     let mut phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
    //     for t in 0..10 {
    //         let s = std::time::Instant::now();
    //         phmm = phmm.fit_banded(&template, &xss, CONS_RAD);
    //         let e = std::time::Instant::now();
    //         let lk: f64 = xss
    //             .iter()
    //             .map(|x| phmm.likelihood_banded(&template, x, CONS_RAD).unwrap())
    //             .sum();
    //         let time = (e - s).as_millis();
    //         println!("Banded\t{}\t{}\t{}", t, time, lk);
    //     }
    // }
    let mat = (0.8, 0.1, 0.1);
    let ins = (0.8, 0.15, 0.05);
    let del = (0.8, 0.05, 0.15);
    let mut emission = [0.05 / 3f64; 16];
    for i in 0..4 {
        emission[i * 4 + i] = 0.95;
    }
    let hmm = kiley::hmm::guided::PairHiddenMarkovModel::new(mat, ins, del, &emission);
    eprintln!("{}", String::from_utf8_lossy(&template));
    eprintln!("{}", String::from_utf8_lossy(&xss[0]));
    {
        let mut hmm = hmm.clone();
        let lk: f64 = xss
            .iter()
            .map(|x| hmm.likelihood(&template, x, CONS_RAD))
            .sum();
        println!("Guided\t{}\t{}\t{:.4}", 0, 0, lk);
        for t in 1..TIME {
            let s = std::time::Instant::now();
            hmm.fit_naive(&template, &xss, CONS_RAD);
            let e = std::time::Instant::now();
            let lk: f64 = xss
                .iter()
                .map(|x| hmm.likelihood(&template, x, CONS_RAD))
                .sum();
            let time = (e - s).as_millis();
            println!("Guided\t{}\t{}\t{:.4}", t, time, lk);
        }
        eprintln!("{}", hmm);
    }
    // {
    // let mut hmm = hmm.clone();
    // let ops: Vec<_> = xss
    //     .iter()
    //     .map(|xs| kiley::bialignment::guided::bootstrap_ops(template.len(), xs.len()))
    //     .collect();
    // let lk: f64 = xss
    //     .iter()
    //     .map(|x| hmm.likelihood(&template, x, CONS_RAD))
    //     .sum();
    // println!("Banded\t{}\t{:.2}\t{:.4}", 0, 0, lk);
    // for t in 1..TIME {
    //     let s = std::time::Instant::now();
    //     hmm.fit_guided(&template, &xss, &ops, CONS_RAD);
    //     let e = std::time::Instant::now();
    //     let lk: f64 = xss
    //         .iter()
    //         .map(|x| hmm.likelihood(&template, x, CONS_RAD))
    //         .sum();
    //     let time = (e - s).as_millis();
    //     println!("Banded\t{}\t{}\t{:.4}", t, time, lk);
    // }
    // eprintln!("{}", hmm);
    // }
}
