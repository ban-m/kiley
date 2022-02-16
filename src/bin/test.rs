const SEED: u64 = 1293890;
const DRAFT: kiley::gen_seq::Profile = kiley::gen_seq::Profile {
    sub: 0.005,
    del: 0.005,
    ins: 0.005,
};
const CONS_COV: usize = 10;
const CONS_RAD: usize = 10;
const CONS_LEN: usize = 100;
fn main() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let prof = &kiley::gen_seq::PROFILE;
    let template = kiley::gen_seq::generate_seq(&mut rng, CONS_LEN);
    let draft = kiley::gen_seq::introduce_randomness(&template, &mut rng, &DRAFT);
    let xss: Vec<_> = (0..CONS_COV)
        .map(|_| kiley::gen_seq::introduce_randomness(&template, &mut rng, prof))
        .collect();
    println!("{}", kiley::bialignment::edit_dist(&template, &draft));
    {
        use kiley::gphmm::*;
        let phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
        let s = std::time::Instant::now();
        let polished = phmm.polish_banded_batch(&draft, &xss, CONS_RAD, 3);
        let e = std::time::Instant::now();
        let dist = kiley::bialignment::edit_dist(&template, &polished);
        println!("{},{}", (e - s).as_millis(), dist);
    }
    {
        let mat = (0.8, 0.1, 0.1);
        let ins = (0.8, 0.15, 0.05);
        let del = (0.85, 0.15);
        let mut emission = [0.05 / 3f64; 16];
        for i in 0..4 {
            emission[i * 4 + i] = 0.95;
        }
        let hmm = kiley::hmm::guided::PairHiddenMarkovModel::new(mat, ins, del, &emission);
        let s = std::time::Instant::now();
        let polished = hmm.polish_until_converge(&draft, &xss, CONS_RAD);
        let e = std::time::Instant::now();
        let dist = kiley::bialignment::edit_dist(&template, &polished);
        println!("{},{}", (e - s).as_millis(), dist);
    }
}
