const SEED: u64 = 129004923;
const CONS_RAD: usize = 2;
const CONS_LEN: usize = 10;
const TIME: usize = 4;
fn main() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(SEED);
    let template = kiley::gen_seq::generate_seq(&mut rng, CONS_LEN);
    let mut xs = template.clone();
    xs.insert(7, b'T');
    let xss = vec![xs];
    let mut hmm = kiley::hmm::guided::PairHiddenMarkovModel::default();
    eprintln!("{}", String::from_utf8_lossy(&template));
    eprintln!("{}", String::from_utf8_lossy(&xss[0]));
    {
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
    }
    eprintln!("{}", hmm);
}
