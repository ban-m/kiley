use kiley::gen_seq;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoroshiro128PlusPlus;
fn main() -> std::io::Result<()> {
    env_logger::init();
    let len = 500;
    let seed = 2032;
    let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    let seq = gen_seq::generate_seq(&mut rng, len);
    let profile = gen_seq::Profile {
        sub: 0.02,
        del: 0.02,
        ins: 0.02,
    };
    let draft = gen_seq::introduce_randomness(&seq, &mut rng, &profile);
    let hmm = kiley::hmm::PairHiddenMarkovModel::default();
    for _ in 0..200 {
        let len = rng.gen_range(300..500);
        let query = gen_seq::introduce_randomness(&seq[len..], &mut rng, &profile);
        let ops = kiley::bialignment::global(&draft, &query, 1, -1, -1, -3).1;
        let len = hmm
            .modification_table(&draft, &query, 20, &ops)
            .unwrap()
            .0
            .len();
        println!("{len}");
    }
    Ok(())
}
