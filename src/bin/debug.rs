// use kiley::alignment::naive::alignment;
use kiley::gen_seq;
use rand::SeedableRng;
use rand_xoshiro::Xoroshiro128PlusPlus;
fn main() {
    let length = 200;
    let seed = 934820;
    let mut rng: Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    let template = gen_seq::generate_seq(&mut rng, length);
    let coverage = 20;
    let seqs: Vec<_> = (0..coverage)
        .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &gen_seq::PROFILE))
        .collect();
    let consensus = kiley::consensus_kiley(&seqs, 102910, 10);
    eprintln!("{}", String::from_utf8_lossy(&template));
    eprintln!("{}", String::from_utf8_lossy(&consensus));
}
