use kiley::gen_seq;
use rand::SeedableRng;
fn main() -> std::io::Result<()> {
    env_logger::init();
    let error_rate: f64 = 0.15;
    let prof = gen_seq::PROFILE.norm().mul(error_rate);
    let len: usize = 1000;
    let seed = 219348;
    let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    let seq = gen_seq::introduce_randomness(&template, &mut rng, &prof);
    use kiley::padseq::*;
    let xs = PadSeq::new(template.as_slice());
    let seq = PadSeq::new(seq.as_slice());
    use kiley::gphmm::*;
    let three_cond = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.05, 0.9);
    use kiley::gphmm::banded::ProfileBanded;
    let radius = 30;
    let profile = ProfileBanded::new(&three_cond, &xs, &seq, radius).unwrap();
    let trans1 = profile.transition_probability();
    let trans2 = profile.transition_probability_old();
    for (x, y) in trans1.iter().zip(trans2.iter()) {
        eprintln!("{},{}", x, y);
        assert!((x - y).abs() < 0.0001, "{},{}", x, y)
    }
    Ok(())
}
