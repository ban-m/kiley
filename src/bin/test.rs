// use kiley::bialignment::*;
use std::io::*;
fn main() -> std::io::Result<()> {
    env_logger::init();
    // use rand::SeedableRng;
    // let seed = 234890;
    // let len = 2000;
    // let error_rate = 0.15;
    // use kiley::gen_seq;
    // let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    // let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    // let prof = gen_seq::PROFILE.norm().mul(error_rate);
    // let xs: Vec<_> = (0..30)
    //     .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &prof))
    //     .collect();
    // for x in xs.iter() {
    //     log::debug!("{}", edlib_sys::global_dist(&template, x));
    // }
    // let cons = kiley::ternary_consensus_by_chunk(&xs, 100);
    // log::debug!("CONS:{}", edlib_sys::global_dist(&template, &cons));
    let args: Vec<_> = std::env::args().collect();
    let inputs: Vec<Vec<_>> = std::fs::File::open(&args[1])
        .map(BufReader::new)?
        .lines()
        .filter_map(|x| x.ok())
        .filter(|x| !x.is_empty())
        .map(|x| x.bytes().collect())
        .collect();
    println!("{}", inputs.len());
    let consensus = kiley::consensus(&inputs, 132, 10, 100);
    println!(">New\n{}", String::from_utf8_lossy(&consensus));
    Ok(())
}
