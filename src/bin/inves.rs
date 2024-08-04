// const LEN: usize = 2_000;
// const COVERAGE: usize = 20;
// const RADIUS: usize = 50;
// const PACK: usize = 4;
// const SEED: u64 = 309482;
// use kiley::{gen_seq::Generate, hmm::TrainingDataPack};
// use rand::{Rng, SeedableRng};
// use rand_xoshiro::Xoroshiro128StarStar;

// macro_rules! elapsed {
//     ($a:expr) => {{
//         let start = std::time::Instant::now();
//         let return_value = $a;
//         let end = std::time::Instant::now();
//         (return_value, (end - start))
//     }};
// }

fn main() {
    use std::io::BufRead;
    let hmm = kiley::hmm::PairHiddenMarkovModelOnStrands::default();
    let args: Vec<_> = std::env::args().collect();
    let dataset: Vec<_> = std::fs::File::open(&args[1])
        .map(std::io::BufReader::new)
        .unwrap()
        .lines()
        .map_while(Result::ok)
        .map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            fields[3].as_bytes().to_vec()
        })
        .collect();
    let mut draft = dataset[0].to_vec();
    let reads = &dataset[1..];
    let config = kiley::hmm::HMMPolishConfig::new(40, reads.len(), 0);
    let mut opss: Vec<_> = reads
        .iter()
        .map(|read| {
            hmm.forward()
                .align_antidiagonal_bootstrap(&draft, read, 40)
                .1
        })
        .collect();
    let mut hmm = hmm;
    let strands = vec![true; reads.len()];
    println!("start");
    for t in 0..1 {
        println!("{t}");
        let training_datapack = vec![kiley::hmm::TrainingDataPack::new(
            &draft, &strands, reads, &opss,
        )];
        hmm.fit_antidiagonal_par_multiple(&training_datapack, 40);
        println!("{t}\n{hmm}");
        draft = hmm.polish_until_converge_antidiagonal(&draft, reads, &mut opss, &strands, &config);
        let seq: Vec<_> = opss
            .iter()
            .map(|ops| ops.iter().filter(|&&o| o != kiley::Op::Match).count())
            .collect();
        println!("{t}\t{seq:?}");
    }
    for (read, ops) in reads.iter().zip(opss.iter()) {
        if 500 < ops.iter().filter(|&&o| o != kiley::Op::Match).count() {
            let (d, o, r) = kiley::op::recover(&draft, read, ops);
            for ((d, o), r) in d.chunks(200).zip(o.chunks(200)).zip(r.chunks(200)) {
                println!("{}", String::from_utf8_lossy(d));
                println!("{}", String::from_utf8_lossy(o));
                println!("{}\n", String::from_utf8_lossy(r));
            }
        }
    }
}
