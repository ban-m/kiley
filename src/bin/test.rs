// use std::io::{BufRead, BufReader};

// use rand::SeedableRng;

fn main() {
    // let lines: Vec<_> = std::env::args()
    //     .nth(1)
    //     .and_then(|name| std::fs::File::open(name).ok())
    //     .map(BufReader::new)
    //     .unwrap()
    //     .lines()
    //     .filter_map(|x| x.ok())
    //     .filter(|x| !x.is_empty())
    //     .filter(|x| x.starts_with("ALN"))
    //     .map(|x| x.split('\t').nth(1).unwrap().to_string())
    //     .collect();
    // let (mut xs, mut ys) = (vec![], vec![]);
    // for chunk in lines.chunks_exact(3) {
    //     xs.extend(chunk[0].bytes().filter(|&x| x != b' '));
    //     ys.extend(chunk[2].bytes().filter(|&x| x != b' '));
    // }
    //let (score, ops) = kiley::bialignment::global(&xs, &ys, 2, -5, -6, -2);
    // println!("{},{},{}", score, xs.len(), ys.len());
    // let (xr, ar, yr) = kiley::recover(&xs, &ys, &ops);
    // for ((xr, ar), yr) in xr.chunks(200).zip(ar.chunks(200)).zip(yr.chunks(200)) {
    //     eprintln!("ALN\t{}", String::from_utf8_lossy(xr));
    //     eprintln!("ALN\t{}", String::from_utf8_lossy(ar));
    //     eprintln!("ALN\t{}\n", String::from_utf8_lossy(yr));
    // }
    // let (score, xstart, xend, ystart, yend, ops) =
    //     kiley::bialignment::local(&xs, &ys, 5, -5, -6, -2);
    // println!("{},{},{}", score, xs.len(), ys.len());
    // let (xr, ar, yr) = kiley::recover(&xs[xstart..xend], &ys[ystart..yend], &ops);
    // for ((xr, ar), yr) in xr.chunks(200).zip(ar.chunks(200)).zip(yr.chunks(200)) {
    //     eprintln!("ALN\t{}", String::from_utf8_lossy(xr));
    //     eprintln!("ALN\t{}", String::from_utf8_lossy(ar));
    //     eprintln!("ALN\t{}\n", String::from_utf8_lossy(yr));
    // }

    // use kiley::gen_seq;
    // let coverage = 10;
    // let rad = 20;
    // let prof = gen_seq::ProfileWithContext::default();
    // let seed = 36;
    // let len = 2000;
    // let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(seed);
    // let template: Vec<_> = gen_seq::generate_seq(&mut rng, len);
    // let seqs: Vec<_> = (0..coverage)
    //     .map(|_| gen_seq::introduce_randomness_with_context(&template, &mut rng, &prof))
    //     .collect();
    // use kiley::bialignment::edit_dist;
    // let (gtime, gdist, after) = {
    //     let start = std::time::Instant::now();
    //     let draft = kiley::ternary_consensus_by_chunk(&seqs, rad);
    //     let consensus = kiley::bialignment::guided::polish_until_converge(&draft, &seqs, rad);
    //     let end = std::time::Instant::now();
    //     let before = edit_dist(&template, &draft);
    //     let after = edit_dist(&template, &consensus);
    //     ((end - start).as_millis(), before, after)
    // };
    // println!("{}\t{}\t{}", gtime, gdist, after);
}
