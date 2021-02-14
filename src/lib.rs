pub mod alignment;
pub mod gen_seq;
use poa_hmm::POA;
use rand::seq::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

pub fn consensus<T: std::borrow::Borrow<[u8]>>(seqs: &[T], seed: u64, repnum: usize) -> Vec<u8> {
    consensus_kiley(seqs, seed, repnum)
}

pub fn consensus_kiley<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    seed: u64,
    repnum: usize,
) -> Vec<u8> {
    assert!(seqs.len() > 2);
    let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
    let mut consensus: Vec<Vec<u8>> = vec![];
    let len = seqs.len();
    for t in 1..repnum + 1 {
        consensus = (0..len)
            .map(|_| {
                if t == 1 {
                    let mut picked = seqs.choose_multiple(&mut rng, 3);
                    let xs: &[u8] = picked.next().unwrap().borrow();
                    let ys = picked.next().unwrap().borrow();
                    let zs = picked.next().unwrap().borrow();
                    let rad = ((xs.len() + ys.len() + zs.len()) / 30 / t).max(5);
                    get_consensus_kiley(xs, ys, zs, rad)
                } else {
                    let mut picked = consensus.choose_multiple(&mut rng, 3);
                    let xs = picked.next().unwrap();
                    let ys = picked.next().unwrap();
                    let zs = picked.next().unwrap();
                    let rad = ((xs.len() + ys.len() + zs.len()) / 30 / t).max(5);
                    get_consensus_kiley(xs, ys, zs, rad)
                }
            })
            .collect();
    }
    consensus.pop().unwrap()
}

pub fn get_consensus_kiley(xs: &[u8], ys: &[u8], zs: &[u8], rad: usize) -> Vec<u8> {
    let (_dist, aln) = alignment::banded::alignment(xs, ys, zs, rad);
    // eprintln!("{}", String::from_utf8_lossy(xs));
    // eprintln!("{}", String::from_utf8_lossy(ys));
    // eprintln!("{}", String::from_utf8_lossy(zs));
    // eprintln!("Score:{},{}", dist, rad);
    let (mut x, mut y, mut z) = (0, 0, 0);
    let mut buffer = vec![];
    for op in aln {
        match op {
            alignment::Op::XInsertion => {
                // Insertion from x. Discard.
                x += 1;
            }
            alignment::Op::YInsertion => {
                y += 1;
            }
            alignment::Op::ZInsertion => {
                z += 1;
            }
            alignment::Op::XDeletion => {
                // '-' or ys[y], or zs[z].
                // Actually, it is hard to determine...
                if ys[y] == zs[z] {
                    buffer.push(ys[y]);
                } else {
                    // TODO: Consider here.
                    match buffer.len() % 3 {
                        0 => buffer.push(ys[y]),
                        1 => buffer.push(zs[z]),
                        _ => {}
                    }
                }
                y += 1;
                z += 1;
            }
            alignment::Op::YDeletion => {
                if xs[x] == zs[z] {
                    buffer.push(xs[x]);
                } else {
                    match buffer.len() % 3 {
                        0 => buffer.push(xs[x]),
                        1 => buffer.push(zs[z]),
                        _ => {}
                    }
                }
                x += 1;
                z += 1;
            }
            alignment::Op::ZDeletion => {
                if ys[y] == xs[x] {
                    buffer.push(ys[y]);
                } else {
                    match buffer.len() % 3 {
                        0 => buffer.push(xs[x]),
                        1 => buffer.push(ys[y]),
                        _ => {}
                    }
                }
                x += 1;
                y += 1;
            }
            alignment::Op::Match => {
                if xs[x] == ys[y] {
                    buffer.push(xs[x]);
                } else if xs[x] == zs[z] {
                    buffer.push(xs[x]);
                } else if ys[y] == zs[z] {
                    buffer.push(zs[z])
                } else {
                    // TODO: consider here.
                    match buffer.len() % 3 {
                        0 => buffer.push(xs[x]),
                        1 => buffer.push(ys[y]),
                        _ => buffer.push(zs[z]),
                    }
                }
                x += 1;
                y += 1;
                z += 1;
            }
        }
    }
    buffer
}

pub fn consensus_poa<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    seed: u64,
    subchunk: usize,
    repnum: usize,
    read_type: &str,
) -> Vec<u8> {
    let seqs: Vec<_> = seqs.iter().map(|x| x.borrow()).collect();
    #[inline]
    fn score(x: u8, y: u8) -> i32 {
        if x == y {
            1
        } else {
            -1
        }
    }
    let max_len = match seqs.iter().map(|x| x.len()).max() {
        Some(res) => res,
        None => return vec![],
    };
    let rad = match read_type {
        "CCS" => max_len / 20,
        "CLR" => max_len / 10,
        "ONT" => max_len / 10,
        _ => unreachable!(),
    };
    if seqs.len() <= 10 {
        POA::from_slice_default(&seqs).consensus()
    } else {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
        let subseq: Vec<_> = (0..repnum)
            .map(|_| {
                let subchunk: Vec<_> = seqs.choose_multiple(&mut rng, subchunk).copied().collect();
                POA::from_slice_banded(&subchunk, (-1, -1, &score), rad).consensus()
            })
            .collect();
        let subseq: Vec<_> = subseq.iter().map(|e| e.as_slice()).collect();
        POA::from_slice_banded(&subseq, (-1, -1, &score), max_len / 10).consensus()
    }
}

#[cfg(test)]
mod test {
    use super::gen_seq;
    use super::*;
    use rayon::prelude::*;
    #[test]
    fn super_long_multi_consensus_rand() {
        let bases = b"ACTG";
        let coverage = 70;
        let start = 20;
        let len = 1000;
        let result = (start..coverage)
            .into_par_iter()
            .filter(|&cov| {
                let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(cov as u64);
                let template1: Vec<_> = (0..len)
                    .filter_map(|_| bases.choose(&mut rng))
                    .copied()
                    .collect();
                let seqs: Vec<_> = (0..cov)
                    .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
                    .collect();
                let seqs: Vec<_> = seqs.iter().map(|e| e.as_slice()).collect();
                let consensus = consensus(&seqs, cov as u64, 10);
                let dist = edit_dist(&consensus, &template1);
                eprintln!("LONG:{}", dist);
                dist <= 2
            })
            .count();
        assert!(result > 40, "{}", result);
    }
    #[test]
    fn lowcomplexity() {
        let bases = b"ACTG";
        let cov = 20;
        let len = 3;
        let repnum = 100;
        let result = (0..50)
            .into_iter()
            .filter(|&i| {
                let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i as u64);
                let template1: Vec<_> = (0..len)
                    .filter_map(|_| bases.choose(&mut rng))
                    .copied()
                    .collect();
                let template1: Vec<_> = (0..repnum)
                    .flat_map(|_| template1.iter())
                    .copied()
                    .collect();
                let seqs: Vec<_> = (0..cov)
                    .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
                    .collect();
                let seqs: Vec<_> = seqs.iter().map(|e| e.as_slice()).collect();
                let consensus = consensus(&seqs, cov as u64, 10);
                let dist = edit_dist(&consensus, &template1);
                eprintln!(
                    "{}\n{}\n{}",
                    dist,
                    String::from_utf8_lossy(&template1),
                    String::from_utf8_lossy(&consensus)
                );
                dist <= 20
            })
            .count();
        assert!(result > 40, "{}", result);
    }
    fn edit_dist(x1: &[u8], x2: &[u8]) -> u32 {
        let mut dp = vec![vec![0; x2.len() + 1]; x1.len() + 1];
        for (i, row) in dp.iter_mut().enumerate() {
            row[0] = i as u32;
        }
        for j in 0..=x2.len() {
            dp[0][j] = j as u32;
        }
        for (i, x1_b) in x1.iter().enumerate() {
            for (j, x2_b) in x2.iter().enumerate() {
                let m = if x1_b == x2_b { 0 } else { 1 };
                dp[i + 1][j + 1] = (dp[i][j + 1] + 1).min(dp[i + 1][j] + 1).min(dp[i][j] + m);
            }
        }
        dp[x1.len()][x2.len()]
    }
    #[test]
    fn step_consensus_kiley() {
        let length = 100;
        for seed in 0..100u64 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
            let template = gen_seq::generate_seq(&mut rng, length);
            let xs = gen_seq::introduce_randomness(&template, &mut rng, &gen_seq::PROFILE);
            let ys = gen_seq::introduce_randomness(&template, &mut rng, &gen_seq::PROFILE);
            let zs = gen_seq::introduce_randomness(&template, &mut rng, &gen_seq::PROFILE);
            let consensus = get_consensus_kiley(&xs, &ys, &zs, 10);
            let xdist = edit_dist(&xs, &template);
            let ydist = edit_dist(&ys, &template);
            let zdist = edit_dist(&zs, &template);
            let prev_dist = xdist.min(ydist).min(zdist);
            let dist = edit_dist(&consensus, &template);
            eprintln!("{}", String::from_utf8_lossy(&template));
            eprintln!("{}", String::from_utf8_lossy(&consensus));
            eprintln!("LONG:{},{},{}=>{}", xdist, ydist, zdist, dist);
            assert!(dist <= prev_dist);
        }
    }
    #[test]
    fn short_consensus_kiley() {
        let coverage = 20;
        let len = 100;
        for i in 0..10u64 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
            let template = gen_seq::generate_seq(&mut rng, len);
            let seqs: Vec<_> = (0..coverage)
                .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &gen_seq::PROFILE))
                .collect();
            let consensus = consensus_kiley(&seqs, i, 10);
            let dist = edit_dist(&consensus, &template);
            eprintln!("T:{}", String::from_utf8_lossy(&template));
            eprintln!("C:{}", String::from_utf8_lossy(&consensus));
            eprintln!("SHORT:{}", dist);
            assert!(dist <= 2);
        }
    }
    #[test]
    fn long_consensus_kiley() {
        let coverage = 70;
        let start = 20;
        let len = 500;
        let result = (start..coverage)
            .into_par_iter()
            .filter(|&cov| {
                let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(cov as u64);
                let template1: Vec<_> = gen_seq::generate_seq(&mut rng, len);
                let seqs: Vec<_> = (0..cov)
                    .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
                    .collect();
                let consensus = consensus_kiley(&seqs, cov as u64, 10);
                let dist = edit_dist(&consensus, &template1);
                eprintln!("LONG:{}", dist);
                dist <= 2
            })
            .count();
        assert!(result > 40, "{}", result);
    }
    #[test]
    fn lowcomplexity_kiley() {
        let bases = b"ACTG";
        let cov = 20;
        let len = 3;
        let repnum = 100;
        let result = (0..50)
            .into_iter()
            .filter(|&i| {
                let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i as u64);
                let template1: Vec<_> = (0..len)
                    .filter_map(|_| bases.choose(&mut rng))
                    .copied()
                    .collect();
                let template1: Vec<_> = (0..repnum)
                    .flat_map(|_| template1.iter())
                    .copied()
                    .collect();
                let seqs: Vec<_> = (0..cov)
                    .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
                    .collect();
                let consensus = consensus_kiley(&seqs, cov as u64, 10);
                let dist = edit_dist(&consensus, &template1);
                eprintln!(
                    "{}\n{}\n{}",
                    dist,
                    String::from_utf8_lossy(&template1),
                    String::from_utf8_lossy(&consensus)
                );
                dist <= 20
            })
            .count();
        assert!(result > 40, "{}", result);
    }
}
