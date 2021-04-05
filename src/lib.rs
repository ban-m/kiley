#![feature(is_sorted)]
pub mod bialignment;
pub mod fasta;
pub mod gen_seq;
pub mod sam;
pub mod trialignment;
// use poa_hmm::POA;
pub use bialignment::polish_until_converge_banded;
pub mod hmm;
mod padseq;
use rand::seq::*;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// Take consensus and polish it. It consists of three step.
/// 1: Make consensus by ternaly alignments.
/// 2: Polish consensus by ordinary pileup method
/// 3: polish consensus by increment polishing method
/// If the radius was too small for the dataset, return None.
/// In such case, please increase radius(recommend doubling) and try again.
pub fn consensus<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    seed: u64,
    repnum: usize,
    radius: usize,
) -> Option<Vec<u8>> {
    let consensus = ternary_consensus(seqs, seed, repnum, radius);
    let consensus = polish_by_pileup(&consensus, seqs, radius);
    bialignment::polish_until_converge_banded(&consensus, &seqs, radius)
}

/// Almost the same as `consensus`, but it automatically scale the radius
/// if needed, until it reaches the configured upper bound.
pub fn consensus_bounded<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    seed: u64,
    repnum: usize,
    radius: usize,
    max_radius: usize,
) -> Option<Vec<u8>> {
    let consensus = ternary_consensus(seqs, seed, repnum, radius);
    let consensus = polish_by_pileup(&consensus, seqs, radius);
    let mut radius = radius;
    loop {
        let consensus = bialignment::polish_until_converge_banded(&consensus, &seqs, radius);
        if consensus.is_some() {
            break consensus;
        } else if max_radius <= radius {
            break None;
        } else {
            radius *= 2;
            radius = radius.min(max_radius);
        }
    }
}

pub fn ternary_consensus<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    seed: u64,
    repnum: usize,
    radius: usize,
) -> Vec<u8> {
    let fold_num = (1..)
        .take_while(|&x| 3usize.pow(x as u32) <= seqs.len())
        .count();
    if repnum <= fold_num {
        consensus_inner(seqs, radius)
    } else {
        let mut seqs: Vec<_> = seqs.iter().map(|x| x.borrow()).collect();
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
        let xs = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
        seqs.shuffle(&mut rng);
        let ys = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
        seqs.shuffle(&mut rng);
        let zs = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
        get_consensus_kiley(&xs, &ys, &zs, radius).1
    }
}

fn consensus_inner<T: std::borrow::Borrow<[u8]>>(seqs: &[T], radius: usize) -> Vec<u8> {
    let mut consensus: Vec<_> = seqs.iter().map(|x| x.borrow().to_vec()).collect();
    for _ in 1.. {
        consensus = consensus
            .chunks_exact(3)
            .map(|xs| get_consensus_kiley(&xs[0], &xs[1], &xs[2], radius).1)
            .collect();
        if consensus.len() < 3 {
            break;
        }
    }
    consensus.pop().unwrap()
}

pub fn get_consensus_kiley(xs: &[u8], ys: &[u8], zs: &[u8], rad: usize) -> (u32, Vec<u8>) {
    let (dist, aln) = trialignment::banded::alignment(xs, ys, zs, rad);
    (dist, correct_by_alignment(xs, ys, zs, &aln))
}

pub fn correct_by_alignment(xs: &[u8], ys: &[u8], zs: &[u8], aln: &[trialignment::Op]) -> Vec<u8> {
    let (mut x, mut y, mut z) = (0, 0, 0);
    let mut buffer = vec![];
    for &op in aln {
        match op {
            trialignment::Op::XInsertion => {
                x += 1;
            }
            trialignment::Op::YInsertion => {
                y += 1;
            }
            trialignment::Op::ZInsertion => {
                z += 1;
            }
            trialignment::Op::XDeletion => {
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
            trialignment::Op::YDeletion => {
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
            trialignment::Op::ZDeletion => {
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
            trialignment::Op::Match => {
                if xs[x] == ys[y] || xs[x] == zs[z] {
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
    use poa_hmm::POA;
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

pub fn polish_by_pileup<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    xs: &[T],
    radius: usize,
) -> Vec<u8> {
    let mut matches = vec![[0; 4]; template.len()];
    let mut insertions = vec![[0; 4]; template.len() + 1];
    let mut deletions = vec![0; template.len()];
    use bialignment::edit_dist_banded;
    use bialignment::Op;
    for x in xs.iter() {
        let x = x.borrow();
        let (_dist, ops): (u32, Vec<Op>) = match edit_dist_banded(&template, x, radius) {
            Some(res) => res,
            None => continue,
        };
        let (mut i, mut j) = (0, 0);
        for op in ops {
            match op {
                Op::Mat => {
                    matches[i][padseq::convert_to_twobit(&x[j]) as usize] += 1;
                    i += 1;
                    j += 1;
                }
                Op::Del => {
                    deletions[i] += 1;
                    i += 1;
                }
                Op::Ins => {
                    insertions[i][padseq::convert_to_twobit(&x[j]) as usize] += 1;
                    j += 1;
                }
            }
        }
    }
    let mut template = vec![];
    for ((i, m), &d) in insertions.iter().zip(matches.iter()).zip(deletions.iter()) {
        let insertion = i.iter().sum::<usize>();
        let coverage = m.iter().sum::<usize>() + d;
        if coverage / 2 < insertion {
            if let Some(base) = i
                .iter()
                .enumerate()
                .max_by_key(|x| x.1)
                .map(|(idx, _)| b"ACGT"[idx])
            {
                template.push(base);
            }
        }
        if d < coverage / 2 {
            if let Some(base) = m
                .iter()
                .enumerate()
                .max_by_key(|x| x.1)
                .map(|(idx, _)| b"ACGT"[idx])
            {
                template.push(base);
            }
        }
    }
    // We neglect the last insertion.
    template
}

#[cfg(test)]
mod test {
    use super::gen_seq;
    use super::*;
    use rayon::prelude::*;
    #[test]
    fn super_long_multi_consensus_rand() {
        let bases = b"ACTG";
        let coverage = 60;
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
                let consensus = consensus(&seqs, cov as u64, 7, 30).unwrap();
                let dist = edit_dist(&consensus, &template1);
                eprintln!("LONG:{}", dist);
                dist <= 2
            })
            .count();
        assert!(result > 30, "{}", result);
    }
    // #[test]
    // fn lowcomplexity() {
    //     let bases = b"ACTG";
    //     let cov = 20;
    //     let len = 3;
    //     let repnum = 100;
    //     let result = (0..20)
    //         .into_iter()
    //         .filter(|&i| {
    //             let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i as u64);
    //             let template1: Vec<_> = (0..len)
    //                 .filter_map(|_| bases.choose(&mut rng))
    //                 .copied()
    //                 .collect();
    //             let template1: Vec<_> = (0..repnum)
    //                 .flat_map(|_| template1.iter())
    //                 .copied()
    //                 .collect();
    //             let seqs: Vec<_> = (0..cov)
    //                 .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
    //                 .collect();
    //             let seqs: Vec<_> = seqs.iter().map(|e| e.as_slice()).collect();
    //             let consensus = consensus(&seqs, cov as u64, 7);
    //             let dist = edit_dist(&consensus, &template1);
    //             eprintln!(
    //                 "{}\n{}\n{}",
    //                 dist,
    //                 String::from_utf8_lossy(&template1),
    //                 String::from_utf8_lossy(&consensus)
    //             );
    //             dist <= 20
    //         })
    //         .count();
    //     assert!(result > 10, "{}", result);
    // }
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
            let (_, consensus) = get_consensus_kiley(&xs, &ys, &zs, 10);
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
            let consensus = consensus(&seqs, i, 7, 20).unwrap();
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
                let consensus = consensus(&seqs, cov as u64, 7, 10).unwrap();
                let dist = edit_dist(&consensus, &template1);
                eprintln!("LONG:{}", dist);
                dist <= 2
            })
            .count();
        assert!(result > 40, "{}", result);
    }
    #[test]
    fn lowcomplexity_kiley() {
        // TODO: Make module to pass this test.
        //     let bases = b"ACTG";
        //     let cov = 20;
        //     let len = 3;
        //     let repnum = 100;
        //     let result = (0..50)
        //         .into_iter()
        //         .filter(|&i| {
        //             let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i as u64);
        //             let template1: Vec<_> = (0..len)
        //                 .filter_map(|_| bases.choose(&mut rng))
        //                 .copied()
        //                 .collect();
        //             let template1: Vec<_> = (0..repnum)
        //                 .flat_map(|_| template1.iter())
        //                 .copied()
        //                 .collect();
        //             let seqs: Vec<_> = (0..cov)
        //                 .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
        //                 .collect();
        //             let consensus = consensus(&seqs, cov as u64, 6, 10).unwrap();
        //             let dist = edit_dist(&consensus, &template1);
        //             eprintln!(
        //                 "{}\n{}\n{}",
        //                 dist,
        //                 String::from_utf8_lossy(&template1),
        //                 String::from_utf8_lossy(&consensus)
        //             );
        //             dist <= 20
        //         })
        //         .count();
        //     assert!(result > 40, "{}", result);
    }
}
