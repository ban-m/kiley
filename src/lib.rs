use poa_hmm::POA;
use rand::seq::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

pub fn consensus(
    seqs: &[&[u8]],
    seed: u64,
    subchunk: usize,
    repnum: usize,
    read_type: &str,
) -> Vec<u8> {
    let max_len = match seqs.iter().map(|x| x.len()).max() {
        Some(res) => res,
        None => return vec![],
    };
    let rad = match read_type {
        "CCS" => max_len / 10,
        "CLR" => max_len / 3,
        "ONT" => max_len / 3,
        _ => unreachable!(),
    };
    if seqs.len() <= 10 {
        POA::from_slice_default(&seqs).consensus()
    } else {
        let subchunks = repnum * seqs.len() / subchunk;
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
        let subseq: Vec<_> = (0..subchunks)
            .map(|_| {
                let subchunk: Vec<_> = seqs.choose_multiple(&mut rng, subchunk).copied().collect();
                POA::from_slice_banded(&subchunk, (-1, -1, &score), rad).consensus()
            })
            .collect();
        let subseq: Vec<_> = subseq.iter().map(|e| e.as_slice()).collect();
        POA::from_slice_banded(&subseq, (-1, -1, &score), max_len / 10).consensus()
    }
}

fn score(x: u8, y: u8) -> i32 {
    if x == y {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use poa_hmm::*;
    use rayon::prelude::*;
    #[test]
    fn super_long_multi_consensus_rand() {
        let bases = b"ACTG";
        let coverage = 70;
        let start = 20;
        let len = 2000;
        let result = (start..coverage)
            .into_par_iter()
            .filter(|&cov| {
                let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(cov as u64);
                let template1: Vec<_> = (0..len)
                    .filter_map(|_| bases.choose(&mut rng))
                    .copied()
                    .collect();
                let seqs: Vec<_> = (0..cov)
                    .map(|_| {
                        gen_sample::introduce_randomness(&template1, &mut rng, &gen_sample::PROFILE)
                    })
                    .collect();
                let seqs: Vec<_> = seqs.iter().map(|e| e.as_slice()).collect();
                let consensus = consensus(&seqs, cov as u64, 5, 3, "CLR");
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
                    .map(|_| {
                        gen_sample::introduce_randomness(&template1, &mut rng, &gen_sample::PROFILE)
                    })
                    .collect();
                let seqs: Vec<_> = seqs.iter().map(|e| e.as_slice()).collect();
                let consensus = consensus(&seqs, cov as u64, 10, 3, "CLR");
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
}
