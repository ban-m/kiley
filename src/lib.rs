#[macro_use]
extern crate log;
pub mod bialignment;
mod dptable;
pub mod fasta;
pub mod gen_seq;
pub mod hmm;
pub mod op;
mod padseq;
mod polishing;
pub use op::Op;
use std::collections::HashMap;
pub mod sam;
// Samll value.
const EP: f64 = -10000000000000000000000000000000f64;

/// Configuration struct for polishing a sequence.
#[derive(Debug, Clone)]
pub struct PolishConfig {
    radius: usize,
    hmm: hmm::PairHiddenMarkovModel,
    chunk_size: usize,
    overlap: usize,
    // max_coverage: usize,
    // seed: u64,
}

impl PolishConfig {
    pub fn new(
        radius: usize,
        chunk_size: usize,
        //max_coverage: usize,
        overlap: usize,
        //        seed: u64,
    ) -> Self {
        let hmm = hmm::PairHiddenMarkovModel::default();
        Self {
            radius,
            hmm,
            chunk_size,
            //  max_coverage,
            //  seed,
            overlap,
        }
    }
    pub fn with_model(
        radius: usize,
        chunk_size: usize,
        // max_coverage: usize,
        overlap: usize,
        // seed: u64,
        hmm: hmm::PairHiddenMarkovModel,
    ) -> Self {
        Self {
            radius,
            hmm,
            chunk_size,
            overlap,
            //  max_coverage,
            //  seed,
        }
    }
}

/// A record representing a sequence with an id.
/// It implements `std::convert::From<SeqRecord<I,S>>` for `(I,S)`. Thus,
/// We can extract the inner information by converting an instance into a tuple `(id, sequence)`.
pub struct SeqRecord<I, S>
where
    I: std::borrow::Borrow<str>,
    S: std::borrow::Borrow<[u8]>,
{
    id: I,
    seq: S,
}

impl<I, S> SeqRecord<I, S>
where
    I: std::borrow::Borrow<str>,
    S: std::borrow::Borrow<[u8]>,
{
    pub fn new(id: I, seq: S) -> Self {
        Self { id, seq }
    }
}

impl<I, S> std::convert::From<SeqRecord<I, S>> for (I, S)
where
    I: std::borrow::Borrow<str>,
    S: std::borrow::Borrow<[u8]>,
{
    fn from(SeqRecord { id, seq }: SeqRecord<I, S>) -> Self {
        (id, seq)
    }
}

/// Only Alignment with Cigar fields would be used in the polishing stage.
/// It is the task for caller to filter erroneous alignmnet before calling this function.
pub fn polish<I, S>(
    templates: &[SeqRecord<I, S>],
    queries: &[SeqRecord<I, S>],
    alignments: &[sam::Record],
    config: &PolishConfig,
) -> Vec<SeqRecord<String, Vec<u8>>>
where
    I: std::borrow::Borrow<str>,
    S: std::borrow::Borrow<[u8]>,
{
    let queries: HashMap<_, _> = queries
        .iter()
        .map(|r| (r.id.borrow().to_string(), r))
        .collect();
    templates
        .iter()
        .map(|template| {
            let alignments: Vec<_> = alignments
                .iter()
                .filter(|aln| aln.r_name() == template.id.borrow())
                .filter_map(|aln| {
                    let seq = queries.get(aln.r_name())?;
                    Some((aln, *seq))
                })
                .collect();
            use polishing::*;
            polish_single(template, &alignments, config)
        })
        .collect()
}

fn edlib_global(target: &[u8], query: &[u8]) -> Vec<op::Op> {
    let task = edlib_sys::AlignTask::Alignment;
    let mode = edlib_sys::AlignMode::Global;
    let aln = edlib_sys::align(query, target, mode, task);
    use op::Op::*;
    const EDLIB2KILEY: [op::Op; 4] = [Match, Ins, Del, Mismatch];
    assert!(aln.operations().is_some());
    match aln.operations() {
        Some(aln) => aln.iter().map(|x| EDLIB2KILEY[*x as usize]).collect(),
        None => vec![],
    }
}

pub fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.;
    }
    let max = xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    let sum = xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    assert!(sum >= 0., "{:?}->{}", xs, sum);
    max + sum
}

// /// Polish draft sequence by queries.
// /// TODO:Maybe re use the alignment, rather than re-compute?
// pub fn polish_chunk_by_parts<T: std::borrow::Borrow<[u8]>>(
//     draft: &[u8],
//     queries: &[T],
//     config: &PolishConfig,
// ) -> Vec<u8> {
//     assert!(!draft.is_empty());
//     let subchunk_size = 100;
//     if draft.len() < subchunk_size + 10 * 3 {
//         let rad = draft.len() / 3;
//         return bialignment::guided::polish_until_converge(draft, queries, rad);
//     }
//     let mut draft = draft.to_vec();
//     let mut config = config.clone();
//     for offset in 0..2 {
//         let subchunk_size = subchunk_size + 10 * offset;
//         config.radius = 20;
//         // 1. Partition each reads into 100bp.
//         let chunk_start_position: Vec<_> = (0..)
//             .map(|i| i * subchunk_size)
//             .take_while(|l| l + subchunk_size <= draft.len())
//             .collect();
//         let part_queries: Vec<_> = queries
//             .iter()
//             .map(|query| partition_query(&draft, query.borrow(), &chunk_start_position))
//             .collect();
//         let mut chunks: Vec<_> = chunk_start_position
//             .windows(2)
//             .map(|w| (&draft[w[0]..w[1]], vec![]))
//             .collect();
//         let pos = *chunk_start_position
//             .last()
//             .unwrap_or_else(|| panic!("{}", line!()));
//         chunks.push((&draft[pos..], vec![]));
//         for query in part_queries.iter() {
//             for &(pos, seq) in query.iter() {
//                 chunks[pos].1.push(seq);
//             }
//         }
//         // Maybe we need filter out very short/long chunk.
//         chunks.iter_mut().for_each(|(_, qs)| {
//             let (sum, sumsq) = qs
//                 .iter()
//                 .map(|x| x.len())
//                 .fold((0, 0), |(sum, sumsq), len| (sum + len, sumsq + len * len));
//             let mean = sum as f64 / qs.len() as f64;
//             let sd = (sumsq as f64 / qs.len() as f64 - mean * mean)
//                 .sqrt()
//                 .max(1f64);
//             // I hope some seqeunce would reserved.
//             if qs.iter().any(|x| (x.len() as f64 - mean).abs() < 3f64 * sd) {
//                 qs.retain(|qs| (qs.len() as f64 - mean).abs() < 3f64 * sd);
//             }
//         });
//         // 2. Polish each segments.
//         let polished_segs = polish_multiple(&chunks, &config);
//         draft = polished_segs.into_iter().flatten().collect();
//     }
//     draft
// }

// fn polish_multiple(chunks: &[(&[u8], Vec<&[u8]>)], config: &PolishConfig) -> Vec<Vec<u8>> {
//     chunks
//         .iter()
//         .map(|(drf, qs)| config.hmm.polish_until_converge(drf, qs, config.radius))
//         .collect()
// }

// // Polish chunk.
// fn polish_chunk<T: std::borrow::Borrow<[u8]>>(
//     draft: &[u8],
//     queries: &[T],
//     config: &PolishConfig,
// ) -> Vec<u8> {
//     config
//         .hmm
//         .polish_until_converge(draft, queries, config.radius)
// }

// /// Take consensus and polish it. It consists of three step.
// /// 1: Make consensus by ternaly alignments.
// /// 2: Polish consensus by ordinary pileup method
// /// 3: polish consensus by increment polishing method
// pub fn consensus<T: std::borrow::Borrow<[u8]>>(
//     seqs: &[T],
//     _seed: u64,
//     _repnum: usize,
//     radius: usize,
// ) -> Vec<u8> {
//     assert!(!seqs.is_empty());
//     let config = PolishConfig::new(radius, 0, seqs.len(), 0, 0);
//     let lens = seqs.iter().map(|x| x.borrow().len());
//     let min = lens.clone().min().unwrap();
//     let max = lens.clone().max().unwrap();
//     match (min <= 10, max < 200) {
//         (true, true) => seqs
//             .iter()
//             .map(|x| x.borrow())
//             .max_by_key(|x| x.len())
//             .unwrap()
//             .to_vec(),
//         (false, true) => {
//             let consensus = ternary_consensus(seqs, 3290, 4, min / 2);
//             polish_chunk(&consensus, seqs, &config)
//         }
//         (_, false) => {
//             let consensus = ternary_consensus_by_chunk(seqs, radius);
//             let consensus: Vec<_> =
//                 bialignment::polish_until_converge_banded(&consensus, seqs, radius);
//             polish_chunk_by_parts(&consensus, seqs, &config)
//         }
//     }
// }

// /// Almost the same as `consensus`, but it automatically scale the radius
// /// if needed, until it reaches the configured upper bound.
// pub fn consensus_bounded<T: std::borrow::Borrow<[u8]>>(
//     seqs: &[T],
//     seed: u64,
//     repnum: usize,
//     radius: usize,
//     max_radius: usize,
// ) -> Option<Vec<u8>> {
//     let consensus = ternary_consensus(seqs, seed, repnum, radius);
//     let consensus = polish_by_pileup(&consensus, seqs);
//     let consensus = padseq::PadSeq::new(consensus.as_slice());
//     let seqs: Vec<_> = seqs
//         .iter()
//         .map(|x| padseq::PadSeq::new(x.borrow()))
//         .collect();
//     use bialignment::*;
//     for radius in (1..).map(|i| i * radius).take_while(|&r| r <= max_radius) {
//         let (mut cons, _) = match polish_by_focused_banded(&consensus, &seqs, radius, 20, 25) {
//             Some(res) => res,
//             None => continue,
//         };
//         while let Some((improved, _)) = polish_by_batch_banded(&cons, &seqs, radius, 20) {
//             cons = improved;
//         }
//         return Some(cons.into());
//     }
//     None
// }

// use trialignment::banded::Aligner;
// pub fn ternary_consensus_by_chunk<T: std::borrow::Borrow<[u8]>>(
//     seqs: &[T],
//     chunk_size: usize,
// ) -> Vec<u8> {
//     let max = chunk_size * 2;
//     let mut aligner = Aligner::new(max, max, max, chunk_size / 2);
//     if seqs.iter().all(|x| x.borrow().len() < chunk_size) {
//         let mut xs: Vec<_> = seqs.iter().map(|x| x.borrow()).collect();
//         let mean: usize = xs.iter().map(|x| x.len()).sum::<usize>() / xs.len();
//         let thr = mean / 4;
//         xs.retain(|qs| {
//             let diff = qs.len().max(mean) - qs.len().min(mean);
//             diff < thr
//         });
//         let radius = xs.iter().map(|x| x.len()).min();
//         let radius = radius.map(|min| (min / 3).max(chunk_size / 3) + 1);
//         match radius {
//             Some(radius) => return consensus_inner(&xs, radius, &mut aligner),
//             None => return vec![],
//         }
//     }
//     let draft = &seqs[0].borrow();
//     // 1. Partition each reads into `chunk-size`bp.
//     // Calculate the range of each window.
//     let chunk_start_position: Vec<_> = (0..)
//         .map(|i| i * chunk_size)
//         .take_while(|l| l + chunk_size <= draft.len()) // ?
//         .collect();
//     let mut chunks: Vec<Vec<&[u8]>> = chunk_start_position
//         .windows(2)
//         .map(|w| vec![&draft[w[0]..w[1]]])
//         .collect();
//     if chunks.is_empty() {
//         chunks.push(vec![draft]);
//     };
//     let pos = *chunk_start_position.last().unwrap_or(&0);
//     chunks.push(vec![&draft[pos..]]);
//     for seq in seqs.iter() {
//         for (pos, x) in partition_query(draft, seq.borrow(), &chunk_start_position) {
//             chunks[pos].push(x);
//         }
//     }
//     // Maybe we need filter out very short chunk.
//     chunks.iter_mut().for_each(|qs| {
//         let mean: usize = qs.iter().map(|x| x.len()).sum();
//         let thr = mean / qs.len() * 3 / 4;
//         qs.retain(|qs| thr < qs.len());
//     });
//     // 2. Polish each segments.
//     let polished: Vec<_> = chunks
//         .iter()
//         .flat_map(|xs| match xs.iter().map(|x| x.len()).min() {
//             Some(min) => {
//                 let radius = (min / 3).max(chunk_size / 3) + 1;
//                 consensus_inner(xs, radius, &mut aligner)
//             }
//             None => vec![],
//         })
//         .collect();
//     polished
// }

// pub fn ternary_consensus<T: std::borrow::Borrow<[u8]>>(
//     seqs: &[T],
//     seed: u64,
//     repnum: usize,
//     radius: usize,
// ) -> Vec<u8> {
//     assert!(!seqs.is_empty());
//     let max_len = seqs.iter().map(|x| x.borrow().len()).max().unwrap();
//     let mut aligner = Aligner::new(max_len, max_len, max_len, radius);
//     let fold_num = (1..)
//         .take_while(|&x| 3usize.pow(x as u32) <= seqs.len())
//         .count();
//     if repnum <= fold_num {
//         consensus_inner(seqs, radius, &mut aligner)
//     } else {
//         let mut seqs: Vec<_> = seqs.iter().map(|x| x.borrow()).collect();
//         let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
//         let xs = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
//         seqs.shuffle(&mut rng);
//         let ys = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
//         seqs.shuffle(&mut rng);
//         let zs = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
//         aligner.consensus(&xs, &ys, &zs, radius).1
//     }
// }

// fn consensus_inner<T: std::borrow::Borrow<[u8]>>(
//     seqs: &[T],
//     radius: usize,
//     aligner: &mut Aligner,
// ) -> Vec<u8> {
//     let mut consensus: Vec<_> = seqs.iter().map(|x| x.borrow().to_vec()).collect();
//     while 3 <= consensus.len() {
//         consensus = consensus
//             .chunks_exact(3)
//             .map(|xs| aligner.consensus(&xs[0], &xs[1], &xs[2], radius).1)
//             .collect();
//     }
//     consensus.pop().unwrap()
// }

// pub fn polish_by_pileup<T: std::borrow::Borrow<[u8]>>(template: &[u8], xs: &[T]) -> Vec<u8> {
//     let mut matches = vec![[0; 4]; template.len()];
//     // the 0-th element is the insertion before the 0-th base.
//     // TODO: Maybe @-delemitered sequence would be much memory efficient.
//     let mut insertions = vec![vec![]; template.len() + 1];
//     let mut deletions = vec![0; template.len()];
//     for x in xs.iter() {
//         let x = x.borrow();
//         let ops = edlib_global(template, x);
//         let (mut i, mut j) = (0, 0);
//         let mut ins_buffer = vec![];
//         for &op in ops.iter() {
//             match op {
//                 Op::Match | Op::Mismatch => {
//                     matches[i][padseq::convert_to_twobit(&x[j]) as usize] += 1;
//                     i += 1;
//                     j += 1;
//                 }
//                 Op::Ins => {
//                     ins_buffer.push(x[j]);
//                     j += 1;
//                 }
//                 Op::Del => {
//                     deletions[i] += 1;
//                     i += 1;
//                 }
//             }
//             if op != Op::Ins && !ins_buffer.is_empty() {
//                 insertions[i].push(ins_buffer.clone());
//                 ins_buffer.clear();
//             }
//         }
//         if !ins_buffer.is_empty() {
//             insertions[i].push(ins_buffer);
//         }
//     }
//     let mut template = vec![];
//     for ((i, m), &d) in insertions.iter().zip(matches.iter()).zip(deletions.iter()) {
//         let insertion = i.iter().len();
//         let coverage = m.iter().sum::<usize>() + d;
//         if coverage / 2 < insertion {
//             let tot_ins_len: usize = i.iter().map(|x| x.len()).sum();
//             let cons = if tot_ins_len / insertion < 3 {
//                 naive_consensus(i)
//             } else {
//                 de_bruijn_consensus(i)
//             };
//             template.extend(cons);
//         }
//         if d < coverage / 2 {
//             if let Some(base) = m
//                 .iter()
//                 .enumerate()
//                 .max_by_key(|x| x.1)
//                 .map(|(idx, _)| b"ACGT"[idx])
//             {
//                 template.push(base);
//             }
//         }
//     }
//     template
// }

// pub fn consensus_by_pileup_affine<T: std::borrow::Borrow<[u8]>>(
//     xs: &[T],
//     alignment_parameters: (i32, i32, i32, i32),
//     radius: usize,
//     repeat_num: usize,
// ) -> Vec<u8> {
//     let mut consensus = xs[0].borrow().to_vec();
//     for _ in 0..repeat_num {
//         consensus = polish_by_pileup_affine(&consensus, xs, alignment_parameters, radius);
//     }
//     consensus
// }

// pub fn polish_by_pileup_affine<T: std::borrow::Borrow<[u8]>>(
//     template: &[u8],
//     xs: &[T],
//     (mat, mism, open, ext): (i32, i32, i32, i32),
//     radius: usize,
// ) -> Vec<u8> {
//     let mut matches = vec![[0; 4]; template.len()];
//     // the 0-th element is the insertion before the 0-th base.
//     // TODO: Maybe @-delemitered sequence would be much memory efficient.
//     let mut insertions = vec![vec![]; template.len() + 1];
//     let mut deletions = vec![0; template.len()];
//     for x in xs.iter() {
//         let x = x.borrow();
//         let (_, ops) = bialignment::global_banded(template, x, mat, mism, open, ext, radius);
//         let (mut i, mut j) = (0, 0);
//         let mut ins_buffer = vec![];
//         for &op in ops.iter() {
//             match op {
//                 Op::Match | Op::Mismatch => {
//                     matches[i][padseq::convert_to_twobit(&x[j]) as usize] += 1;
//                     i += 1;
//                     j += 1;
//                 }
//                 Op::Ins => {
//                     ins_buffer.push(x[j]);
//                     j += 1;
//                 }
//                 Op::Del => {
//                     deletions[i] += 1;
//                     i += 1;
//                 }
//             }
//             if op != Op::Ins && !ins_buffer.is_empty() {
//                 insertions[i].push(ins_buffer.clone());
//                 ins_buffer.clear();
//             }
//         }
//         if !ins_buffer.is_empty() {
//             insertions[i].push(ins_buffer);
//         }
//     }

//     let mut template = vec![];
//     for ((i, m), &d) in insertions.iter().zip(matches.iter()).zip(deletions.iter()) {
//         let insertion = i.iter().len();
//         let coverage = m.iter().sum::<usize>() + d;
//         if coverage / 2 < insertion {
//             let mut counts = [0; 4];
//             for s in i.iter().filter(|s| !s.is_empty()) {
//                 counts[padseq::convert_to_twobit(&s[0]) as usize] += 1;
//             }
//             let (max, _) = counts.iter().enumerate().max_by_key(|x| x.1).unwrap();
//             // println!("{:?}->{}", i, b"ACGT"[max] as char);
//             template.push(b"ACGT"[max]);
//         }
//         if d < coverage / 2 {
//             if let Some(base) = m
//                 .iter()
//                 .enumerate()
//                 .max_by_key(|x| x.1)
//                 .map(|(idx, _)| b"ACGT"[idx])
//             {
//                 template.push(base);
//             }
//         }
//     }
//     template
// }

// // Take a consensus from seqeunces.
// // This is very naive consensus, majority voting.
// fn naive_consensus(xs: &[Vec<u8>]) -> Vec<u8> {
//     let mut counts: HashMap<(u64, usize), u64> = HashMap::new();
//     for x in xs.iter() {
//         let len = x.len();
//         let hash = x
//             .iter()
//             .map(padseq::convert_to_twobit)
//             .fold(0, |acc, base| (acc << 2) | base as u64);
//         *counts.entry((hash, len)).or_default() += 1;
//     }
//     let (&(kmer, k), _max) = counts
//         .iter()
//         .max_by_key(|x| x.1)
//         .unwrap_or_else(|| panic!("{}", line!()));
//     (0..k)
//         .map(|idx| {
//             let base = (kmer >> (2 * (k - 1 - idx))) & 0b11;
//             b"ACGT"[base as usize]
//         })
//         .collect()
// }

// // Take a consensus sequence from `xs`.
// // I use a naive de Bruijn graph to do that.
// // In other words, I recorded all the 3-mers to the hash map,
// // removing all the lightwehgit edge,
// fn de_bruijn_consensus(xs: &[Vec<u8>]) -> Vec<u8> {
//     // TODO: Implement this function.
//     match xs.iter().max_by_key(|x| x.len()) {
//         Some(res) => res.to_vec(),
//         None => vec![],
//     }
// }

// #[cfg(test)]
// mod test {
//     use super::gen_seq;
//     use super::*;
//     #[test]
//     fn super_long_multi_consensus_rand() {
//         let bases = b"ACTG";
//         let coverage = 60;
//         let start = 20;
//         let len = 1000;
//         let result = (start..coverage)
//             .into_par_iter()
//             .filter(|&cov| {
//                 let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(cov as u64);
//                 let template1: Vec<_> = (0..len)
//                     .filter_map(|_| bases.choose(&mut rng))
//                     .copied()
//                     .collect();
//                 let seqs: Vec<_> = (0..cov)
//                     .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
//                     .collect();
//                 let seqs: Vec<_> = seqs.iter().map(|e| e.as_slice()).collect();
//                 let consensus = consensus(&seqs, cov as u64, 7, 30);
//                 let dist = edit_dist(&consensus, &template1);
//                 eprintln!("LONG:{}", dist);
//                 dist <= 2
//             })
//             .count();
//         assert!(result > 30, "{}", result);
//     }

//     fn edit_dist(x1: &[u8], x2: &[u8]) -> u32 {
//         let mut dp = vec![vec![0; x2.len() + 1]; x1.len() + 1];
//         for (i, row) in dp.iter_mut().enumerate() {
//             row[0] = i as u32;
//         }
//         for j in 0..=x2.len() {
//             dp[0][j] = j as u32;
//         }
//         for (i, x1_b) in x1.iter().enumerate() {
//             for (j, x2_b) in x2.iter().enumerate() {
//                 let m = (x1_b != x2_b) as u32;
//                 dp[i + 1][j + 1] = (dp[i][j + 1] + 1).min(dp[i + 1][j] + 1).min(dp[i][j] + m);
//             }
//         }
//         dp[x1.len()][x2.len()]
//     }

//     #[test]
//     fn short_consensus_kiley() {
//         let coverage = 20;
//         let len = 100;
//         for i in 0..10u64 {
//             let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(i);
//             let template = gen_seq::generate_seq(&mut rng, len);
//             let seqs: Vec<_> = (0..coverage)
//                 .map(|_| gen_seq::introduce_randomness(&template, &mut rng, &gen_seq::PROFILE))
//                 .collect();
//             let consensus = consensus(&seqs, i, 7, 20);
//             let dist = edit_dist(&consensus, &template);
//             eprintln!("T:{}", String::from_utf8_lossy(&template));
//             eprintln!("C:{}", String::from_utf8_lossy(&consensus));
//             eprintln!("SHORT:{}", dist);
//             assert!(dist <= 2);
//         }
//     }
//     #[test]
//     fn long_consensus_kiley() {
//         let coverage = 70;
//         let start = 20;
//         let len = 500;
//         let result = (start..coverage)
//             .into_par_iter()
//             .filter(|&cov| {
//                 eprintln!("{}", cov);
//                 let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(cov as u64);
//                 let template1: Vec<_> = gen_seq::generate_seq(&mut rng, len);
//                 let seqs: Vec<_> = (0..cov)
//                     .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
//                     .collect();
//                 let consensus = consensus(&seqs, cov as u64, 7, 100);
//                 let dist = edit_dist(&consensus, &template1);
//                 eprintln!("LONG:{}", dist);
//                 dist <= 2
//             })
//             .count();
//         assert!(result > 40, "{}", result);
//     }
//     #[test]
//     fn lowcomplexity_kiley() {}
// }
