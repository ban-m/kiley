#![feature(is_sorted)]
#[macro_use]
extern crate log;
pub mod bialignment;
pub mod fasta;
pub mod gen_seq;
pub mod hmm;
pub mod padseq;
pub mod sam;
pub mod trialignment;
pub use bialignment::polish_until_converge_banded;
mod dptable;
pub mod gphmm;
pub use op::recover;
pub mod op;
pub use op::Op;
use rand::seq::*;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PolishConfig {
    radius: usize,
    hmm: hmm::guided::PairHiddenMarkovModel,
    chunk_size: usize,
    overlap: usize,
    max_coverage: usize,
    seed: u64,
}

impl PolishConfig {
    pub fn new(
        radius: usize,
        chunk_size: usize,
        max_coverage: usize,
        overlap: usize,
        seed: u64,
    ) -> Self {
        let hmm = hmm::guided::PairHiddenMarkovModel::default();
        Self {
            radius,
            hmm,
            chunk_size,
            max_coverage,
            seed,
            overlap,
        }
    }
    pub fn with_model(
        radius: usize,
        chunk_size: usize,
        max_coverage: usize,
        overlap: usize,
        seed: u64,
        hmm: hmm::guided::PairHiddenMarkovModel,
    ) -> Self {
        Self {
            radius,
            hmm,
            chunk_size,
            overlap,
            max_coverage,
            seed,
        }
    }
}

pub struct SeqRecord<I, S>
where
    I: std::borrow::Borrow<str>,
    S: std::borrow::Borrow<[u8]>,
{
    pub id: I,
    pub seq: S,
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
            polish_single(template, &alignments, config)
        })
        .collect()
}

pub fn polish_single<I, S, J, T>(
    template: &SeqRecord<I, S>,
    alignments: &[(&sam::Record, &SeqRecord<J, T>)],
    config: &PolishConfig,
) -> SeqRecord<String, Vec<u8>>
where
    I: std::borrow::Borrow<str>,
    S: std::borrow::Borrow<[u8]>,
    J: std::borrow::Borrow<str>,
    T: std::borrow::Borrow<[u8]>,
{
    let chunks = register_all_alignments(template, alignments, config);
    // for (i, (temp, seqs, _)) in chunks.iter().enumerate() {
    //     debug!("POLISH\tREF\t{i}\t{}\t{}", temp.len(), seqs.len());
    // }
    // if log_enabled!(log::Level::Trace) {
    //     for (i, (temp, seqs, _)) in chunks.iter().enumerate() {
    //         trace!("POLISH\tREF\t{i}\t{}\t{}", temp.len(), seqs.len());
    // for seq in seqs.iter() {
    //     trace!("POLISH\tQRY\t{i}\t{}", seq.len());
    // }
    //     }
    // }
    use bialignment::guided::polish_until_converge_with;
    let polished = chunks
        .into_par_iter()
        .map(|(draft, seqs, mut ops)| match seqs.len() < 5 {
            true => draft.to_vec(),
            false => {
                let draft = polish_until_converge_with(draft, &seqs, &mut ops, config.radius);
                config
                    .hmm
                    .polish_until_converge_with(&draft, &seqs, &mut ops, config.radius)
            }
        })
        .fold(Vec::new, |cons: Vec<u8>, chunk: Vec<u8>| {
            merge(cons, chunk, config.overlap)
        })
        .reduce(Vec::new, |cons: Vec<u8>, chunk: Vec<u8>| {
            merge(cons, chunk, config.overlap)
        });
    SeqRecord::new(template.id.borrow().to_string(), polished)
}

use std::collections::HashMap;
fn register_all_alignments<'a, I, S, J, T>(
    template: &'a SeqRecord<I, S>,
    alignments: &[(&'a sam::Record, &'a SeqRecord<J, T>)],
    config: &PolishConfig,
) -> Vec<(&'a [u8], Vec<Vec<u8>>, Vec<Vec<op::Op>>)>
where
    I: std::borrow::Borrow<str>,
    S: std::borrow::Borrow<[u8]>,
    J: std::borrow::Borrow<str>,
    T: std::borrow::Borrow<[u8]>,
{
    let len = template.seq.borrow().len();
    let stride = config.chunk_size - config.overlap;
    let mut chunks: Vec<_> = (0..)
        .map(|i| (i * stride, i * stride + config.chunk_size))
        .take_while(|&(_, e)| e <= len)
        .map(|(start, end)| {
            let seqs = Vec::with_capacity(alignments.len());
            let ops = Vec::with_capacity(alignments.len());
            (&template.seq.borrow()[start..end], seqs, ops)
        })
        .collect();
    for (aln, seq) in alignments {
        assert_eq!(aln.q_name(), seq.id.borrow());
        let split_read = split_query(seq.seq.borrow(), aln, len, config);
        for (position, seq, op) in split_read {
            chunks[position].1.push(seq);
            chunks[position].2.push(op);
        }
    }
    chunks
}

// Connect chunk into the end of cons.
fn merge(mut cons: Vec<u8>, mut chunk: Vec<u8>, overlap: usize) -> Vec<u8> {
    if cons.is_empty() {
        chunk
    } else {
        let split_len = 2 * overlap;
        let cons_trailing = cons.split_off(cons.len().max(split_len) - split_len);
        let chunk_trailing = chunk.split_off(split_len.min(chunk.len()));
        let merged_seq = merge_seq(&cons_trailing, &chunk);
        cons.extend(merged_seq);
        cons.extend(chunk_trailing);
        cons
    }
}

//Marge two sequence. For each error, we choose the above sequcne if we are in the first half, vise varsa.
fn merge_seq(above: &[u8], below: &[u8]) -> Vec<u8> {
    let (_, ops) = overlap_aln(above, below);
    let (mut a_pos, mut b_pos) = (0, 0);
    let mut seq = vec![];
    for op in ops {
        match op {
            Op::Del => {
                if a_pos == 0 || a_pos < above.len() / 2 {
                    seq.push(above[a_pos]);
                }
                a_pos += 1;
            }
            Op::Ins => {
                if a_pos == above.len() || above.len() / 2 < a_pos {
                    seq.push(below[b_pos]);
                }
                b_pos += 1;
            }
            Op::Mismatch | Op::Match => {
                if a_pos < above.len() / 2 {
                    seq.push(above[a_pos]);
                } else {
                    seq.push(below[b_pos]);
                }
                a_pos += 1;
                b_pos += 1;
            }
        }
    }
    assert_eq!(a_pos, above.len());
    assert_eq!(b_pos, below.len());
    seq
}

fn overlap_aln(xs: &[u8], ys: &[u8]) -> (i32, Vec<Op>) {
    let mut dp = vec![vec![0; ys.len() + 1]; xs.len() + 1];
    for (i, x) in xs.iter().enumerate().map(|(i, &x)| (i + 1, x)) {
        for (j, y) in ys.iter().enumerate().map(|(j, &y)| (j + 1, y)) {
            let mat = if x == y { 1 } else { -1 };
            dp[i][j] = (dp[i - 1][j - 1] + mat)
                .max(dp[i - 1][j] - 1)
                .max(dp[i][j - 1] - 1);
        }
    }
    let (score, (mut i, mut j)) = (1..ys.len() + 1)
        .map(|j| (xs.len(), j))
        .map(|(i, j)| (dp[i][j], (i, j)))
        .max_by_key(|x| x.0)
        .unwrap();
    let mut ops: Vec<_> = std::iter::repeat(Op::Ins).take(ys.len() - j).collect();
    while 0 < i && 0 < j {
        let mat = if xs[i - 1] == ys[j - 1] { 1 } else { -1 };
        if dp[i][j] == dp[i - 1][j - 1] + mat {
            if mat == 1 {
                ops.push(Op::Match);
            } else {
                ops.push(Op::Mismatch);
            }
            i -= 1;
            j -= 1;
        } else if dp[i][j] == dp[i - 1][j] - 1 {
            ops.push(Op::Del);
            i -= 1;
        } else if dp[i][j] == dp[i][j - 1] - 1 {
            ops.push(Op::Ins);
            j -= 1;
        } else {
            unreachable!()
        }
    }
    ops.extend(std::iter::repeat(Op::Del).take(i));
    ops.extend(std::iter::repeat(Op::Ins).take(j));
    ops.reverse();
    (score, ops)
}

// Split query into (chunk-id, aligned seq)-array.
// If the alignment does not have CIGAR string, return empty array.
fn split_query(
    query: &[u8],
    aln: &sam::Record,
    reflen: usize,
    config: &PolishConfig,
) -> Vec<(usize, Vec<u8>, Vec<op::Op>)> {
    let mut ops: Vec<_> = aln
        .cigar()
        .iter()
        .rev() // Rev-ed!
        .flat_map(|&op| match op {
            sam::Op::Align(l) | sam::Op::Match(l) | sam::Op::Mismatch(l) => vec![Op::Match; l],
            sam::Op::HardClip(l) | sam::Op::SoftClip(l) | sam::Op::Insertion(l) => {
                vec![Op::Ins; l]
            }
            sam::Op::Deletion(l) => vec![Op::Del; l],
            _ => unreachable!(),
        })
        .collect();
    if ops.is_empty() || aln.pos() == 0 {
        return vec![];
    }
    let (mut ref_position, mut query_position) = (aln.pos() - 1, 0);
    let break_len = config.chunk_size - config.overlap;
    let initial_chunk_id = if ref_position % break_len == 0 {
        ref_position / break_len
    } else {
        ref_position / break_len + 1
    };
    let chunk_start = initial_chunk_id * break_len;
    // Seek by first clippings.
    while ops.last() == Some(&Op::Ins) {
        query_position += 1;
        ops.pop();
    }
    // Seek until reached to the chunk_start.
    assert!(ref_position <= chunk_start);
    while ref_position < chunk_start {
        match ops.pop() {
            Some(Op::Mismatch | Op::Match) => {
                ref_position += 1;
                query_position += 1;
            }
            Some(Op::Ins) => query_position += 1,
            Some(Op::Del) => ref_position += 1,
            None => return vec![],
        }
    }
    assert_eq!(ref_position, chunk_start);
    let query = if aln.is_forward() {
        query.to_vec()
    } else {
        revcmp(query)
    };
    let query = &query[query_position..];
    seq_into_subchunks(query, config, ops, reflen - ref_position)
        .into_iter()
        .enumerate()
        .map(|(i, (subseq, ops))| (i + initial_chunk_id, subseq, ops))
        .collect()
}

// Cigar is reversed. So, by poping the lemente, we can read the alignment.
fn seq_into_subchunks(
    query: &[u8],
    config: &PolishConfig,
    mut ops: Vec<Op>,
    _reflen: usize,
) -> Vec<(Vec<u8>, Vec<Op>)> {
    let break_len = config.chunk_size - config.overlap;
    let mut q_pos = 0;
    let mut split_seqs = vec![];
    while let Some((popped_ops, q_len)) = peek(config.chunk_size, &ops) {
        split_seqs.push((query[q_pos..q_pos + q_len].to_vec(), popped_ops));
        q_pos += skip(break_len, &mut ops);
    }
    split_seqs
}
// `ops` shoule be reversed!
fn peek(len: usize, ops: &[Op]) -> Option<(Vec<Op>, usize)> {
    let (mut q_pos, mut r_pos) = (0, 0);
    let mut popped_ops = Vec::with_capacity(len * 3 / 2);
    for &op in ops.iter().rev() {
        popped_ops.push(op);
        match op {
            Op::Match | Op::Mismatch => {
                q_pos += 1;
                r_pos += 1;
            }
            Op::Ins => q_pos += 1,
            Op::Del => r_pos += 1,
        }
        if len <= r_pos {
            break;
        }
    }
    (len <= r_pos).then(|| (popped_ops, q_pos))
}

// `ops` shoule be reversed!
fn skip(len: usize, ops: &mut Vec<Op>) -> usize {
    let (mut q_pos, mut r_pos) = (0, 0);
    while let Some(op) = ops.pop() {
        match op {
            Op::Match | Op::Mismatch => {
                q_pos += 1;
                r_pos += 1;
            }
            Op::Ins => q_pos += 1,
            Op::Del => r_pos += 1,
        }
        if len <= r_pos {
            break;
        }
    }
    q_pos
}

// let mut target = (1..).flat_map(|i| {
//     let start = i * break_len;
//     [start, start + config.overlap]
// });
// let mut current_target = target.next().unwrap();
// // Reference position, query position.
// let mut chunk_position: Vec<(usize, usize)> = vec![(0, 0)];
// while let Some(op) = ops.pop() {
//     match op {
//         Op::Mismatch | Op::Match => {
//             q_pos += 1;
//             r_pos += 1;
//         }
//         Op::Ins => q_pos += 1,
//         Op::Del => r_pos += 1,
//     }
//     if current_target == r_pos {
//         chunk_position.push((current_target, q_pos));
//         current_target = target.next().unwrap();
//     } else if reflen == r_pos {
//         chunk_position.push((reflen, q_pos));
//         // We should break here, as there would be some junk trailing clips.
//         break;
//     }
// }
// if chunk_position.len() < 2 {
//     return vec![];
// }
// chunk_position
//     .iter()
//     .enumerate()
//     .filter(|(_, (r_pos, _))| r_pos % break_len == 0)
//     .filter_map(|(idx, &(ref_start, start_pos))| {
//         // If this chunk is at the end of the contig, and this alignment comsumes all the reference,
//         if reflen <= ref_start + config.chunk_size && r_pos == reflen {
//             Some(query[start_pos..q_pos].to_vec())
//         } else {
//             chunk_position[idx..]
//                 .iter()
//                 .find(|&&(r_pos, _)| r_pos == ref_start + config.chunk_size)
//                 .map(|&(_, end_pos)| query[start_pos..end_pos].to_vec())
//         }
//     })
//     .map(|x| (x, vec![]))
//     .collect()
// }

fn revcmp(xs: &[u8]) -> Vec<u8> {
    xs.iter()
        .map(padseq::convert_to_twobit)
        .map(|x| b"TGCA"[x as usize])
        .rev()
        .collect()
}

// pub fn fit_model_from_multiple(
//     training: &[(&[u8], &[Vec<u8>])],
//     config: &PolishConfig,
// ) -> GPHMM {
//     use padseq::PadSeq;
//     let (mut drafts, queries): (Vec<_>, Vec<_>) = training
//         .iter()
//         .map(|&(x, ys)| {
//             let x = PadSeq::new(x);
//             let ys: Vec<_> = ys.iter().map(|y| PadSeq::new(y.as_slice())).collect();
//             (x, ys)
//         })
//         .unzip();
//     let mut phmm = config.phmm.clone();
//     let get_lk = |model: &GPHMM<Cond>, drafts: &[PadSeq]| -> f64 {
//         queries
//             .par_iter()
//             .zip(drafts.par_iter())
//             .map(|(qs, d)| {
//                 qs.iter()
//                     .filter_map(|q| model.likelihood_banded_inner(d, q, config.radius))
//                     .sum::<f64>()
//             })
//             .sum()
//     };
//     let mut lk = get_lk(&phmm, &drafts);
//     loop {
//         let new_drafts = drafts
//             .par_iter()
//             .zip(queries.par_iter())
//             .map(
//                 |(d, qs)| match phmm.correct_banded_batch(d, qs, config.radius, 20) {
//                     Some(res) => res,
//                     None => d.clone(),
//                 },
//             )
//             .collect();
//         if drafts == new_drafts {
//             break phmm;
//         }
//         drafts = new_drafts;
//         phmm = fit_multiple_inner(&phmm, &drafts, &queries, config);
//         let new_lk = get_lk(&phmm, &drafts);
//         if new_lk < lk {
//             break phmm;
//         }
//         lk = new_lk;
//     }
// }

// fn fit_multiple_inner(
//     model: &GPHMM<Cond>,
//     drafts: &[padseq::PadSeq],
//     queries: &[Vec<padseq::PadSeq>],
//     config: &PolishConfig<Cond>,
// ) -> GPHMM<Cond> {
//     use gphmm::banded::ProfileBanded;
//     let radius = config.radius as isize;
//     let profiles: Vec<_> = queries
//         .par_iter()
//         .zip(drafts.par_iter())
//         .flat_map(|(qs, d)| {
//             qs.iter()
//                 .filter_map(|q| ProfileBanded::new(model, d, q, radius))
//                 .collect::<Vec<_>>()
//         })
//         .collect();
//     // debug!("Profiled {} alignments.", profiles.len());
//     // let start = std::time::Instant::now();
//     let initial_distribution = model.par_estimate_initial_distribution_banded(&profiles);
//     // let init = std::time::Instant::now();
//     let transition_matrix = model.par_estimate_transition_prob_banded(&profiles);
//     // let trans = std::time::Instant::now();
//     let observation_matrix = model.par_estimate_observation_prob_banded(&profiles);
//     // let obs = std::time::Instant::now();
//     // debug!(
//     //     "ESTIM\t{}\t{}\t{}",
//     //     (init - start).as_millis(),
//     //     (trans - init).as_millis(),
//     //     (obs - trans).as_millis()
//     // );
//     // debug!("Re-estimated parameters.");
//     GPHMM::from_raw_elements(
//         model.states(),
//         transition_matrix,
//         observation_matrix,
//         initial_distribution,
//     )
// }

// pub fn fit_model<T: std::borrow::Borrow<[u8]>>(
//     draft: &[u8],
//     queries: &[T],
//     config: &PolishConfig<Cond>,
// ) -> GPHMM<Cond> {
//     use padseq::PadSeq;
//     let mut draft = PadSeq::new(draft);
//     let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
//     let mut phmm = config.phmm.clone();
//     let get_lk = |model: &GPHMM<Cond>, draft: &PadSeq| -> f64 {
//         queries
//             .iter()
//             .filter_map(|q| model.likelihood_banded_inner(draft, q, config.radius))
//             .sum()
//     };
//     let mut lk = get_lk(&phmm, &draft);
//     loop {
//         let (new_phmm, new_lk) = phmm.fit_banded_inner(&draft, &queries, config.radius);
//         if new_lk - lk < 0.1f64 {
//             break phmm;
//         }
//         phmm = new_phmm;
//         lk = new_lk;
//         draft = phmm
//             .correction_until_convergence_banded_inner(&draft, &queries, config.radius)
//             .unwrap();
//     }
// }

fn edlib_global(target: &[u8], query: &[u8]) -> Vec<Op> {
    let task = edlib_sys::AlignTask::Alignment;
    let mode = edlib_sys::AlignMode::Global;
    let aln = edlib_sys::align(query, target, mode, task);
    use Op::*;
    const EDLIB2KILEY: [Op; 4] = [Match, Ins, Del, Mismatch];
    aln.operations()
        .unwrap()
        .iter()
        .map(|x| EDLIB2KILEY[*x as usize])
        .collect()
}

// Split reads into size intervals. Note that the last chunks is merged the 2nd last one.
fn partition_query<'a>(
    draft: &[u8],
    query: &'a [u8],
    split_positions: &[usize],
) -> Vec<(usize, &'a [u8])> {
    let ops = edlib_global(draft, query);
    let (mut i, mut j) = (0, 0);
    let mut q_split_position = vec![];
    let mut target_poss = split_positions.iter();
    let mut target_pos = match target_poss.next() {
        Some(&res) => res,
        None => return vec![(0, query)],
    };
    for op in ops {
        match op {
            Op::Match | Op::Mismatch => {
                if i == target_pos {
                    q_split_position.push(j);
                    target_pos = match target_poss.next() {
                        Some(&res) => res,
                        None => break,
                    };
                }
                i += 1;
                j += 1;
            }
            Op::Del => {
                if i == target_pos {
                    q_split_position.push(j);
                    target_pos = match target_poss.next() {
                        Some(&res) => res,
                        None => break,
                    };
                }
                i += 1;
            }
            Op::Ins => j += 1,
        }
    }
    assert_eq!(q_split_position.len(), split_positions.len());
    let mut split: Vec<_> = q_split_position
        .windows(2)
        .enumerate()
        .map(|(bin, w)| (bin, &query[w[0]..w[1]]))
        .collect();
    let query_last = *q_split_position.last().unwrap();
    split.push((split_positions.len() - 1, &query[query_last..]));
    split
}

/// Polish draft sequence by queries.
/// TODO:Maybe re use the alignment, rather than re-compute?
pub fn polish_chunk_by_parts<T: std::borrow::Borrow<[u8]>>(
    draft: &[u8],
    queries: &[T],
    config: &PolishConfig,
) -> Vec<u8> {
    assert!(!draft.is_empty());
    let subchunk_size = 100;
    let mut draft = draft.to_vec();
    let mut config = config.clone();
    for offset in 0..2 {
        let subchunk_size = subchunk_size + 10 * offset;
        config.radius = 20;
        // 1. Partition each reads into 100bp.
        let chunk_start_position: Vec<_> = (0..)
            .map(|i| i * subchunk_size)
            .take_while(|l| l + subchunk_size <= draft.len())
            .collect();
        let part_queries: Vec<_> = queries
            .iter()
            .map(|query| partition_query(&draft, query.borrow(), &chunk_start_position))
            .collect();
        let mut chunks: Vec<_> = chunk_start_position
            .windows(2)
            .map(|w| (&draft[w[0]..w[1]], vec![]))
            .collect();
        let pos = *chunk_start_position.last().unwrap();
        chunks.push((&draft[pos..], vec![]));
        for query in part_queries.iter() {
            for &(pos, seq) in query.iter() {
                chunks[pos].1.push(seq);
            }
        }
        // Maybe we need filter out very short/long chunk.
        chunks.iter_mut().for_each(|(_, qs)| {
            let (sum, sumsq) = qs
                .iter()
                .map(|x| x.len())
                .fold((0, 0), |(sum, sumsq), len| (sum + len, sumsq + len * len));
            let mean = sum as f64 / qs.len() as f64;
            let sd = (sumsq as f64 / qs.len() as f64 - mean * mean)
                .sqrt()
                .max(1f64);
            // I hope some seqeunce would reserved.
            if qs.iter().any(|x| (x.len() as f64 - mean).abs() < 3f64 * sd) {
                qs.retain(|qs| (qs.len() as f64 - mean).abs() < 3f64 * sd);
            }
        });
        // 2. Polish each segments.
        let polished_segs = polish_multiple(&chunks, &config);
        draft = polished_segs.into_iter().flatten().collect();
    }
    draft
}

fn polish_multiple(chunks: &[(&[u8], Vec<&[u8]>)], config: &PolishConfig) -> Vec<Vec<u8>> {
    chunks
        .iter()
        .map(|(drf, qs)| config.hmm.polish_until_converge(drf, qs, config.radius))
        .collect()
    // use padseq::PadSeq;
    // let (mut drafts, queries): (Vec<_>, Vec<_>) = chunks
    //     .iter()
    //     .map(|&(x, ref ys)| {
    // let x = PadSeq::new(x);
    // let ys: Vec<_> = ys.iter().map(|&y| PadSeq::new(y)).collect();
    //     (x, ys)
    // })
    // .unzip();
    // for (i, (d, qs)) in drafts.iter_mut().zip(queries.iter()).enumerate() {
    //     let mut skip = 7;
    //     let orig = d.len();
    // while let Some(res) = config
    //     .phmm
    //     .correct_banded_batch(d, qs, config.radius, skip % d.len())
    // {
    //     skip += 1;
    //     *d = res;
    //     if d.len() * 3 < skip {
    //         warn!("CHUNK\tReached max cycle\t{}\t{}", i, config.seed);
    //         let draft = String::from_utf8(d.clone().into()).unwrap();
    //         warn!("CHUNK\tREF\t{}\t{}\t{}", draft.len(), orig, draft);
    //         break;
    //     }
    // }
    // }
    // drafts.into_iter().map(|x| x.into()).collect()
}

// Polish chunk.
fn polish_chunk<T: std::borrow::Borrow<[u8]>>(
    draft: &[u8],
    queries: &[T],
    config: &PolishConfig,
) -> Vec<u8> {
    config
        .hmm
        .polish_until_converge(&draft, &queries, config.radius)
    // let mut draft = draft.to_vec();
    // let mut hmm = config.hmm.clone();
    // let mut lk = std::f64::NEG_INFINITY;
    // for _  {
    // draft = hmm.polish_until_converge(&draft, &queries, config.radius);
    // if let Some(seq) = hmm.correct_banded_batch(&draft, &queries, config.radius, 15) {
    //     draft = seq;
    // }
    // let (new_hmm, new_lk) = hmm.fit_banded_inner(&draft, &queries, config.radius);
    // if lk < new_lk {
    //     hmm = new_hmm;
    //     lk = new_lk;
    // } else {
    //     break draft.into();
    // }
    // }
}

/// Take consensus and polish it. It consists of three step.
/// 1: Make consensus by ternaly alignments.
/// 2: Polish consensus by ordinary pileup method
/// 3: polish consensus by increment polishing method
pub fn consensus<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    _seed: u64,
    _repnum: usize,
    radius: usize,
) -> Vec<u8> {
    let config = PolishConfig::new(radius, 0, seqs.len(), 0, 0);
    let lens = seqs.iter().map(|x| x.borrow().len());
    let min = lens.clone().min().unwrap();
    let max = lens.clone().max().unwrap();
    match (min <= 10, max < 200) {
        (true, true) => seqs
            .iter()
            .map(|x| x.borrow())
            .max_by_key(|x| x.len())
            .unwrap()
            .to_vec(),
        (false, true) => {
            let consensus = ternary_consensus(seqs, 3290, 4, min / 2);
            polish_chunk(&consensus, seqs, &config)
        }
        (_, false) => {
            let consensus = ternary_consensus_by_chunk(seqs, radius);
            let consensus: Vec<_> =
                bialignment::polish_until_converge_banded(&consensus, seqs, radius);
            polish_chunk_by_parts(&consensus, seqs, &config)
        }
    }
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
    let consensus = polish_by_pileup(&consensus, seqs);
    let consensus = padseq::PadSeq::new(consensus.as_slice());
    let seqs: Vec<_> = seqs
        .iter()
        .map(|x| padseq::PadSeq::new(x.borrow()))
        .collect();
    use bialignment::*;
    for radius in (1..).map(|i| i * radius).take_while(|&r| r <= max_radius) {
        let (mut cons, _) = match polish_by_focused_banded(&consensus, &seqs, radius, 20, 25) {
            Some(res) => res,
            None => continue,
        };
        while let Some((improved, _)) = polish_by_batch_banded(&cons, &seqs, radius, 20) {
            cons = improved;
        }
        return Some(cons.into());
    }
    None
}

use trialignment::banded::Aligner;
pub fn ternary_consensus_by_chunk<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    chunk_size: usize,
) -> Vec<u8> {
    let max = chunk_size * 2;
    let mut aligner = Aligner::new(max, max, max, chunk_size / 2);
    if seqs.iter().all(|x| x.borrow().len() < chunk_size) {
        let mut xs: Vec<_> = seqs.iter().map(|x| x.borrow()).collect();
        let mean: usize = xs.iter().map(|x| x.len()).sum::<usize>() / xs.len();
        let thr = mean / 4;
        xs.retain(|qs| {
            let diff = qs.len().max(mean) - qs.len().min(mean);
            diff < thr
        });
        let min = xs.iter().map(|x| x.len()).min().unwrap();
        let radius = (min / 3).max(chunk_size / 3) + 1;
        return consensus_inner(&xs, radius, &mut aligner);
    }
    let draft = &seqs[0].borrow();
    // 1. Partition each reads into `chunk-size`bp.
    // Calculate the range of each window.
    let chunk_start_position: Vec<_> = (0..)
        .map(|i| i * chunk_size)
        .take_while(|l| l + chunk_size <= draft.len()) // ?
        .collect();
    let mut chunks: Vec<_> = chunk_start_position
        .windows(2)
        .map(|w| vec![&draft[w[0]..w[1]]])
        .collect();
    if chunks.is_empty() {
        chunks.push(vec![draft]);
    };
    let pos = *chunk_start_position.last().unwrap_or(&0);
    chunks.push(vec![&draft[pos..]]);
    for seq in seqs.iter() {
        for (pos, x) in partition_query(draft, seq.borrow(), &chunk_start_position) {
            chunks[pos].push(&x);
        }
    }
    // Maybe we need filter out very short chunk.
    chunks.iter_mut().for_each(|qs| {
        let mean: usize = qs.iter().map(|x| x.len()).sum();
        let thr = mean / qs.len() * 3 / 4;
        qs.retain(|qs| thr < qs.len());
    });
    // 2. Polish each segments.
    let polished: Vec<_> = chunks
        .iter()
        .flat_map(|xs| {
            let min = xs.iter().map(|x| x.len()).min().unwrap();
            let radius = (min / 3).max(chunk_size / 3) + 1;
            consensus_inner(xs, radius, &mut aligner)
        })
        .collect();
    polished
}

pub fn ternary_consensus<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    seed: u64,
    repnum: usize,
    radius: usize,
) -> Vec<u8> {
    let max_len = seqs.iter().map(|x| x.borrow().len()).max().unwrap();
    let mut aligner = Aligner::new(max_len, max_len, max_len, radius);
    let fold_num = (1..)
        .take_while(|&x| 3usize.pow(x as u32) <= seqs.len())
        .count();
    if repnum <= fold_num {
        consensus_inner(seqs, radius, &mut aligner)
    } else {
        let mut seqs: Vec<_> = seqs.iter().map(|x| x.borrow()).collect();
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
        let xs = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
        seqs.shuffle(&mut rng);
        let ys = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
        seqs.shuffle(&mut rng);
        let zs = ternary_consensus(&seqs, rng.gen(), repnum - 1, radius);
        aligner.consensus(&xs, &ys, &zs, radius).1
    }
}

fn consensus_inner<T: std::borrow::Borrow<[u8]>>(
    seqs: &[T],
    radius: usize,
    aligner: &mut Aligner,
) -> Vec<u8> {
    let mut consensus: Vec<_> = seqs.iter().map(|x| x.borrow().to_vec()).collect();
    while 3 <= consensus.len() {
        consensus = consensus
            .chunks_exact(3)
            .map(|xs| aligner.consensus(&xs[0], &xs[1], &xs[2], radius).1)
            .collect();
    }
    consensus.pop().unwrap()
}

// pub fn consensus_poa<T: std::borrow::Borrow<[u8]>>(
//     seqs: &[T],
//     seed: u64,
//     subchunk: usize,
//     repnum: usize,
//     read_type: &str,
// ) -> Vec<u8> {
//     use poa_hmm::POA;
//     let seqs: Vec<_> = seqs.iter().map(|x| x.borrow()).collect();
//     #[inline]
//     fn score(x: u8, y: u8) -> i32 {
//         if x == y {
//             1
//         } else {
//             -1
//         }
//     }
//     let max_len = match seqs.iter().map(|x| x.len()).max() {
//         Some(res) => res,
//         None => return vec![],
//     };
//     let rad = match read_type {
//         "CCS" => max_len / 20,
//         "CLR" => max_len / 10,
//         "ONT" => max_len / 10,
//         _ => unreachable!(),
//     };
//     if seqs.len() <= 10 {
//         POA::from_slice_default(&seqs).consensus()
//     } else {
//         let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(seed);
//         let subseq: Vec<_> = (0..repnum)
//             .map(|_| {
//                 let subchunk: Vec<_> = seqs.choose_multiple(&mut rng, subchunk).copied().collect();
//                 POA::from_slice_banded(&subchunk, (-1, -1, &score), rad).consensus()
//             })
//             .collect();
//         let subseq: Vec<_> = subseq.iter().map(|e| e.as_slice()).collect();
//         POA::from_slice_banded(&subseq, (-1, -1, &score), max_len / 10).consensus()
//     }
// }

pub fn polish_by_pileup<T: std::borrow::Borrow<[u8]>>(template: &[u8], xs: &[T]) -> Vec<u8> {
    let mut matches = vec![[0; 4]; template.len()];
    // the 0-th element is the insertion before the 0-th base.
    // TODO: Maybe @-delemitered sequence would be much memory efficient.
    let mut insertions = vec![vec![]; template.len() + 1];
    let mut deletions = vec![0; template.len()];
    for x in xs.iter() {
        let x = x.borrow();
        let ops = edlib_global(template, x);
        let (mut i, mut j) = (0, 0);
        let mut ins_buffer = vec![];
        for &op in ops.iter() {
            match op {
                Op::Match | Op::Mismatch => {
                    matches[i][padseq::convert_to_twobit(&x[j]) as usize] += 1;
                    i += 1;
                    j += 1;
                }
                Op::Ins => {
                    ins_buffer.push(x[j]);
                    j += 1;
                }
                Op::Del => {
                    deletions[i] += 1;
                    i += 1;
                }
            }
            if op != Op::Ins && !ins_buffer.is_empty() {
                insertions[i].push(ins_buffer.clone());
                ins_buffer.clear();
            }
        }
        if !ins_buffer.is_empty() {
            insertions[i].push(ins_buffer);
        }
    }
    let mut template = vec![];
    for ((i, m), &d) in insertions.iter().zip(matches.iter()).zip(deletions.iter()) {
        let insertion = i.iter().len();
        let coverage = m.iter().sum::<usize>() + d;
        if coverage / 2 < insertion {
            let tot_ins_len: usize = i.iter().map(|x| x.len()).sum();
            let cons = if tot_ins_len / insertion < 3 {
                naive_consensus(i)
            } else {
                de_bruijn_consensus(i)
            };
            template.extend(cons);
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
    template
}

pub fn consensus_by_pileup_affine<T: std::borrow::Borrow<[u8]>>(
    xs: &[T],
    alignment_parameters: (i32, i32, i32, i32),
    radius: usize,
    repeat_num: usize,
) -> Vec<u8> {
    let mut consensus = xs[0].borrow().to_vec();
    for _ in 0..repeat_num {
        consensus = polish_by_pileup_affine(&consensus, xs, alignment_parameters, radius);
    }
    consensus
}

pub fn polish_by_pileup_affine<T: std::borrow::Borrow<[u8]>>(
    template: &[u8],
    xs: &[T],
    (mat, mism, open, ext): (i32, i32, i32, i32),
    radius: usize,
) -> Vec<u8> {
    let mut matches = vec![[0; 4]; template.len()];
    // the 0-th element is the insertion before the 0-th base.
    // TODO: Maybe @-delemitered sequence would be much memory efficient.
    let mut insertions = vec![vec![]; template.len() + 1];
    let mut deletions = vec![0; template.len()];
    for x in xs.iter() {
        let x = x.borrow();
        let (_, ops) = bialignment::global_banded(template, x, mat, mism, open, ext, radius);
        let (mut i, mut j) = (0, 0);
        let mut ins_buffer = vec![];
        for &op in ops.iter() {
            match op {
                Op::Match | Op::Mismatch => {
                    matches[i][padseq::convert_to_twobit(&x[j]) as usize] += 1;
                    i += 1;
                    j += 1;
                }
                Op::Ins => {
                    ins_buffer.push(x[j]);
                    j += 1;
                }
                Op::Del => {
                    deletions[i] += 1;
                    i += 1;
                }
            }
            if op != Op::Ins && !ins_buffer.is_empty() {
                insertions[i].push(ins_buffer.clone());
                ins_buffer.clear();
            }
        }
        if !ins_buffer.is_empty() {
            insertions[i].push(ins_buffer);
        }
    }
    // println!("DUMP");
    // for (idx, ((m, d), i)) in matches
    //     .iter()
    //     .zip(deletions.iter())
    //     .zip(insertions.iter())
    //     .enumerate()
    // {
    //     println!("{}\t{:?}\t{}\t{:?}", idx, i, d, m);
    // }
    let mut template = vec![];
    for ((i, m), &d) in insertions.iter().zip(matches.iter()).zip(deletions.iter()) {
        let insertion = i.iter().len();
        let coverage = m.iter().sum::<usize>() + d;
        if coverage / 2 < insertion {
            let mut counts = [0; 4];
            for s in i.iter().filter(|s| !s.is_empty()) {
                counts[padseq::convert_to_twobit(&s[0]) as usize] += 1;
            }
            let (max, _) = counts.iter().enumerate().max_by_key(|x| x.1).unwrap();
            // println!("{:?}->{}", i, b"ACGT"[max] as char);
            template.push(b"ACGT"[max]);
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
    template
}

// Take a consensus from seqeunces.
// This is very naive consensus, majority voting.
fn naive_consensus(xs: &[Vec<u8>]) -> Vec<u8> {
    let mut counts: HashMap<(u64, usize), u64> = HashMap::new();
    for x in xs.iter() {
        let len = x.len();
        let hash = x
            .iter()
            .map(padseq::convert_to_twobit)
            .fold(0, |acc, base| (acc << 2) | base as u64);
        *counts.entry((hash, len)).or_default() += 1;
    }
    let (&(kmer, k), _max) = counts.iter().max_by_key(|x| x.1).unwrap();
    (0..k)
        .map(|idx| {
            let base = (kmer >> (2 * (k - 1 - idx))) & 0b11;
            b"ACGT"[base as usize]
        })
        .collect()
}

// Take a consensus sequence from `xs`.
// I use a naive de Bruijn graph to do that.
// In other words, I recorded all the 3-mers to the hash map,
// removing all the lightwehgit edge,
fn de_bruijn_consensus(xs: &[Vec<u8>]) -> Vec<u8> {
    // TODO: Implement this function.
    xs.iter().max_by_key(|x| x.len()).unwrap().to_vec()
}

#[cfg(test)]
mod test {
    use super::gen_seq;
    use super::*;
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
                let consensus = consensus(&seqs, cov as u64, 7, 30);
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
            let (_, consensus) =
                trialignment::banded::Aligner::new(xs.len(), ys.len(), zs.len(), 10)
                    .consensus(&xs, &ys, &zs, 10);
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
            let consensus = consensus(&seqs, i, 7, 20);
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
                eprintln!("{}", cov);
                let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(cov as u64);
                let template1: Vec<_> = gen_seq::generate_seq(&mut rng, len);
                let seqs: Vec<_> = (0..cov)
                    .map(|_| gen_seq::introduce_randomness(&template1, &mut rng, &gen_seq::PROFILE))
                    .collect();
                let consensus = consensus(&seqs, cov as u64, 7, 100);
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
