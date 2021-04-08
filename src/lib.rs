#![feature(is_sorted)]
#[macro_use]
extern crate log;
pub mod bialignment;
pub mod fasta;
pub mod gen_seq;
pub mod hmm;
mod padseq;
pub mod sam;
pub mod trialignment;
pub use bialignment::polish_until_converge_banded;
pub mod gphmm;
use gphmm::{Cond, HMMType, GPHMM};
use rand::seq::*;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PolishConfig<M: HMMType> {
    radius: usize,
    phmm: GPHMM<M>,
    chunk_size: usize,
    max_coverage: usize,
}

impl PolishConfig<Cond> {
    pub fn new(radius: usize, chunk_size: usize, max_coverage: usize) -> Self {
        let phmm = GPHMM::<Cond>::new_three_state(0.9, 0.05, 0.1, 0.95);
        Self {
            radius,
            phmm,
            chunk_size,
            max_coverage,
        }
    }
    pub fn with_model(
        radius: usize,
        chunk_size: usize,
        max_coverage: usize,
        phmm: GPHMM<Cond>,
    ) -> Self {
        Self {
            radius,
            phmm,
            chunk_size,
            max_coverage,
        }
    }
}

use fasta::FASTARecord;

/// Only Alignment with Cigar fields would be used in the polishing stage.
/// It is the task for caller to filter erroneous alignmnet before calling this function.
pub fn polish(
    template: &[FASTARecord],
    queries: &[FASTARecord],
    alignments: &[sam::Record],
    config: &PolishConfig<Cond>,
) -> Vec<FASTARecord> {
    use std::collections::HashMap;
    let mut chunks: HashMap<String, Vec<Vec<Vec<u8>>>> = template
        .iter()
        .map(|(id, seq)| {
            let len = if seq.len() % config.chunk_size == 0 {
                seq.len() / config.chunk_size
            } else {
                seq.len() / config.chunk_size + 1
            };
            let slots: Vec<_> = vec![vec![]; len];
            (id.clone(), slots)
        })
        .collect();
    let reflen: HashMap<String, usize> = template
        .iter()
        .map(|(id, seq)| (id.clone(), seq.len()))
        .collect();
    let queries: HashMap<String, _> = queries
        .iter()
        .map(|(x, y)| (x.to_string(), y.as_slice()))
        .collect();
    // Record alignments.
    for aln in alignments.iter().filter(|a| 0 < a.pos()) {
        let query = queries.get(aln.q_name());
        let ref_slots = chunks.get_mut(aln.r_name());
        let (query, ref_slots) = match (query, ref_slots) {
            (Some(q), Some(r)) => (q, r),
            _ => continue,
        };
        let reflen = *reflen.get(aln.r_name()).unwrap();
        let split_read = split_query(query, aln, reflen, config);
        for (position, seq) in split_read {
            ref_slots[position].push(seq);
        }
    }
    debug!("Record alignemnts.");
    // Model fitting.
    let training: Vec<(&[u8], &[Vec<u8>])> = {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(config.chunk_size as u64);
        let mut training = vec![];
        for _ in 0..5 {
            let (id, seq) = template.choose(&mut rng).unwrap();
            let chunks = chunks.get(id).unwrap();
            let i = rng.gen_range(0..chunks.len());
            let queries = chunks[i].as_slice();
            if 5 < queries.len() {
                let start = i * config.chunk_size;
                let end = (start + config.chunk_size).min(seq.len());
                let draft: &[u8] = &seq[start..end];
                training.push((draft, queries));
            }
        }
        training
    };
    debug!("Fit Model...");
    let model = fit_model_from_multiple(&training, &config);
    debug!("Fit Model:{}", model);
    // Polishing.
    template
        .iter()
        .map(|(id, seq)| {
            let chunks = chunks.get(id).unwrap();
            debug!("CHUNKS\t{}", chunks.len());
            let polished: Vec<_> = chunks
                .par_iter()
                .enumerate()
                .filter_map(|(i, queries)| {
                    let start = i * config.chunk_size;
                    let end = (start + config.chunk_size).min(seq.len());
                    let draft = &seq[start..end];
                    if queries.len() < 5 {
                        return Some(draft.to_vec());
                    }
                    debug!("POLISH\tSTART\t{}\t{}\t{}", i, queries.len(), draft.len());
                    let start = std::time::Instant::now();
                    let cons =
                        model.correct_until_convergence_banded(&draft, &queries, config.radius);
                    let end = std::time::Instant::now();
                    debug!("POLISH\tEND\t{}\t{}", i, (end - start).as_millis());
                    cons
                })
                .flat_map(std::convert::identity)
                .collect();
            (id.clone(), polished)
        })
        .collect()
}

// Split query into (chunk-id, aligned seq)-array.
// If the alignment does not have CIGAR string, return empty array(NOt good).
fn split_query<M: HMMType>(
    query: &[u8],
    aln: &sam::Record,
    reflen: usize,
    config: &PolishConfig<M>,
) -> Vec<(usize, Vec<u8>)> {
    let mut cigar = aln.cigar();
    cigar.reverse();
    if cigar.is_empty() {
        return vec![];
    }
    let query = if aln.is_forward() {
        query.to_vec()
    } else {
        revcmp(query)
    };
    let (mut ref_position, mut query_position) = (aln.pos() - 1, 0);
    let initial_chunk_id = if ref_position % config.chunk_size == 0 {
        ref_position / config.chunk_size
    } else {
        ref_position / config.chunk_size + 1
    };
    let chunk_start = initial_chunk_id * config.chunk_size;
    // Seek by first clippings.
    match cigar.last().unwrap() {
        sam::Op::SoftClip(l) | sam::Op::HardClip(l) => {
            query_position += l;
            cigar.pop();
        }
        _ => {}
    }
    // Seek until reached to the chunk_start.
    while ref_position < chunk_start {
        if let Some(op) = cigar.pop() {
            match op {
                sam::Op::Match(l) | sam::Op::Mismatch(l) | sam::Op::Align(l) => {
                    ref_position += l;
                    query_position += l;
                }
                sam::Op::SoftClip(l) | sam::Op::HardClip(l) => query_position += l,
                sam::Op::Insertion(l) => query_position += l,
                sam::Op::Deletion(l) => ref_position += l,
                sam::Op::Skipped(_) | sam::Op::Padding(_) => {}
            }
            if chunk_start < ref_position {
                use sam::Op;
                let size = ref_position - chunk_start;
                ref_position = chunk_start;
                match op {
                    sam::Op::Align(_) => {
                        cigar.push(Op::Align(size));
                        query_position -= size;
                    }
                    sam::Op::Match(_) => {
                        cigar.push(Op::Match(size));
                        query_position -= size;
                    }
                    sam::Op::Mismatch(_) => {
                        cigar.push(Op::Mismatch(size));
                        query_position -= size;
                    }
                    sam::Op::Deletion(_) => cigar.push(Op::Deletion(size)),
                    sam::Op::Skipped(_)
                    | sam::Op::Insertion(_)
                    | sam::Op::SoftClip(_)
                    | sam::Op::HardClip(_)
                    | sam::Op::Padding(_) => unreachable!(),
                }
            }
            if chunk_start == ref_position {
                break;
            }
        } else {
            return vec![];
        }
    }
    assert_eq!(ref_position, chunk_start);
    let query = &query[query_position..];
    let chunks = seq_into_subchunks(query, config.chunk_size, cigar, reflen - ref_position);
    chunks
        .into_iter()
        .enumerate()
        .map(|(i, sx)| (i + initial_chunk_id, sx))
        .collect()
}

// Cigar is reversed. So, by poping the lemente, we can read the alignment.
fn seq_into_subchunks(
    query: &[u8],
    len: usize,
    mut cigar: Vec<sam::Op>,
    reflen: usize,
) -> Vec<Vec<u8>> {
    use sam::Op;
    let (mut q_pos, mut r_pos) = (0, 0);
    let mut target = len;
    let mut chunk_position = vec![0];
    while let Some(op) = cigar.pop() {
        match op {
            sam::Op::Align(l) | sam::Op::Match(l) | sam::Op::Mismatch(l) => {
                q_pos += l;
                r_pos += l;
                if target < r_pos {
                    let size = r_pos - target;
                    r_pos -= size;
                    q_pos -= size;
                    match op {
                        Op::Align(_) => cigar.push(Op::Align(size)),
                        Op::Match(_) => cigar.push(Op::Match(size)),
                        Op::Mismatch(_) => cigar.push(Op::Mismatch(size)),
                        _ => unreachable!(),
                    }
                }
            }
            sam::Op::Insertion(l) | sam::Op::SoftClip(l) | sam::Op::HardClip(l) => q_pos += l,
            sam::Op::Deletion(l) => {
                r_pos += l;
                if target < r_pos {
                    let size = r_pos - target;
                    r_pos -= size;
                    cigar.push(Op::Deletion(size));
                }
            }
            sam::Op::Skipped(_) => {}
            sam::Op::Padding(_) => {}
        }
        if target == r_pos {
            chunk_position.push(q_pos);
            target += len;
        } else if reflen == r_pos {
            chunk_position.push(q_pos);
            // We should break here, as there would be some junk trailing clips.
            break;
        }
    }
    chunk_position
        .windows(2)
        .map(|w| query[w[0]..w[1]].to_vec())
        .collect()
}

fn revcmp(xs: &[u8]) -> Vec<u8> {
    xs.iter()
        .map(padseq::convert_to_twobit)
        .map(|x| b"TGCA"[x as usize])
        .rev()
        .collect()
}

pub fn fit_model_from_multiple(
    training: &[(&[u8], &[Vec<u8>])],
    config: &PolishConfig<Cond>,
) -> GPHMM<Cond> {
    use padseq::PadSeq;
    let (mut drafts, queries): (Vec<_>, Vec<_>) = training
        .iter()
        .map(|&(x, ys)| {
            let x = PadSeq::new(x);
            let ys: Vec<_> = ys.iter().map(|y| PadSeq::new(y.as_slice())).collect();
            (x, ys)
        })
        .unzip();
    let mut phmm = config.phmm.clone();
    let get_lk = |model: &GPHMM<Cond>, drafts: &[PadSeq]| -> f64 {
        queries
            .par_iter()
            .zip(drafts.par_iter())
            .map(|(qs, d)| {
                qs.iter()
                    .filter_map(|q| model.likelihood_banded_inner(d, q, config.radius))
                    .sum::<f64>()
            })
            .sum()
    };
    debug!("Start fitting model.");
    let mut lk = get_lk(&phmm, &drafts);
    debug!("Initial LK:{:.3}", lk);
    loop {
        let new_drafts = loop {
            let new_phmm = fit_multiple_inner(&phmm, &drafts, &queries, config);
            let new_lk = get_lk(&new_phmm, &drafts);
            debug!("{:.3}->{:.3}", lk, new_lk);
            if new_lk - lk < 0.1f64 {
                let new_drafts = drafts
                    .par_iter()
                    .zip(queries.par_iter())
                    .map(|(d, qs)| {
                        new_phmm
                            .correction_until_convergence_banded_inner(d, qs, config.radius)
                            .unwrap()
                    })
                    .collect();
                break new_drafts;
            } else {
                lk = new_lk;
                phmm = new_phmm;
            }
        };
        if drafts == new_drafts {
            break phmm;
        } else {
            drafts = new_drafts;
        }
    }
}

fn fit_multiple_inner(
    model: &GPHMM<Cond>,
    drafts: &[padseq::PadSeq],
    queries: &[Vec<padseq::PadSeq>],
    config: &PolishConfig<Cond>,
) -> GPHMM<Cond> {
    use gphmm::banded::ProfileBanded;
    let radius = config.radius as isize;
    let profiles: Vec<_> = queries
        .par_iter()
        .zip(drafts.par_iter())
        .flat_map(|(qs, d)| {
            qs.iter()
                .filter_map(|q| ProfileBanded::new(model, d, q, radius))
                .collect::<Vec<_>>()
        })
        .collect();
    debug!("Profiled {} alignments.", profiles.len());
    let initial_distribution = model.estimate_initial_distribution_banded(&profiles);
    let transition_matrix = model.estimate_transition_prob_banded(&profiles);
    let observation_matrix = model.estimate_observation_prob_banded(&profiles);
    debug!("Re-estimated parameters.");
    GPHMM::from_raw_elements(
        model.states(),
        transition_matrix,
        observation_matrix,
        initial_distribution,
    )
}

pub fn fit_model<T: std::borrow::Borrow<[u8]>>(
    draft: &[u8],
    queries: &[T],
    config: &PolishConfig<Cond>,
) -> GPHMM<Cond> {
    use padseq::PadSeq;
    let mut draft = PadSeq::new(draft);
    let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
    let mut phmm = config.phmm.clone();
    let get_lk = |model: &GPHMM<Cond>, draft: &PadSeq| -> f64 {
        queries
            .iter()
            .filter_map(|q| model.likelihood_banded_inner(draft, q, config.radius))
            .sum()
    };
    let mut lk = get_lk(&phmm, &draft);
    loop {
        let new_cons = loop {
            let start = std::time::Instant::now();
            let new_phmm = phmm.fit_banded_inner(&draft, &queries, config.radius);
            let fit = std::time::Instant::now();
            let new_lk = get_lk(&new_phmm, &draft);
            debug!("FIT\t{}", (fit - start).as_millis());
            debug!("{:.3}->{:.3}", lk, new_lk);
            if new_lk - lk < 0.1f64 {
                debug!("Break");
                break phmm
                    .correction_until_convergence_banded_inner(&draft, &queries, config.radius)
                    .unwrap();
            } else {
                lk = new_lk;
                phmm = new_phmm;
            }
        };
        if draft == new_cons {
            break phmm;
        } else {
            draft = new_cons;
        }
    }
}

// Polish chunk.
// TODO:This function should not fail.
fn polish_chunk<T: std::borrow::Borrow<[u8]>>(
    draft: &[u8],
    queries: &[T],
    config: &PolishConfig<Cond>,
) -> Option<Vec<u8>> {
    use padseq::PadSeq;
    let mut draft = PadSeq::new(draft);
    let queries: Vec<_> = queries.iter().map(|x| PadSeq::new(x.borrow())).collect();
    let mut phmm = config.phmm.clone();
    let get_lk = |model: &GPHMM<Cond>, draft: &PadSeq| -> f64 {
        queries
            .iter()
            .filter_map(|q| model.likelihood_banded_inner(draft, q, config.radius))
            .sum()
    };
    let mut lk = get_lk(&phmm, &draft);
    loop {
        let new_cons = loop {
            let start = std::time::Instant::now();
            let new_phmm = phmm.fit_banded_inner(&draft, &queries, config.radius);
            let fit = std::time::Instant::now();
            let new_lk = get_lk(&new_phmm, &draft);
            debug!("FIT\t{}", (fit - start).as_millis());
            debug!("{:.3}->{:.3}", lk, new_lk);
            if new_lk - lk < 0.1f64 {
                debug!("Break");
                break phmm
                    .correction_until_convergence_banded_inner(&draft, &queries, config.radius)
                    .unwrap();
            } else {
                lk = new_lk;
                phmm = new_phmm;
            }
        };
        if draft == new_cons {
            debug!("Polished");
            debug!("ModelDiff:{:?}", config.phmm.dist(&phmm));
            break Some(draft.into());
        } else {
            draft = new_cons;
        }
    }
}

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
    use std::time::Instant;
    let start = Instant::now();
    let consensus = ternary_consensus(seqs, seed, repnum, radius);
    let ternary = Instant::now();
    let consensus = polish_by_pileup(&consensus, seqs, radius);
    let correct = Instant::now();
    let config = PolishConfig::new(radius, 0, seqs.len());
    let consensus = polish_chunk(&consensus, &seqs, &config);
    let polish = Instant::now();
    eprintln!(
        "{}\t{}\t{}",
        (ternary - start).as_millis(),
        (correct - ternary).as_millis(),
        (polish - correct).as_millis()
    );
    consensus
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
