//! Kiley -- removing errors in a sequence.
//! # What is kiley
//! Kiley is a tiny library to remove errors in a DNA sequence, i.e., a string consisting of ACGT, by other DNA sequences ("reads").
//! The core concept of the kiley is to use "perturbation matrix" that records how the distance between the consensus sequence and reads when we modify the sequence.

#[macro_use]
extern crate log;
/// Implementing alignments between two sequences.
pub mod bialignment;
mod dptable;
/// Codes to generate random sequences (used for benchmarking)
pub mod gen_seq;
/// Implementations of hidden Markov models and their variants.
pub mod hmm;
/// Alignment operations
pub mod op;
mod padseq;
mod polishing;
/// Alignment operations.
pub use op::Op;
use std::collections::HashMap;
// Small value.
const EP: f64 = -10000000000000000000000000000000f64;

/// Configuration struct for polishing a sequence.
#[derive(Debug, Clone)]
pub struct PolishConfig {
    radius: usize,
    hmm: hmm::PairHiddenMarkovModel,
    chunk_size: usize,
    overlap: usize,
}

impl PolishConfig {
    /// Create a new configuration of the polishing process with the default hidden Markov model.
    pub fn new(radius: usize, chunk_size: usize, overlap: usize) -> Self {
        let hmm = hmm::PairHiddenMarkovModel::default();
        Self {
            radius,
            hmm,
            chunk_size,
            overlap,
        }
    }
    /// Create a new configuration of the polishing process with the given hidden Markov model.
    pub fn with_model(
        radius: usize,
        chunk_size: usize,
        overlap: usize,
        hmm: hmm::PairHiddenMarkovModel,
    ) -> Self {
        Self {
            radius,
            hmm,
            chunk_size,
            overlap,
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
    alignments: &[bio_utils::sam::Record],
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

// Batch logsumexp
fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.;
    }
    let max = xs.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    let sum = xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    assert!(sum >= 0., "{:?}->{}", xs, sum);
    max + sum
}

// Streaming logsumexp.
#[derive(Debug, Clone, Copy)]
struct LogSumExp {
    accum: f64,
    max: f64,
}

impl LogSumExp {
    fn new() -> Self {
        Self {
            accum: 0f64,
            max: std::f64::NEG_INFINITY,
        }
    }
}

impl std::ops::Add<f64> for LogSumExp {
    type Output = Self;
    fn add(self, rhs: f64) -> Self::Output {
        let Self { accum, max } = self;
        if rhs < max {
            Self {
                accum: accum + (rhs - max).exp(),
                max,
            }
        } else {
            Self {
                accum: accum * (max - rhs).exp() + 1f64,
                max: rhs,
            }
        }
    }
}

impl std::ops::AddAssign<f64> for LogSumExp {
    fn add_assign(&mut self, rhs: f64) {
        *self = *self + rhs;
    }
}

impl std::convert::From<LogSumExp> for f64 {
    fn from(LogSumExp { accum, max }: LogSumExp) -> Self {
        accum.ln() + max
    }
}
