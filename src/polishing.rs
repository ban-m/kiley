use crate::bialignment;
use crate::hmm::HMMPolishConfig;
use crate::op::Op;
use crate::padseq;
use crate::PolishConfig;
use crate::SeqRecord;
use bio_utils::sam;
use rayon::prelude::*;

pub(crate) fn polish_single<I, S, J, T>(
    template: &SeqRecord<I, S>,
    alignments: &[(&bio_utils::sam::Record, &SeqRecord<J, T>)],
    config: &PolishConfig,
) -> SeqRecord<String, Vec<u8>>
where
    I: std::borrow::Borrow<str>,
    S: std::borrow::Borrow<[u8]>,
    J: std::borrow::Borrow<str>,
    T: std::borrow::Borrow<[u8]>,
{
    let chunks = register_all_alignments(template, alignments, config);
    use bialignment::guided::polish_until_converge_with;
    let polished = chunks
        .into_par_iter()
        .map(|(draft, seqs, mut ops)| match seqs.len() < 5 {
            true => draft.to_vec(),
            false => {
                let draft = polish_until_converge_with(draft, &seqs, &mut ops, config.radius);
                let pconfig = HMMPolishConfig::new(config.radius, seqs.len(), 0);
                config
                    .hmm
                    .polish_until_converge_guided(&draft, &seqs, &mut ops, &pconfig)
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

type AlnInfo<'a> = (&'a [u8], Vec<Vec<u8>>, Vec<Vec<Op>>);
fn register_all_alignments<'a, I, S, J, T>(
    template: &'a SeqRecord<I, S>,
    alignments: &[(&'a sam::Record, &'a SeqRecord<J, T>)],
    config: &PolishConfig,
) -> Vec<AlnInfo<'a>>
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
    let (score, (mut i, mut j)) = (0..ys.len() + 1)
        .map(|j| (xs.len(), j))
        .map(|(i, j)| (dp[i][j], (i, j)))
        .max_by_key(|x| x.0)
        .unwrap_or_else(|| panic!("{}", line!()));
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
) -> Vec<(usize, Vec<u8>, Vec<Op>)> {
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
    (len <= r_pos).then_some((popped_ops, q_pos))
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

fn revcmp(xs: &[u8]) -> Vec<u8> {
    xs.iter()
        .map(padseq::convert_to_twobit)
        .map(|x| b"TGCA"[x as usize])
        .rev()
        .collect()
}
