//! This is a tiny library to compute ReAligner algorithm.

/// This is the default window size of ReAligner.
/// At every iteration step, the
/// profile is aligned to sequence.
/// Usual S-W-G algorithm is O(N^2) where N is the
/// length of the sequence, thus it becomes the
/// bottleneck of the algorithm.
/// To manage this, I used a `banded` SW algorithm.
pub const WINDOW_SIZE: usize = 500;

/// This is the halting threshold used in ReAligner by default.
/// Here, the termination would be determied by the "average" thereshold.
/// In other words, if `THRESHOLD * NUM_OF_SEQ > divergence of sequence`, then
/// realignemnt terminates.
pub const THRESHOLD: f64 = 0.3;
const SMALL: f64 = -10_000.0;
const INS: f64 = -0.4;
const DEL: f64 = -0.6;
type Read = Vec<u8>;
type DP = Vec<Vec<f64>>;
type Trace = Vec<Vec<Ed>>;
type Offsets = Vec<usize>;
/// Alignment operations.
/// Note that there is a special operation named "null gap".
/// This is equivelent to 'start-end' gap in the alphabet used.
/// For example, Vec[NullDel(3),Match(10),In(1),Match(4),Del(1),Match(3),In(1)]
/// is equavalent to
/// MSA  : XXXXXXXXXXXXX-XXXXXXXX-
/// ALIGN:    |||||||||| |||| |||
/// QUERY: ___XXXXXXXXXXXXXXX-XXXX
pub enum Op {
    Match(u16),
    In(u16),
    Del(u16),
    NullDel(u16),
}

/// The struct to represent multiple sequence alignment.
/// Each element is a vector on an alphabet {'A', 'C', 'G', 'T', '-', '_'}.
/// Where '-' represents a gap, meanwhile '_' represents a "end/start gap".
/// Strictly speaking, if there is no {'A','C', 'G', 'T'} in the following position,
/// or previous position,
/// they are consided to be end/start gap respectively.
#[derive(Clone, PartialEq, Eq)]
pub struct MultipleSequenceAlignment {
    /// Sequence with gaps
    seq: Vec<Vec<u8>>,
    /// Length of MSA
    len: usize,
}

impl MultipleSequenceAlignment {
    /// Construct a new MSA by using the first sequence as "seed" sequence, align the rest of the sequence onto it.
    pub fn from_raw_reads<T: std::borrow::Borrow<[u8]>>(input: &[T], radius: usize) -> Self {
        assert!(input.len() > 1);
        let seed = input[0].borrow();
        use crate::bialignment;
        let alns: Vec<_> = input
            .iter()
            .skip(1)
            .filter_map(|y| {
                let y = y.borrow();
                bialignment::edit_dist_banded(seed, y, radius).map(|(_, ops)| (y, ops))
            })
            .collect();
        let mut insertion_len = vec![0; seed.len()];
        for (_, aln) in alns.iter() {
            let mut r_pos = 0;
            let mut current_ins_len = 0;
            for op in aln {
                match op {}
            }
        }
        Self {
            seq: vec![],
            len: 0,
        }
    }
    pub fn new(input: &[Read]) -> Self {
        let mut seq: Vec<_> = input.to_vec();
        seq.iter_mut().for_each(Self::supress_trailing_gap);
        let len = seq[0].len();
        Self { seq, len }
    }
    fn average_divergence(&self) -> f64 {
        let seq: Vec<_> = self.seq.iter().map(|e| e.as_slice()).collect();
        let freqs = Self::summarize(&seq);
        let sum = freqs
            .into_iter()
            .map(|(max_base, freq)| match max_base {
                b'A' => 1. - freq[0],
                b'C' => 1. - freq[1],
                b'G' => 1. - freq[2],
                b'T' => 1. - freq[3],
                b'-' => 1. - freq[4],
                _ => unreachable!(),
            })
            .sum::<f64>();
        sum / self.len as f64
    }
    fn split(&mut self) -> Read {
        self.seq.pop().unwrap()
    }
    fn determin(query: &[u8]) -> (usize, usize) {
        let start = query.iter().take_while(|&&e| e == b'_').count();
        let end = query.iter().rev().take_while(|&&e| e == b'_').count();
        (start, query.len() - end)
    }
    fn get_focus(&self, start: usize, end: usize, w: usize) -> (usize, usize) {
        // Determin start and end positions
        let start = if start < w { 0 } else { start - w };
        let end = if end + w < self.len {
            end + w
        } else {
            self.len
        };
        (start, end)
    }
    fn summarize(msa: &[&[u8]]) -> Vec<(u8, [f64; 5])> {
        let mut freqs = vec![[0; 5]; msa[0].len()];
        let mut count = vec![0; msa[0].len()];
        for read in msa {
            for (idx, base) in read.iter().enumerate() {
                match base {
                    b'A' => freqs[idx][0] += 1,
                    b'C' => freqs[idx][1] += 1,
                    b'G' => freqs[idx][2] += 1,
                    b'T' => freqs[idx][3] += 1,
                    b'-' => freqs[idx][4] += 1,
                    b'_' => {}
                    _ => unreachable!(),
                }
                count[idx] += match base {
                    b'A' | b'C' | b'G' | b'T' | b'-' => 1,
                    b'_' => 0,
                    _ => unreachable!(),
                };
            }
        }
        freqs
            .into_iter()
            .zip(count.into_iter())
            .map(|(freq, count)| {
                let max_base = match *freq.iter().max().unwrap() {
                    x if x == freq[0] => b'A',
                    x if x == freq[1] => b'C',
                    x if x == freq[2] => b'G',
                    x if x == freq[3] => b'T',
                    x if x == freq[4] => b'-',
                    _ => unreachable!(),
                };
                let count = f64::from(count);
                let freq = [
                    f64::from(freq[0]) / count,
                    f64::from(freq[1]) / count,
                    f64::from(freq[2]) / count,
                    f64::from(freq[3]) / count,
                    f64::from(freq[4]) / count,
                ];
                (max_base, freq)
            })
            .collect()
    }
    fn match_score(base: u8, &(max_base, frequency): &(u8, [f64; 5])) -> f64 {
        let base = if base == b'_' { b'-' } else { base };
        let a = if base == max_base { 1. } else { -1. };
        let b = match base {
            b'A' => frequency[0],
            b'C' => frequency[1],
            b'G' => frequency[2],
            b'T' => frequency[3],
            b'-' => frequency[4],
            _ => unreachable!(),
        };
        0.5 * a + 0.5 * b
    }
    fn calc_offset(prev_max: usize, w: usize, end: usize) -> usize {
        if prev_max < w {
            0
        } else if end < prev_max + w {
            end - 2 * w
        } else {
            prev_max - w
        }
    }
    fn calc_max_position(dp: &[f64]) -> usize {
        dp.iter()
            .enumerate()
            .fold(
                (0, -1.),
                |(midx, max), (idx, &s)| if s > max { (idx, s) } else { (midx, max) },
            )
            .0
    }
    // Fallback for very big w
    fn fill_dp_fb(msa: &[&[u8]], query: &[u8]) -> (DP, Trace) {
        // eprintln!("Fallback");
        let height = query.len();
        let width = msa[0].len();
        let mut dp = vec![vec![0.; width + 1]; height + 1];
        // (0..=width).for_each(|j| dp[0][j] == 0);
        let mut traceback = vec![vec![Ed::Term; width + 1]; height + 1];
        let summary = Self::summarize(msa);
        // Initialization
        (1..=height).for_each(|i| dp[i][0] = SMALL);
        for i in 1..=height {
            for j in 1..=width {
                let ins = dp[i - 1][j] + INS;
                let del = dp[i][j - 1] + DEL;
                let mat = dp[i - 1][j - 1] + Self::match_score(query[i - 1], &summary[j - 1]);
                dp[i][j] = ins.max(del).max(mat);
                traceback[i][j] = match dp[i][j] {
                    x if (x - mat).abs() < std::f64::EPSILON => Ed::Mat,
                    x if (x - ins).abs() < std::f64::EPSILON => Ed::Ins,
                    x if (x - del).abs() < std::f64::EPSILON => Ed::Del,
                    _ => unreachable!(),
                };
            }
        }
        (dp, traceback)
    }
    fn calc_dp(msa: &[&[u8]], query: &[u8], w: usize) -> Vec<Ed> {
        if 2 * w > query.len() {
            let (dp, traceback) = Self::fill_dp_fb(msa, query);
            // Traceback.
            let max_j = Self::calc_max_position(dp.last().unwrap());
            Self::calc_traceback_fb(traceback, max_j)
        } else {
            let (dp, traceback, offsets) = Self::fill_dp(msa, query, w);
            let max_j = Self::calc_max_position(dp.last().unwrap());
            Self::calc_traceback(traceback, max_j, offsets, msa[0].len())
        }
    }
    fn fill_dp(msa: &[&[u8]], query: &[u8], w: usize) -> (DP, Trace, Offsets) {
        let height = query.len();
        let width = msa[0].len();
        let mut dp = vec![vec![0.; 2 * w + 1]; height + 1];
        dp[0][0] = 0.;
        // Note that the offsets should be monotonously increasing.
        let mut offsets = vec![0; height + 1];
        let mut traceback = vec![vec![Ed::Term; 2 * w + 1]; height + 1];
        let summary = Self::summarize(msa);
        // The index of maximum score in previous column
        let mut prev_max = 0;
        for i in 1..=height {
            let w_offset =
                Self::calc_offset(prev_max + offsets[i - 1], w, width).max(offsets[i - 1]);
            offsets[i] = w_offset;
            for j in 0..=2 * w {
                if w_offset == 0 && j == 0 {
                    dp[i][j] = SMALL;
                    continue;
                }
                let prev_j = w_offset + j - offsets[i - 1];
                let ins = if prev_j <= 2 * w {
                    dp[i - 1][prev_j] + INS
                } else {
                    SMALL
                };
                let del = if j != 0 { dp[i][j - 1] + DEL } else { SMALL };
                let mat = if prev_j <= 2 * w + 1 && prev_j != 0 {
                    dp[i - 1][prev_j - 1]
                        + Self::match_score(query[i - 1], &summary[j + w_offset - 1])
                } else {
                    SMALL
                };
                dp[i][j] = ins.max(del).max(mat);
                traceback[i][j] = match dp[i][j] {
                    x if (x - mat).abs() < std::f64::EPSILON => Ed::Mat,
                    x if (x - ins).abs() < std::f64::EPSILON => Ed::Ins,
                    x if (x - del).abs() < std::f64::EPSILON => Ed::Del,
                    _ => unreachable!(),
                };
            }
            prev_max = Self::calc_max_position(&dp[i]);
        }
        (dp, traceback, offsets)
    }
    fn calc_traceback(
        traceback: Vec<Vec<Ed>>,
        prev_max: usize,
        offsets: Vec<usize>,
        width: usize,
    ) -> Vec<Ed> {
        let (mut i, mut j) = (traceback.len() - 1, prev_max);
        let mut operations = vec![Ed::Del; width - j - offsets.last().unwrap()];
        let mut op = traceback[i][j];
        operations.push(op);
        loop {
            let next = match op {
                Ed::Term => break,
                Ed::Mat => (i - 1, offsets[i] + j - offsets[i - 1] - 1),
                Ed::Ins => (i - 1, offsets[i] + j - offsets[i - 1]),
                Ed::Del => (i, j - 1),
            };
            i = next.0;
            j = next.1;
            op = traceback[i][j];
            operations.push(op);
        }
        operations.pop();
        assert!(i == 0);
        assert!(offsets[i] == 0);
        operations.extend(vec![Ed::Del; j]);
        operations.reverse();
        operations
    }
    fn calc_traceback_fb(traceback: Vec<Vec<Ed>>, prev_max: usize) -> Vec<Ed> {
        let (mut i, mut j) = (traceback.len() - 1, prev_max);
        let mut operations = vec![Ed::Del; traceback[0].len() - 1 - j];
        let mut op = traceback[i][j];
        operations.push(op);
        loop {
            match op {
                Ed::Term => break,
                Ed::Mat => {
                    i -= 1;
                    j -= 1
                }
                Ed::Ins => i -= 1,
                Ed::Del => j -= 1,
            };
            op = traceback[i][j];
            operations.push(op);
        }
        operations.pop();
        assert!(i == 0);
        operations.extend(vec![Ed::Del; j]);
        operations.reverse();
        operations
    }
    fn query_trim_gap(query: &[u8]) -> Vec<u8> {
        query
            .iter()
            .filter(|&&e| e != b'-' && e != b'_')
            .copied()
            .collect()
    }
    fn trim_gap(&mut self, start: usize, end: usize) -> (usize, usize) {
        let summary = Self::all_gap_positions(&self.seq);
        let trailing_gap = summary[..start].iter().filter(|&&e| e).count();
        let contained_gap = summary[..end].iter().filter(|&&e| e).count();
        let seq: Vec<Vec<u8>> = self
            .seq
            .iter()
            .map(|read| {
                read.iter()
                    .zip(summary.iter())
                    .filter(|&(_, &g)| !g)
                    .map(|e| e.0)
                    .copied()
                    .collect()
            })
            .collect();
        *self = Self::new(&seq);
        (start - trailing_gap, end - contained_gap)
    }
    // The edit operations and trimmed query.
    fn align(&mut self, query: &[u8], w: usize) -> (Vec<Ed>, Vec<u8>) {
        let (start, end) = Self::determin(query);
        let query = Self::query_trim_gap(query);
        let (rstart, rend) = self.get_focus(start, end, w);
        let (rstart, rend) = self.trim_gap(rstart, rend);
        let msa: Vec<_> = self.seq.iter().map(|read| &read[rstart..rend]).collect();
        // DP matrix.
        let operations = Self::calc_dp(&msa, &query, w);
        let mut front = vec![Ed::Del; rstart];
        front.extend(operations);
        front.extend(vec![Ed::Del; self.len - rend]);
        (front, query)
    }
    fn supress_trailing_gap(read: &mut Vec<u8>) {
        read.iter_mut()
            .take_while(|b| b == &&b'-' || b == &&b'_')
            .for_each(|base| *base = b'_');
        read.iter_mut()
            .rev()
            .take_while(|b| b == &&b'-' || b == &&b'_')
            .for_each(|base| *base = b'_');
    }
    fn format_ref(read: &[u8], aln: &[Ed]) -> Vec<u8> {
        let mut res = aln
            .iter()
            .fold((vec![], 0), |(mut res, idx), op| match op {
                Ed::Mat | Ed::Del => {
                    res.push(read[idx]);
                    (res, idx + 1)
                }
                Ed::Ins => {
                    res.push(b'-');
                    (res, idx)
                }
                Ed::Term => unreachable!(),
            })
            .0;
        Self::supress_trailing_gap(&mut res);
        res
    }
    fn format_que(query: &[u8], aln: &[Ed]) -> Vec<u8> {
        let (start, end) = Self::determin(query);
        let query = &query[start..end];
        let mut res = aln
            .iter()
            .fold((vec![], 0), |(mut res, idx), op| match op {
                Ed::Mat | Ed::Ins => {
                    res.push(query[idx]);
                    (res, idx + 1)
                }
                Ed::Del => {
                    res.push(b'-');
                    (res, idx)
                }
                Ed::Term => unreachable!(),
            })
            .0;
        Self::supress_trailing_gap(&mut res);
        res
    }
    fn all_gap_positions(seq: &[Vec<u8>]) -> Vec<bool> {
        (0..seq[0].len())
            .map(|i| seq.iter().all(|read| read[i] == b'-' || read[i] == b'_'))
            .collect()
    }
    fn merge(&mut self, query: &[u8], alignment: &[Ed]) {
        let mut seq: Vec<_> = vec![Self::format_que(query, alignment)];
        seq.extend(
            self.seq
                .iter_mut()
                .map(|read| Self::format_ref(read, alignment)),
        );
        *self = MultipleSequenceAlignment::new(&seq);
    }
    pub fn realign_with(&mut self, thr: f64, window: usize) {
        let mut count = 0;
        while self.average_divergence() > thr && count < 200 {
            count += 1;
            let read = self.split();
            let (alignment, read) = self.align(&read, window);
            self.merge(&read, &alignment);
        }
    }
    pub fn realign(&mut self) {
        self.realign_with(THRESHOLD, WINDOW_SIZE);
    }
}

impl std::fmt::Debug for MultipleSequenceAlignment {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "Score:{}", self.average_divergence())?;
        let digit = 200;
        let line = self.len / digit + 1;
        for i in 0..line - 1 {
            for read in &self.seq {
                let end = ((i + 1) * digit).min(self.len);
                let print = String::from_utf8_lossy(&read[i * digit..end]);
                writeln!(f, "{}", print)?;
            }
        }
        for read in &self.seq[0..self.seq.len() - 1] {
            let print = String::from_utf8_lossy(&read[(line - 1) * digit..]);
            writeln!(f, "{}", print)?;
        }
        let print = String::from_utf8_lossy(&self.seq.last().unwrap()[(line - 1) * digit..]);
        write!(f, "{}", print)?;
        Ok(())
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ed {
    Mat,
    Ins,
    Del,
    Term,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn MSA() -> [Vec<u8>; 5] {
        [
            b"AATCACT-CT-CCTCATGCT".to_vec(),
            b"_____CTACTACCTCATGC_".to_vec(),
            b"AATCACT--TAC-CTCATG_".to_vec(),
            b"____CACTCTACCTCATCCT".to_vec(),
            b"AACTACTACTTGCGCATG__".to_vec(),
        ]
    }
    #[test]
    fn works() {}
    #[test]
    fn msa_new() {
        let _msa = MultipleSequenceAlignment::new(&MSA());
    }
    #[test]
    fn average_divergence_zero() {
        let msa = MultipleSequenceAlignment::new(&[
            b"AAAAAAAA".to_vec(),
            b"AAAAAAAA".to_vec(),
            b"AAAAAAAA".to_vec(),
        ]);
        assert!(msa.average_divergence().abs() < std::f64::EPSILON);
        let msa = MultipleSequenceAlignment::new(&[
            b"ATATA____".to_vec(),
            b"ATATA____".to_vec(),
            b"____ATATA".to_vec(),
            b"____ATATA".to_vec(),
            b"__ATATA__".to_vec(),
            b"____ATATA".to_vec(),
            b"ATATA____".to_vec(),
        ]);
        assert!(msa.average_divergence().abs() < std::f64::EPSILON);
    }
    #[test]
    fn average_divergence() {
        let msa = MultipleSequenceAlignment::new(&MSA());
        let mis = vec![0, 0, 1, 1, 1, 1, 1, 3, 1, 0, 2, 1, 1, 2, 1, 1, 1, 2, 1, 0];
        let cnt = vec![3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 2];
        let answer = mis
            .into_iter()
            .zip(cnt.into_iter())
            .map(|(x, y)| x as f64 / y as f64)
            .sum::<f64>()
            / 20.;
        //   AATCACTACTACCTCATGCT
        //   00111113102112111210
        //   33334555555555555542
        // b"AATCACT-CT-CCTCATGCT",
        // b"_____CTACTACCTCATGC_",
        // b"AATCACT--TAC-CTCATG_",
        // b"____CACTCTACCTCATCCT",
        // b"AACTACTACTTGCGCATG__",
        eprintln!("{} vs {}", answer, msa.average_divergence());
        assert!((msa.average_divergence() - answer).abs() < 0.01);
    }
    #[test]
    fn split_test() {
        let mut msa = MultipleSequenceAlignment::new(&MSA());
        let mut raw = MSA().to_vec();
        while let Some(read) = raw.pop() {
            assert_eq!(msa.split(), read);
        }
    }
    #[test]
    fn determine() {
        let mut msa = MultipleSequenceAlignment::new(&MSA());
        let read = msa.split();
        let (start, end) = MultipleSequenceAlignment::determin(&read);
        assert_eq!(start, 0);
        assert_eq!(end, 18);
        let read = msa.split();
        let (start, end) = MultipleSequenceAlignment::determin(&read);
        assert_eq!(start, 4);
        assert_eq!(end, 20);
        let read = msa.split();
        let (start, end) = MultipleSequenceAlignment::determin(&read);
        assert_eq!(start, 0);
        assert_eq!(end, 19);
        let read = msa.split();
        let (start, end) = MultipleSequenceAlignment::determin(&read);
        assert_eq!(start, 5);
        assert_eq!(end, 19);
    }
    #[test]
    fn get_focus() {
        let mut msa = MultipleSequenceAlignment::new(&MSA());
        let read = msa.split();
        let (qs, qe) = MultipleSequenceAlignment::determin(&read);
        let (start, end) = msa.get_focus(qs, qe, 0);
        assert_eq!(start, 0);
        assert_eq!(end, 18);
        let read = msa.split();
        let (qs, qe) = MultipleSequenceAlignment::determin(&read);
        let (start, end) = msa.get_focus(qs, qe, 2);
        assert_eq!(start, 2);
        assert_eq!(end, 20);
        let read = msa.split();
        let (qs, qe) = MultipleSequenceAlignment::determin(&read);
        let (start, end) = msa.get_focus(qs, qe, 2);
        assert_eq!(start, 0);
        assert_eq!(end, 20);
    }
    #[test]
    fn summarize() {
        let msa = MSA();
        let msa: Vec<_> = msa.iter().map(|e| e.as_slice()).collect();
        let summary = MultipleSequenceAlignment::summarize(&msa);
        let max_base: Vec<_> = summary.iter().map(|e| e.0).collect();
        let max_base_answer = b"AATCACTACTACCTCATGCT".to_vec();
        assert_eq!(max_base, max_base_answer);
        assert_eq!(summary[0].1, [3. / 3., 0., 0., 0., 0.]);
    }
    // #[test]
    // fn match_score() {
    //     let score = MultipleSequenceAlignment::match_score(b'A', &(b'A', [0.8, 0.1, 0., 0., 0.1]));
    //     assert_eq!(score, (1. + 0.8) / 2.);
    //     let score = MultipleSequenceAlignment::match_score(b'C', &(b'A', [0.8, 0.1, 0., 0., 0.1]));
    //     assert_eq!(score, (0. + 0.1) / 2.);
    //     let score = MultipleSequenceAlignment::match_score(b'-', &(b'A', [0.8, 0.1, 0., 0., 0.1]));
    //     assert_eq!(score, (0. + 0.1) / 2.);
    //     let score = MultipleSequenceAlignment::match_score(b'_', &(b'A', [0.8, 0.1, 0., 0., 0.1]));
    //     assert_eq!(score, (0. + 0.1) / 2.);
    // }
    #[test]
    fn calc_offset() {
        let offset = MultipleSequenceAlignment::calc_offset(4, 2, 10);
        assert_eq!(offset, 2);
        assert_eq!(MultipleSequenceAlignment::calc_offset(4, 5, 20), 0);
        assert_eq!(MultipleSequenceAlignment::calc_offset(7, 4, 10), 2);
    }
    #[test]
    fn calc_max_position() {
        let dp = [1., 2., 3., 4.];
        assert_eq!(MultipleSequenceAlignment::calc_max_position(&dp), 3);
        let dp = [2., 4., 2., 1.];
        assert_eq!(MultipleSequenceAlignment::calc_max_position(&dp), 1);
    }
    #[test]
    fn dp() {
        let query = b"AATATATA";
        let msa = vec![query.to_vec()];
        let msa: Vec<_> = msa.iter().map(|e| e.as_slice()).collect();
        let ed = MultipleSequenceAlignment::calc_dp(&msa, query, 10);
        eprintln!("{:?}", ed);
        assert!(true);
    }
    #[test]
    fn fill_dp() {
        let query = b"ATCCAGTCA".to_vec();
        let msa = vec![query.as_slice()];
        let w = 1;
        let (dp, trace, offsets) = MultipleSequenceAlignment::fill_dp(&msa, &query, w);
        for i in 0..trace.len() {
            for _ in 0..offsets[i] {
                eprint!(" ");
            }
            for op in &trace[i] {
                eprint!(
                    "{}",
                    match op {
                        Ed::Term => 'T',
                        Ed::Del => 'D',
                        Ed::Ins => 'I',
                        Ed::Mat => 'M',
                    }
                );
            }
            eprintln!();
        }
        let max = dp
            .last()
            .unwrap()
            .into_iter()
            .fold(0., |x, &y| if x < y { y } else { x });
        assert!((max - query.len() as f64) < 0.01);
    }
    #[test]
    fn fill_dp2() {
        let query = b"TTATACTA".to_vec();
        let msa = b"TTAGTCTA".to_vec();
        let msa = vec![msa.as_slice()];
        let answer = vec![
            Ed::Mat,
            Ed::Mat,
            Ed::Mat,
            Ed::Del,
            Ed::Mat,
            Ed::Ins,
            Ed::Mat,
            Ed::Mat,
            Ed::Mat,
        ];
        let msa = MultipleSequenceAlignment::calc_dp(&msa, &query, 2);
        assert_eq!(msa, answer);
    }
    #[test]
    fn traceback() {
        let query = b"AATATATA".to_vec();
        let msa = vec![query.as_slice()];
        let ed = MultipleSequenceAlignment::calc_dp(&msa, &query, 10);
        assert_eq!(ed, vec![Ed::Mat; 8]);
        let ed = MultipleSequenceAlignment::calc_dp(&msa, &query, 2);
        assert_eq!(ed, vec![Ed::Mat; 8])
    }
    #[test]
    fn format_ref() {
        let read = b"ATATACA";
        let ops = vec![
            Ed::Ins,
            Ed::Mat,
            Ed::Mat,
            Ed::Del,
            Ed::Ins,
            Ed::Mat,
            Ed::Del,
            Ed::Del,
            Ed::Mat,
            Ed::Ins,
        ];
        assert_eq!(
            MultipleSequenceAlignment::format_ref(read, &ops),
            b"_ATA-TACA_"
        );
        assert_eq!(
            MultipleSequenceAlignment::format_que(read, &ops),
            b"ATA-TA--CA"
        );
    }
    #[test]
    fn msa_test() {
        let mut msa = MultipleSequenceAlignment::new(&MSA());
        let msa = msa.realign_with(0.1, 20);
        eprintln!("{:?}", msa);
        assert!(true);
    }
    #[test]
    fn msa_test1() {
        let msa = vec![b"C--A".to_vec(), b"C-TA".to_vec()];
        let query = b"CT-A";
        let mut msa = MultipleSequenceAlignment::new(&msa);
        let (ed, query) = msa.align(query, 20);
        msa.merge(&query, &ed);
        assert_eq!(
            msa.seq,
            vec![b"CTA".to_vec(), b"C-A".to_vec(), b"CTA".to_vec(),]
        );
    }
}
