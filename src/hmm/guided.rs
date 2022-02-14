use crate::dptable::DPTable;
use crate::op::Op;

const fn base_table() -> [usize; 256] {
    let mut slots = [0; 256];
    slots[b'A' as usize] = 0;
    slots[b'C' as usize] = 1;
    slots[b'G' as usize] = 2;
    slots[b'T' as usize] = 3;
    slots[b'a' as usize] = 0;
    slots[b'c' as usize] = 1;
    slots[b'g' as usize] = 2;
    slots[b't' as usize] = 3;
    slots
}
const BASE_TABLE: [usize; 256] = base_table();

const MIN_LK: f64 = -1000000000000000000000000000000000f64;
const COPY_SIZE: usize = 3;
const DEL_SIZE: usize = 3;
// for each position, four type of mutation,
// four type of insertion,
// 1bp, 2bp, 3bp copy,
// 1bp, 2bp, 3bp deletion,
// so, in total, there are 4 + 4 + 3 + 3 = 14 patterns of modifications.
pub const NUM_ROW: usize = 8 + COPY_SIZE + DEL_SIZE;
// After introducing mutation, we would take INACTIVE_TIME bases just as-is.
const INACTIVE_TIME: usize = 5;

#[derive(Debug, Clone)]
pub struct PairHiddenMarkovModel {
    /// Prob from mat. Logged.
    /// Pr{Mat->Mat},
    pub mat_mat: f64,
    /// Pr{Mat->Ins}
    pub mat_ins: f64,
    /// Pr{Mat->Del}
    pub mat_del: f64,
    /// Pr{Ins->Mat}
    pub ins_mat: f64,
    /// Pr{Ins->Ins}
    pub ins_ins: f64,
    /// Pr{Ins->Del}
    pub ins_del: f64,
    /// Pr{Del->Mat}.
    pub del_mat: f64,
    /// Pr{Del->Del}
    pub del_del: f64,
    // 4 * ref_base + query_base = Pr{Query|Ref}
    pub mat_emit: [f64; 16],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Match,
    Del,
    Ins,
}

// Equallity check for f64.
fn same(x: f64, y: f64) -> bool {
    (x - y).abs() < 0.0000000000000000000000001
}

impl std::default::Default for PairHiddenMarkovModel {
    fn default() -> Self {
        let mat = (0.90, 0.05, 0.05);
        let ins = (0.85, 0.10, 0.05);
        let del = (0.90, 0.10);
        let emits = vec![
            vec![0.97, 0.01, 0.01, 0.01],
            vec![0.01, 0.97, 0.01, 0.01],
            vec![0.01, 0.01, 0.97, 0.01],
            vec![0.01, 0.01, 0.01, 0.97],
        ]
        .concat();
        Self::new(mat, ins, del, &emits)
    }
}

impl PairHiddenMarkovModel {
    pub fn new(
        (mat_mat, mat_ins, mat_del): (f64, f64, f64),
        (ins_mat, ins_ins, ins_del): (f64, f64, f64),
        (del_mat, del_del): (f64, f64),
        mat_emit: &[f64],
    ) -> Self {
        assert!(0f64 <= mat_mat && 0f64 <= mat_ins && 0f64 <= mat_del);
        assert!(0f64 <= ins_mat && 0f64 <= ins_ins && 0f64 <= ins_del);
        assert!(0f64 <= del_mat && 0f64 <= del_del);
        assert!(mat_emit.iter().all(|&x| 0f64 <= x));
        let mat = mat_mat + mat_ins + mat_del;
        let ins = ins_mat + ins_ins + ins_del;
        let del = del_mat + del_del;
        let mut mat_emit_norm = [0f64; 16];
        for (from, to) in mat_emit
            .chunks_exact(4)
            .zip(mat_emit_norm.chunks_exact_mut(4))
        {
            let sum: f64 = from.iter().sum();
            to.iter_mut().zip(from).for_each(|(x, y)| *x = *y / sum);
        }
        mat_emit_norm.iter_mut().for_each(|x| *x = x.ln());
        Self {
            mat_mat: mat_mat.ln() - mat.ln(),
            mat_ins: mat_ins.ln() - mat.ln(),
            mat_del: mat_del.ln() - mat.ln(),
            ins_mat: ins_mat.ln() - ins.ln(),
            ins_ins: ins_ins.ln() - ins.ln(),
            ins_del: ins_del.ln() - ins.ln(),
            del_mat: del_mat.ln() - del.ln(),
            del_del: del_del.ln() - del.ln(),
            mat_emit: mat_emit_norm,
        }
    }
    fn zeros() -> Self {
        Self {
            mat_mat: 0f64,
            mat_ins: 0f64,
            mat_del: 0f64,
            ins_mat: 0f64,
            ins_ins: 0f64,
            ins_del: 0f64,
            del_mat: 0f64,
            del_del: 0f64,
            mat_emit: [0f64; 16],
        }
    }
    // Log
    fn obs(&self, r: u8, q: u8) -> f64 {
        let index = (BASE_TABLE[r as usize] << 2) | BASE_TABLE[q as usize];
        self.mat_emit[index as usize]
    }
    // logged
    fn del(&self, _r: u8) -> f64 {
        0f64 // with one probability, deletions state output - mark.
    }
    // Logged
    fn ins(&self, _q: u8) -> f64 {
        (0.25f64).ln()
    }
    pub fn align(&self, rs: &[u8], qs: &[u8], radius: usize) -> (f64, Vec<Op>) {
        let mut ops = bootstrap_ops(rs.len(), qs.len());
        let mut memory = Memory::with_capacity(rs.len(), radius);
        memory
            .fill_ranges
            .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
        use crate::bialignment::guided::re_fill_fill_range;
        re_fill_fill_range(qs.len(), rs.len(), &ops, radius, &mut memory.fill_ranges);
        memory.initialize();
        let lk = self.update_aln_path(&mut memory, rs, qs, &mut ops);
        assert!(lk <= 0f64, "{},{:?}", lk, self);
        (lk, ops)
    }
    fn update_aln_path(&self, memory: &mut Memory, rs: &[u8], qs: &[u8], ops: &mut Vec<Op>) -> f64 {
        self.fill_viterbi(memory, rs, qs);
        ops.clear();
        let mut qpos = qs.len();
        let mut rpos = memory.fill_ranges.last().unwrap().1 - 1;
        assert_eq!(rpos, rs.len());
        let (mut state, score) = {
            let mat = memory.pre_mat.get(qpos, rpos);
            let ins = memory.pre_ins.get(qpos, rpos);
            let del = memory.pre_del.get(qpos, rpos);
            if ins <= mat && del <= mat {
                (State::Match, mat)
            } else if ins <= del && mat <= del {
                (State::Del, del)
            } else {
                assert!(del <= ins && mat <= ins);
                (State::Ins, ins)
            }
        };
        while 0 < qpos && 0 < rpos {
            let (r, q) = (rs[rpos - 1], qs[qpos - 1]);
            match state {
                State::Match => {
                    let prev_lk = memory.pre_mat.get(qpos, rpos);
                    let mat_mat =
                        memory.pre_mat.get(qpos - 1, rpos - 1) + self.mat_mat + self.obs(r, q);
                    let del_mat =
                        memory.pre_del.get(qpos - 1, rpos - 1) + self.del_mat + self.obs(r, q);
                    state = if same(prev_lk, mat_mat) {
                        State::Match
                    } else if same(prev_lk, del_mat) {
                        State::Del
                    } else {
                        let ins_mat =
                            memory.pre_ins.get(qpos - 1, rpos - 1) + self.ins_mat + self.obs(r, q);
                        assert!(
                            same(prev_lk, ins_mat),
                            "{},{},{},{}",
                            prev_lk,
                            ins_mat,
                            mat_mat,
                            del_mat
                        );
                        State::Ins
                    };
                    qpos -= 1;
                    rpos -= 1;
                    match q == r {
                        true => ops.push(Op::Match),
                        false => ops.push(Op::Mismatch),
                    }
                }
                State::Del => {
                    let prev_lk = memory.pre_del.get(qpos, rpos);
                    let mat_del = memory.pre_mat.get(qpos, rpos - 1) + self.mat_del + self.del(r);
                    let del_del = memory.pre_del.get(qpos, rpos - 1) + self.del_del + self.del(r);
                    let ins_del = memory.pre_ins.get(qpos, rpos - 1) + self.ins_del + self.del(r);
                    state = if same(prev_lk, mat_del) {
                        State::Match
                    } else if same(prev_lk, del_del) {
                        State::Del
                    } else {
                        assert!(same(prev_lk, ins_del));
                        State::Ins
                    };
                    rpos -= 1;
                    ops.push(Op::Del);
                }
                State::Ins => {
                    let prev_lk = memory.pre_ins.get(qpos, rpos);
                    let mat_ins = memory.pre_mat.get(qpos - 1, rpos) + self.mat_ins + self.ins(q);
                    let ins_ins = memory.pre_ins.get(qpos - 1, rpos) + self.ins_ins + self.ins(q);
                    state = if same(prev_lk, mat_ins) {
                        State::Match
                    } else {
                        assert!(same(prev_lk, ins_ins));
                        State::Ins
                    };
                    qpos -= 1;
                    ops.push(Op::Ins);
                }
            }
        }
        ops.extend(std::iter::repeat(Op::Del).take(rpos));
        ops.extend(std::iter::repeat(Op::Ins).take(qpos));
        ops.reverse();
        score
    }
    // Viterbi algorithm.
    fn fill_viterbi(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        // 1. Initialize.
        {
            let mut ins_cum = self.mat_ins - self.ins_ins;
            for i in 1..qs.len() + 1 {
                ins_cum += self.ins(qs[i - 1]) + self.ins_ins;
                memory.pre_mat.set(i, 0, MIN_LK);
                memory.pre_ins.set(i, 0, ins_cum);
                memory.pre_del.set(i, 0, MIN_LK);
            }
        }
        {
            let mut del_cum = self.mat_del - self.del_del;
            for j in 1..rs.len() + 1 {
                del_cum += self.del(rs[j - 1]) + self.del_del;
                memory.pre_mat.set(0, j, MIN_LK);
                memory.pre_ins.set(0, j, MIN_LK);
                memory.pre_del.set(0, j, del_cum);
            }
        }
        {
            memory.pre_mat.set(0, 0, 0f64);
            memory.pre_del.set(0, 0, MIN_LK);
            memory.pre_ins.set(0, 0, MIN_LK);
        }
        // 2. Recur.
        for (i, &(start, end)) in memory.fill_ranges.iter().enumerate().skip(1) {
            let q = qs[i - 1];
            for j in start.max(1)..end {
                let r = rs[j - 1];
                let mat = (memory.pre_mat.get(i - 1, j - 1) + self.mat_mat)
                    .max(memory.pre_del.get(i - 1, j - 1) + self.del_mat)
                    .max(memory.pre_ins.get(i - 1, j - 1) + self.ins_mat);
                memory.pre_mat.set(i, j, mat + self.obs(r, q));
                let ins = (memory.pre_mat.get(i - 1, j) + self.mat_ins)
                    .max(memory.pre_ins.get(i - 1, j) + self.ins_ins);
                memory.pre_ins.set(i, j, ins + self.ins(q));
                let del = (memory.pre_mat.get(i, j - 1) + self.mat_del)
                    .max(memory.pre_ins.get(i, j - 1) + self.ins_del)
                    .max(memory.pre_del.get(i, j - 1) + self.del_del);
                memory.pre_del.set(i, j, del + self.del(r));
            }
        }
    }
    fn fill_pre_dp(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        // 1. Initialization.
        {
            let mut ins_cum = self.mat_ins - self.ins_ins;
            for i in 1..qs.len() + 1 {
                ins_cum += self.ins(qs[i - 1]) + self.ins_ins;
                memory.pre_mat.set(i, 0, MIN_LK);
                memory.pre_ins.set(i, 0, ins_cum);
                memory.pre_del.set(i, 0, MIN_LK);
            }
        }
        {
            let mut del_cum = self.mat_del - self.del_del;
            for j in 1..rs.len() + 1 {
                del_cum += self.del(rs[j - 1]) + self.del_del;
                memory.pre_mat.set(0, j, MIN_LK);
                memory.pre_ins.set(0, j, MIN_LK);
                memory.pre_del.set(0, j, del_cum);
            }
        }
        {
            memory.pre_mat.set(0, 0, 0f64);
            memory.pre_del.set(0, 0, MIN_LK);
            memory.pre_ins.set(0, 0, MIN_LK);
        }
        // 2. Recur
        for (i, &(start, end)) in memory.fill_ranges.iter().enumerate().skip(1) {
            let q = qs[i - 1];
            for j in start.max(1)..end {
                let r = rs[j - 1];
                let mat_mat = memory.pre_mat.get(i - 1, j - 1) + self.mat_mat;
                let del_mat = memory.pre_del.get(i - 1, j - 1) + self.del_mat;
                let ins_mat = memory.pre_ins.get(i - 1, j - 1) + self.ins_mat;
                let lk = logsumexp3(mat_mat, del_mat, ins_mat) + self.obs(r, q);
                memory.pre_mat.set(i, j, lk);
                let mat_del = memory.pre_mat.get(i, j - 1) + self.mat_del;
                let del_del = memory.pre_del.get(i, j - 1) + self.del_del;
                let ins_del = memory.pre_ins.get(i, j - 1) + self.ins_del;
                let lk = logsumexp3(mat_del, ins_del, del_del) + self.del(r);
                memory.pre_del.set(i, j, lk);
                let mat_ins = memory.pre_mat.get(i - 1, j) + self.mat_ins;
                let ins_ins = memory.pre_ins.get(i - 1, j) + self.ins_ins;
                let lk = logsumexp2(mat_ins, ins_ins) + self.ins(q);
                memory.pre_ins.set(i, j, lk);
            }
        }
    }
    fn fill_post_dp(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        // 1. Initialization
        {
            let mut ins_acc = 0f64;
            for i in (0..qs.len()).rev() {
                ins_acc += self.ins(qs[i]);
                memory.post_mat.set(i, rs.len(), self.mat_ins + ins_acc);
                memory.post_del.set(i, rs.len(), MIN_LK);
                ins_acc += self.ins_ins;
                memory.post_ins.set(i, rs.len(), ins_acc);
            }
        }
        {
            let mut del_acc = 0f64;
            for j in (0..rs.len()).rev() {
                del_acc += self.del(rs[j]);
                memory.post_mat.set(qs.len(), j, self.mat_del + del_acc);
                memory.post_ins.set(qs.len(), j, self.ins_del + del_acc);
                del_acc += self.del_del;
                memory.post_del.set(qs.len(), j, del_acc);
            }
        }
        {
            memory.post_del.set(qs.len(), rs.len(), 0f64);
            memory.post_mat.set(qs.len(), rs.len(), 0f64);
            memory.post_ins.set(qs.len(), rs.len(), 0f64);
        }
        // 2. Recur
        for (i, &(start, end)) in memory.fill_ranges.iter().enumerate().rev().skip(1) {
            let q = qs[i];
            for j in (start..end.min(rs.len())).rev() {
                let r = rs[j];
                let (mat, ins, del) = (self.obs(r, q), self.ins(q), self.del(r));
                let mat_mat = self.mat_mat + mat + memory.post_mat.get(i + 1, j + 1);
                let mat_del = self.mat_del + del + memory.post_del.get(i, j + 1);
                let mat_ins = self.mat_ins + ins + memory.post_ins.get(i + 1, j);
                memory
                    .post_mat
                    .set(i, j, logsumexp3(mat_mat, mat_del, mat_ins));
                let ins_mat = self.ins_mat + mat + memory.post_mat.get(i + 1, j + 1);
                let ins_del = self.ins_del + del + memory.post_del.get(i, j + 1);
                let ins_ins = self.ins_ins + ins + memory.post_ins.get(i + 1, j);
                memory
                    .post_ins
                    .set(i, j, logsumexp3(ins_mat, ins_del, ins_ins));
                let del_mat = self.del_mat + mat + memory.post_mat.get(i + 1, j + 1);
                let del_del = self.del_del + del + memory.post_del.get(i, j + 1);
                memory.post_del.set(i, j, logsumexp2(del_mat, del_del));
            }
        }
    }
    fn fill_mod_table(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        assert_eq!(memory.fill_ranges.len(), qs.len() + 1);
        let total_len = NUM_ROW * (rs.len() + 1);
        memory.mod_table.truncate(total_len);
        memory.mod_table.iter_mut().for_each(|x| *x = MIN_LK);
        if memory.mod_table.len() < total_len {
            let len = total_len - memory.mod_table.len();
            memory.mod_table.extend(std::iter::repeat(MIN_LK).take(len));
        }
        let mut stream_lse = vec![(MIN_LK, 0f64); NUM_ROW * (rs.len() + 1)];
        let mut slots = vec![MIN_LK; 8 + COPY_SIZE + DEL_SIZE];
        for (i, &(start, end)) in memory.fill_ranges.iter().enumerate().take(qs.len()) {
            let q = qs[i];
            for j in start..end {
                slots.iter_mut().for_each(|x| *x = MIN_LK);
                // Change the j-th base into...
                let mutates = b"ACGT".iter().map(|&b| {
                    let mat = {
                        let mat_mat =
                            self.mat_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j + 1);
                        let mat_del = self.mat_del + self.del(b) + memory.post_del.get(i, j + 1);
                        memory.pre_mat.get(i, j) + logsumexp2(mat_mat, mat_del)
                    };
                    let del = {
                        let del_mat =
                            self.del_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j + 1);
                        let del_del = self.del_del + self.del(b) + memory.post_del.get(i, j + 1);
                        memory.pre_del.get(i, j) + logsumexp2(del_mat, del_del)
                    };
                    let ins = {
                        let ins_mat =
                            self.ins_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j + 1);
                        let ins_del = self.ins_del + self.del(b) + memory.post_del.get(i, j + 1);
                        memory.pre_ins.get(i, j) + logsumexp2(ins_mat, ins_del)
                    };
                    logsumexp3(mat, del, ins)
                });
                // Insert before the j-th base ...
                let inserts = b"ACGT".iter().map(|&b| {
                    let mat = {
                        let mat_mat = self.mat_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j);
                        let mat_del = self.mat_del + self.del(b) + memory.post_del.get(i, j);
                        memory.pre_mat.get(i, j) + logsumexp2(mat_mat, mat_del)
                    };
                    let del = {
                        let del_mat = self.del_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j);
                        let del_del = self.del_del + self.del(b) + memory.post_del.get(i, j);
                        memory.pre_del.get(i, j) + logsumexp2(del_mat, del_del)
                    };
                    let ins = {
                        let ins_mat = self.ins_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j);
                        let ins_del = self.ins_del + self.del(b) + memory.post_del.get(i, j);
                        memory.pre_ins.get(i, j) + logsumexp2(ins_mat, ins_del)
                    };
                    logsumexp3(mat, del, ins)
                });
                // Copying the j..j+c bases ...
                let copies = (0..COPY_SIZE).map(|len| {
                    if j + len + 1 <= rs.len() {
                        let r = rs[j];
                        let mat = {
                            let mat_mat =
                                self.mat_mat + self.obs(r, q) + memory.post_mat.get(i + 1, j + 1);
                            let mat_del =
                                self.mat_del + self.del(r) + memory.post_del.get(i, j + 1);
                            memory.pre_mat.get(i, j + len + 1) + logsumexp2(mat_mat, mat_del)
                        };
                        let del = {
                            let del_mat =
                                self.del_mat + self.obs(r, q) + memory.post_mat.get(i + 1, j + 1);
                            let del_del =
                                self.del_del + self.del(r) + memory.post_del.get(i, j + 1);
                            memory.pre_del.get(i, j + len + 1) + logsumexp2(del_mat, del_del)
                        };
                        let ins = {
                            let ins_mat =
                                self.ins_mat + self.obs(r, q) + memory.post_mat.get(i + 1, j + 1);
                            let ins_del =
                                self.ins_del + self.del(r) + memory.post_del.get(i, j + 1);
                            memory.pre_ins.get(i, j + len + 1) + logsumexp2(ins_mat, ins_del)
                        };
                        logsumexp3(mat, del, ins)
                    } else {
                        MIN_LK
                    }
                });
                // deleting the j..j+d bases..
                let deletes = (0..DEL_SIZE).filter(|d| j + d + 1 < rs.len()).map(|len| {
                    let post = j + len + 1;
                    let r = rs[post];
                    let mat_mat =
                        self.mat_mat + self.obs(r, q) + memory.post_mat.get(i + 1, post + 1);
                    let mat_del = self.mat_del + self.del(r) + memory.post_del.get(i, post + 1);
                    let mat = memory.pre_mat.get(i, j) + logsumexp2(mat_mat, mat_del);
                    let del_mat =
                        self.del_mat + self.obs(r, q) + memory.post_mat.get(i + 1, post + 1);
                    let del_del = self.del_del + self.del(r) + memory.post_del.get(i, post + 1);
                    let del = memory.pre_del.get(i, j) + logsumexp2(del_mat, del_del);
                    let ins_mat =
                        self.ins_mat + self.obs(r, q) + memory.post_mat.get(i + 1, post + 1);
                    let ins_del = self.ins_del + self.del(r) + memory.post_del.get(i, post + 1);
                    let ins = memory.pre_ins.get(i, j) + logsumexp2(ins_mat, ins_del);
                    logsumexp3(mat, del, ins)
                });
                let row_start = NUM_ROW * j;
                slots
                    .iter_mut()
                    .zip(mutates.chain(inserts).chain(copies).chain(deletes))
                    .for_each(|(x, y)| *x = y);
                stream_logsumexp(stream_lse.iter_mut().skip(row_start), slots.iter());
            }
        }
        if let Some((start, end)) = memory.fill_ranges.last().copied() {
            let i = memory.fill_ranges.len() - 1;
            for j in start..end {
                slots.iter_mut().for_each(|x| *x = MIN_LK);
                // Change the j-th base into ...
                let mutates = b"ACGT".iter().map(|&b| {
                    let mat = memory.pre_mat.get(i, j) + self.mat_del;
                    let del = memory.pre_del.get(i, j) + self.del_del;
                    let ins = memory.pre_ins.get(i, j) + self.ins_del;
                    logsumexp3(mat, del, ins) + self.del(b) + memory.post_del.get(i, j + 1)
                });
                // Insertion before the j-th base ...
                let inserts = b"ACGT".iter().map(|&b| {
                    let mat = memory.pre_mat.get(i, j) + self.mat_del;
                    let del = memory.pre_del.get(i, j) + self.del_del;
                    let ins = memory.pre_ins.get(i, j) + self.ins_del;
                    logsumexp3(mat, del, ins) + self.del(b) + memory.post_del.get(i, j)
                });
                // Copying the j..j+c bases....
                let copies = (0..COPY_SIZE).map(|len| {
                    if j + len + 1 <= rs.len() {
                        let r = rs[j];
                        let mat = memory.pre_mat.get(i, j + len + 1) + self.mat_del;
                        let del = memory.pre_del.get(i, j + len + 1) + self.del_del;
                        let ins = memory.pre_ins.get(i, j + len + 1) + self.ins_del;
                        logsumexp3(mat, del, ins) + self.del(r) + memory.post_del.get(i, j + 1)
                    } else {
                        MIN_LK
                    }
                });
                // Deleting the j..j+d bases
                let deletes = (0..DEL_SIZE).map(|len| {
                    let post = j + len + 1;
                    if let Some(&r) = rs.get(post) {
                        let mat = memory.pre_mat.get(i, j) + self.mat_del;
                        let del = memory.pre_del.get(i, j) + self.del_del;
                        let ins = memory.pre_ins.get(i, j) + self.ins_del;
                        logsumexp3(mat, del, ins) + self.del(r) + memory.post_del.get(i, post + 1)
                    } else {
                        let mat = memory.pre_mat.get(i, j);
                        let del = memory.pre_del.get(i, j);
                        let ins = memory.pre_ins.get(i, j);
                        logsumexp3(mat, del, ins)
                    }
                });
                let row_start = NUM_ROW * j;
                slots
                    .iter_mut()
                    .zip(mutates.chain(inserts).chain(copies).chain(deletes))
                    .for_each(|(x, y)| *x = y);
                stream_logsumexp(stream_lse.iter_mut().skip(row_start), slots.iter());
            }
        }
        memory
            .mod_table
            .iter_mut()
            .zip(stream_lse)
            .for_each(|(x, (max, acc))| *x = max + acc.ln());
    }
    fn update(
        &self,
        memory: &mut Memory,
        rs: &[u8],
        qs: &[u8],
        radius: usize,
        ops: &mut Vec<Op>,
    ) -> f64 {
        use crate::bialignment::guided::re_fill_fill_range;
        memory.fill_ranges.clear();
        memory
            .fill_ranges
            .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
        re_fill_fill_range(qs.len(), rs.len(), &ops, radius, &mut memory.fill_ranges);
        memory.initialize();
        self.fill_pre_dp(memory, rs, qs);
        self.fill_post_dp(memory, rs, qs);
        self.fill_mod_table(memory, rs, qs);
        let lk = memory.post_mat.get(0, 0);
        let lk2 = {
            let mat = memory.pre_mat.get(qs.len(), rs.len());
            let del = memory.pre_del.get(qs.len(), rs.len());
            let ins = memory.pre_ins.get(qs.len(), rs.len());
            logsumexp3(mat, del, ins)
        };
        assert!((lk - lk2).abs() < 0.001, "{},{}", lk, lk2);
        memory.initialize();
        self.update_aln_path(memory, rs, qs, ops);
        lk
    }
    pub fn likelihood(&self, rs: &[u8], qs: &[u8], radius: usize) -> f64 {
        let mut memory = Memory::with_capacity(rs.len(), radius);
        let mut ops = bootstrap_ops(rs.len(), qs.len());
        self.update(&mut memory, rs, qs, radius, &mut ops)
    }
    pub fn modification_table(
        &self,
        rs: &[u8],
        qs: &[u8],
        radius: usize,
        ops: &mut Vec<Op>,
    ) -> Vec<f64> {
        let mut memory = Memory::with_capacity(rs.len(), radius);
        let _lk = self.update(&mut memory, rs, qs, radius, ops);
        memory.mod_table
    }
    /// With bootstrap operations. The returned operations would be
    /// consistent with the returned sequence.
    pub fn polish_until_converge_with<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        xs: &[T],
        ops: &mut [Vec<Op>],
        radius: usize,
    ) -> Vec<u8> {
        let mut template = template.to_vec();
        let len = template.len().min(21);
        let mut modif_table = Vec::new();
        let mut memory = Memory::with_capacity(template.len(), radius);
        for inactive in (0..100).map(|x| INACTIVE_TIME + (x * INACTIVE_TIME) % len) {
            modif_table.clear();
            let lk: f64 = ops
                .iter_mut()
                .zip(xs.iter())
                .map(|(ops, seq)| {
                    let lk = self.update(&mut memory, &template, seq.borrow(), radius, ops);
                    match modif_table.is_empty() {
                        true => modif_table.extend_from_slice(&memory.mod_table),
                        false => {
                            modif_table
                                .iter_mut()
                                .zip(memory.mod_table.iter())
                                .for_each(|(x, y)| *x += y);
                        }
                    }
                    lk
                })
                .sum();
            match polish_guided(&template, &modif_table, lk, inactive) {
                Some(next) => template = next,
                None => break,
            }
        }
        template
    }
    pub fn polish_until_converge<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        xs: &[T],
        radius: usize,
    ) -> Vec<u8> {
        let mut ops: Vec<_> = xs
            .iter()
            .map(|seq| bootstrap_ops(template.len(), seq.borrow().len()))
            .collect();
        self.polish_until_converge_with(template, xs, &mut ops, radius)
    }
    pub fn fit<T: std::borrow::Borrow<[u8]>>(&mut self, template: &[u8], xs: &[T], radius: usize) {
        let rs = template;
        let mut ops: Vec<_> = xs
            .iter()
            .map(|seq| bootstrap_ops(template.len(), seq.borrow().len()))
            .collect();
        let mut memory = Memory::with_capacity(template.len(), radius);
        for _ in 0..10 {
            let mut next = PairHiddenMarkovModel::zeros();
            for (ops, seq) in ops.iter_mut().zip(xs.iter()) {
                let qs = seq.borrow();
                use crate::bialignment::guided::re_fill_fill_range;
                memory.fill_ranges.clear();
                memory
                    .fill_ranges
                    .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
                re_fill_fill_range(qs.len(), rs.len(), &ops, radius, &mut memory.fill_ranges);
                memory.initialize();
                self.update_aln_path(&mut memory, rs, qs, ops);
                self.fill_pre_dp(&mut memory, rs, qs);
                self.fill_post_dp(&mut memory, rs, qs);
                self.register(&memory, rs, qs, radius, &mut next);
            }
            next.normalize();
            *self = next;
        }
    }
    fn register(&self, _: &Memory, _: &[u8], _: &[u8], _: usize, _next: &mut Self) {
        // Transition
        todo!()
    }
    fn normalize(&mut self) {
        let mat_sum = self.mat_mat + self.mat_ins + self.mat_del;
        self.mat_mat /= mat_sum;
        self.mat_ins /= mat_sum;
        self.mat_del /= mat_sum;
        let ins_sum = self.ins_mat + self.ins_del + self.ins_ins;
        self.ins_mat /= ins_sum;
        self.ins_del /= ins_sum;
        self.ins_ins /= ins_sum;
        let del_sum = self.del_mat + self.del_del;
        self.del_mat /= del_sum;
        self.del_del /= del_sum;
        for obss in self.mat_emit.chunks_mut(4) {
            let sum: f64 = obss.iter().sum();
            obss.iter_mut().for_each(|x| *x /= sum);
        }
    }
}

struct Memory {
    pre_mat: DPTable<f64>,
    pre_del: DPTable<f64>,
    pre_ins: DPTable<f64>,
    post_mat: DPTable<f64>,
    post_del: DPTable<f64>,
    post_ins: DPTable<f64>,
    fill_ranges: Vec<(usize, usize)>,
    mod_table: Vec<f64>,
}

impl Memory {
    fn with_capacity(rlen: usize, radius: usize) -> Self {
        let fill_ranges = Vec::with_capacity(rlen * 3);
        let mod_table = Vec::with_capacity(rlen * 3);
        Self {
            fill_ranges,
            mod_table,
            pre_mat: DPTable::with_capacity(rlen, radius, MIN_LK),
            pre_del: DPTable::with_capacity(rlen, radius, MIN_LK),
            pre_ins: DPTable::with_capacity(rlen, radius, MIN_LK),
            post_mat: DPTable::with_capacity(rlen, radius, MIN_LK),
            post_del: DPTable::with_capacity(rlen, radius, MIN_LK),
            post_ins: DPTable::with_capacity(rlen, radius, MIN_LK),
        }
    }
    fn initialize(&mut self) {
        self.pre_mat.initialize(MIN_LK, &self.fill_ranges);
        self.pre_ins.initialize(MIN_LK, &self.fill_ranges);
        self.pre_del.initialize(MIN_LK, &self.fill_ranges);
        self.post_mat.initialize(MIN_LK, &self.fill_ranges);
        self.post_del.initialize(MIN_LK, &self.fill_ranges);
        self.post_ins.initialize(MIN_LK, &self.fill_ranges);
    }
}

fn logsumexp3(x: f64, y: f64, z: f64) -> f64 {
    let max = x.max(y).max(z);
    max + ((x - max).exp() + (y - max).exp() + (z - max).exp()).ln()
}

fn logsumexp2(x: f64, y: f64) -> f64 {
    let max = x.max(y);
    max + ((x - max).exp() + (y - max).exp()).ln()
}

// Override x by log(exp(x) + exp(y)).
// fn override_sum<'a, 'b, I: Iterator<Item = &'a mut f64>, J: Iterator<Item = &'b f64>>(
//     xs: I,
//     ys: J,
// ) {
//     xs.zip(ys).for_each(|(x, &y)| {
//         if y < *x {
//             *x += (1f64 + (y - *x).exp()).ln();
//         } else {
//             *x = y + (1f64 + (*x - y).exp()).ln();
//         };
//     });
// }

fn stream_logsumexp<'a, 'b, I, J>(x: I, y: J)
where
    I: Iterator<Item = &'a mut (f64, f64)>,
    J: Iterator<Item = &'b f64>,
{
    for ((max, acc), &y) in x.zip(y) {
        if y < *max {
            *acc += (y - *max).exp();
        } else {
            *acc = *acc * (*max - y).exp() + 1f64;
            *max = y;
        }
    }
}

fn bootstrap_ops(rlen: usize, qlen: usize) -> Vec<Op> {
    let (mut qpos, mut rpos) = (0, 0);
    let mut ops = Vec::with_capacity(rlen + qlen);
    let map_coef = rlen as f64 / qlen as f64;
    while qpos < qlen && rpos < rlen {
        let corresp_rpos = (qpos as f64 * map_coef).round() as usize;
        if corresp_rpos < rpos {
            // rpos is too fast. Move qpos only.
            qpos += 1;
            ops.push(Op::Ins);
        } else if corresp_rpos == rpos {
            qpos += 1;
            rpos += 1;
            ops.push(Op::Match);
        } else {
            rpos += 1;
            ops.push(Op::Del);
        }
    }
    ops.extend(std::iter::repeat(Op::Ins).take(qlen - qpos));
    ops.extend(std::iter::repeat(Op::Del).take(rlen - rpos));
    ops
}

fn polish_guided(
    template: &[u8],
    modif_table: &[f64],
    current_lk: f64,
    inactive: usize,
) -> Option<Vec<u8>> {
    let mut improved = Vec::with_capacity(template.len() * 11 / 10);
    let mut modif_table = modif_table.chunks_exact(NUM_ROW);
    let mut pos = 0;
    while let Some(row) = modif_table.next() {
        let (op, &lk) = row
            .iter()
            .enumerate()
            .min_by(|x, y| (x.1).partial_cmp(&y.1).unwrap())
            .unwrap();
        if current_lk < lk && pos < template.len() {
            if op < 4 {
                // Mutateion.
                improved.push(b"ACGT"[op]);
                pos += 1;
            } else if op < 8 {
                // Insertion before this base.
                improved.push(b"ACGT"[op - 4]);
                improved.push(template[pos]);
                pos += 1;
            } else if op < 8 + COPY_SIZE {
                // copy from here to here + 1 + (op - 8) base
                let len = (op - 8) + 1;
                improved.extend(&template[pos..pos + len]);
                improved.push(template[pos]);
                pos += 1;
            } else if op < NUM_ROW {
                // Delete from here to here + 1 + (op - 8 - COPY_SIZE).
                let del_size = op - 8 - COPY_SIZE + 1;
                pos += del_size;
                (0..del_size - 1).filter_map(|_| modif_table.next()).count();
            } else {
                unreachable!()
            }
            improved.extend(template.iter().skip(pos).take(inactive));
            pos = (pos + inactive).min(template.len());
            (0..inactive).filter_map(|_| modif_table.next()).count();
        } else if pos < template.len() {
            improved.push(template[pos]);
            pos += 1;
        } else if current_lk < lk && 4 <= op && op < 8 {
            // Here, we need to consider the last insertion...
            improved.push(b"ACGT"[op - 4]);
        }
    }
    assert_eq!(pos, template.len());
    (improved != template).then(|| improved)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::{gen_seq, hmm::logsumexp};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;
    type PHMM = PairHiddenMarkovModel;
    #[test]
    fn align() {
        let phmm = PHMM::default();
        let (lk, ops) = phmm.align(b"ACCG", b"ACCG", 5);
        eprintln!("{:?}\t{:.3}", ops, lk);
        assert_eq!(ops, vec![Op::Match; 4]);
        let (lk, ops) = phmm.align(b"ACCG", b"", 2);
        eprintln!("{:?}\t{:.3}", ops, lk);
        assert_eq!(ops, vec![Op::Del; 4]);
        let (lk, ops) = phmm.align(b"", b"ACCG", 2);
        assert_eq!(ops, vec![Op::Ins; 4]);
        eprintln!("{:?}\t{:.3}", ops, lk);
        let (lk, ops) = phmm.align(b"ATGCCGCACAGTCGAT", b"ATCCGC", 5);
        eprintln!("{:?}\t{:.3}", ops, lk);
        use Op::*;
        let answer = vec![vec![Match; 2], vec![Del], vec![Match; 4], vec![Del; 9]].concat();
        assert_eq!(ops, answer);
    }
    #[test]
    fn likelihood() {
        let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198);
        let template = gen_seq::generate_seq(&mut rng, 300);
        let profile = gen_seq::PROFILE;
        let hmm = PHMM::default();
        let radius = 50;
        let same_lk = hmm.likelihood(&template, &template, radius);
        let (lk, _) = hmm.align(&template, &template, radius);
        let expect = (template.len() as f64 - 1f64) * hmm.mat_mat
            + (template.len() as f64) * hmm.mat_emit[0];
        assert!(lk < same_lk);
        assert!(expect < same_lk);
        for i in 0..10 {
            let seq = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let lk = hmm.likelihood(&template, &seq, radius);
            assert!(lk < same_lk, "{},{},{}", i, same_lk, lk);
            let seq2 = gen_seq::introduce_randomness(&seq, &mut rng, &profile);
            let lk2 = hmm.likelihood(&template, &seq2, radius);
            assert!(lk2 < lk, "{},{},{}", i, lk2, lk);
        }
    }
    #[test]
    fn likelihood2() {
        let seq1 = b"AC";
        let seq2 = b"CT";
        use Op::*;
        // There are no del->ins transition. why?
        let ops = vec![
            vec![Match; 2],
            vec![Ins, Del, Match],
            // vec![Del, Ins, Match],
            vec![Ins, Match, Del],
            vec![Match, Ins, Del],
            vec![Del, Match, Ins],
            // vec![Match, Del, Ins],
            // vec![Del, Del, Ins, Ins],
            // vec![Del, Ins, Del, Ins],
            // vec![Del, Ins, Ins, Del],
            // vec![Ins, Del, Del, Ins],
            // vec![Ins, Del, Ins, Del],
            vec![Ins, Ins, Del, Del],
        ];
        let hmm = PHMM::default();
        let lks: Vec<_> = ops.iter().map(|ops| hmm.eval(seq1, seq2, ops)).collect();
        eprintln!("{:?}", lks);
        let lk_e = logsumexp(&lks);
        let radius = 5;
        let lk = hmm.likelihood(seq1, seq2, radius);
        assert!((lk_e - lk).abs() < 0.001, "{},{}", lk_e, lk);
    }
    impl PairHiddenMarkovModel {
        fn eval(&self, rs: &[u8], qs: &[u8], ops: &[Op]) -> f64 {
            use Op::*;
            let mut lk = 0f64;
            let mut current = Match;
            let (mut qpos, mut rpos) = (0, 0);
            for &op in ops.iter() {
                lk += match (current, op) {
                    (Match, Match) => self.mat_mat,
                    (Match, Ins) => self.mat_ins,
                    (Match, Del) => self.mat_del,
                    (Ins, Match) => self.ins_mat,
                    (Ins, Ins) => self.ins_ins,
                    (Ins, Del) => self.ins_del,
                    (Del, Match) => self.del_mat,
                    (Del, Del) => self.del_del,
                    _ => panic!(),
                };
                match op {
                    Match => {
                        lk += self.obs(qs[qpos], rs[rpos]);
                        qpos += 1;
                        rpos += 1;
                    }
                    Ins => {
                        lk += self.ins(qs[qpos]);
                        qpos += 1;
                    }
                    Del => {
                        lk += self.del(rs[rpos]);
                        rpos += 1;
                    }
                    _ => panic!(),
                };
                current = op;
            }
            lk
        }
    }
    #[test]
    fn modification_table() {
        for seed in 0..10 {
            let mut rng: Xoshiro256StarStar = SeedableRng::seed_from_u64(32198 + seed);
            let template = gen_seq::generate_seq(&mut rng, 70);
            let profile = gen_seq::PROFILE;
            let hmm = PHMM::default();
            let radius = 50;
            let query = gen_seq::introduce_randomness(&template, &mut rng, &profile);
            let (modif_table, mut ops) = {
                let mut memory = Memory::with_capacity(template.len(), radius);
                let mut ops = bootstrap_ops(template.len(), query.len());
                let lk = hmm.update(&mut memory, &template, &query, radius, &mut ops);
                println!("LK\t{}", lk);
                (memory.mod_table, ops)
            };
            let mut memory = Memory::with_capacity(template.len(), radius);
            let mut mod_version = template.clone();
            println!("{}", seed);
            // Mutation error
            for (j, modif_table) in modif_table
                .chunks_exact(NUM_ROW)
                .take(template.len())
                .enumerate()
            {
                println!("{}", j);
                let orig = mod_version[j];
                for (&base, lk_m) in b"ACGT".iter().zip(modif_table) {
                    mod_version[j] = base;
                    let lk = hmm.update(&mut memory, &mod_version, &query, radius, &mut ops);
                    assert!((lk - lk_m).abs() < 0.0001, "{},{},mod", lk, lk_m);
                    println!("M\t{}\t{}", j, (lk - lk_m).abs());
                    mod_version[j] = orig;
                }
                // Insertion error
                for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                    mod_version.insert(j, base);
                    let lk = hmm.update(&mut memory, &mod_version, &query, radius, &mut ops);
                    assert!((lk - lk_m).abs() < 0.0001, "{},{}", lk, lk_m);
                    println!("I\t{}\t{}", j, (lk - lk_m).abs());
                    mod_version.remove(j);
                }
                // Copying mod
                for len in (0..COPY_SIZE).filter(|c| j + c + 1 <= template.len()) {
                    let lk_m = modif_table[8 + len];
                    let mod_version: Vec<_> = template[..j + len + 1]
                        .iter()
                        .chain(template[j..].iter())
                        .copied()
                        .collect();
                    let lk = hmm.update(&mut memory, &mod_version, &query, radius, &mut ops);
                    println!("C\t{}\t{}\t{}", j, len, (lk - lk_m).abs());
                    assert!((lk - lk_m).abs() < 0.0001);
                }
                // Deletion error
                for len in (0..DEL_SIZE).filter(|d| j + d + 1 <= template.len()) {
                    let lk_m = modif_table[8 + COPY_SIZE + len];
                    let mod_version: Vec<_> = template[..j]
                        .iter()
                        .chain(template[j + len + 1..].iter())
                        .copied()
                        .collect();
                    let lk = hmm.update(&mut memory, &mod_version, &query, radius, &mut ops);
                    println!("D\t{}\t{}\t{}", j, len, lk - lk_m);
                    assert!(
                        (lk - lk_m).abs() < 0.01,
                        "{},{},{}",
                        lk,
                        lk_m,
                        String::from_utf8_lossy(&template)
                    );
                }
            }
            let modif_table = modif_table
                .chunks_exact(NUM_ROW)
                .nth(template.len())
                .unwrap();
            for (&base, lk_m) in b"ACGT".iter().zip(&modif_table[4..]) {
                mod_version.push(base);
                let lk = hmm.update(&mut memory, &mod_version, &query, radius, &mut ops);
                assert!((lk - lk_m).abs() < 1.0001);
                mod_version.pop();
            }
        }
    }
}
