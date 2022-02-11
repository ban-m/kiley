use crate::dptable::DPTable;
use crate::op::Op;

const MIN_LK: f64 = -1000000000000000000000000000000000f64;
const COPY_SIZE: usize = 3;
const DEL_SIZE: usize = 3;
// for each position, four type of mutation,
// four type of insertion,
// 1bp, 2bp, 3bp copy,
// 1bp, 2bp, 3bp deletion,
// so, in total, there are 4 + 4 + 3 + 3 = 14 patterns of modifications.
const NUM_ROW: usize = 8 + COPY_SIZE + DEL_SIZE;
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

impl PairHiddenMarkovModel {
    // Log
    fn obs(&self, r: u8, q: u8) -> f64 {
        self.mat_emit[((r << 2) | q) as usize]
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
        let ops = bootstrap_ops(rs.len(), qs.len());
        let mut memory = Memory::with_capacity(rs.len(), radius);
        memory
            .fill_ranges
            .extend(std::iter::repeat((rs.len() + 1, 0)).take(qs.len() + 1));
        use crate::bialignment::guided::re_fill_fill_range;
        re_fill_fill_range(qs.len(), rs.len(), &ops, radius, &mut memory.fill_ranges);
        self.fill_viterbi(&mut memory, rs, qs);
        let mut ops = vec![];
        let lk = self.update_aln_path(&mut memory, rs, qs, &mut ops);
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
            match state {
                State::Match => {
                    let (r, q) = (rs[rpos], qs[qpos]);
                    let prev_lk = memory.pre_mat.get(qpos, rpos) - self.obs(r, q);
                    let mat_mat = memory.pre_mat.get(qpos - 1, rpos - 1) + self.mat_mat;
                    let del_mat = memory.pre_del.get(qpos - 1, rpos - 1) + self.del_mat;
                    state = if same(prev_lk, mat_mat) {
                        State::Match
                    } else if same(prev_lk, del_mat) {
                        State::Del
                    } else {
                        let ins_mat = memory.pre_ins.get(qpos - 1, rpos - 1) + self.ins_mat;
                        assert!(same(prev_lk, ins_mat,));
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
                    let prev_lk = memory.pre_del.get(qpos, rpos) - self.del(rs[rpos]);
                    let mat_del = memory.pre_mat.get(qpos, rpos - 1) + self.mat_del;
                    let del_del = memory.pre_del.get(qpos, rpos - 1) + self.del_del;
                    let ins_del = memory.pre_ins.get(qpos, rpos - 1) + self.ins_del;
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
                    let prev_lk = memory.pre_ins.get(qpos, rpos) - self.ins(qs[qpos]);
                    let mat_ins = memory.pre_mat.get(qpos - 1, rpos) + self.mat_ins;
                    let ins_ins = memory.pre_ins.get(qpos - 1, rpos) + self.ins_ins;
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
                let r = rs[j];
                let mat_mat = memory.pre_mat.get(i - 1, j - 1) + self.mat_mat;
                let del_mat = memory.pre_del.get(i - 1, j - 1) + self.del_mat;
                let ins_mat = memory.pre_ins.get(i - 1, j - 1) + self.ins_mat;
                let lk = logsumexp3(mat_mat, del_mat, ins_mat) + self.obs(r, q);
                memory.pre_mat.set(i, j, lk);
                let mat_del = memory.pre_mat.get(i, j - 1) + self.mat_del;
                let ins_del = memory.pre_ins.get(i, j - 1) + self.ins_del;
                let del_del = memory.pre_del.get(i, j - 1) + self.del_del;
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
                let ins = self.ins(qs[i]);
                let mat_ins = self.mat_ins + ins + ins_acc;
                memory.post_mat.set(i, rs.len(), mat_ins);
                memory.post_del.set(i, rs.len(), MIN_LK);
                ins_acc += self.ins_ins + ins;
                memory.post_ins.set(i, rs.len(), ins_acc);
            }
        }
        {
            let mut del_acc = 0f64;
            for j in (0..rs.len()).rev() {
                let del = self.del(rs[j]);
                let mat_del = self.mat_del + del + del_acc;
                let ins_del = self.ins_del + del + del_acc;
                memory.post_mat.set(qs.len(), j, mat_del);
                memory.post_ins.set(qs.len(), j, ins_del);
                del_acc += self.del_del + del;
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
        self.update_aln_path(memory, rs, qs, ops);
        self.fill_pre_dp(memory, rs, qs);
        self.fill_post_dp(memory, rs, qs);
        self.fill_mod_table(memory, rs, qs);
        // As the initial state is 0, Pr(Observations|state==0) would be the answer.
        memory.post_mat.get(0, 0)
    }
    pub fn polish_until_converge<T: std::borrow::Borrow<[u8]>>(
        &self,
        template: &[u8],
        xs: &[T],
        radius: usize,
    ) -> Vec<u8> {
        let mut template = template.to_vec();
        let len = template.len().min(21);
        let mut ops: Vec<_> = xs
            .iter()
            .map(|seq| bootstrap_ops(template.len(), seq.borrow().len()))
            .collect();
        let mut modif_table = Vec::new();
        let mut aligner = Memory::with_capacity(template.len(), radius);
        for inactive in (0..100).map(|x| INACTIVE_TIME + (x * INACTIVE_TIME) % len) {
            modif_table.clear();
            let lk: f64 = ops
                .iter_mut()
                .zip(xs.iter())
                .map(|(ops, seq)| {
                    let lk = self.update(&mut aligner, &template, seq.borrow(), radius, ops);
                    match modif_table.is_empty() {
                        true => modif_table.extend_from_slice(&aligner.mod_table),
                        false => {
                            modif_table
                                .iter_mut()
                                .zip(aligner.mod_table.iter())
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
    fn fill_mod_table(&self, memory: &mut Memory, rs: &[u8], qs: &[u8]) {
        assert_eq!(memory.fill_ranges.len(), qs.len() + 1);
        let total_len = NUM_ROW * (rs.len() + 1);
        memory.mod_table.truncate(total_len);
        memory.mod_table.iter_mut().for_each(|x| *x = MIN_LK);
        if memory.mod_table.len() < total_len {
            let len = total_len - memory.mod_table.len();
            memory.mod_table.extend(std::iter::repeat(MIN_LK).take(len));
        }
        let mut slots = vec![0f64; 4.max(DEL_SIZE).max(COPY_SIZE)];
        for (i, &(start, end)) in memory.fill_ranges.iter().enumerate().take(qs.len()) {
            let q = qs[i];
            for j in start..end {
                let row_start = NUM_ROW * j;
                // Change the j-th base into...
                slots.iter_mut().zip(b"ACGT").for_each(|(x, &b)| {
                    let mat_mat = self.mat_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j + 1);
                    let mat_del = self.mat_del + self.del(b) + memory.post_del.get(i, j + 1);
                    let mat = memory.pre_mat.get(i, j) + logsumexp2(mat_mat, mat_del);
                    let del_mat = self.del_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j + 1);
                    let del_del = self.del_del + self.del(b) + memory.post_del.get(i, j + 1);
                    let del = memory.pre_del.get(i, j) + logsumexp2(del_mat, del_del);
                    let ins_mat = self.ins_mat + self.obs(b, q) + memory.post_mat.get(i + 1, j + 1);
                    let ins_del = self.ins_del + self.del(b) + memory.post_del.get(i, j + 1);
                    let ins = memory.pre_ins.get(i, j) + logsumexp2(ins_mat, ins_del);
                    *x = logsumexp3(mat, del, ins);
                });
                memory
                    .mod_table
                    .iter_mut()
                    .skip(row_start)
                    .zip(slots.iter().take(4))
                    .for_each(|(x, &m)| *x = (*x).max(m));
                // Insert before the j-th base ...
                slots.iter_mut().zip(b"ACGT").for_each(|(x, &b)| {});
                override_max(
                    memory.mod_table.iter_mut().skip(row_start + 4),
                    slots.iter().take(4),
                );
                // Copying the j..j+c bases ...
                slots
                    .iter_mut()
                    .take(COPY_SIZE)
                    .enumerate()
                    .filter(|(c, _)| j + c + 1 <= rs.len())
                    .for_each(|(len, x)| {});
                override_max(
                    memory.mod_table.iter_mut().skip(row_start + 8),
                    slots.iter().take(COPY_SIZE),
                );
                // deleting the j..j+d bases..
                slots
                    .iter_mut()
                    .take(DEL_SIZE)
                    .enumerate()
                    .filter(|(d, _)| j + d + 1 <= rs.len())
                    .for_each(|(len, x)| {});
                override_max(
                    memory.mod_table.iter_mut().skip(row_start + 8 + COPY_SIZE),
                    slots.iter().take(DEL_SIZE),
                );
            }
        }
        // Insertion at the last position.
        if let Some((start, end)) = memory.fill_ranges.last().copied() {
            let i = memory.fill_ranges.len() - 1;
            for j in start..end {
                let row_start = NUM_ROW * j;
                // Change the j-th base into ...
                slots.iter_mut().zip(b"ACGT").for_each(|(x, &b)| {
                    let mat_del = self.mat_del + self.del(b) + memory.post_del.get(i, j + 1);
                    let mat = memory.pre_mat.get(i, j) + mat_del;
                    let del_del = self.del_del + self.del(b) + memory.post_del.get(i, j + 1);
                    let del = memory.pre_del.get(i, j) + del_del;
                    *x = logsumexp2(mat, del);
                });
                override_max(
                    memory.mod_table.iter_mut().skip(row_start),
                    slots.iter().take(4),
                );
                // Insertion before the j-th base ...
                slots.iter_mut().zip(b"ACGT").for_each(|(x, &b)| {});
                override_max(
                    memory.mod_table.iter_mut().skip(row_start + 4),
                    slots.iter().take(4),
                );
                // Copying the j..j+c bases....
                slots
                    .iter_mut()
                    .take(COPY_SIZE)
                    .enumerate()
                    .filter(|(c, _)| j + c + 1 <= rs.len())
                    .for_each(|(len, x)| {});
                override_max(
                    memory.mod_table.iter_mut().skip(row_start + 8),
                    slots.iter().take(COPY_SIZE),
                );
                // Deleting the j..j+d bases
                slots
                    .iter_mut()
                    .take(DEL_SIZE)
                    .enumerate()
                    .filter(|(d, _)| j + d + 1 <= rs.len())
                    .for_each(|(len, x)| {});
                override_max(
                    memory.mod_table.iter_mut().skip(row_start + 8 + COPY_SIZE),
                    slots.iter().take(DEL_SIZE),
                );
            }
        }
    }
    fn fit<T: std::borrow::Borrow<[u8]>>(
        &mut self,
        template: &[u8],
        xs: &[T],
        radius: usize,
    ) -> Vec<u8> {
        todo!()
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

fn override_max<'a, 'b, I: Iterator<Item = &'a mut f64>, J: Iterator<Item = &'b f64>>(
    xs: I,
    ys: J,
) {
    xs.zip(ys).for_each(|(x, y)| *x = (*x).max(*y));
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
        if lk < current_lk && pos < template.len() {
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
        } else if lk < current_lk && 4 <= op && op < 8 {
            // Here, we need to consider the last insertion...
            improved.push(b"ACGT"[op - 4]);
        }
    }
    assert_eq!(pos, template.len());
    (improved != template).then(|| improved)
}
