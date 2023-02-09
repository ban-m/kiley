use super::{COPY_SIZE, DEL_SIZE, NUM_ROW};
use crate::op::Op;

// Anti diagonal version of the HMM.
// The index is (a, i) where a is the anti diagonal (i + j = a) and i is the position of the *query*.

impl super::PairHiddenMarkovModel {
    pub fn likelihood_antidiagonal(
        &self,
        rs: &[u8],
        qs: &[u8],
        ops: &[Op],
        radius: usize,
    ) -> (Vec<crate::Op>, f64) {
        let filling_regions = filling_region(ops, radius, rs.len(), qs.len());
        assert_eq!(filling_regions.len(), rs.len() + rs.len() + 1);
        let mut dptable = DPTable::new(filling_regions);
        for ad in 0..rs.len() + qs.len() + 1 {
            let (start, end) = dptable.filling_regions[ad];
            for i in start..end {
                let j = ad - i;
            }
        }
        todo!()
    }
    pub fn align_antidiagonal(&self, rs: &[u8], qs: &[u8], ops: &[Op], radius: f64) -> f64 {
        todo!()
    }
}

// Index of the anti-diagonal (a = 0, ..., qs.len() + rs.len())
fn filling_region(ops: &[Op], radius: usize, rlen: usize, qlen: usize) -> Vec<(usize, usize)> {
    let mut position_at_antidiagonal = vec![0];
    let mut qpos = 0usize;
    for op in ops.iter() {
        match op {
            Op::Mismatch | Op::Match => {
                position_at_antidiagonal.push(qpos);
                qpos += 1;
                position_at_antidiagonal.push(qpos);
            }
            Op::Ins => {
                qpos += 1;
                position_at_antidiagonal.push(qpos);
            }
            Op::Del => position_at_antidiagonal.push(qpos),
        }
    }
    todo!()
    // position_at_antidiagonal
    //     .iter()
    //     .enumerate()
    //     .map(|(anti_d, &x)| {
    //         let radius_start = radius.max(x).max(x.saturating_sub(anti_d));
    //         let radius_end = radius.min()
    //         let start = x.saturating_sub(radius);
    //         let end = (x + radius).min(qlen) + 1;
    //         (start, end)
    //     })
    //     .collect()
}

#[derive(Debug, Clone)]
struct DPTable<T: std::fmt::Debug + Clone + Copy> {
    inner: Vec<T>,
    filling_regions: Vec<(usize, usize)>,
}

impl DPTable<(f64, f64, f64)> {
    fn new(filling_regions: Vec<(usize, usize)>) -> Self {
        let total_cells: usize = filling_regions.iter().map(|(s, e)| e - s).sum();
        let inner = vec![(0f64, 0f64, 0f64); total_cells];
        Self {
            filling_regions,
            inner,
        }
    }
}
