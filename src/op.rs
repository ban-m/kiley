#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Mismatch,
    Match,
    Ins,
    Del,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        let op = match self {
            Op::Mismatch => 'X',
            Op::Match => '=',
            Op::Ins => 'I',
            Op::Del => 'D',
        };
        f.write_char(op)
    }
}

/// xs is the reference, ys is the query.
pub fn recover(xs: &[u8], ys: &[u8], ops: &[Op]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut i, mut j) = (0, 0);
    let (mut xr, mut yr, mut aln) = (vec![], vec![], vec![]);
    for &op in ops {
        match op {
            Op::Mismatch | Op::Match => {
                xr.push(xs[i]);
                yr.push(ys[j]);
                if xs[i] == ys[j] {
                    aln.push(b'|');
                } else {
                    aln.push(b'X');
                }
                i += 1;
                j += 1;
            }
            Op::Del => {
                xr.push(xs[i]);
                aln.push(b' ');
                yr.push(b' ');
                i += 1;
            }
            Op::Ins => {
                xr.push(b' ');
                aln.push(b' ');
                yr.push(ys[j]);
                j += 1;
            }
        }
    }
    (xr, aln, yr)
}

/// Edit operation (Used in polishing phase)
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Edit {
    Subst,
    // Insertion before a specified position.
    Insertion,
    Copy(usize),
    Deletion(usize),
}

/// Fix(or "lift-over") the edit operation `ops` so that the modified operation would be compatible
/// with the reference after all the `updated_position` editation applied.
pub fn fix_alignment_path<T>(ops: &mut Vec<Op>, updated_position: T, qlen: usize, rlen: usize)
where
    T: std::iter::Iterator<Item = (usize, Edit)>,
{
    let mut new_ops = Vec::with_capacity(ops.len());
    let mut prev_pos = 0;
    let mut op_idx = 0;
    for (position, op) in updated_position {
        op_idx += take_n_reference_base(ops, &mut new_ops, op_idx, position - prev_pos);
        prev_pos = match op {
            Edit::Subst => {
                // Mism. Just take another operation.
                op_idx += take_n_reference_base(ops, &mut new_ops, op_idx, 1);
                position + 1
            }
            Edit::Insertion => {
                // Insertion. Add a new deletion operation(get rid of the new inserted element)
                new_ops.push(Op::Del);
                op_idx += take_n_reference_base(ops, &mut new_ops, op_idx, 1);
                position + 1
            }
            Edit::Copy(len) => {
                new_ops.extend(std::iter::repeat(Op::Del).take(len));
                op_idx += take_n_reference_base(ops, &mut new_ops, op_idx, 1);
                position + 1
            }
            Edit::Deletion(len) => {
                op_idx += skip_n_reference_base(ops, &mut new_ops, op_idx, len);
                position + len
            }
        };
    }
    for &op in ops.iter().skip(op_idx) {
        op_idx += 1;
        new_ops.push(op);
    }
    // assert_eq!(op_idx, ops.len());
    let ref_len = new_ops.iter().filter(|&&op| op != Op::Ins).count();
    let qry_len = new_ops.iter().filter(|&&op| op != Op::Del).count();
    assert_eq!((rlen, qlen), (ref_len, qry_len));
    *ops = new_ops;
}

fn take_n_reference_base(ops: &[Op], new_ops: &mut Vec<Op>, start: usize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let mut take_so_far = 0;
    ops.iter()
        .skip(start)
        .take_while(|&&op| {
            new_ops.push(op);
            take_so_far += (op != Op::Ins) as usize;
            take_so_far < len
        })
        .count()
        + 1
}
fn skip_n_reference_base(ops: &[Op], new_ops: &mut Vec<Op>, start: usize, len: usize) -> usize {
    let mut take_so_far = 0;
    ops.iter()
        .skip(start)
        .take_while(|&&op| {
            take_so_far += (op != Op::Ins) as usize;
            if op != Op::Del {
                new_ops.push(Op::Ins);
            }
            take_so_far < len
        })
        .count()
        + 1
}

#[cfg(test)]
pub mod tests {
    use super::*;
    #[test]
    fn take_skip() {
        use super::Op::*;
        {
            let ops = vec![Match, Mismatch, Del, Ins, Del, Match];
            let index = 1;
            let mut buffer = vec![];
            let proc = take_n_reference_base(&ops, &mut buffer, index, 2);
            assert_eq!(proc, 2);
            assert_eq!(buffer, vec![Mismatch, Del]);
        }
        {
            let ops = vec![Match, Mismatch, Del, Ins, Del, Match];
            let index = 2;
            let mut buffer = vec![];
            let proc = skip_n_reference_base(&ops, &mut buffer, index, 2);
            assert_eq!(proc, 3);
            assert_eq!(buffer, vec![Ins]);
        }
    }
}
