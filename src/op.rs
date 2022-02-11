#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Mismatch,
    Match,
    Ins,
    Del,
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
