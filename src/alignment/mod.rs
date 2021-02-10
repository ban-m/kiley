/// Edit operation on a ternary alignment.
/// Compared to the three operation (insertion to the reference, deletion from the reference, match between the query and the reference) in
/// binary sequence alignment, there are seven operations in a ternary alignment.
/// In other words, we can express edit operations by whether or not query/reference is consumed in the operation.
/// For example, we can write an insertion to the reference as (+1, 0) and a match as (+1, +1), or for short, 11.
/// Likewise, we can write down a operation on a ternary alignment by indicating whether or not a sequence is consumed, as 001 or 111.
/// To decode an operation to this style, offset tuples, `x.to_tuple()` would suffice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// (1,0,0) in offset style
    XInsertion,
    /// (0,1,0) in offset style
    YInsertion,
    /// (0,0,1) in offset style
    ZInsertion,
    /// (0,1,1) in offset style
    XDeletion,
    /// (1,0,1) in offset style
    YDeletion,
    /// (1,1,0) in offset style
    ZDeletion,
    /// (1,1,1) in offset style
    Match,
}

impl Op {}

impl std::convert::From<u8> for Op {
    fn from(val: u8) -> Op {
        match val {
            0b001 => Op::XInsertion,
            0b010 => Op::YInsertion,
            0b100 => Op::ZInsertion,
            0b110 => Op::XDeletion,
            0b101 => Op::YDeletion,
            0b011 => Op::ZDeletion,
            0b111 => Op::Match,
            _ => panic!(),
        }
    }
}

impl std::convert::From<Op> for u8 {
    fn from(val: Op) -> u8 {
        match val {
            Op::XInsertion => 0b001,
            Op::YInsertion => 0b010,
            Op::ZInsertion => 0b100,
            Op::XDeletion => 0b110,
            Op::YDeletion => 0b101,
            Op::ZDeletion => 0b011,
            Op::Match => 0b111,
        }
    }
}

impl std::convert::From<Op> for (bool, bool, bool) {
    fn from(val: Op) -> (bool, bool, bool) {
        match val {
            Op::XInsertion => (true, false, false),
            Op::YInsertion => (false, true, false),
            Op::ZInsertion => (false, false, true),
            Op::XDeletion => (false, true, true),
            Op::YDeletion => (true, false, true),
            Op::ZDeletion => (true, true, false),
            Op::Match => (true, true, true),
        }
    }
}

pub mod banded;
pub mod naive;
pub mod wavefront;
pub fn recover(xs: &[u8], ys: &[u8], zs: &[u8], ops: &[Op]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut x, mut y, mut z) = (0, 0, 0);
    let (mut x_res, mut y_res, mut z_res) = (vec![], vec![], vec![]);
    for &op in ops.iter() {
        let (x_proc, y_proc, z_proc): (bool, bool, bool) = op.into();
        if x_proc {
            x_res.push(xs[x]);
            x += 1;
        } else {
            x_res.push(b'-');
        }
        if y_proc {
            y_res.push(ys[y]);
            y += 1;
        } else {
            y_res.push(b'-');
        }
        if z_proc {
            z_res.push(zs[z]);
            z += 1;
        } else {
            z_res.push(b'-');
        }
    }
    (x_res, y_res, z_res)
}
