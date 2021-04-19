//! Alignment packages. In contrast to other packages, it contains alignment modules for three sequence, not two.
pub mod banded;
pub mod naive;
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
// Return matching array, MA.
// For three combination of THREE bit bases x, y, and, z,
// we have MA[x << 6 | y <<  3 | z ] = the penalty to align x,y, and z.
// For example, if x == y and y == z,
// MA[x << 6 | y << 3 | z ] = 0, as there is no mimatch.
// Another example is x = '-'(=0b100), y = 'G'(=0b00), and z = 'G',
// MA[100_010_010] = 1.
// Note that the match score among NULL base and others would be very large value,
// inhibiting the alignment between NULL base to others.
const LARGE: u32 = 10_000;
pub const MA32: [u32; 512] = get_match_table_u32();
const fn get_match_table_u32() -> [u32; 512] {
    let mut scores = [0u32; 512];
    let mut x = 0;
    while x < 6 {
        let mut y = 0;
        while y < 6 {
            let mut z = 0;
            while z < 6 {
                // match score of x, y, and z.
                let match_score = if x == 5 || y == 5 || z == 5 {
                    // We do not allow any alignment among NULL base and others.
                    LARGE
                } else if (x == 4 && y == 4) || (y == 4 && z == 4) || (z == 4 && x == 4) {
                    // If there's two gaps, the penalty is 1,
                    // as we could have three gaps by mutating the rest.
                    1
                } else if x == 4 {
                    // If there's only one gap,
                    // the panalty is 1 or 2,depending on
                    // whether the other two bases are qual or not.
                    1 + (y != z) as u32
                } else if y == 4 {
                    1 + (x != z) as u32
                } else if z == 4 {
                    1 + (x != y) as u32
                } else {
                    // If there is no gap character, the match score
                    // is the usual one.
                    // Note that we do not care abound efficiency,
                    // as this function is a constant function.
                    match (x == y, y == z, z == x) {
                        (true, true, true) => 0,
                        (true, false, false) => 1,
                        (false, true, false) => 1,
                        (false, false, true) => 1,
                        (false, false, false) => 2,
                        _ => 2,
                    }
                };
                let position = (x << 6) | (y << 3) | z;
                scores[position] = match_score;
                z += 1;
            }
            y += 1;
        }
        x += 1;
    }
    // NULL-NULL-NULL should be no penalty alignment.
    // This allows to move (0,0,0) -> (1,1,1) without any penalty,
    // making correct intialization.
    scores[5 << 6 | 5 << 3 | 5] = 0;
    scores
}

// Same as the previous function, only different in types.
// const MA16: [u16; 512] = get_match_table_u16();
// const fn get_match_table_u16() -> [u16; 512] {
//     let mut scores = [0u16; 512];
//     let mut x = 0;
//     while x < 5 {
//         let mut y = 0;
//         while y < 5 {
//             let mut z = 0;
//             while z < 5 {
//                 // match score of x, y, and z.
//                 let position = (x << 6) | (y << 3) | z;
//                 if (x == 4 && y == 4) || (y == 4 && z == 4) || (z == 4 && x == 4) {
//                     // If there's two gaps, the penalty is 1,
//                     // as we could have three gaps by mutating the rest.
//                     scores[position] = 1;
//                 } else if x == 4 {
//                     // If there's only one gap,
//                     // the panalty is 1 or 2,depending on
//                     // whether the other two bases are qual or not.
//                     scores[position] = 1 + (y != z) as u16;
//                 } else if y == 4 {
//                     scores[position] = 1 + (x != z) as u16;
//                 } else if z == 4 {
//                     scores[position] = 1 + (x != y) as u16;
//                 } else {
//                     // If there is no gap character, the match score
//                     // is the usual one.
//                     // Note that we do not care abound efficiency,
//                     // as this function is a constant function.
//                     scores[position] = match (x == y, y == z, z == x) {
//                         (true, true, true) => 0,
//                         (true, false, false) => 1,
//                         (false, true, false) => 1,
//                         (false, false, true) => 1,
//                         (false, false, false) => 2,
//                         _ => 2,
//                     };
//                 }
//                 z += 1;
//             }
//             y += 1;
//         }
//         x += 1;
//     }
//     scores
// }

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

#[cfg(test)]
mod test {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn compare_naive_banded() {
        let length = 100;
        let p = crate::gen_seq::PROFILE;
        for i in 0..20u64 {
            let mut rng: rand_xoshiro::Xoroshiro128PlusPlus = SeedableRng::seed_from_u64(i);
            let template = crate::gen_seq::generate_seq(&mut rng, length);
            let x = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
            let y = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
            let z = crate::gen_seq::introduce_randomness(&template, &mut rng, &p);
            let (naive_score, _) = naive::alignment(&x, &y, &z);
            let (banded_score, _) = banded::alignment_u32(&x, &y, &z, 20);
            assert_eq!(naive_score, banded_score);
            let (banded_score, _) = banded::alignment_u16(&x, &y, &z, 20);
            assert_eq!(naive_score, banded_score);
        }
    }
}
