// Three bit encoding for each base, gap, and "sentinel" base.
pub const ADENINE: u8 = 0b00;
pub const CYTOSINE: u8 = 0b01;
pub const GUANINE: u8 = 0b10;
pub const THYMINE: u8 = 0b11;
pub const GAP: u8 = 0b100;
pub const NULL: u8 = 0b101;
// Convert a char to two bit encoding.
pub const fn convert_to_twobit(base: &u8) -> u8 {
    match *base {
        b'A' | b'a' => ADENINE,
        b'C' | b'c' => CYTOSINE,
        b'G' | b'g' => GUANINE,
        b'T' | b't' => THYMINE,
        b'-' => GAP,
        _ => NULL,
    }
}
#[derive(Debug, Clone)]
pub struct PadSeq(Vec<u8>);

// Leading and trailing sequnce size. Filled with NULL.
const OFFSET: usize = 3;
impl PadSeq {
    pub fn new<T: std::borrow::Borrow<[u8]>>(xs: T) -> Self {
        let seq: Vec<_> = std::iter::repeat(NULL)
            .take(OFFSET)
            .chain(xs.borrow().iter().map(convert_to_twobit))
            .chain(std::iter::repeat(NULL).take(OFFSET))
            .collect();
        PadSeq(seq)
    }
    pub fn get(&self, index: isize) -> Option<&u8> {
        self.0.get((index + OFFSET as isize) as usize)
    }
    pub fn len(&self) -> usize {
        self.0.len() - 2 * OFFSET
    }
}

impl std::ops::Index<isize> for PadSeq {
    type Output = u8;
    fn index(&self, index: isize) -> &Self::Output {
        self.get(index).unwrap()
    }
}