// Three bit encoding for each base, gap, and "sentinel" base.
pub const ADENINE: u8 = 0b00;
pub const CYTOSINE: u8 = 0b01;
pub const GUANINE: u8 = 0b10;
pub const THYMINE: u8 = 0b11;
pub const GAP: u8 = 0b100;
pub const NULL: u8 = 0b101;
const fn lookup_table() -> [u8; 256] {
    let mut slots = [NULL; 256];
    slots[b'A' as usize] = ADENINE;
    slots[b'a' as usize] = ADENINE;
    slots[b'C' as usize] = CYTOSINE;
    slots[b'c' as usize] = CYTOSINE;
    slots[b'G' as usize] = GUANINE;
    slots[b'g' as usize] = GUANINE;
    slots[b'T' as usize] = THYMINE;
    slots[b't' as usize] = THYMINE;
    slots[b'-' as usize] = GAP;
    slots
}
pub const LOOKUP_TABLE: [u8; 256] = lookup_table();
// Convert a char to two bit encoding.
pub const fn convert_to_twobit(base: &u8) -> u8 {
    LOOKUP_TABLE[*base as usize]
}
#[derive(Debug, Clone)]
pub struct PadSeq(Vec<u8>);

// Leading and trailing sequnce size. Filled with NULL.
impl PadSeq {
    const OFFSET: usize = 3;
    pub fn new<T: std::borrow::Borrow<[u8]>>(xs: T) -> Self {
        let seq: Vec<_> = std::iter::repeat(NULL)
            .take(Self::OFFSET)
            .chain(xs.borrow().iter().map(convert_to_twobit))
            .chain(std::iter::repeat(NULL).take(Self::OFFSET))
            .collect();
        PadSeq(seq)
    }
    pub fn from(mut xs: Vec<u8>) -> Self {
        xs.iter_mut().for_each(|x| *x = convert_to_twobit(x));
        xs.extend(std::iter::repeat(NULL).take(Self::OFFSET));
        xs.reverse();
        xs.extend(std::iter::repeat(NULL).take(Self::OFFSET));
        xs.reverse();
        PadSeq(xs)
    }
    pub fn get(&self, index: isize) -> Option<&u8> {
        self.0.get((index + Self::OFFSET as isize) as usize)
    }
    pub fn get_mut(&mut self, index: isize) -> Option<&mut u8> {
        self.0.get_mut((index + Self::OFFSET as isize) as usize)
    }
    pub fn len(&self) -> usize {
        self.0.len() - 2 * Self::OFFSET
    }
    pub fn remove(&mut self, index: isize) -> u8 {
        self.0.remove((index + Self::OFFSET as isize) as usize)
    }
    pub fn insert(&mut self, index: isize, base: u8) {
        self.0
            .insert((index + Self::OFFSET as isize) as usize, base)
    }
    /// Return [start..end)
    pub fn get_range(&self, start: isize, end: isize) -> &[u8] {
        let start = (start + Self::OFFSET as isize) as usize;
        let end = (end + Self::OFFSET as isize) as usize;
        &self.0[start..end]
    }
}

impl std::convert::From<PadSeq> for Vec<u8> {
    fn from(PadSeq(mut inner): PadSeq) -> Vec<u8> {
        inner.retain(|&x| x != NULL && x != GAP);
        inner.iter_mut().for_each(|x| *x = b"ACGT"[*x as usize]);
        inner
    }
}

impl std::convert::AsRef<[u8]> for PadSeq {
    fn as_ref(&self) -> &[u8] {
        &self.0[Self::OFFSET..self.0.len() - Self::OFFSET]
    }
}

impl std::ops::Index<isize> for PadSeq {
    type Output = u8;
    fn index(&self, index: isize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl std::ops::IndexMut<isize> for PadSeq {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padseq() {
        let xs = b"CACAGTCGATGCTAGCTAGTACGTACGTACGT";
        let xs_converted: Vec<_> = xs.iter().map(convert_to_twobit).collect();
        let xs = PadSeq::new(xs.as_ref());
        assert_eq!(xs.as_ref(), xs_converted);
    }
}
