pub struct DPTable<T> {
    // Total memory
    mem: Vec<T>,
    // Offset of each row and the acceptable range of j.
    offsets: Vec<(usize, usize, usize)>,
    // Upperbound of the edit distance.
    upperbound: T,
}

impl<T: Copy> DPTable<T> {
    pub fn upperbound(&self) -> T {
        self.upperbound
    }
    // Set the [i,j] to target, here i and j are the original corrdinte.
    pub fn set(&mut self, i: usize, j: usize, target: T) {
        if let Some(&(ofs, s, e)) = self.offsets.get(i) {
            if (s..e).contains(&j) {
                if let Some(slot) = self.mem.get_mut(ofs + j - s) {
                    *slot = target;
                }
            }
        }
    }
    pub fn get(&self, i: usize, j: usize) -> T {
        match self.offsets.get(i) {
            Some(&(ofs, s, e)) if (s..e).contains(&j) => self.mem[ofs + j - s],
            _ => self.upperbound,
        }
    }
    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        match self.offsets.get(i) {
            Some(&(ofs, s, e)) if (s..e).contains(&j) => self.mem.get_mut(ofs + j - s),
            _ => None,
        }
    }
    // Strip line from [i][j..j+len].
    // It would be truncated appropriately with respect to the band in it.
    pub fn get_line(&self, i: usize, j_start: usize, len: usize) -> &[T] {
        match self.offsets.get(i) {
            Some(&(ofs, start, end)) if (start..end).contains(&j_start) => {
                let j_end = (j_start + len).min(end);
                &self.mem[ofs + j_start - start..ofs + j_end - start]
            }
            _ => &self.mem[0..0],
        }
    }
    pub fn new(fill_range: &[(usize, usize)], ub: T) -> Self {
        let (mut offsets, mut total_cells) = (Vec::with_capacity(fill_range.len()), 0);
        for &(start, end) in fill_range {
            offsets.push((total_cells, start, end));
            total_cells += end - start;
        }
        let mut mem = Vec::with_capacity(total_cells * 12 / 10);
        mem.extend(std::iter::repeat(ub).take(total_cells));
        Self {
            mem,
            offsets,
            upperbound: ub,
        }
    }
    pub fn with_capacity(rlen: usize, radius: usize, ub: T) -> Self {
        Self {
            mem: Vec::with_capacity(rlen * radius * 2),
            offsets: Vec::with_capacity(rlen * 2),
            upperbound: ub,
        }
    }
    pub fn initialize(&mut self, ub: T, fill_range: &[(usize, usize)]) {
        self.offsets.truncate(fill_range.len());
        if self.offsets.len() < fill_range.len() {
            let len = fill_range.len() - self.offsets.len();
            self.offsets.extend(std::iter::repeat((0, 0, 0)).take(len));
        }
        let mut total_cells = 0;
        for (i, &(start, end)) in fill_range.iter().enumerate() {
            self.offsets[i] = (total_cells, start, end);
            total_cells += end - start;
        }
        self.mem.truncate(total_cells);
        if self.mem.len() < total_cells {
            let len = total_cells - self.mem.len();
            self.mem.extend(std::iter::repeat(ub).take(len));
        }
        self.upperbound = ub;
    }
}
