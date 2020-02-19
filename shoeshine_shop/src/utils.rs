// TODO: get rid of this
/// Wrapper around an array of two elements
#[derive(Debug, Default)]
pub struct Queue<T: Copy> {
    buffer: [T; 2],
    end:    usize,
}

impl<T: Copy> Queue<T> {
    pub fn pop_front(&mut self) -> T {
        self.end -= 1;
        self.buffer.swap(0, 1);
        self.buffer[1]
    }
    pub fn push_back(&mut self, val: T) {
        self.buffer[self.end] = val;
        self.end += 1;
    }
}

// TODO: get rid of this
/// Sorted array on stack with capacity 3
#[derive(Debug, Default)]
pub struct PriorityQueue3<T> {
    buffer: [T; 3],
    len:    usize,
}

impl<T: PartialOrd + Copy + Default + std::fmt::Debug> PriorityQueue3<T> {
    pub fn new() -> Self {
        PriorityQueue3 { buffer: [T::default(); 3], len: 0 }
    }

    pub fn push(&mut self, val: T) {
        let mut i = 0;
        while i < self.len && val > self.buffer[i] {
            i += 1;
        }

        let mut k = self.len;
        while k > i {
            self.buffer.swap(k, k - 1);
            k -= 1;
        }

        self.buffer[i] = val;
        self.len += 1;
    }

    pub fn pop(&mut self) -> T {
        for k in 0..self.len - 1 {
            self.buffer.swap(k, k + 1);
        }

        self.len -= 1;
        self.buffer[self.len]
    }
}
