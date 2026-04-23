use std::collections::BinaryHeap;
use std::cmp::Ordering;

struct HeapEntry<T> {
    score: f32,
    data: T,
}

impl<T> PartialEq for HeapEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<T> Eq for HeapEntry<T> {}

impl<T> PartialOrd for HeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for HeapEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

pub struct MaxHeap<T> {
    heap: BinaryHeap<HeapEntry<T>>,
}

impl<T> MaxHeap<T> {
    pub fn new() -> Self {
        Self { heap: BinaryHeap::new() }
    }

    pub fn push(&mut self, score: f32, data: T) {
        self.heap.push(HeapEntry { score, data });
    }

    pub fn pop(&mut self) -> Option<(f32, T)> {
        self.heap.pop().map(|e| (e.score, e.data))
    }
}
