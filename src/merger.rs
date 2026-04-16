use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use tempfile::NamedTempFile;

//
// Writer
//
pub struct Writer {
    file: NamedTempFile,
    writer: BufWriter<File>,
    row_size: usize,
}

impl Writer {
    pub fn new(row_size: usize) -> io::Result<Self> {
        let file = NamedTempFile::new()?;
        let writer = BufWriter::new(file.reopen()?);
        Ok(Self { file, writer, row_size })
    }

    pub fn write_record(&mut self, value: u32, keys: &[(u32, u32)], data: &[u8]) -> io::Result<()> {
        let count = keys.len() as u32;
        if data.len() != count as usize * self.row_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("data length must be count * {}", self.row_size),
            ));
        }

        self.writer.write_all(&value.to_ne_bytes())?;
        self.writer.write_all(&count.to_ne_bytes())?;
        for &(major, minor) in keys {
            self.writer.write_all(&major.to_ne_bytes())?;
            self.writer.write_all(&minor.to_ne_bytes())?;
        }
        self.writer.write_all(data)?;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<NamedTempFile> {
        self.writer.flush()?;
        Ok(self.file)
    }
}

//
// Merger
//
pub struct MergedEntry {
    pub value: u32,
    pub keys: Vec<(u32, u32)>,
    pub data: Vec<u8>,
}

pub struct Merger {
    heap: BinaryHeap<HeapEntry>,
    readers: Vec<BufReader<File>>,
    row_size: usize,
}

impl Merger {
    pub fn from_tempfiles(tempfiles: Vec<NamedTempFile>, row_size: usize) -> io::Result<Self> {
        let mut readers = Vec::with_capacity(tempfiles.len());
        for tf in tempfiles {
            readers.push(BufReader::new(tf.reopen()?));
        }

        let mut heap = BinaryHeap::new();
        for (i, reader) in readers.iter_mut().enumerate() {
            if let Some(record) = read_record(reader, row_size)? {
                heap.push(HeapEntry { record, source: i });
            }
        }

        Ok(Self { heap, readers, row_size })
    }
}

impl Iterator for Merger {
    type Item = io::Result<MergedEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.heap.pop()?;

        let mut combined_keys = current.record.keys;
        let mut combined_data = current.record.data;

        while let Some(next) = self.heap.peek() {
            if next.record.value != current.record.value {
                break;
            }

            let next = self.heap.pop().unwrap();
            combined_keys.extend(next.record.keys);
            combined_data.extend(next.record.data);

            if let Some(record) = match read_record(&mut self.readers[next.source], self.row_size) {
                Ok(Some(r)) => Some(r),
                Ok(None) => None,
                Err(e) => return Some(Err(e)),
            } {
                self.heap.push(HeapEntry {
                    record,
                    source: next.source,
                });
            }
        }

        if let Some(record) = match read_record(&mut self.readers[current.source], self.row_size) {
            Ok(Some(r)) => Some(r),
            Ok(None) => None,
            Err(e) => return Some(Err(e)),
        } {
            self.heap.push(HeapEntry {
                record,
                source: current.source,
            });
        }

        // Sort keys and align data rows
        let mut keyed_data: Vec<_> = combined_keys
            .into_iter()
            .zip(combined_data.chunks_exact(self.row_size).map(|c| c.to_vec()))
            .collect();

        keyed_data.sort_by_key(|(k, _)| *k);
        let (sorted_keys, sorted_data): (Vec<_>, Vec<_>) = keyed_data.into_iter().unzip();

        Some(Ok(MergedEntry {
            value: current.record.value,
            keys: sorted_keys,
            data: sorted_data.concat(),
        }))
    }
}

//
// Internal Record Reading
//
struct Record {
    value: u32,
    keys: Vec<(u32, u32)>,
    data: Vec<u8>,
}

struct HeapEntry {
    record: Record,
    source: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.record.value.cmp(&self.record.value)
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.record.value == other.record.value
    }
}
impl Eq for HeapEntry {}

fn read_record(reader: &mut BufReader<File>, row_size: usize) -> io::Result<Option<Record>> {
    let mut buf4 = [0u8; 4];

    if reader.read_exact(&mut buf4).is_err() {
        return Ok(None);
    }
    let value = u32::from_ne_bytes(buf4);

    reader.read_exact(&mut buf4)?;
    let count = u32::from_ne_bytes(buf4);

    let mut keys = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let mut major = [0u8; 4];
        let mut minor = [0u8; 4];
        reader.read_exact(&mut major)?;
        reader.read_exact(&mut minor)?;
        keys.push((u32::from_ne_bytes(major), u32::from_ne_bytes(minor)));
    }

    let mut data = vec![0u8; count as usize * row_size];
    reader.read_exact(&mut data)?;

    Ok(Some(Record { value, keys, data }))
}

//
// Tests
//
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_merger_sorted_keys() -> io::Result<()> {
        let mut w1 = Writer::new(64)?;
        let keys1 = vec![(2, 3), (5, 1)];
        let data1 = vec![1u8; 128];
        w1.write_record(42, &keys1, &data1)?;
        let tf1 = w1.finish()?;

        let mut w2 = Writer::new(64)?;
        let keys2 = vec![(1, 9)];
        let data2 = vec![2u8; 64];
        w2.write_record(42, &keys2, &data2)?;
        let tf2 = w2.finish()?;

        let merger = Merger::from_tempfiles(vec![tf1, tf2], 64)?;
        let entries: Vec<_> = merger.collect::<Result<_, _>>()?;

        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.value, 42);
        assert_eq!(entry.keys, vec![(1, 9), (2, 3), (5, 1)]);
        assert_eq!(entry.data.len(), 192);
        assert_eq!(entry.data[..64], vec![2u8; 64][..]); // key (1,9) should still match data from w2

        Ok(())
    }

    #[test]
    fn test_key_sorting_preserves_data_alignment() -> io::Result<()> {
        let mut w1 = Writer::new(64)?;
        let keys1 = vec![(2, 5), (1, 1)];
        let data1 = [b'A'; 64].into_iter().chain([b'B'; 64]).collect::<Vec<_>>();
        w1.write_record(100, &keys1, &data1)?;
        let tf1 = w1.finish()?;

        let mut w2 = Writer::new(64)?;
        let keys2 = vec![(3, 0)];
        let data2 = vec![b'C'; 64];
        w2.write_record(100, &keys2, &data2)?;
        let tf2 = w2.finish()?;

        let merger = Merger::from_tempfiles(vec![tf1, tf2], 64)?;
        let merged: Vec<_> = merger.collect::<Result<_, _>>()?;
        assert_eq!(merged.len(), 1);

        let m = &merged[0];
        assert_eq!(m.value, 100);
        assert_eq!(m.keys, vec![(1, 1), (2, 5), (3, 0)]);

        let rows: Vec<&[u8]> = m.data.chunks_exact(64).collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![b'B'; 64].as_slice()); // (1,1)
        assert_eq!(rows[1], vec![b'A'; 64].as_slice()); // (2,5)
        assert_eq!(rows[2], vec![b'C'; 64].as_slice()); // (3,0)

        Ok(())
    }

    #[test]
    fn test_merger_with_duplicate_keys() -> io::Result<()> {
        let mut w = Writer::new(64)?;
        let keys = vec![(1, 2), (1, 2), (2, 0)];
        let data = vec![b'A'; 64]
            .into_iter()
            .chain(vec![b'B'; 64])
            .chain(vec![b'C'; 64])
            .collect::<Vec<_>>();
        w.write_record(7, &keys, &data)?;
        let tf = w.finish()?;

        let merger = Merger::from_tempfiles(vec![tf], 64)?;
        let entries: Vec<_> = merger.collect::<Result<_, _>>()?;
        let e = &entries[0];

        assert_eq!(e.keys, vec![(1, 2), (1, 2), (2, 0)]);
        assert_eq!(e.data.len(), 3 * 64);
        assert_eq!(&e.data[0..64], vec![b'A'; 64].as_slice());
        assert_eq!(&e.data[64..128], vec![b'B'; 64].as_slice());
        assert_eq!(&e.data[128..], vec![b'C'; 64].as_slice());

        Ok(())
    }
}
