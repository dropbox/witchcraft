use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use tempfile::NamedTempFile;

//
// Writer: constructs mergeable files using NamedTempFile
//
pub struct Writer {
    file: NamedTempFile,
    writer: BufWriter<File>,
}

impl Writer {
    pub fn new() -> io::Result<Self> {
        let file = NamedTempFile::new()?;
        let writer = BufWriter::new(file.reopen()?);
        Ok(Self { file, writer })
    }

    pub fn write_record(
        &mut self,
        value: u32,
        count: u32,
        tags: &[u8],
        data: &[u8],
    ) -> io::Result<()> {
        if tags.len() != count as usize * 4 || data.len() != count as usize * 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mismatched lengths",
            ));
        }

        self.writer.write_all(&value.to_ne_bytes())?;
        self.writer.write_all(&count.to_ne_bytes())?;
        self.writer.write_all(tags)?;
        self.writer.write_all(data)?;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<NamedTempFile> {
        self.writer.flush()?;
        Ok(self.file)
    }
}

//
// Merger: performs in-place n-way merge from tempfiles, emits merged records
//
pub struct MergedEntry {
    pub value: u32,
    pub tags: Vec<u8>,
    pub data: Vec<u8>,
}

pub struct Merger {
    heap: BinaryHeap<HeapEntry>,
    readers: Vec<BufReader<File>>,
}

impl Merger {
    pub fn from_tempfiles(tempfiles: Vec<NamedTempFile>) -> io::Result<Self> {
        let mut readers = Vec::with_capacity(tempfiles.len());
        for tf in tempfiles {
            readers.push(BufReader::new(tf.reopen()?));
        }

        let mut heap = BinaryHeap::new();
        for (i, reader) in readers.iter_mut().enumerate() {
            if let Some(record) = read_record(reader)? {
                heap.push(HeapEntry {
                    record,
                    source_index: i,
                });
            }
        }

        Ok(Self { heap, readers })
    }
}

impl Iterator for Merger {
    type Item = io::Result<MergedEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.heap.pop()?;
        let mut combined_tags = current.record.tags;
        let mut combined_data = current.record.data;

        while let Some(next) = self.heap.peek() {
            if next.record.value != current.record.value {
                break;
            }

            let HeapEntry {
                record,
                source_index,
            } = self.heap.pop().unwrap();
            combined_tags.extend(record.tags);
            combined_data.extend(record.data);

            match read_record(&mut self.readers[source_index]) {
                Ok(Some(following)) => self.heap.push(HeapEntry {
                    record: following,
                    source_index,
                }),
                Ok(None) => (),
                Err(e) => return Some(Err(e)),
            }
        }

        match read_record(&mut self.readers[current.source_index]) {
            Ok(Some(next_record)) => self.heap.push(HeapEntry {
                record: next_record,
                source_index: current.source_index,
            }),
            Ok(None) => (),
            Err(e) => return Some(Err(e)),
        }

        Some(Ok(MergedEntry {
            value: current.record.value,
            tags: combined_tags,
            data: combined_data,
        }))
    }
}

//
// Internal record and heap management
//
struct Record {
    value: u32,
    tags: Vec<u8>,
    data: Vec<u8>,
}

struct HeapEntry {
    record: Record,
    source_index: usize,
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

fn read_record(reader: &mut BufReader<File>) -> io::Result<Option<Record>> {
    let mut buf4 = [0u8; 4];

    if reader.read_exact(&mut buf4).is_err() {
        return Ok(None);
    }
    let value = u32::from_ne_bytes(buf4);

    reader.read_exact(&mut buf4)?;
    let count = u32::from_ne_bytes(buf4);

    let mut tags = vec![0u8; count as usize * 4];
    reader.read_exact(&mut tags)?;

    let mut data = vec![0u8; count as usize * 64];
    reader.read_exact(&mut data)?;

    Ok(Some(Record { value, tags, data }))
}

//
// Tests
//
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_and_merger_with_named_tempfiles() -> io::Result<()> {
        let mut w1 = Writer::new()?;
        let tags1 = [100u32.to_ne_bytes(), 101u32.to_ne_bytes()].concat();
        let data1 = vec![0xAA; 2 * 64];
        w1.write_record(10, 2, &tags1, &data1)?;
        let tf1 = w1.finish()?;

        let mut w2 = Writer::new()?;
        let tags2 = 200u32.to_ne_bytes().to_vec();
        let data2 = vec![0xBB; 64];
        w2.write_record(10, 1, &tags2, &data2)?;
        let tf2 = w2.finish()?;

        let merger = Merger::from_tempfiles(vec![tf1, tf2])?;
        let result: Vec<_> = merger.collect::<Result<_, _>>()?;

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, 10);
        assert_eq!(result[0].tags.len(), 3 * 4);
        assert_eq!(result[0].data.len(), 3 * 64);
        Ok(())
    }
}
