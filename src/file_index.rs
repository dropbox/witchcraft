use anyhow::Result;
use crate::direct_io::DirectFileWriter;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const ROWID_RECORD_BYTES: usize = std::mem::size_of::<u64>() + std::mem::size_of::<u32>();

pub(crate) fn sync_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn unique_tmp_path(path: &Path, label: &str) -> PathBuf {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(format!(".{label}.{}.tmp", FileBackedIndex::nonce()));
    PathBuf::from(tmp)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RowidRecord {
    pub(crate) rowid: u64,
    pub(crate) rows: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct FileIndexGeneration {
    pub(crate) level: u32,
    pub(crate) num_embeddings: usize,
    pub(crate) bucket_data_file: String,
    pub(crate) rowids_file: String,
}

pub(crate) struct FileBackedIndex {
    parent: PathBuf,
    prefix: String,
    manifest_path: PathBuf,
    rowid_buffer_path: PathBuf,
}

impl FileBackedIndex {
    pub(crate) fn new(base_path: PathBuf) -> Self {
        let parent = base_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let prefix = base_path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "witchcraft".to_string());
        Self {
            manifest_path: parent.join(format!("{prefix}.index")),
            rowid_buffer_path: parent.join(format!("{prefix}.rowids.buffer")),
            parent,
            prefix,
        }
    }

    pub(crate) fn path_for(&self, file_name: &str) -> PathBuf {
        let path = Path::new(file_name);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.parent.join(path)
        }
    }

    fn nonce() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    }

    pub(crate) fn bucket_data_file_name(&self, level: u32) -> String {
        format!("{}.buckets.{}.{}", self.prefix, level, Self::nonce())
    }

    pub(crate) fn rowids_file_name(&self, level: u32) -> String {
        format!("{}.rowids.{}.{}", self.prefix, level, Self::nonce())
    }

    pub(crate) fn rowid_buffer_path(&self) -> &PathBuf {
        &self.rowid_buffer_path
    }

    pub(crate) fn read_manifest(&self) -> Result<Vec<FileIndexGeneration>> {
        let text = match std::fs::read_to_string(&self.manifest_path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(err) => return Err(err.into()),
        };
        let mut lines = text.lines();
        anyhow::ensure!(
            lines.next() == Some("WITCHCRAFT_INDEX_V1"),
            "bad C index manifest header in {}",
            self.manifest_path.display()
        );
        let mut generations = vec![];
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split('\t');
            let level = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing level in C index manifest"))?
                .parse::<u32>()?;
            let num_embeddings = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing embedding count in C index manifest"))?
                .parse::<usize>()?;
            let bucket_data_file = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing bucket file in C index manifest"))?
                .to_string();
            let rowids_file = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing rowids file in C index manifest"))?
                .to_string();
            anyhow::ensure!(
                parts.next().is_none(),
                "extra fields in C index manifest"
            );
            generations.push(FileIndexGeneration {
                level,
                num_embeddings,
                bucket_data_file,
                rowids_file,
            });
        }
        generations.sort_by_key(|generation| generation.level);
        Ok(generations)
    }

    pub(crate) fn write_manifest(&self, generations: &[FileIndexGeneration]) -> Result<()> {
        std::fs::create_dir_all(&self.parent)?;
        let tmp = unique_tmp_path(&self.manifest_path, "manifest");
        let mut file = DirectFileWriter::create_append_new(&tmp)?;
        writeln!(file, "WITCHCRAFT_INDEX_V1")?;
        for generation in generations {
            writeln!(
                file,
                "{}\t{}\t{}\t{}",
                generation.level,
                generation.num_embeddings,
                generation.bucket_data_file,
                generation.rowids_file
            )?;
        }
        file.finish()?;
        std::fs::rename(&tmp, &self.manifest_path)?;
        sync_parent_dir(&self.manifest_path)?;
        Ok(())
    }

    pub(crate) fn append_rowid_record(&self, rowid: u64, rows: u32) -> Result<()> {
        std::fs::create_dir_all(&self.parent)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.rowid_buffer_path)?;
        file.write_all(&rowid.to_le_bytes())?;
        file.write_all(&rows.to_le_bytes())?;
        file.flush()?;
        Ok(())
    }

    pub(crate) fn generation_files(&self) -> Result<Vec<PathBuf>> {
        Ok(self
            .read_manifest()?
            .into_iter()
            .map(|generation| self.path_for(&generation.bucket_data_file))
            .collect())
    }

    fn current_rowid_records(&self) -> Result<Vec<RowidRecord>> {
        let mut inputs =
            vec![sort_dedup_rowid_records(read_rowid_records(&self.rowid_buffer_path)?)];
        for generation in self.read_manifest()? {
            inputs.push(read_rowid_records(&self.path_for(&generation.rowids_file))?);
        }
        Ok(nway_merge_rowid_records(&inputs))
    }

    pub(crate) fn active_rowids(&self) -> Result<HashMap<u64, bool>> {
        Ok(self
            .current_rowid_records()?
            .into_iter()
            .map(|record| (record.rowid, record.rows > 0))
            .collect())
    }

    pub(crate) fn remove_generation_files(&self, generations: &[FileIndexGeneration]) {
        for generation in generations {
            let _ = std::fs::remove_file(self.path_for(&generation.bucket_data_file));
            let _ = std::fs::remove_file(self.path_for(&generation.rowids_file));
        }
    }
}

pub(crate) fn read_rowid_records(path: &PathBuf) -> Result<Vec<RowidRecord>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(err) => return Err(err.into()),
    };
    let mut bytes = vec![];
    file.read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() % ROWID_RECORD_BYTES == 0,
        "rowid file {} length {} is not divisible by {ROWID_RECORD_BYTES}",
        path.display(),
        bytes.len()
    );
    let mut records = Vec::with_capacity(bytes.len() / ROWID_RECORD_BYTES);
    for chunk in bytes.chunks_exact(ROWID_RECORD_BYTES) {
        let rowid = u64::from_le_bytes(chunk[..8].try_into()?);
        let rows = u32::from_le_bytes(chunk[8..12].try_into()?);
        records.push(RowidRecord { rowid, rows });
    }
    Ok(records)
}

pub(crate) fn write_rowid_records(path: &PathBuf, records: &[RowidRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = unique_tmp_path(path, "rowids");
    let mut file = DirectFileWriter::create_append_new(&tmp)?;
    for record in records {
        file.write_all(&record.rowid.to_le_bytes())?;
        file.write_all(&record.rows.to_le_bytes())?;
    }
    file.finish()?;
    std::fs::rename(&tmp, path)?;
    sync_parent_dir(path)?;
    Ok(())
}

pub(crate) fn sort_dedup_rowid_records(records: Vec<RowidRecord>) -> Vec<RowidRecord> {
    let mut latest: HashMap<u64, (usize, u32)> = HashMap::new();
    for (idx, record) in records.into_iter().enumerate() {
        latest.insert(record.rowid, (idx, record.rows));
    }
    let mut records: Vec<RowidRecord> = latest
        .into_iter()
        .map(|(rowid, (_idx, rows))| RowidRecord { rowid, rows })
        .collect();
    records.sort_unstable_by_key(|record| record.rowid);
    records
}

pub(crate) fn nway_merge_rowid_records(inputs: &[Vec<RowidRecord>]) -> Vec<RowidRecord> {
    let mut heap = BinaryHeap::new();
    for (source, records) in inputs.iter().enumerate() {
        if let Some(record) = records.first() {
            heap.push(RowidHeapEntry {
                rowid: record.rowid,
                rows: record.rows,
                source,
                index: 0,
            });
        }
    }

    let mut merged = vec![];
    while let Some(first) = heap.pop() {
        let rowid = first.rowid;
        let mut candidates = vec![first];

        while matches!(heap.peek(), Some(next) if next.rowid == rowid) {
            candidates.push(heap.pop().unwrap());
        }

        for entry in &candidates {
            let next_index = entry.index + 1;
            if let Some(record) = inputs[entry.source].get(next_index) {
                heap.push(RowidHeapEntry {
                    rowid: record.rowid,
                    rows: record.rows,
                    source: entry.source,
                    index: next_index,
                });
            }
        }

        let winner = candidates
            .into_iter()
            .min_by_key(|entry| entry.source)
            .unwrap();
        merged.push(RowidRecord {
            rowid,
            rows: winner.rows,
        });
    }
    merged
}

pub(crate) fn rowid_records_embedding_count(records: &[RowidRecord]) -> usize {
    records.iter().map(|record| record.rows as usize).sum()
}

pub(crate) fn active_rowid_records(records: &[RowidRecord]) -> impl Iterator<Item = RowidRecord> + '_ {
    records.iter().copied().filter(|record| record.rows > 0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RowidHeapEntry {
    rowid: u64,
    rows: u32,
    source: usize,
    index: usize,
}

impl Ord for RowidHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rowid
            .cmp(&self.rowid)
            .then_with(|| other.source.cmp(&self.source))
    }
}

impl PartialOrd for RowidHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rowid_records_roundtrip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("level.rowids");
        let records = vec![
            RowidRecord { rowid: 2, rows: 3 },
            RowidRecord { rowid: 9, rows: 0 },
        ];

        write_rowid_records(&path, &records)?;

        assert_eq!(read_rowid_records(&path)?, records);
        Ok(())
    }

    #[test]
    fn sort_dedup_rowid_records_keeps_latest_update() {
        let records = vec![
            RowidRecord { rowid: 7, rows: 3 },
            RowidRecord { rowid: 2, rows: 4 },
            RowidRecord { rowid: 7, rows: 0 },
        ];

        assert_eq!(
            sort_dedup_rowid_records(records),
            vec![
                RowidRecord { rowid: 2, rows: 4 },
                RowidRecord { rowid: 7, rows: 0 },
            ]
        );
    }

    #[test]
    fn nway_merge_prefers_newer_inputs_for_duplicate_rowids() {
        let inputs = vec![
            vec![
                RowidRecord { rowid: 3, rows: 0 },
                RowidRecord { rowid: 9, rows: 2 },
            ],
            vec![
                RowidRecord { rowid: 1, rows: 5 },
                RowidRecord { rowid: 3, rows: 8 },
            ],
        ];

        assert_eq!(
            nway_merge_rowid_records(&inputs),
            vec![
                RowidRecord { rowid: 1, rows: 5 },
                RowidRecord { rowid: 3, rows: 0 },
                RowidRecord { rowid: 9, rows: 2 },
            ]
        );
    }

    #[test]
    fn manifest_roundtrip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let index = FileBackedIndex::new(dir.path().join("standalone"));
        let generations = vec![FileIndexGeneration {
            level: 2,
            num_embeddings: 17,
            bucket_data_file: "standalone.buckets.2".to_string(),
            rowids_file: "standalone.rowids.2".to_string(),
        }];

        index.write_manifest(&generations)?;

        let loaded = index.read_manifest()?;
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].level, generations[0].level);
        assert_eq!(loaded[0].num_embeddings, generations[0].num_embeddings);
        assert_eq!(loaded[0].bucket_data_file, generations[0].bucket_data_file);
        assert_eq!(loaded[0].rowids_file, generations[0].rowids_file);
        Ok(())
    }
}
