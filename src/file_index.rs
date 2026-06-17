use anyhow::Result;
use crate::app_id::APP_ID_U32;
use crate::file_writer::NewFileWriter;
use log::warn;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub(crate) const GENERATION_DATA_APP_ID: u32 = APP_ID_U32;
pub(crate) const GENERATION_DATA_VERSION: u32 = 3;
pub(crate) const GENERATION_DATA_HEADER_BYTES: usize =
    8 * std::mem::size_of::<u32>() + std::mem::size_of::<u64>();

const ROWID_RECORD_BYTES: usize = std::mem::size_of::<u64>() + std::mem::size_of::<u32>();
const ROWIDS_OFFSET_FIELD: usize = 8 * std::mem::size_of::<u32>();

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
    pub(crate) data_file: String,
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

    pub(crate) fn generation_file_name(&self, level: u32) -> String {
        format!("{}.buckets.{}.{}", self.prefix, level, Self::nonce())
    }

    pub(crate) fn rowid_buffer_path(&self) -> &PathBuf {
        &self.rowid_buffer_path
    }

    #[cfg(feature = "sqlite")]
    pub(crate) fn clear(&self) -> Result<()> {
        let generations = self.read_manifest()?;
        self.remove_generation_files(&generations);
        let _ = std::fs::remove_file(&self.manifest_path);
        let _ = std::fs::remove_file(&self.rowid_buffer_path);
        Ok(())
    }

    pub(crate) fn read_manifest(&self) -> Result<Vec<FileIndexGeneration>> {
        let text = match std::fs::read_to_string(&self.manifest_path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(err) => return Err(err.into()),
        };
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty file index manifest {}", self.manifest_path.display()))?;
        if let Some(reason) = file_index_manifest_stale_reason(header) {
            warn!(
                "file index manifest {} is stale ({reason}), resetting file-backed index",
                self.manifest_path.display()
            );
            self.remove_index_sidecars();
            return Ok(vec![]);
        }
        let mut generations = vec![];
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split('\t');
            let level = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing level in file index manifest"))?
                .parse::<u32>()?;
            let num_embeddings = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing embedding count in file index manifest"))?
                .parse::<usize>()?;
            let data_file = parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing generation file in file index manifest"))?
                .to_string();
            anyhow::ensure!(
                parts.next().is_none(),
                "extra fields in file index manifest"
            );
            generations.push(FileIndexGeneration {
                level,
                num_embeddings,
                data_file,
            });
        }
        generations.sort_by_key(|generation| generation.level);
        if let Some(reason) = self.index_sidecars_stale_reason(&generations)? {
            warn!("file-backed index sidecars are stale ({reason}), resetting file-backed index");
            self.remove_index_sidecars();
            return Ok(vec![]);
        }
        Ok(generations)
    }

    pub(crate) fn write_manifest(&self, generations: &[FileIndexGeneration]) -> Result<()> {
        std::fs::create_dir_all(&self.parent)?;
        let tmp = unique_tmp_path(&self.manifest_path, "manifest");
        let mut file = NewFileWriter::create_new(&tmp)?;
        writeln!(file, "{}\t{}", GENERATION_DATA_APP_ID, GENERATION_DATA_VERSION)?;
        for generation in generations {
            writeln!(
                file,
                "{}\t{}\t{}",
                generation.level,
                generation.num_embeddings,
                generation.data_file
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
            .map(|generation| self.path_for(&generation.data_file))
            .collect())
    }

    pub(crate) fn buffered_rowid_records(&self) -> Result<Vec<RowidRecord>> {
        Ok(sort_dedup_rowid_records(read_rowid_records(&self.rowid_buffer_path)?))
    }

    fn current_rowid_records(&self) -> Result<Vec<RowidRecord>> {
        let mut inputs = vec![self.buffered_rowid_records()?];
        inputs.push(self.indexed_rowid_records()?);
        Ok(nway_merge_rowid_records(&inputs))
    }

    pub(crate) fn indexed_rowid_records(&self) -> Result<Vec<RowidRecord>> {
        let mut inputs = vec![];
        for generation in self.read_manifest()? {
            inputs.push(self.generation_rowid_records(&generation)?);
        }
        Ok(nway_merge_rowid_records(&inputs))
    }

    pub(crate) fn generation_rowid_records(
        &self,
        generation: &FileIndexGeneration,
    ) -> Result<Vec<RowidRecord>> {
        read_generation_rowid_records(&self.path_for(&generation.data_file))
    }

    #[cfg(feature = "sqlite")]
    pub(crate) fn all_rowid_records(&self) -> Result<Vec<RowidRecord>> {
        self.current_rowid_records()
    }

    #[cfg(feature = "sqlite")]
    pub(crate) fn indexed_embedding_count(&self) -> Result<usize> {
        Ok(self
            .read_manifest()?
            .into_iter()
            .map(|generation| generation.num_embeddings)
            .sum())
    }

    #[cfg(feature = "sqlite")]
    pub(crate) fn indexed_rowid_map(&self) -> Result<HashMap<u64, u32>> {
        Ok(self
            .indexed_rowid_records()?
            .into_iter()
            .map(|record| (record.rowid, record.rows))
            .collect())
    }

    #[cfg(feature = "sqlite")]
    pub(crate) fn all_rowid_map(&self) -> Result<HashMap<u64, u32>> {
        Ok(self
            .all_rowid_records()?
            .into_iter()
            .map(|record| (record.rowid, record.rows))
            .collect())
    }

    #[cfg(all(test, feature = "sqlite"))]
    pub(crate) fn level_embedding_counts(&self) -> Result<Vec<(u32, usize)>> {
        Ok(self
            .read_manifest()?
            .into_iter()
            .map(|generation| (generation.level, generation.num_embeddings))
            .collect())
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
            let _ = std::fs::remove_file(self.path_for(&generation.data_file));
        }
    }

    fn remove_index_sidecars(&self) {
        let bucket_prefix = format!("{}.buckets.", self.prefix);
        let rowids_prefix = format!("{}.rowids.", self.prefix);
        let residuals_prefix = format!("{}.residuals.", self.prefix);
        if let Ok(entries) = std::fs::read_dir(&self.parent) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(&bucket_prefix)
                    || name.starts_with(&rowids_prefix)
                    || name.starts_with(&residuals_prefix)
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        let _ = std::fs::remove_file(&self.manifest_path);
    }

    fn index_sidecars_stale_reason(
        &self,
        generations: &[FileIndexGeneration],
    ) -> Result<Option<String>> {
        for generation in generations {
            match generation_sidecar_stale_reason(&self.path_for(&generation.data_file))? {
                Some(reason) => return Ok(Some(format!("{}: {reason}", generation.data_file))),
                None => {}
            }
        }
        Ok(None)
    }
}

fn file_index_manifest_stale_reason(header: &str) -> Option<String> {
    let mut parts = header.split('\t');
    let Some(app_id) = parts.next().and_then(|part| part.parse::<u32>().ok()) else {
        return Some("missing or invalid app id".to_string());
    };
    let Some(version) = parts.next().and_then(|part| part.parse::<u32>().ok()) else {
        return Some("missing or invalid version".to_string());
    };
    if parts.next().is_some() {
        return Some("extra manifest header fields".to_string());
    }
    if app_id != GENERATION_DATA_APP_ID {
        return Some(format!("app id {app_id:#x} is not supported"));
    }
    if version != GENERATION_DATA_VERSION {
        return Some(format!("version {version} is not supported"));
    }
    None
}

pub(crate) fn read_rowid_records(path: &PathBuf) -> Result<Vec<RowidRecord>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(err) => return Err(err.into()),
    };
    let mut bytes = vec![];
    file.read_to_end(&mut bytes)?;
    parse_rowid_records(&bytes, &path.display().to_string())
}

fn generation_rowids_offset_from_header(
    header: &[u8; GENERATION_DATA_HEADER_BYTES],
    file_len: u64,
) -> Result<u64> {
    let app_id = u32::from_le_bytes(header[..4].try_into()?);
    anyhow::ensure!(
        app_id == GENERATION_DATA_APP_ID,
        "generation sidecar app id {app_id:#x} is not supported"
    );
    let version = u32::from_le_bytes(header[4..8].try_into()?);
    anyhow::ensure!(
        version == GENERATION_DATA_VERSION,
        "generation sidecar version {version} is not supported"
    );
    let rowids_offset = u64::from_le_bytes(
        header[ROWIDS_OFFSET_FIELD..ROWIDS_OFFSET_FIELD + 8].try_into()?,
    );
    anyhow::ensure!(
        rowids_offset <= file_len,
        "generation sidecar rowid offset {} exceeds file length {}",
        rowids_offset,
        file_len
    );
    Ok(rowids_offset)
}

fn generation_sidecar_header(
    path: &PathBuf,
) -> Result<Option<([u8; GENERATION_DATA_HEADER_BYTES], u64)>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let file_len = file.metadata()?.len();
    let header_bytes = u64::try_from(GENERATION_DATA_HEADER_BYTES)?;
    if file_len < header_bytes {
        return Ok(None);
    }
    let mut header = [0u8; GENERATION_DATA_HEADER_BYTES];
    match file.read_exact(&mut header) {
        Ok(()) => Ok(Some((header, file_len))),
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(err) => Err(err.into()),
    }
}

fn generation_sidecar_stale_reason(path: &PathBuf) -> Result<Option<String>> {
    let Some((header, file_len)) = generation_sidecar_header(path)? else {
        return Ok(Some("missing or truncated sidecar".to_string()));
    };
    match generation_rowids_offset_from_header(&header, file_len) {
        Ok(_) => Ok(None),
        Err(err) => Ok(Some(err.to_string())),
    }
}

pub(crate) fn read_generation_rowid_records(path: &PathBuf) -> Result<Vec<RowidRecord>> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let header_bytes = u64::try_from(GENERATION_DATA_HEADER_BYTES)?;
    anyhow::ensure!(
        file_len >= header_bytes,
        "generation sidecar {} is too small",
        path.display()
    );
    let mut header = [0u8; GENERATION_DATA_HEADER_BYTES];
    file.read_exact(&mut header)?;
    let rowids_offset = generation_rowids_offset_from_header(&header, file_len)?;
    file.seek(SeekFrom::Start(rowids_offset))?;
    let mut bytes = vec![];
    file.read_to_end(&mut bytes)?;
    parse_rowid_records(&bytes, &path.display().to_string())
}

fn parse_rowid_records(bytes: &[u8], source: &str) -> Result<Vec<RowidRecord>> {
    anyhow::ensure!(
        bytes.len() % ROWID_RECORD_BYTES == 0,
        "rowid data in {} length {} is not divisible by {ROWID_RECORD_BYTES}",
        source,
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

pub(crate) fn write_rowid_records_to_writer(
    writer: &mut impl Write,
    records: &[RowidRecord],
) -> Result<()> {
    for record in records {
        writer.write_all(&record.rowid.to_le_bytes())?;
        writer.write_all(&record.rows.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_rowid_records(path: &PathBuf, records: &[RowidRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = unique_tmp_path(path, "rowids");
    let mut file = NewFileWriter::create_new(&tmp)?;
    write_rowid_records_to_writer(&mut file, records)?;
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

    fn write_generation_test_header(
        file: &mut impl Write,
        version: u32,
    ) -> Result<()> {
        let header_bytes = u32::try_from(GENERATION_DATA_HEADER_BYTES)?;
        file.write_all(&GENERATION_DATA_APP_ID.to_le_bytes())?;
        file.write_all(&version.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?;
        file.write_all(&128u32.to_le_bytes())?;
        file.write_all(&crate::BUCKET_CENTER_FORMAT.to_le_bytes())?;
        file.write_all(&header_bytes.to_le_bytes())?;
        file.write_all(&header_bytes.to_le_bytes())?;
        file.write_all(&u64::from(header_bytes).to_le_bytes())?;
        Ok(())
    }

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
        let data_file = "standalone.buckets.2".to_string();
        let data_path = index.path_for(&data_file);
        let mut file = NewFileWriter::create_new(&data_path)?;
        write_generation_test_header(&mut file, GENERATION_DATA_VERSION)?;
        file.finish()?;

        let generations = vec![FileIndexGeneration {
            level: 2,
            num_embeddings: 17,
            data_file,
        }];

        index.write_manifest(&generations)?;

        let loaded = index.read_manifest()?;
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].level, generations[0].level);
        assert_eq!(loaded[0].num_embeddings, generations[0].num_embeddings);
        assert_eq!(loaded[0].data_file, generations[0].data_file);
        Ok(())
    }

    #[test]
    fn generation_sidecar_rowids_roundtrip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("level.buckets");
        let records = vec![
            RowidRecord { rowid: 11, rows: 4 },
            RowidRecord { rowid: 19, rows: 0 },
        ];
        let mut file = NewFileWriter::create_new(&path)?;
        write_generation_test_header(&mut file, GENERATION_DATA_VERSION)?;
        write_rowid_records_to_writer(&mut file, &records)?;
        file.finish()?;

        assert_eq!(read_generation_rowid_records(&path)?, records);
        Ok(())
    }

    #[test]
    fn read_manifest_resets_stale_generation_sidecars() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let index = FileBackedIndex::new(dir.path().join("standalone"));
        let data_file = "standalone.buckets.0.stale".to_string();
        let data_path = index.path_for(&data_file);
        let mut file = NewFileWriter::create_new(&data_path)?;
        write_generation_test_header(&mut file, GENERATION_DATA_VERSION - 1)?;
        file.finish()?;

        index.write_manifest(&[FileIndexGeneration {
            level: 0,
            num_embeddings: 11,
            data_file,
        }])?;

        assert!(index.read_manifest()?.is_empty());
        assert!(!data_path.exists());
        assert!(index.read_manifest()?.is_empty());
        Ok(())
    }
}
