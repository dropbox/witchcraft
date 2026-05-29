use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAGIC: [u8; 8] = *b"WEMB0001";
const HASH_CHARS: usize = 32;

#[derive(Clone, Debug)]
pub struct CachedEmbeddings {
    pub model: String,
    pub counts: String,
    pub embedding_count: usize,
    pub embeddings: Vec<u8>,
}

pub trait EmbeddingCache {
    fn get(&self, hash: &str) -> Result<Option<CachedEmbeddings>>;
    fn put(&self, hash: &str, embeddings: &CachedEmbeddings) -> Result<()>;
}

#[derive(Clone, Debug)]
pub struct FileEmbeddingCache {
    root: PathBuf,
}

impl FileEmbeddingCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for_hash(&self, hash: &str) -> Result<PathBuf> {
        anyhow::ensure!(
            hash.len() == HASH_CHARS && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "embedding cache key must be a {HASH_CHARS}-character hex chunk hash: {hash}"
        );
        Ok(self.root.join(hash))
    }

    fn temp_path_for_hash(&self, hash: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        self.root
            .join(format!(".{hash}.{}.{}.tmp", std::process::id(), nonce))
    }
}

impl EmbeddingCache for FileEmbeddingCache {
    fn get(&self, hash: &str) -> Result<Option<CachedEmbeddings>> {
        let path = self.path_for_hash(hash)?;
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err).with_context(|| {
                format!("failed to open embedding cache entry {}", path.display())
            }),
        };

        let mut magic = [0u8; MAGIC.len()];
        file.read_exact(&mut magic)
            .with_context(|| format!("failed to read embedding cache header {}", path.display()))?;
        anyhow::ensure!(
            magic == MAGIC,
            "bad embedding cache magic in {}",
            path.display()
        );

        let model_len = read_u32(&mut file)? as usize;
        let counts_len = read_u32(&mut file)? as usize;
        let embedding_count = read_u64(&mut file)? as usize;
        let embeddings_len = read_u64(&mut file)? as usize;

        let mut model = vec![0u8; model_len];
        file.read_exact(&mut model)
            .with_context(|| format!("failed to read model id from {}", path.display()))?;
        let model = String::from_utf8(model)
            .with_context(|| format!("model id is not UTF-8 in {}", path.display()))?;

        let mut counts = vec![0u8; counts_len];
        file.read_exact(&mut counts)
            .with_context(|| format!("failed to read counts from {}", path.display()))?;
        let counts = String::from_utf8(counts)
            .with_context(|| format!("counts are not UTF-8 in {}", path.display()))?;

        let mut embeddings = vec![0u8; embeddings_len];
        file.read_exact(&mut embeddings)
            .with_context(|| format!("failed to read embeddings from {}", path.display()))?;

        Ok(Some(CachedEmbeddings {
            model,
            counts,
            embedding_count,
            embeddings,
        }))
    }

    fn put(&self, hash: &str, embeddings: &CachedEmbeddings) -> Result<()> {
        let path = self.path_for_hash(hash)?;
        fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create {}", self.root.display()))?;

        let tmp_path = self.temp_path_for_hash(hash);
        let result = (|| -> Result<()> {
            let mut file = File::create(&tmp_path).with_context(|| {
                format!("failed to create temporary cache entry {}", tmp_path.display())
            })?;

            file.write_all(&MAGIC)?;
            write_u32(&mut file, embeddings.model.len())?;
            write_u32(&mut file, embeddings.counts.len())?;
            write_u64(&mut file, embeddings.embedding_count)?;
            write_u64(&mut file, embeddings.embeddings.len())?;
            file.write_all(embeddings.model.as_bytes())?;
            file.write_all(embeddings.counts.as_bytes())?;
            file.write_all(&embeddings.embeddings)?;
            file.flush()?;
            fs::rename(&tmp_path, &path).with_context(|| {
                format!(
                    "failed to move temporary cache entry {} to {}",
                    tmp_path.display(),
                    path.display()
                )
            })?;
            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&tmp_path);
        }
        result
    }
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0u8; std::mem::size_of::<u32>()];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; std::mem::size_of::<u64>()];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_u32(writer: &mut impl Write, value: usize) -> Result<()> {
    writer.write_all(&u32::try_from(value)?.to_le_bytes())?;
    Ok(())
}

fn write_u64(writer: &mut impl Write, value: usize) -> Result<()> {
    writer.write_all(&u64::try_from(value)?.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CachedEmbeddings, EmbeddingCache, FileEmbeddingCache};

    #[test]
    fn file_cache_round_trips_cached_embeddings() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FileEmbeddingCache::new(dir.path());
        let hash = "0123456789abcdef0123456789abcdef";
        let value = CachedEmbeddings {
            model: "xtr-base-en".to_string(),
            counts: "1,2,3".to_string(),
            embedding_count: 6,
            embeddings: vec![1, 2, 3, 4],
        };

        cache.put(hash, &value).unwrap();
        let loaded = cache.get(hash).unwrap().unwrap();

        assert_eq!(loaded.model, value.model);
        assert_eq!(loaded.counts, value.counts);
        assert_eq!(loaded.embedding_count, value.embedding_count);
        assert_eq!(loaded.embeddings, value.embeddings);
    }

    #[test]
    fn file_cache_missing_entry_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FileEmbeddingCache::new(dir.path());
        assert!(cache
            .get("0123456789abcdef0123456789abcdef")
            .unwrap()
            .is_none());
    }
}
