use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

type JobResult = Result<(), String>;

enum IndexJob {
    Add {
        uuid: uuid::Uuid,
        date: Option<iso8601_timestamp::Timestamp>,
        metadata: String,
        body: String,
        lengths: Option<Vec<usize>>,
    },
    Remove {
        uuid: uuid::Uuid,
    },
    Index {
        done: mpsc::Sender<JobResult>,
    },
    Clear {
        done: mpsc::Sender<JobResult>,
    },
    Shutdown,
}

struct Reader {
    db: crate::DB,
    embedder: Arc<Mutex<crate::Embedder>>,
    cache: crate::EmbeddingsCache,
}

impl Reader {
    fn new(db: crate::DB, embedder: Arc<Mutex<crate::Embedder>>) -> Self {
        Self {
            db,
            embedder,
            cache: crate::EmbeddingsCache::new(16),
        }
    }

    fn search(
        &mut self,
        q: &str,
        threshold: f32,
        top_k: usize,
    ) -> Result<Vec<(f32, String, Vec<String>, u32, String)>, String> {
        let embedder = self
            .embedder
            .lock()
            .map_err(|_| "embedder lock poisoned".to_string())?;
        crate::search(
            &self.db,
            &embedder,
            &mut self.cache,
            q,
            threshold,
            top_k,
            true,
            None,
        )
        .map_err(|e| e.to_string())
    }

    fn score(&mut self, q: &str, sentences: &[String]) -> Result<Vec<f32>, String> {
        let embedder = self
            .embedder
            .lock()
            .map_err(|_| "embedder lock poisoned".to_string())?;
        crate::score_query_sentences(&embedder, &mut self.cache, &q.to_string(), sentences)
            .map_err(|e| e.to_string())
    }
}

fn runtime_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(message.into())
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn run_job_safely<F>(job: F) -> JobResult
where
    F: FnOnce() -> JobResult,
{
    match catch_unwind(AssertUnwindSafe(job)) {
        Ok(result) => result,
        Err(payload) => Err(format!("indexer panicked: {}", panic_message(payload))),
    }
}

fn run_index_job(
    write_db: &mut crate::DB,
    embedder: &Arc<Mutex<crate::Embedder>>,
    device: &candle_core::Device,
) -> JobResult {
    {
        let embedder = embedder
            .lock()
            .map_err(|_| "embedder lock poisoned".to_string())?;
        loop {
            match crate::embed_chunks(write_db, &embedder, Some(10)) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => return Err(format!("embed_chunks failed: {e}")),
            }
        }
    }

    crate::index_chunks(write_db, device).map_err(|e| format!("index_chunks failed: {e}"))
}

fn wait_for_job(done: mpsc::Receiver<JobResult>) -> PyResult<()> {
    match done.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(runtime_error(e)),
        Err(_) => Err(runtime_error("indexer thread exited before completing job")),
    }
}

/// Semantic search index with hybrid dense/sparse retrieval.
///
/// Args:
///     db_name: Path to the SQLite database file (created if absent).
///     assets: Directory containing model weights and tokenizer files.
#[pyclass]
pub struct Witchcraft {
    reader: Reader,
    tx: mpsc::Sender<IndexJob>,
    handle: Option<thread::JoinHandle<()>>,
}

// Safe: Reader (which owns a rusqlite::Connection, a !Send/!Sync type) is only
// ever accessed from the Python thread while the GIL is held, ensuring
// exclusive single-threaded access.  The indexer thread owns a separate
// write connection to the same file and never shares state with Reader.
unsafe impl Send for Witchcraft {}
unsafe impl Sync for Witchcraft {}

impl Drop for Witchcraft {
    fn drop(&mut self) {
        let _ = self.tx.send(IndexJob::Shutdown);
        if let Some(h) = self.handle.take() {
            if let Err(payload) = h.join() {
                tracing::warn!(
                    "witchcraft: indexer thread panicked: {}",
                    panic_message(payload)
                );
            }
        }
    }
}

#[pymethods]
impl Witchcraft {
    #[new]
    fn new(db_name: String, assets: String) -> PyResult<Self> {
        let db_path = PathBuf::from(&db_name);
        let assets_path = PathBuf::from(&assets);

        // Create the write DB first so the file exists before DB::new_reader opens it.
        let mut write_db = crate::DB::new(db_path.clone())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Load the embedder once and share it between the reader and indexer thread.
        let device = crate::make_device();
        let embedder = Arc::new(Mutex::new(
            crate::Embedder::new(&device, &assets_path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        ));

        let reader_db = crate::DB::new_reader(db_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let reader = Reader::new(reader_db, Arc::clone(&embedder));

        let thread_embedder = Arc::clone(&embedder);
        let (tx, rx) = mpsc::channel::<IndexJob>();
        let handle = thread::spawn(move || {
            while let Ok(job) = rx.recv() {
                match job {
                    IndexJob::Shutdown => break,
                    IndexJob::Clear { done } => {
                        write_db.clear();
                        let _ = done.send(Ok(()));
                    }
                    IndexJob::Add {
                        uuid,
                        date,
                        metadata,
                        body,
                        lengths,
                    } => {
                        if let Err(e) = write_db.add_doc(&uuid, date, &metadata, &body, lengths) {
                            tracing::warn!("witchcraft: add_doc failed: {e}");
                        }
                    }
                    IndexJob::Remove { uuid } => {
                        if let Err(e) = write_db.remove_doc(&uuid) {
                            tracing::warn!("witchcraft: remove_doc failed: {e}");
                        }
                    }
                    IndexJob::Index { done } => {
                        let result = run_job_safely(|| {
                            run_index_job(&mut write_db, &thread_embedder, &device)
                        });
                        let _ = done.send(result);
                    }
                }
            }
        });

        Ok(Self {
            reader,
            tx,
            handle: Some(handle),
        })
    }

    /// Search the index.
    ///
    /// Args:
    ///     q: Query string.
    ///     threshold: Minimum similarity score (0–1). Defaults to 0.3.
    ///     top_k: Maximum results to return. Defaults to 10.
    ///
    /// Returns:
    ///     List of dicts with keys: score, metadata, body, idx, date.
    #[pyo3(signature = (q, threshold=0.3, top_k=10))]
    fn search(
        &mut self,
        py: Python<'_>,
        q: String,
        threshold: f64,
        top_k: usize,
    ) -> PyResult<Vec<PyObject>> {
        let results = self
            .reader
            .search(&q, threshold as f32, top_k)
            .map_err(runtime_error)?;
        results
            .into_iter()
            .map(|(score, metadata, bodies, idx, date)| {
                let sub = (idx as usize).min(bodies.len().saturating_sub(1));
                let body = bodies.get(sub).cloned().unwrap_or_default();
                let dict = PyDict::new(py);
                dict.set_item("score", score as f64)?;
                dict.set_item("metadata", &metadata)?;
                dict.set_item("body", &body)?;
                dict.set_item("idx", idx)?;
                dict.set_item("date", &date)?;
                Ok(dict.into_any().unbind())
            })
            .collect()
    }

    /// Score how well each sentence matches the query.
    ///
    /// Args:
    ///     q: Query string.
    ///     sentences: Candidate sentences to score.
    ///
    /// Returns:
    ///     List of similarity scores (one per sentence, 0–1).
    fn score(&mut self, q: String, sentences: Vec<String>) -> PyResult<Vec<f32>> {
        self.reader.score(&q, &sentences).map_err(runtime_error)
    }

    /// Add or update a document in the index.
    ///
    /// Args:
    ///     uuid: Stable identifier for the document (UUID string).
    ///     date: ISO 8601 timestamp (e.g. "2024-01-15T10:00:00Z").
    ///     metadata: Arbitrary JSON metadata string.
    ///     body: Document text.
    ///     lengths: Optional list of codepoint lengths for pre-split chunks.
    #[pyo3(signature = (uuid, date, metadata, body, lengths=None))]
    fn add(
        &self,
        uuid: String,
        date: String,
        metadata: String,
        body: String,
        lengths: Option<Vec<u32>>,
    ) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(&uuid)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let date = iso8601_timestamp::Timestamp::parse(&date).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid ISO 8601 date: {date}"))
        })?;
        let lengths = lengths.map(|v| v.into_iter().map(|l| l as usize).collect());
        if self
            .tx
            .send(IndexJob::Add {
                uuid,
                date: Some(date),
                metadata,
                body,
                lengths,
            })
            .is_err()
        {
            return Err(runtime_error("indexer thread has exited"));
        }
        Ok(())
    }

    /// Remove a document by UUID.
    fn remove(&self, uuid: String) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(&uuid)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if self.tx.send(IndexJob::Remove { uuid }).is_err() {
            return Err(runtime_error("indexer thread has exited"));
        }
        Ok(())
    }

    /// Embed and index all pending documents, blocking until complete.
    fn index(&self) -> PyResult<()> {
        let (done_tx, done_rx) = mpsc::channel();
        self.tx
            .send(IndexJob::Index { done: done_tx })
            .map_err(|_| runtime_error("indexer thread has exited"))?;
        wait_for_job(done_rx)
    }

    /// Clear all documents from the index.
    fn clear(&self) -> PyResult<()> {
        let (done_tx, done_rx) = mpsc::channel();
        self.tx
            .send(IndexJob::Clear { done: done_tx })
            .map_err(|_| runtime_error("indexer thread has exited"))?;
        wait_for_job(done_rx)
    }

    /// Shut down the background indexer and wait for it to finish.
    fn shutdown(&mut self) -> PyResult<()> {
        if self.handle.is_none() {
            return Ok(());
        }

        let send_result = self.tx.send(IndexJob::Shutdown);
        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(()) => {
                    send_result.map_err(|_| runtime_error("indexer thread has exited"))?;
                }
                Err(payload) => {
                    return Err(runtime_error(format!(
                        "indexer thread panicked: {}",
                        panic_message(payload)
                    )));
                }
            }
        }
        Ok(())
    }
}

#[pymodule]
fn witchcraft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Witchcraft>()?;
    Ok(())
}
