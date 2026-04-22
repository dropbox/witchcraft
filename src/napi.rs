/* Node module API for Witchcraft */

use crate::types::SqlStatement;
use iso8601_timestamp::Timestamp;
use napi::{Env, ScopedTask};
use napi_derive::napi;

use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    sync::{mpsc, Arc, Mutex, OnceLock},
    thread::{self, JoinHandle},
};

use uuid::Uuid;

use log::{info, warn, LevelFilter, Log, Metadata, Record};
use napi::bindgen_prelude::*; // Env, Function, Result, etc.
use napi::threadsafe_function::ThreadsafeCallContext;
use once_cell::sync::OnceCell;

use std::alloc::{GlobalAlloc, Layout, System};

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

struct StatsAllocator;

unsafe impl GlobalAlloc for StatsAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static STATSALLOCATOR: StatsAllocator = StatsAllocator;

fn memory_stats() {
    let allocated = ALLOCATED.load(Ordering::Relaxed) as f64;
    let mb = (1 << 20) as f64;
    stats("allocated-megabytes", (allocated / mb).round());
}

#[napi(object)]
pub struct LogEvent {
    pub level: String,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
}

// Store the threadsafe function using a boxed type-erased version
static LOGFN: OnceCell<Box<dyn Fn(LogEvent) + Send + Sync>> = OnceCell::new();

struct JsLogger;

impl Log for JsLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // SECURITY: Filter out openvino-finder logs below ERROR level to prevent PII leakage.
        // openvino-finder (dependency of openvino crate) logs file paths at INFO level like:
        // "Attempting to find library: libopenvino_c.so"
        // "Found library at path: /Users/username/Library/..."
        // These paths may contain usernames or other PII that must not be exposed via the
        // logging callback. Only ERROR-level failures are allowed through.
        if metadata.target().starts_with("openvino_finder") {
            metadata.level() <= log::Level::Error
        } else {
            true
        }
    }

    fn log(&self, record: &Record) {
        if let Some(callback) = LOGFN.get() {
            let evt = LogEvent {
                level: record.level().to_string().to_lowercase(),
                message: record.args().to_string(),
                file: record.file().map(|s| s.to_string()),
                line: record.line(),
            };
            callback(evt);
        } else {
            println!(
                "[{}] {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

static LOGGER: JsLogger = JsLogger;

#[napi]
pub fn set_log_callback(
    callback: Function<(LogEvent,), Unknown>,
    max_level: Option<String>,
) -> Result<()> {
    use napi::threadsafe_function::ThreadsafeFunctionCallMode;

    // SAFETY: This lifetime extension is safe because ThreadsafeFunction's build process
    // creates a proper Node.js reference (via napi_create_threadsafe_function) that
    // increments the JavaScript function's reference count and keeps it alive for the
    // lifetime of the ThreadsafeFunction. The Node.js runtime guarantees the callback
    // remains valid until the TSFN is explicitly released or the environment shuts down.
    //
    // The transmute is necessary because Rust's borrow checker cannot understand the
    // FFI boundary where Node-API's reference counting takes over lifetime management.
    // This pattern is documented in napi-rs for long-lived threadsafe functions.
    let callback_static: Function<'static, (LogEvent,), Unknown> =
        unsafe { std::mem::transmute(callback) };

    let tsfn = callback_static.build_threadsafe_function().build_callback(
        |ctx: ThreadsafeCallContext<(LogEvent,)>| {
            // Extract the first element from the tuple (LogEvent,) to pass just the LogEvent
            Ok(ctx.value.0)
        },
    )?;

    // Wrap the threadsafe function in a closure that we can store
    let _ = LOGFN.set(Box::new(move |evt: LogEvent| {
        let _ = tsfn.call((evt,), ThreadsafeFunctionCallMode::NonBlocking);
    }));

    let lvl = match max_level.as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("error") => LevelFilter::Error,
        Some("warn") => LevelFilter::Warn,
        Some("info") => LevelFilter::Info,
        Some("debug") => LevelFilter::Debug,
        Some("trace") => LevelFilter::Trace,
        Some("off") => LevelFilter::Off,
        _ => LevelFilter::Trace,
    };
    let _ = log::set_logger(&LOGGER).map(|_| log::set_max_level(lvl));
    Ok(())
}

#[napi(object)]
pub struct StatsEvent {
    pub name: String,
    pub number: f64,
}

// Store the threadsafe function using a boxed type-erased version
static STATSFN: OnceCell<Box<dyn Fn(StatsEvent) + Send + Sync>> = OnceCell::new();

#[napi]
pub fn set_stats_callback(callback: Function<(StatsEvent,), Unknown>) -> Result<()> {
    use napi::threadsafe_function::ThreadsafeFunctionCallMode;

    // SAFETY: This lifetime extension is safe because ThreadsafeFunction's build process
    // creates a proper Node.js reference (via napi_create_threadsafe_function) that
    // increments the JavaScript function's reference count and keeps it alive for the
    // lifetime of the ThreadsafeFunction. The Node.js runtime guarantees the callback
    // remains valid until the TSFN is explicitly released or the environment shuts down.
    //
    // The transmute is necessary because Rust's borrow checker cannot understand the
    // FFI boundary where Node-API's reference counting takes over lifetime management.
    // This pattern is documented in napi-rs for long-lived threadsafe functions.
    let callback_static: Function<'static, (StatsEvent,), Unknown> =
        unsafe { std::mem::transmute(callback) };

    let tsfn = callback_static.build_threadsafe_function().build_callback(
        |ctx: ThreadsafeCallContext<(StatsEvent,)>| {
            // Extract the first element from the tuple (StatsEvent,) to pass just the StatsEvent
            Ok(ctx.value.0)
        },
    )?;

    // Wrap the threadsafe function in a closure that we can store
    let _ = STATSFN.set(Box::new(move |evt: StatsEvent| {
        let _ = tsfn.call((evt,), ThreadsafeFunctionCallMode::NonBlocking);
    }));

    Ok(())
}

pub fn stats(name: &str, number: f64) {
    if let Some(callback) = STATSFN.get() {
        let evt = StatsEvent {
            name: name.to_string(),
            number: number,
        };
        callback(evt);
    }
}

#[napi(object)]
pub struct ProgressEvent {
    pub progress: f64,
    pub phase: String,
}

// Store the threadsafe function for progress updates
static PROGRESSFN: OnceCell<Box<dyn Fn(ProgressEvent) + Send + Sync>> = OnceCell::new();

#[napi]
pub fn set_progress_callback(callback: Function<(ProgressEvent,), Unknown>) -> Result<()> {
    use napi::threadsafe_function::ThreadsafeFunctionCallMode;

    // SAFETY: This lifetime extension is safe because ThreadsafeFunction's build process
    // creates a proper Node.js reference (via napi_create_threadsafe_function) that
    // increments the JavaScript function's reference count and keeps it alive for the
    // lifetime of the ThreadsafeFunction. The Node.js runtime guarantees the callback
    // remains valid until the TSFN is explicitly released or the environment shuts down.
    //
    // The transmute is necessary because Rust's borrow checker cannot understand the
    // FFI boundary where Node-API's reference counting takes over lifetime management.
    // This pattern is documented in napi-rs for long-lived threadsafe functions.
    let callback_static: Function<'static, (ProgressEvent,), Unknown> =
        unsafe { std::mem::transmute(callback) };

    let tsfn = callback_static.build_threadsafe_function().build_callback(
        |ctx: ThreadsafeCallContext<(ProgressEvent,)>| {
            // Pass the progress event with value and phase
            Ok(ctx.value.0)
        },
    )?;

    // Wrap the threadsafe function in a closure that we can store
    let _ = PROGRESSFN.set(Box::new(move |evt: ProgressEvent| {
        let _ = tsfn.call((evt,), ThreadsafeFunctionCallMode::NonBlocking);
    }));

    Ok(())
}

pub fn progress_update(progress: f64, phase: &str) {
    if let Some(callback) = PROGRESSFN.get() {
        let evt = ProgressEvent {
            progress,
            phase: phase.to_string(),
        };
        callback(evt);
    }
}

enum Job {
    Add {
        uuid: Uuid,
        date: Option<Timestamp>,
        metadata: String,
        body: String,
        lengths: Option<Vec<usize>>,
    },
    Remove {
        uuid: Uuid,
    },
    Index,
    Clear,
    Delete {
        sql_filter: crate::types::SqlStatementInternal,
    },
    Shutdown,
}

pub struct Indexer {
    tx: mpsc::Sender<Job>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

static INDEXER: OnceLock<Indexer> = OnceLock::new();
static CLEAR: AtomicBool = AtomicBool::new(false);
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

fn accepting_commands() -> bool {
    SHUTDOWN.load(Ordering::Relaxed) == false
}

fn drain_commands() -> bool {
    CLEAR.load(Ordering::Relaxed) || SHUTDOWN.load(Ordering::Relaxed)
}

impl Indexer {
    pub fn new(db_name: String, assets: String) -> Self {
        let db_path = PathBuf::from(db_name.clone());
        let (tx, rx) = mpsc::channel::<Job>();
        let mut db = match crate::DB::new(db_path) {
            Ok(db) => Some(db),
            Err(v) => {
                warn!("database `{}' could not be opened {}", db_name, v);
                None
            }
        };
        let device = crate::make_device();
        let embedder = match crate::Embedder::new(&device, &PathBuf::from(&assets)) {
            Ok(embedder) => Some(embedder),
            Err(v) => {
                warn!(
                    "embedder with assets in `{}` could not be created {}",
                    assets, v
                );
                None
            }
        };

        let handle = thread::spawn(move || {
            while let Ok(job) = rx.recv() {
                match job {
                    Job::Clear => {
                        if let Some(db) = db.as_mut() {
                            db.clear();
                        }
                        CLEAR.store(false, Ordering::Release);
                    }
                    Job::Delete { ref sql_filter } => {
                        if let Some(db) = db.as_mut() {
                            let _ = db.delete_with_filter(sql_filter);
                        }
                        CLEAR.store(false, Ordering::Release);
                    }
                    Job::Shutdown => {
                        if let Some(db) = db.as_mut() {
                            db.shutdown();
                        }
                        return;
                    }
                    _ => {}
                }

                if drain_commands() {
                    continue;
                }

                if let Some(db) = db.as_mut() {
                    match job {
                        Job::Add {
                            uuid,
                            date,
                            metadata,
                            body,
                            lengths,
                        } => match db.add_doc(&uuid, date, &metadata, &body, lengths) {
                            Ok(()) => {}
                            Err(v) => {
                                warn!("add_doc failed! {}", v);
                            }
                        },
                        Job::Remove { uuid } => match db.remove_doc(&uuid) {
                            Ok(()) => {}
                            Err(v) => {
                                warn!("remove_doc failed! {}", v);
                            }
                        },
                        Job::Index => {
                            loop {
                                match &embedder {
                                    Some(embedder) => {
                                        let now = std::time::Instant::now();
                                        let got =
                                            match crate::embed_chunks(&db, &embedder, Some(10)) {
                                                Ok(got) => got,
                                                Err(_v) => {
                                                    break;
                                                }
                                            };
                                        let dt = now.elapsed().as_secs_f64();
                                        stats("embed-docs-per-second", ((got as f64) / dt).round());

                                        if got == 0 || drain_commands() {
                                            break;
                                        }
                                    }
                                    None => {
                                        break;
                                    }
                                };
                            }
                            if crate::count_unindexed_embeddings(&db).unwrap_or(0) > 1024 {
                                let now = std::time::Instant::now();
                                match crate::index_chunks(&db, &device) {
                                    Ok(()) => {}
                                    Err(v) => {
                                        warn!("index_chunks failed! {}", v);
                                    }
                                }
                                stats("indexing-time-ms", now.elapsed().as_millis() as f64);
                                match db.file_size() {
                                    Ok(size) => {
                                        let mb = (1 << 20) as f64;
                                        stats(
                                            "database-size-megabytes",
                                            (size as f64 / mb).round(),
                                        );
                                    }
                                    Err(v) => {
                                        warn!("db.file_size() failed! {}", v);
                                    }
                                }
                            }
                            memory_stats();
                        }
                        _ => {}
                    }
                }
            }
        });
        Indexer {
            tx,
            handle: Mutex::new(Some(handle)),
        }
    }
    pub fn global(db_name: String, assets: String) -> &'static Self {
        INDEXER.get_or_init(|| Indexer::new(db_name, assets))
    }

    pub fn add(
        &self,
        uuid: Uuid,
        date: Option<Timestamp>,
        metadata: String,
        body: String,
        lengths: Option<Vec<usize>>,
    ) {
        if accepting_commands() {
            let _ = self.tx.send(Job::Add {
                uuid,
                date,
                metadata,
                body,
                lengths,
            });
        }
    }

    pub fn remove(&self, uuid: Uuid) {
        if accepting_commands() {
            let _ = self.tx.send(Job::Remove { uuid });
        }
    }

    pub fn index(&self) {
        if accepting_commands() {
            let _ = self.tx.send(Job::Index);
        }
    }

    pub fn clear(&self) {
        if accepting_commands() {
            CLEAR.store(true, Ordering::Release);
            let _ = self.tx.send(Job::Clear);
        }
    }

    pub fn delete(&self, sql_filter: crate::types::SqlStatementInternal) {
        if accepting_commands() {
            CLEAR.store(true, Ordering::Release);
            let _ = self.tx.send(Job::Delete { sql_filter });
        }
    }

    pub fn shutdown(&self) {
        if accepting_commands() {
            SHUTDOWN.store(true, Ordering::Release);
            let _ = self.tx.send(Job::Shutdown);
            if let Some(h) = self.handle.lock().unwrap().take() {
                h.join().unwrap();
            }
        }
    }
}

struct WitchcraftInner {
    db: Option<crate::DB>,
    cache: crate::EmbeddingsCache,
    embedder: Option<crate::Embedder>,
}

impl WitchcraftInner {
    pub fn new(db_name: String, assets: String) -> Self {
        let _indexer = Indexer::global(db_name.clone(), assets.clone());
        let db = crate::DB::new_reader(db_name.into()).ok();
        let cache = crate::EmbeddingsCache::new(16);
        let device = crate::make_device();
        let assets = std::path::PathBuf::from(assets);
        let embedder = crate::Embedder::new(&device, &assets).ok();

        Self {
            db: db,
            cache: cache,
            embedder: embedder,
        }
    }

    pub fn search(
        &mut self,
        q: &String,
        threshold: f32,
        top_k: usize,
        sql_filter: Option<&crate::types::SqlStatementInternal>,
    ) -> Vec<(f32, String, Vec<String>, u32, String)> {
        self.embedder
            .as_ref()
            .and_then(|embedder| self.db.as_ref().map(|db| (embedder, db)))
            .map_or_else(
                || {
                    warn!("no embedder or db");
                    Vec::new()
                },
                |(embedder, db)| {
                    let now = std::time::Instant::now();
                    let results = crate::search(
                        db,
                        embedder,
                        &mut self.cache,
                        &q,
                        threshold,
                        top_k,
                        true,
                        sql_filter,
                    )
                    .unwrap_or_else(|e| {
                        warn!("error {e} querying");
                        Vec::new()
                    });
                    stats("search-latency-ms", now.elapsed().as_millis() as f64);
                    results
                },
            )
    }

    pub fn score(&mut self, q: &String, sentences: &Vec<String>) -> Vec<f32> {
        self.embedder
            .as_ref()
            .filter(|_| !sentences.is_empty())
            .map_or_else(
                || Vec::new(),
                |embedder| {
                    crate::score_query_sentences(embedder, &mut self.cache, q, sentences)
                        .unwrap_or_else(|e| {
                            warn!("error {e} scoring");
                            Vec::new()
                        })
                },
            )
    }
}

static INNER: OnceLock<Arc<Mutex<WitchcraftInner>>> = OnceLock::new();
fn inner(db_name: String, assets: String) -> Arc<Mutex<WitchcraftInner>> {
    INNER
        .get_or_init(|| Arc::new(Mutex::new(WitchcraftInner::new(db_name, assets))))
        .clone()
}

pub struct SearchTask {
    inner: Arc<Mutex<WitchcraftInner>>,
    q: String,
    threshold: f32,
    top_k: usize,
    sql_filter: Option<crate::types::SqlStatementInternal>,
}

impl<'env> ScopedTask<'env> for SearchTask {
    type Output = Vec<(f32, String, Vec<String>, u32, String)>;
    type JsValue = Object<'env>;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(self.inner.lock().unwrap().search(
            &self.q,
            self.threshold,
            self.top_k,
            self.sql_filter.as_ref(),
        ))
    }

    fn resolve(&mut self, env: &'env Env, out: Self::Output) -> Result<Self::JsValue> {
        let mut outer: Array<'env> = env.create_array(out.len() as u32)?;
        for (i, (score, metadata, bodies, idx, _date)) in out.into_iter().enumerate() {
            let idx = (idx as usize).min(bodies.len().saturating_sub(1));
            let body = bodies.get(idx).cloned().unwrap_or_default();
            let mut obj = Object::new(env)?;
            obj.set("score", score as f64)?;
            obj.set("metadata", metadata)?;
            obj.set("body", body)?;
            obj.set("idx", idx as u32)?;
            outer.set(i as u32, obj)?;
        }
        outer.coerce_to_object()
    }
}

pub struct ScoreTask {
    inner: Arc<Mutex<WitchcraftInner>>,
    q: String,
    sentences: Vec<String>,
}

impl<'env> ScopedTask<'env> for ScoreTask {
    type Output = Vec<f32>;
    type JsValue = Object<'env>;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(self.inner.lock().unwrap().score(&self.q, &self.sentences))
    }

    fn resolve(&mut self, env: &'env Env, output: Self::Output) -> Result<Self::JsValue> {
        let mut rows: Vec<f32> = Vec::with_capacity(output.len());
        for score in output {
            rows.push(score);
        }
        let arr = Array::from_vec(env, rows)?;
        // If you need to return Object<'env>, coerce the Array to Object
        arr.coerce_to_object()
    }
}

#[napi(js_name = "Witchcraft")]
pub struct Witchcraft {
    db_name: String,
    assets: String,
    indexer: &'static Indexer,
}

#[napi]
impl Witchcraft {
    #[napi(constructor)]
    pub fn new(db_name: String, assets: String) -> Self {
        info!("warp running");
        let indexer = Indexer::global(db_name.clone(), assets.clone());
        Self {
            db_name,
            assets,
            indexer,
        }
    }
    #[napi]
    pub fn search(
        &mut self,
        q: String,
        threshold: f64,
        top_k: u32,
        sql_filter: Option<SqlStatement>,
    ) -> AsyncTask<SearchTask> {
        AsyncTask::new(SearchTask {
            inner: inner(self.db_name.clone(), self.assets.clone()),
            q: q,
            threshold: threshold as f32,
            top_k: top_k as usize,
            sql_filter: sql_filter.map(Into::into),
        })
    }

    #[napi]
    pub fn score(&mut self, q: String, sentences: Vec<String>) -> AsyncTask<ScoreTask> {
        AsyncTask::new(ScoreTask {
            inner: inner(self.db_name.clone(), self.assets.clone()),
            q: q,
            sentences: sentences,
        })
    }

    #[napi]
    pub fn add(
        &self,
        uuid: String,
        date: String,
        metadata: String,
        body: String,
        lengths: Vec<u32>,
    ) {
        let uuid = Uuid::parse_str(&uuid).unwrap();
        let date = Timestamp::parse(date.as_str());

        let lengths = if !lengths.is_empty() {
            Some(lengths.iter().map(|len| *len as usize).collect())
        } else {
            None
        };

        self.indexer.add(uuid, date, metadata, body, lengths);
    }

    #[napi]
    pub fn remove(&self, uuid: String) {
        let uuid = Uuid::parse_str(&uuid).unwrap();
        self.indexer.remove(uuid);
    }

    #[napi]
    pub fn index(&self) {
        self.indexer.index();
    }

    #[napi]
    pub fn clear(&self) {
        self.indexer.clear();
    }

    #[napi]
    pub fn delete(&self, sql_filter: SqlStatement) {
        self.indexer.delete(sql_filter.into());
    }

    #[napi]
    pub fn shutdown(&self) {
        self.indexer.shutdown();
    }
}
