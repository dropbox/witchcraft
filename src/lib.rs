/* Node module API for Warp */

use iso8601_timestamp::Timestamp;
use napi::bindgen_prelude::*;
use napi::{Env, ScopedTask};
use napi_derive::napi;

use std::{
    sync::atomic::{AtomicBool, Ordering},
    sync::{mpsc, Arc, Mutex, OnceLock},
    thread::{self, JoinHandle},
};

use uuid::Uuid;

enum Job {
    Add {
        uuid: Uuid,
        date: Option<Timestamp>,
        metadata: String,
        body: String,
    },
    Remove {
        uuid: Uuid,
    },
    Index,
    Embed,
    Clear,
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
        let (tx, rx) = mpsc::channel::<Job>();
        let mut db = match warp::DB::new(&db_name) {
            Ok(db) => Some(db),
            Err(v) => {
                println!("database `{}' could not be opened {}", db_name, v);
                None
            }
        };
        let device = warp::make_device();
        let embedder = match warp::Embedder::new(&device, &std::path::PathBuf::from(&assets)) {
            Ok(embedder) => Some(embedder),
            Err(v) => {
                println!(
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
                        } => match db.add_doc(&uuid, date, &metadata, &body) {
                            Ok(()) => {}
                            Err(v) => {
                                println!("add_doc failed! {}", v);
                            }
                        },
                        Job::Remove { uuid } => match db.remove_doc(&uuid) {
                            Ok(()) => {}
                            Err(v) => {
                                println!("remove_doc failed! {}", v);
                            }
                        },
                        Job::Index => {
                            if warp::count_unindexed_chunks(&db).unwrap_or(0) > 2048 {
                                match warp::index_chunks(&db, &device) {
                                    Ok(()) => {}
                                    Err(v) => {
                                        println!("index_chunks failed! {}", v);
                                    }
                                }
                            }
                        }
                        Job::Embed => loop {
                            match &embedder {
                                Some(embedder) => {
                                    let got = match warp::embed_chunks(&db, &embedder, Some(10)) {
                                        Ok(got) => got,
                                        Err(_v) => {
                                            break;
                                        }
                                    };
                                    if got == 0 || drain_commands() {
                                        break;
                                    }
                                }
                                None => {
                                    break;
                                }
                            };
                        },
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
    pub fn init_global(db_name: String, assets: String) -> &'static Self {
        INDEXER.get_or_init(|| Indexer::new(db_name, assets))
    }

    pub fn global() -> &'static Self {
        INDEXER
            .get()
            .expect("Indexer not initialized. Call `Indexer::init_global()` first.")
    }

    pub fn add(&self, uuid: Uuid, date: Option<Timestamp>, metadata: String, body: String) {
        if accepting_commands() {
            let _ = self.tx.send(Job::Add {
                uuid,
                date,
                metadata,
                body,
            });
        }
    }

    pub fn remove(&self, uuid: Uuid) {
        if accepting_commands() {
            let _ = self.tx.send(Job::Remove { uuid });
        }
    }

    pub fn embed(&self) {
        if accepting_commands() {
            let _ = self.tx.send(Job::Embed);
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

mod warp;

struct WarpInner {
    db: Option<warp::DB>,
    cache: warp::EmbeddingsCache,
    embedder: Option<warp::Embedder>,
}

impl WarpInner {
    pub fn new(db_name: String, assets: String) -> Self {
        let _indexer = Indexer::init_global(db_name.clone(), assets.clone());
        let db = warp::DB::new_reader(&db_name.clone()).ok();
        let cache = warp::EmbeddingsCache::new(16);
        let device = warp::make_device();
        let assets = std::path::PathBuf::from(assets);
        let embedder = warp::Embedder::new(&device, &assets).ok();

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
        sql_filter: &String,
    ) -> Vec<(f32, String, String)> {
        let filter = if !sql_filter.is_empty() {
            Some(sql_filter.as_str())
        } else {
            None
        };

        self.embedder
            .as_ref()
            .and_then(|embedder| self.db.as_ref().map(|db| (embedder, db)))
            .map_or_else(
                || {
                    println!("no embedder or db");
                    Vec::new()
                },
                |(embedder, db)| {
                    warp::search(
                        db,
                        embedder,
                        &mut self.cache,
                        &q,
                        threshold,
                        top_k,
                        true,
                        filter,
                    )
                    .unwrap_or_else(|e| {
                        println!("error {e} querying");
                        Vec::new()
                    })
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
                    warp::score_query_sentences(embedder, &mut self.cache, q, sentences)
                        .unwrap_or_else(|e| {
                            println!("error {e} scoring");
                            Vec::new()
                        })
                },
            )
    }
}

static INNER: OnceLock<Arc<Mutex<WarpInner>>> = OnceLock::new();
fn inner(db_name: String, assets: String) -> Arc<Mutex<WarpInner>> {
    INNER
        .get_or_init(|| Arc::new(Mutex::new(WarpInner::new(db_name, assets))))
        .clone()
}

pub struct SearchTask {
    inner: Arc<Mutex<WarpInner>>,
    q: String,
    threshold: f32,
    top_k: usize,
    sql_filter: String,
}

impl<'env> ScopedTask<'env> for SearchTask {
    type Output = Vec<(f32, String, String)>;
    type JsValue = Object<'env>;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(self
            .inner
            .lock()
            .unwrap()
            .search(&self.q, self.threshold, self.top_k, &self.sql_filter))
    }

    fn resolve(&mut self, env: &'env Env, out: Self::Output) -> Result<Self::JsValue> {
        let mut outer: Array<'env> = env.create_array(out.len() as u32)?;
        for (i, (score, metadata, body)) in out.into_iter().enumerate() {
            let mut obj = Object::new(env)?;
            obj.set("score", score as f64)?;
            obj.set("metadata", metadata)?;
            obj.set("body", body)?;
            outer.set(i as u32, obj)?;
        }
        outer.coerce_to_object()
    }
}

pub struct ScoreTask {
    inner: Arc<Mutex<WarpInner>>,
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

#[napi(js_name = "Warp")]
pub struct Warp {
    db_name: String,
    assets: String,
}

#[napi]
impl Warp {
    #[napi(constructor)]
    pub fn new(db_name: String, assets: String) -> Self {
        let cwd = std::env::current_dir().unwrap();
        println!(
            "warp running db=`{}' assets=`{}' cwd=`{}'",
            db_name,
            assets,
            cwd.display()
        );
        let _indexer = Indexer::init_global(db_name.clone(), assets.clone());
        Self {
            db_name: db_name,
            assets: assets,
        }
    }
    #[napi]
    pub fn search(
        &mut self,
        q: String,
        threshold: f64,
        top_k: u32,
        sql_filter: String,
    ) -> AsyncTask<SearchTask> {
        AsyncTask::new(SearchTask {
            inner: inner(self.db_name.clone(), self.assets.clone()),
            q: q,
            threshold: threshold as f32,
            top_k: top_k as usize,
            sql_filter: sql_filter,
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
    pub fn add(&self, uuid: String, date: String, metadata: String, body: String) {
        let uuid = Uuid::parse_str(&uuid).unwrap();
        let date = Timestamp::parse(date.as_str());
        Indexer::global().add(uuid, date, metadata, body);
    }

    #[napi]
    pub fn remove(&self, uuid: String) {
        let uuid = Uuid::parse_str(&uuid).unwrap();
        Indexer::global().remove(uuid);
    }

    #[napi]
    pub fn embed(&self) {
        Indexer::global().embed();
    }

    #[napi]
    pub fn index(&self) {
        Indexer::global().index();
    }

    #[napi]
    pub fn clear(&self) {
        Indexer::global().clear();
    }

    #[napi]
    pub fn shutdown(&self) {
        Indexer::global().shutdown();
    }
}
