/* Node module API for Warp */

use napi::bindgen_prelude::*;
use napi::{Env, ScopedTask};
use napi_derive::napi;

use std::{
    sync::atomic::{AtomicBool, Ordering},
    sync::{mpsc, Arc, Mutex, OnceLock},
    thread::{self, JoinHandle},
};

type Job = (String, String, String);

#[derive(Debug)]
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
    pub fn new(db_name: String) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let mut db = warp::DB::new(&db_name);

        let handle = thread::spawn(move || {
            let device = warp::make_device();
            while let Ok(job) = rx.recv() {
                let (command, arg1, arg2) = job;

                if command == "clear" {
                    db.clear();
                    CLEAR.store(false, Ordering::Release);
                } else if command == "shutdown" {
                    db.shutdown();
                    return;
                }

                if drain_commands() {
                    continue;
                }

                if command == "add" {
                    db.add_doc(&arg1, &arg2).unwrap();
                } else if command == "index" {
                    let count = warp::count_unindexed_chunks(&db).unwrap();
                    if count >= 2048 {
                        warp::index_chunks(&db, &device).unwrap();
                    }
                } else if command == "embed" {
                    loop {
                        let got = warp::embed_chunks(&db, &device, Some(10)).unwrap();
                        if got == 0 || drain_commands() {
                            break;
                        }
                    }
                }
            }
        });
        Indexer {
            tx,
            handle: Mutex::new(Some(handle)),
        }
    }
    pub fn init_global(db_name: String) -> &'static Self {
        INDEXER.get_or_init(|| Indexer::new(db_name))
    }

    pub fn global() -> &'static Self {
        INDEXER
            .get()
            .expect("Indexer not initialized. Call `Indexer::init_global()` first.")
    }

    pub fn add(&self, metadata: String, body: String) {
        if accepting_commands() {
            let _ = self.tx.send(("add".to_string(), metadata, body));
        }
    }

    pub fn embed(&self) {
        if accepting_commands() {
            let _ = self
                .tx
                .send(("embed".to_string(), String::new(), String::new()));
        }
    }

    pub fn index(&self) {
        if accepting_commands() {
            let _ = self
                .tx
                .send(("index".to_string(), String::new(), String::new()));
        }
    }

    pub fn clear(&self) {
        if accepting_commands() {
            CLEAR.store(true, Ordering::Release);
            let _ = self
                .tx
                .send(("clear".to_string(), String::new(), String::new()));
        }
    }

    pub fn shutdown(&self) {
        if accepting_commands() {
            SHUTDOWN.store(true, Ordering::Release);
            let _ = self
                .tx
                .send(("shutdown".to_string(), String::new(), String::new()));
            if let Some(h) = self.handle.lock().unwrap().take() {
                h.join().unwrap();
            }
        }
    }
}

mod warp;

struct WarpInner {
    embedder: warp::Embedder,
    db: warp::DB,
    cache: warp::EmbeddingsCache,
}

impl WarpInner {
    pub fn new(db_name: String) -> Self {
        let _indexer = Indexer::init_global(db_name.clone());
        let db = warp::DB::new_reader(&db_name.clone());
        let device = warp::make_device();
        let embedder = warp::Embedder::new(&device);
        let cache = warp::EmbeddingsCache::new(16);

        Self {
            embedder: embedder,
            db: db,
            cache: cache,
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
        match warp::search(
            &self.db,
            &self.embedder,
            &mut self.cache,
            &q,
            threshold,
            top_k,
            true,
            filter,
        ) {
            Ok(v) => v,
            Err(e) => {
                println!("error {} querying", e);
                [].to_vec()
            }
        }
    }

    pub fn score(&mut self, q: &String, sentences: &Vec<String>) -> Vec<f32> {
        if sentences.len() != 0 {
            match warp::score_query_sentences(&self.embedder, &mut self.cache, &q, &sentences) {
                Ok(v) => v,
                Err(e) => {
                    println!("error {} scoring", e);
                    [].to_vec()
                }
            }
        } else {
            [].to_vec()
        }
    }
}

static INNER: OnceLock<Arc<Mutex<WarpInner>>> = OnceLock::new();
fn inner(db_name: String) -> Arc<Mutex<WarpInner>> {
    INNER
        .get_or_init(|| Arc::new(Mutex::new(WarpInner::new(db_name))))
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
        // Outer array
        let mut outer: Array<'env> = env.create_array(out.len() as u32)?;
        for (i, (score, a, b)) in out.into_iter().enumerate() {
            // Triplet [number, string, string]
            let mut triplet: Array<'env> = env.create_array(3)?;
            triplet.set(0, score as f64)?; // numbers as f64
            triplet.set(1, a)?; // String -> JS string
            triplet.set(2, b)?;
            outer.set(i as u32, triplet)?; // set the triplet into the outer array
        }
        // Return Object<'env> (Array is an Object; coerce to Object)
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
}

#[napi]
impl Warp {
    #[napi(constructor)]
    pub fn new(db_name: String) -> Self {
        let _indexer = Indexer::init_global(db_name.clone());
        Self { db_name: db_name }
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
            inner: inner(self.db_name.clone()),
            q: q,
            threshold: threshold as f32,
            top_k: top_k as usize,
            sql_filter: sql_filter,
        })
    }

    #[napi]
    pub fn score(&mut self, q: String, sentences: Vec<String>) -> AsyncTask<ScoreTask> {
        AsyncTask::new(ScoreTask {
            inner: inner(self.db_name.clone()),
            q: q,
            sentences: sentences,
        })
    }

    #[napi]
    pub fn add(&self, metadata: String, body: String) {
        Indexer::global().add(metadata, body);
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
