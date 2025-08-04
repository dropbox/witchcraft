use candle_core::Device;
use napi_derive::napi;

use std::{
    sync::{mpsc, LazyLock},
    thread::{self, JoinHandle},
};

#[derive(Debug)]
pub struct Indexer {
    tx: mpsc::Sender<Job>,
    db_name: String,
    _handle: JoinHandle<()>,
}

static INDEXER: LazyLock<Indexer> = LazyLock::new(|| {
    Indexer::new()
});

type Job = (String, String, String);

impl Indexer {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let db_name = "mydb.sqlite";
        let db = warp::DB::new(db_name);
        let handle = thread::spawn(move || {
            let device = Device::new_metal(0).unwrap();
            while let Ok(job) = rx.recv() {
                let (command, arg1, arg2) = job;
                if command == "add" {
                    warp::add_doc_from_string(&db, &arg1, &arg2).unwrap();
                } else if command == "index" {
                    let count = warp::count_unindexed_chunks(&db).unwrap();
                    println!("count {}", count);
                    if count >= 2048 {
                        warp::index_chunks(&db, &device).unwrap();
                    }
                } else if command == "embed" {
                    warp::embed_chunks(&db, &device).unwrap();
                }
            }
        });
        Indexer {
            tx,
            db_name: db_name.to_string(),
            _handle: handle,
        }
    }
    pub fn submit(&self, job: Job) {
        let _ = self.tx.send(job);
    }
}

mod warp;

#[napi(js_name = "Warp")]
pub struct Warp {
    db: warp::DB,
    embedder: warp::Embedder,
}

#[napi]
impl Warp {
    #[napi(constructor)]
    pub fn new() -> Self {
        // we depend on INDEXER.db_name to ensure the DB has been created
        let db = warp::DB::new_reader(&INDEXER.db_name);
        let device = Device::new_metal(0).unwrap();
        let embedder = warp::Embedder::new(&device);

        Self {
            db: db,
            embedder: embedder,
        }
    }
    #[napi]
    pub fn search(&self, q: String, threshold: f64) -> Vec<(String, String)> {
        match warp::search(&self.db, &self.embedder, &q, threshold as f32, true) {
            Ok(v) => v,
            Err(e) => {
                println!("error {} querying for {}", e, &q);
                [].to_vec()
            }
        }
    }

    #[napi]
    pub fn add(&self, metadata: String, body: String) {
        INDEXER.submit(("add".to_string(), metadata, body));
    }

    #[napi]
    pub fn embed(&self) {
        INDEXER.submit(("embed".to_string(), "".to_string(), "".to_string()));
    }

    #[napi]
    pub fn index(&self) {
        INDEXER.submit(("index".to_string(), "".to_string(), "".to_string()));
    }
}
