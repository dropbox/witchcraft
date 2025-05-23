use std::env;
use rand::prelude::*;
use rusqlite::{Connection, Statement, Result as SQLResult};
use sha2::{Sha256, Digest};
use min_heap::MinHeap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;
use indicatif::ProgressBar;

mod t5;

mod packops;
use packops::TensorPackOps;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, D, IndexOp};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
}

impl T5ModelBuilder {
    pub fn load() -> Result<(Self, Tokenizer)> {
        let device = Device::new_metal(0)?;
        let path = PathBuf::from(r"xtr.safetensors");
        let weights_filename = vec![path];
        let config = std::fs::read_to_string("xtr-base-en/config.json").unwrap();
        let config: t5::Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file("xtr-base-en/tokenizer.json").map_err(E::msg).unwrap();
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }

}

fn kmeans(data: &Tensor, k: usize, max_iter: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    println!("kmeans...");
    let (n, _) = data.dims2()?;

    let total : u64 = (max_iter * k).try_into().unwrap();
    let bar = ProgressBar::new(total);

    let mut rng = SmallRng::seed_from_u64(0);
    let centroid_idx = rand::seq::index::sample(&mut rng, n, k).into_vec();
    let centroid_idx: Vec<u32> = centroid_idx.iter().map(|&i| i as u32).collect();

    let centroid_idx_tensor = Tensor::from_slice(centroid_idx.as_slice(), (k,), device)?;
    let centroid_idx_tensor = centroid_idx_tensor.to_device(data.device())?;
    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;
    let mut cluster_assignments = Tensor::zeros((n,), DType::U32, device)?;

    for _ in 0..max_iter {
        //let dist = cdist(data, &centers)?;
        let sim = data.matmul(&centers.transpose(D::Minus1, D::Minus2)?)?;
        cluster_assignments = sim.argmax(D::Minus1)?;
        let mut centers_vec = vec![];
        for i in 0..k {
            let mut indices = vec![];
            cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .for_each(|(j, x)| {
                    if *x == i as u32 {
                        indices.push(j as u32);
                    }
                });
            let indices = Tensor::from_slice(indices.as_slice(), (indices.len(),), device)?;
            let indices = indices.to_device(data.device())?;
            let cluster_data = data.index_select(&indices, 0)?;
            let sum = cluster_data.sum(0)?;
            let normalized = sum.broadcast_div(&sum.sqr()?.sum_keepdim(0)?.sqrt()?);
            centers_vec.push(normalized?);
            bar.inc(1);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }
    bar.finish();
    Ok((centers, cluster_assignments))
}

fn write_buckets(db: &DB,
        data: &Tensor,
        document_indices: &Tensor,
        centers: &Tensor,
        device: &Device) -> Result<()> {

    let (k, _) = centers.dims2()?;
    println!("k={}", k);

    let bar = ProgressBar::new(k as u64 + 2);
    let sim = data.matmul(&centers.transpose(D::Minus1, D::Minus2)?)?;
    bar.inc(1);
    let cluster_assignments = sim.argmax(D::Minus1)?;
    bar.inc(1);
    for i in 0..k {
        let center = centers.get(i)?;
        let mut indices = vec![];
        cluster_assignments
            .to_vec1::<u32>()?
            .iter()
            .enumerate()
            .for_each(|(j, x)| {
                if *x == i as u32 {
                    indices.push(j as u32);
                }
            });
        let data_indices = Tensor::from_slice(indices.as_slice(), (indices.len(),), device)?;
        let data_indices = data_indices.to_device(data.device())?;
        let cluster_data = data.index_select(&data_indices, 0)?;

        let residuals = cluster_data.broadcast_sub(&center).unwrap();
        let qmin = 0.0;
        let qmax = 15.0;
        let scale1 = 16.0 / 2.0;
        let zp = (qmax - qmin) / 2.0;
        let residuals = ((&residuals * scale1)? + zp)?.round()?.clamp(qmin, qmax)?;

        let center_bytes = center.to_f32_bytes()?;

        let document_subset = document_indices.index_select(&data_indices, 0)?;
        let indices_bytes = document_subset.to_u32_bytes()?;

        let residuals_bytes = residuals.to_q4_bytes()?;

        db.add_bucket(i as u32, &center_bytes, &indices_bytes, &residuals_bytes).unwrap();
        bar.inc(1);
    }
    bar.finish();
    Ok(())
}

fn match_centroids(
        bucket_sizes_query: &mut Query,
        bucket_query: &mut Query,
        body_query: &mut Query,
        query_embeddings: &Tensor,
        centers: &Tensor) -> Result<()> {
    println!("******************** LOOKUP **********************\n");
    let now = std::time::Instant::now();

    let k = 16;
    let t_prime = 10000;

    let sizes = bucket_sizes_query.u32_vec()?;

    let device = query_embeddings.device();
    let centers = centers.to_device(&device)?;
    let query_centroid_similarity = query_embeddings.matmul(&centers.transpose(D::Minus1, D::Minus2)?).unwrap();

    let sorted_indices = query_centroid_similarity.arg_sort_last_dim(false)?;
    let (m, n) = sorted_indices.dims2()?;

    let mut topk_clusters = vec![];
    let mut missing = vec![];
    for i in 0..m {
        let row = sorted_indices.get(i)?;
        let row_scores_sorted = query_centroid_similarity.get(i)?.gather(&row, D::Minus1)?;
        let mut cumsum = 0;
        let mut score = 0.0f32;
        for j in 0..n {
            let idx = row.get(j)?.to_scalar::<u32>()?;
            if j < k {
                topk_clusters.push(idx);
            }
            let size = sizes[idx as usize];
            if cumsum < t_prime {
                score = row_scores_sorted.get(j)?.to_scalar::<f32>()?;
            }
            cumsum += size;

            if j >= k || cumsum >= t_prime {
                break;
            }
        }
        missing.push(score);
    }
    topk_clusters.sort();
    topk_clusters.dedup();
    let missing_similarities = Tensor::from_vec(missing, m, &Device::Cpu)?;
    println!("finding top-{} clusters took {} ms.", topk_clusters.len(), now.elapsed().as_millis());

    let now = std::time::Instant::now();
    let mut heap = MinHeap::new();

    let mut all_document_embeddings = vec![];
    for i in topk_clusters {
        let (document_indices, document_embeddings) = bucket_query.point3(i)?;
        let center = centers.get(i as usize)?;
        let center = center.to_device(&device)?;

        let document_indices = u8_to_vec_u32(&document_indices);

        let residuals = Tensor::from_q4_bytes(&document_embeddings, 128, &device).unwrap();
        let qmin = 0.0;
        let qmax = 15.0;
        let scale2 = 2.0 / 16.0;
        let zp = (qmax - qmin) / 2.0;
        let residuals = ((&residuals - zp)? * scale2)?;
        let embeddings = residuals.broadcast_add(&center)?;

        let (m, _) = residuals.dims2()?;
        for j in 0..m {
            heap.push((document_indices[j], all_document_embeddings.len()));
            all_document_embeddings.push(embeddings.get(j)?);
        }
    }

    let all_document_embeddings = Tensor::stack(all_document_embeddings.as_slice(), 1)?;
    let all_document_embeddings = all_document_embeddings.to_device(query_embeddings.device())?;

    let sim = query_embeddings.matmul(&all_document_embeddings)?.transpose(0, 1)?;
    let sim = sim.to_device(&Device::Cpu)?;

    let mut last = std::u32::MAX;
    let mut current = Tensor::zeros((n,), DType::F32, &Device::Cpu)?;

    let mut heap2 = MinHeap::new();
    while let Some((idx, i)) = heap.pop() {

        if last != idx || heap.len() == 0 {
            let score = current.mean(0)?.to_scalar::<f32>()?;
            if score >= 0.8 {
                let score_as_u32 = (1000.0 * score) as u32;
                let elem = (score_as_u32, last);
                if heap2.len() < 10 {
                    heap2.push(elem);
                } else {
                    let min = heap2.peek().unwrap();
                    if *min < elem {
                        heap2.pop();
                        heap2.push(elem);
                    }
                }
            }
        }

        let row = sim.get(i)?;
        current = if idx == last {
            row.maximum(&current)?
        } else {
            row.maximum(&missing_similarities)?
        };

        last = idx;
    }
    println!("scoring clusters took {} ms.", now.elapsed().as_millis());
    println!("");

    let mut results = heap2.into_vec();
    results.reverse();

    for (score_as_u32, idx) in results {
        println!("================= score:{} ============", (score_as_u32 as f32) / 1000.0);
        let body = body_query.point4(idx)?;
        println!("{}", body);
        println!("");
    }

    Ok(())
}

fn split_tensor(tensor: &Tensor) -> Vec<Tensor> {
    let dims = tensor.dims();
    let num_rows = dims[0];

    // Collect each row as a separate Tensor of shape [128]
    (0..num_rows)

        .map(|i| {
            let row_tensor = tensor.i(i).unwrap();
            row_tensor.unsqueeze(0).unwrap()
        })

        .collect()
}

fn compute_sha256<P: AsRef<Path>>(path: P) -> Result<String, Box<dyn std::error::Error>> {
    // Open the file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();

    // Read file in chunks
    let mut buffer = [0u8; 0x10000];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    // Finalize and return the hash as a hex string
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}


fn register_documents(db: &DB, dirname: &str) -> Result<()> {

    println!("register documents...");
    let mut entries: Vec<_> = fs::read_dir(dirname)?
        .filter_map(Result::ok) // Filter out errors
        .filter(|e| e.path().is_file()) // Only files
        .collect();

    // Sort entries by file name
    entries.sort_by_key(|e| e.file_name());
    let bar = ProgressBar::new(entries.len() as u64);

    for entry in entries {
        let path = entry.path();
        let filename = path.to_str().unwrap();
        let body = fs::read_to_string(filename)?;
        let hash = compute_sha256(path.clone()).unwrap();
        //println!("{} hash is {}", filename, hash);
        db.add_doc(&filename, &hash, &body).unwrap();
        bar.inc(1);
    }
    bar.finish();

    Ok(())
}

struct Embedder {
    tokenizer: Tokenizer,
    model: t5::T5EncoderModel,
}

impl Embedder {
    fn new() -> Self {
        let (builder, tokenizer) = T5ModelBuilder::load().unwrap();
        let model = builder.build_encoder().unwrap();
        Self {
            tokenizer,
            model
        }
    }
    fn embed(self: &Self, text: &str) -> Result<Tensor> {
        let tokens = self.tokenizer
            .encode(text, true)
            .map_err(E::msg).unwrap()
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], self.model.device()).unwrap().unsqueeze(0).unwrap();
        let embeddings = self.model.forward(&token_ids).unwrap();
        //println!("embedder took {} ms.", now.elapsed().as_millis());
        normalize_l2(&embeddings)
    }
}

pub struct Gatherer<'a> {
    documents: Box<dyn Iterator<Item = (String, String)> + 'a>,
    embedder: &'a Embedder,
}

impl<'a> Gatherer<'a> {
    fn new(query: &'a mut Query, embedder: &'a Embedder) -> Self {
        let documents = Box::new(query.iter1().unwrap().map(Result::unwrap));
        Self {
            documents,
            embedder,
        }
    }
}

impl<'a> Iterator for Gatherer<'a> {
    type Item = (String, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let mut doc_embedding = vec![];
        match self.documents.next() {
            Some((hash, body)) => {

                let now = std::time::Instant::now();
                let embeddings = self.embedder.embed(&body).unwrap();
                println!("embedder took {} ms.", now.elapsed().as_millis());

                let split = split_tensor(&embeddings.get(0).ok()?);
                doc_embedding.extend(split);
                Some((hash, Tensor::cat(&doc_embedding, 0).unwrap()))
            }
            None => {
                None
            }
        }
    }
}

struct DB {
    connection: Connection,
}

struct Query<'connection> {
    stmt: Statement<'connection>,
}

impl DB {
    pub fn new() -> Self {
        //let connection = Connection::open_in_memory().unwrap();
        let connection = Connection::open("mydb.sqlite").unwrap();

        println!("init");
        let query = "CREATE TABLE IF NOT EXISTS document(filename TEXT PRIMARY KEY,
            hash TEXT NOT NULL, body TEXT, UNIQUE(filename, hash))";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS chunk(hash TEXT PRIMARY KEY NOT NULL, embedding BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS bucket(id INTEGER PRIMARY KEY,
            center BLOB NOT NULL, indices BLOB NOT NULL, residuals BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();
        Self { connection }
    }

    fn make_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT
            document.hash,document.body
            FROM document
            LEFT JOIN chunk ON document.hash = chunk.hash
            WHERE chunk.hash IS NULL
            ORDER BY filename")?;
        Ok(Query { stmt })
    }

    fn make_kmeans_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT document.rowid,document.filename,chunk.hash,chunk.embedding FROM document,chunk
            WHERE document.hash == chunk.hash
            ORDER BY document.filename")?;
        Ok(Query { stmt })
    }

    fn make_bucket_center_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT center FROM bucket ORDER BY id")?;
        Ok(Query { stmt })
    }

    fn make_bucket_sizes_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT length(indices)/4 FROM bucket ORDER BY id")?;
        Ok(Query { stmt })
    }

    fn make_bucket_residuals_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT indices,residuals FROM bucket WHERE id = ?1")?;
        Ok(Query { stmt })
    }

    fn make_document_body_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT body FROM document WHERE rowid = ?1")?;
        Ok(Query { stmt })
    }

    fn add_doc(self: &Self, filename: &str, hash: &str, body: &str) -> SQLResult<()> {
        self.connection.execute("INSERT OR IGNORE INTO document VALUES(?1, ?2, ?3)", (&filename, &hash, &body))?;
        Ok(())
    }

    fn add_chunk(self: &Self, hash: &str, embeddings: &Vec<u8>) -> SQLResult<()> {
        self.connection.execute("INSERT OR REPLACE INTO chunk VALUES(?1, ?2)", (&hash, embeddings)).unwrap();
        Ok(())
    }

    fn add_bucket(self: &Self, id: u32, center: &Vec<u8>, indices: &Vec<u8>, residuals: &Vec<u8>) -> SQLResult<()> {
        self.connection.execute("INSERT OR REPLACE INTO bucket VALUES(?1, ?2, ?3, ?4)",
            (id, center, indices, residuals))?;
        Ok(())
    }
}

impl<'connection> Query<'connection> {
    fn iter1(&mut self) -> SQLResult<impl Iterator<Item = SQLResult<(String, String)>> + '_> {
        self.stmt.query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
            ))
        })
    }

    fn iter2(&mut self) -> SQLResult<impl Iterator<Item = SQLResult<(u32, String, String, Vec<u8>)>> + '_> {
        self.stmt.query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
            ))
        })
    }

    fn u32_vec(&mut self) -> SQLResult<Vec<u32>> {
        let iter = self.stmt.query_map([], |row| {
        Ok(
            row.get(0)?,
        )
        })?;
        iter.collect()
    }

    fn point3(&mut self, id: u32) -> SQLResult<(Vec<u8>, Vec<u8>)> {
        self.stmt.query_row([id], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
            ))
        })
    }

    fn point4(&mut self, id: u32) -> SQLResult<String> {
        self.stmt.query_row([id], |row| {
            Ok(
                row.get(0)?,
            )
        })
    }

    fn iter4(&mut self) -> SQLResult<impl Iterator<Item = SQLResult<Vec<u8>>> + '_> {
        self.stmt.query_map([], |row| {
            Ok( row.get(0)? )
        })
    }
}

fn u8_to_vec_u32(bytes: &[u8]) -> Vec<u32> {
    let u32_size = size_of::<u32>();

    assert!(bytes.len() % u32_size == 0);
    let total_u32s = bytes.len() / u32_size;

    let mut u32s = Vec::with_capacity(total_u32s);
    for chunk in bytes.chunks_exact(u32_size) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        u32s.push(u32::from_ne_bytes(arr));
    }
    u32s
}

fn usage(arg0: &String) {
    eprintln!("Usage: {} index | query <text>", arg0);
    std::process::exit(1)
}

fn main() -> Result<()> {

    //let device = Device::Cpu;
    let device = Device::new_metal(0)?;
    let embedder = Embedder::new();

    let db = DB::new();
    let mut query = db.make_query().unwrap();
    let mut kmeans_query = db.make_kmeans_query().unwrap();

    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "index" {

        register_documents(&db, "documents").unwrap();

        let embedding_iter = Gatherer::new(&mut query, &embedder);
        for (hash, embeddings) in embedding_iter {
            println!("for hash {} {:?}", hash, embeddings.dims2().unwrap());
            let bytes = embeddings.to_f32_bytes()?;
            db.add_chunk(&hash, &bytes).unwrap();
        }

        let mut document_indices = Vec::<u32>::new();
        let mut all_embeddings = vec![];
        let mut total = 0;
        println!("read embeddings...");
        for result in kmeans_query.iter2()? {
            let (id, _filename, _hash, embedding) = result?;
            //println!("id={} filename={}", id, _filename);
            let t = Tensor::from_f32_bytes(&embedding, 128, &Device::Cpu)?;
            let split = split_tensor(&t);
            all_embeddings.extend(split);
            let (m, _n) = t.dims2()?;
            for _ in 0..m {
                document_indices.push(id as u32);
            }
            total += m;
        }
        let matrix = Tensor::cat(&all_embeddings, 0).unwrap();
        let now = std::time::Instant::now();
        let k = 1024;
        let (centers, idxs) = kmeans(&matrix, k, 5, &device)?;
        println!("kmeans took {} ms.", now.elapsed().as_millis());
        println!("idxs {}", idxs);

        println!("write buckets...");
        let document_indices = Tensor::from_vec(document_indices, total, &Device::Cpu).unwrap();
        let now = std::time::Instant::now();
        write_buckets(&db, &matrix, &document_indices, &centers, &device).unwrap();
        println!("write buckets took {} ms.", now.elapsed().as_millis());

    } else if args.len() >= 3 && args[1] == "query" {

        let q = &args[2..].join(" ");
        println!("Looking up: {}", q);

        let mut center_query = db.make_bucket_center_query().unwrap();
        let mut bucket_sizes_query = db.make_bucket_sizes_query().unwrap();
        let mut bucket_query = db.make_bucket_residuals_query().unwrap();
        let mut body_query = db.make_document_body_query().unwrap();

        let mut centers = vec![];
        for center in center_query.iter4()? {
            let t = Tensor::from_f32_bytes(&center?, 128, &Device::Cpu)?.flatten_all()?;
            centers.push(t);
        }
        let centers = Tensor::stack(&centers, 0)?;

        let qe = embedder.embed(q)?.get(0)?;
        match_centroids(&mut bucket_sizes_query, &mut bucket_query, &mut body_query, &qe, &centers).unwrap();
    } else {
        usage(&args[0]);
    }

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
