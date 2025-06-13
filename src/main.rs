use std::env;
use rusqlite::{Connection, Statement, Row, Result as SQLResult};
use sha2::{Sha256, Digest};
use min_heap::MinHeap;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Read, BufWriter, Write};
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;
use indicatif::ProgressBar;
use csv;
use serde::Deserialize;
mod t5;

mod packops;
use packops::TensorPackOps;

mod merger;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, D, IndexOp};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

struct T5ModelBuilder {
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
}

impl T5ModelBuilder {
    pub fn load() -> Result<(Self, Tokenizer)> {
        let path = PathBuf::from(r"xtr.safetensors");
        let weights_filename = vec![path];
        let config = std::fs::read_to_string("xtr-base-en/config.json").unwrap();
        let config: t5::Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file("xtr-base-en/tokenizer.json").map_err(E::msg).unwrap();
        Ok((
            Self {
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_encoder(&self, device: &Device) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }

}

fn kmeans(data: &Tensor, k: usize, max_iter: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let (m, n) = data.dims2()?;
    println!("kmeans k={} m={} n={}...", k, m, n);

    let total : u64 = (max_iter * k).try_into().unwrap();
    let bar = ProgressBar::new(total);

    let mut rng = rand::rng();
    let centroid_idx = rand::seq::index::sample(&mut rng, m, k).into_vec();
    let centroid_idx: Vec<u32> = centroid_idx.iter().map(|&i| i as u32).collect();

    let centroid_idx_tensor = Tensor::from_slice(centroid_idx.as_slice(), (k,), device)?;
    let centroid_idx_tensor = centroid_idx_tensor.to_device(data.device())?;
    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;
    let mut cluster_assignments = Tensor::zeros((m,), DType::U32, device)?;

    for _ in 0..max_iter {
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
            if indices.len() > 0 {
                let indices = Tensor::from_slice(indices.as_slice(), (indices.len(),), device)?;
                let indices = indices.to_device(data.device())?;
                let cluster_data = data.index_select(&indices, 0)?;
                let sum = cluster_data.sum(0)?;
                let normalized = sum.broadcast_div(&sum.sqr()?.sum_keepdim(0)?.sqrt()?);
                centers_vec.push(normalized?);
            } else {
                let idx = rand::seq::index::sample(&mut rng, m, 1).into_vec()[0];
                let center = data.get(idx)?;
                let normalized = center.broadcast_div(&center.sqr()?.sum_keepdim(0)?.sqrt()?);
                centers_vec.push(normalized?);
            }
            bar.inc(1);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }
    bar.finish();
    Ok((centers, cluster_assignments))
}

fn write_buckets(db: &DB,
        centers: &Tensor,
        device: &Device) -> Result<()> {

    let (k, _) = centers.dims2()?;
    println!("k={}", k);
    let mut mmuls_total = 0;
    let mut writes_total = 0;

    let embeddings_count = db.query("SELECT sum(length(embedding)/128) FROM chunk")?.point((), |row| {
            Ok( row.get::<_, f32>(0)?,)
        }).unwrap();
    let bar = ProgressBar::new(embeddings_count as u64);

    //let mut document_indices = vec![];
    let mut document_indices = Vec::<u32>::new();
    let mut all_embeddings = vec![];

    let mut query = db.query("SELECT document.rowid,chunk.embedding FROM document,chunk
        WHERE document.hash = chunk.hash
        ORDER BY document.rowid")?;

    let mut results = query.iter((), |row| {
            Ok((
                row.get::<_, u32>(0)?,
                row.get::<_, Vec<u8>>(1)?,
            ))
        })?;

    let mut done = false;
    let mut batch = 0;
    let mut tmpfiles = vec![];
    let centers_cpu = centers.to_device(&Device::Cpu)?;
    while !done {
        match results.next() {
            Some(result) => {
                let (id, embedding) = result?;
                let t = Tensor::from_f32_bytes(&embedding, 128, &Device::Cpu)?;
                let split = split_tensor(&t);
                let m = split.len();
                all_embeddings.extend(split);

                for _ in 0..m {
                    document_indices.push(id as u32);
                }
                batch += m;

                if batch >= 0x10000 {
                    let now = std::time::Instant::now();
                    let data = Tensor::cat(&all_embeddings, 0)?.to_device(&device).unwrap();

                    let sim = data.matmul(&centers.transpose(D::Minus1, D::Minus2)?)?;
                    let cluster_assignments = sim.argmax(D::Minus1)?.to_device(&Device::Cpu)?;
                    mmuls_total += now.elapsed().as_millis();

                    let now = std::time::Instant::now();
                    let mut writer = merger::Writer::new().unwrap();

                    let mut pairs: Vec<(usize, u32)> = cluster_assignments
                        .to_vec1::<u32>()?
                        .iter()
                        .enumerate()
                        .map(|(i, &bucket)| (i, bucket))
                        .collect();
                    pairs.sort_by_key(|&(_, bucket)| bucket);

                    let mut indices_bytes: Vec<u8> = Vec::with_capacity(batch * 4);
                    let mut residuals_bytes: Vec<u8> = Vec::with_capacity(batch * 64);
                    let n = pairs.len();
                    let (_, mut prev_bucket) = pairs[0];
                    let mut count = 0;
                    let mut done = false;

                    for i in 0.. {
                        let (sample, bucket) = if i < n {
                            pairs[i]
                        } else {
                            done = true;
                            (0, std::u32::MAX)
                        };

                        if  (bucket != prev_bucket || done) && count > 0 {
                            assert!(prev_bucket < bucket);
                            writer.write_record(prev_bucket, count, &indices_bytes, &residuals_bytes)?;

                            indices_bytes.clear();
                            residuals_bytes.clear();
                            prev_bucket = bucket;
                            count = 0;
                        }

                        if done {
                            break;
                        }
 
                        let doc_idx = document_indices[sample];
                        indices_bytes.extend_from_slice(&doc_idx.to_ne_bytes());
                        let center = centers_cpu.get(bucket as usize)?;
                        let residual = (all_embeddings[sample].get(0) - &center)?;
                        let residual_quantized = residual.compand()?.quantize(4)?.to_q4_bytes()?;
                        residuals_bytes.extend(&residual_quantized);
                        count += 1;
                    }
                    tmpfiles.push(writer.finish()?);
                    writes_total += now.elapsed().as_millis();
                    bar.inc(batch as u64);

                    document_indices.clear();
                    all_embeddings.clear();
                    batch = 0;
                }

            }
            None => {
                done = true;
            }
        }
    }
    bar.finish();
    println!("mmuls took {} ms.", mmuls_total);
    println!("writes took {} ms.", writes_total);
    println!("merge all");
    let mut merger = merger::Merger::from_tempfiles(tmpfiles)?;
    for result in &mut merger {
        let entry = result?;
        let center = centers_cpu.get(entry.value as usize)?;
        let center_bytes = center.to_f32_bytes()?;
        db.add_bucket(entry.value, &center_bytes, &entry.tags, &entry.data).unwrap();
    }
    Ok(())
}

fn fulltext_search(
    db: &DB,
    q: &String
) -> Result<Vec<u32>> {
    let mut fts_idxs = vec![];
    let mut query = db.query("SELECT rowid,bm25(document_fts) AS score
        FROM document_fts
        WHERE document_fts MATCH ?1 ORDER BY score")?;

    let q: String = q.chars()
         .filter(|c| c.is_alphanumeric() || c.is_whitespace())
         .collect();

    let results = query.iter((&q,), |row| {
            Ok((
                row.get::<_, u32>(0)?,
            ))
        })?;
    for result in results {
        let (rowid, ) = result?;
        fts_idxs.push(rowid);
    }
    println!("full text found {} matches", fts_idxs.len());
    Ok(fts_idxs)
}

fn reciprocal_rank_fusion(
    list1: &[u32],
    list2: &[u32],
    k: f64,
) -> Vec<(u32, f64)> {
    let mut scores: HashMap<u32, f64> = HashMap::new();

    for (rank, &doc_id) in list1.iter().enumerate() {
        let score = 1.0 / (k + rank as f64);
        *scores.entry(doc_id).or_insert(0.0) += score;
    }

    for (rank, &doc_id) in list2.iter().enumerate() {
        let score = 1.0 / (k + rank as f64);
        *scores.entry(doc_id).or_insert(0.0) += score;
    }

    let mut results: Vec<(u32, f64)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort descending by score
    results
}


fn match_centroids(
        bucket_sizes_query: &mut Query,
        bucket_query: &mut Query,
        query_embeddings: &Tensor,
        centers: &Tensor,
        report: bool) -> Result<Vec<(f32, u32)>> {
    let now = std::time::Instant::now();

    let k = 32;
    let t_prime = 40000;

    let cutoff = if report { 0.5 } else { 0.0 };
    let top_k = if report { 10 } else { 100 };

    let sizes = bucket_sizes_query.u32_vec()?;

    let device = query_embeddings.device();
    let centers = centers.to_device(&device)?;

    let query_centroid_similarity = query_embeddings.matmul(&centers.transpose(D::Minus1, D::Minus2)?).unwrap();
    let query_centroid_similarity = query_centroid_similarity.to_device(&Device::Cpu)?;

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
    println!("finding top-{} out of {} clusters took {} ms.", topk_clusters.len(), n, now.elapsed().as_millis());

    let now = std::time::Instant::now();
    let mut all = vec![];
    let mut count = 0;

    let mut all_document_embeddings = vec![];
    for i in topk_clusters {
        let (document_indices, document_embeddings) = bucket_query.point3(i)?;
        let center = centers.get(i as usize)?;

        let document_indices = u8_to_vec_u32(&document_indices);

        //let residuals = Tensor::from_q4_bytes(&document_embeddings, 128, &device)?.dequantize(4)?.inv_compand()?;
        let residuals = Tensor::from_companded_q4_bytes(&document_embeddings, 128, &device)?;
        let embeddings = residuals.broadcast_add(&center)?;
        //let embeddings = embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(0)?.sqrt()?)?;
        all_document_embeddings.push(embeddings);

        let (m, _) = residuals.dims2()?;
        for j in 0..m {
            all.push((document_indices[j], count));
            count += 1;
        }
    }
    println!("dense matrix fill took {} ms.", now.elapsed().as_millis());
    let now = std::time::Instant::now();

    let all_document_embeddings = Tensor::cat(&all_document_embeddings, 0).unwrap();
    let all_document_embeddings = all_document_embeddings.to_device(query_embeddings.device())?;

    println!("tensor stacking took {} ms.", now.elapsed().as_millis());

    let now = std::time::Instant::now();
    let sim = query_embeddings.matmul(&all_document_embeddings.t()?)?.transpose(0, 1).unwrap();
    let sim = sim.to_device(&Device::Cpu)?;

    println!("sim mmul took {} ms.", now.elapsed().as_millis());
    let now = std::time::Instant::now();

    all.sort();
    all.push((std::u32::MAX, 0)); // sentinel, triggers push of last element

    let mut last = std::u32::MAX;
    let mut current = Tensor::zeros((n,), DType::F32, &Device::Cpu)?;

    let mut heap2 = MinHeap::new();
    let mut unique_docs = 0;

    for (idx, i) in all {

        if last != idx {
            unique_docs += 1;
            let score = current.mean(0)?.to_scalar::<f32>()?;
            if score >= cutoff {
                let score_as_u32 = (1000.0 * score) as u32;
                let elem = (score_as_u32, last);
                if heap2.len() < top_k {
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
    println!("scoring {} documents took {} ms.", unique_docs, now.elapsed().as_millis());
    println!("");

    let mut results = vec![];
    while let Some((score_as_u32, idx)) = heap2.pop() {
        results.push((score_as_u32 as f32 / 1000.0, idx));
    }
    results.reverse();
    Ok(results)
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


fn scan_documents_dir(db: &DB, dirname: &str) -> Result<()> {

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

#[derive(Debug, Deserialize)]
struct Record {
    name: String,
    body: String,
}


fn read_csv(db: &DB, csvname: &str) -> Result<()> {

    println!("register documents from CSV {}...", csvname);

    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    for result in rdr.deserialize() {
        let record: Record = result?;
        let filename = record.name;
        let body = record.body;

        let mut hasher = Sha256::new();
        hasher.update(&body);
        let hash = format!("{:x}", hasher.finalize());
        db.add_doc(&filename, &hash, &body).unwrap();
    }

    Ok(())
}

struct Embedder {
    tokenizer: Tokenizer,
    model: t5::T5EncoderModel,
}

impl Embedder {
    fn new(device: &Device) -> Self {
        let (builder, tokenizer) = T5ModelBuilder::load().unwrap();
        let model = builder.build_encoder(&device).unwrap();
        Self {
            tokenizer,
            model
        }
    }
    fn embed(self: &Self, text: &str) -> Result<Tensor> {
        let now = std::time::Instant::now();
        let tokens = self.tokenizer
            .encode(text, true)
            .map_err(E::msg).unwrap()
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], self.model.device()).unwrap().unsqueeze(0).unwrap();
        let embeddings = self.model.forward(&token_ids).unwrap();
        println!("embedder took {} ms.", now.elapsed().as_millis());
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
    pub fn new(db_fn: &str) -> Self {
        //let connection = Connection::open_in_memory().unwrap();
        let connection = Connection::open(db_fn).unwrap();
        connection.pragma_update(None, "synchronous", "OFF").unwrap();

        let query = "CREATE TABLE IF NOT EXISTS document(filename TEXT PRIMARY KEY,
            hash TEXT NOT NULL, body TEXT, UNIQUE(filename, hash))";
        connection.execute(query, ()).unwrap();

        let query = "CREATE VIRTUAL TABLE IF NOT EXISTS document_fts USING fts5(body, content='document', content_rowid='rowid')";
        connection.execute(query, ()).unwrap();

        println!("rebuild...");
        let query = "INSERT INTO document_fts(document_fts) VALUES('rebuild')";
        connection.execute(query, ()).unwrap();
        println!("rebuild done");

        let query = "CREATE TABLE IF NOT EXISTS chunk(hash TEXT PRIMARY KEY NOT NULL, embedding BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS bucket(id INTEGER PRIMARY KEY,
            center BLOB NOT NULL, indices BLOB NOT NULL, residuals BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE INDEX IF NOT EXISTS bucket_index ON bucket(id)";
        connection.execute(query, ()).unwrap();

        Self { connection }
    }

    fn execute(self: &Self, sql: &str) -> SQLResult<()> {
        self.connection.execute(sql, ()).unwrap();
        Ok(())
    }

    fn query(self: &Self, sql: &str) -> SQLResult<Query> {
        let stmt = self.connection.prepare(&sql)?;
        Ok(Query { stmt })
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
        let stmt = self.connection.prepare("SELECT filename,body FROM document WHERE rowid = ?1")?;
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

    fn iter<'stmt, T1, T2, F>(&'stmt mut self, args: T1, map_fn: F) -> SQLResult<impl Iterator<Item = SQLResult<T2>> + 'stmt>
        where
            F: FnMut(&Row) -> SQLResult<T2> + 'stmt,
            T1: rusqlite::Params, {
        self.stmt.query_map(args, map_fn)
    }



    fn point<'stmt, T1, T2, F>(&'stmt mut self, args: T1, map_fn: F) -> SQLResult<T2>
        where F: FnOnce(&Row) -> SQLResult<T2> + 'stmt, T1: rusqlite::Params,
    {
        self.stmt.query_row(args, map_fn)
    }


    fn iter1(&mut self) -> SQLResult<impl Iterator<Item = SQLResult<(String, String)>> + '_> {
        self.stmt.query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
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
    eprintln!("Usage: {} scan | readcsv <file> | embed | index | query <text> | hybrid <text> | querycsv <file> <results-file>", arg0);
    std::process::exit(1)
}

fn main() -> Result<()> {

    //let device = Device::Cpu;
    let device = Device::new_metal(0)?;
    let embedder = Embedder::new(&device);

    let db = DB::new("mydb.sqlite");

    let mut center_query = db.make_bucket_center_query().unwrap();
    let mut bucket_sizes_query = db.make_bucket_sizes_query().unwrap();
    let mut bucket_query = db.make_bucket_residuals_query().unwrap();
    let mut body_query = db.make_document_body_query().unwrap();

    let args: Vec<String> = env::args().collect();
    if args.len() == 3 && args[1] == "scan" {

        scan_documents_dir(&db, &args[2]).unwrap();

    } else if args.len() == 3 && args[1] == "readcsv" {

        read_csv(&db, &args[2]).unwrap();

    } else if args.len() == 2 && &args[1] == "embed" {

        let mut query = db.make_query().unwrap();
        let embedding_iter = Gatherer::new(&mut query, &embedder);
        for (hash, embeddings) in embedding_iter {
            println!("for hash {} {:?}", hash, embeddings.dims2().unwrap());
            let bytes = embeddings.to_f32_bytes()?;
            db.add_chunk(&hash, &bytes).unwrap();
        }

    } else if args.len() == 2 && &args[1] == "index" {

        db.execute("DELETE from bucket").unwrap();

        let chunks_count = db.query("SELECT count(hash) FROM chunk")?.point((), |row| {
                Ok( row.get::<_, f32>(0)?,)
            }).unwrap();
        println!("chunks_count {}", chunks_count);

        let mut avg_query = db.query("SELECT avg(length(embedding))
                FROM chunk WHERE rowid IN (SELECT rowid FROM chunk ORDER BY RANDOM() LIMIT 1000)").unwrap();
        let avg_len = avg_query.point((), |row| {
                Ok(
                    row.get::<_, f32>(0)?,
                )
            })?;
        let avg_embeddings = (avg_len / (128.0 * 4.0)).ceil();
        println!("avg embeddings per chunk {}", avg_embeddings);
        let est_total_embeddings = chunks_count * avg_embeddings;
        let subset_size = chunks_count.sqrt().ceil();
        println!("est_total_embeddings={} samling subset size {}", est_total_embeddings, subset_size);

        let mut kmeans_query1 = db.query("SELECT chunk.embedding FROM chunk WHERE chunk.rowid IN
                (SELECT chunk.rowid FROM chunk ORDER BY RANDOM() LIMIT ?1)")?;

        let mut all_embeddings = vec![];
        println!("read embeddings...");
        for embedding in kmeans_query1.iter((subset_size,), |row| {
            Ok( row.get::<_, Vec<u8>>(0)? )
        })? {
            let t = Tensor::from_f32_bytes(&embedding?, 128, &Device::Cpu)?;
            let split = split_tensor(&t);
            all_embeddings.extend(split);
        }
        let matrix = Tensor::cat(&all_embeddings, 0)?.to_device(&device).unwrap();
        let now = std::time::Instant::now();
        let log2_k = (16.0 * est_total_embeddings.sqrt()).log(2.0).floor() as u32;
        let k = 1 << log2_k;
        println!("est_total_embeddings={} k={}", est_total_embeddings, k);
        let (centers, idxs) = kmeans(&matrix, k as usize, 5, &device)?;
        println!("kmeans took {} ms.", now.elapsed().as_millis());
        println!("idxs {}", idxs);

        println!("write buckets...");
        let now = std::time::Instant::now();
        write_buckets(&db, &centers, &device).unwrap();
        println!("write buckets took {} ms.", now.elapsed().as_millis());

    } else if args.len() >= 3 && (args[1] == "query" || args[1] == "hybrid") {

        let q = &args[2..].join(" ");

        let fts_idxs = if args[1] == "hybrid" {
            println!("Doing full text search for: {}", q);
            fulltext_search(&db, &q)?
        } else {
            [].to_vec()
        };

        println!("Doing semantic search for: {}", q);
        let mut centers = vec![];
        for center in center_query.iter((), |row| {
            Ok( row.get::<_, Vec<u8>>(0)? )
        })? {
            let t = Tensor::from_f32_bytes(&center?, 128, &Device::Cpu)?.flatten_all()?;
            centers.push(t);
        }
        let centers = Tensor::stack(&centers, 0)?;

        let qe = embedder.embed(q)?.get(0)?;
        let sem_matches = match_centroids(&mut bucket_sizes_query, &mut bucket_query, &qe, &centers, true).unwrap();
        let sem_idxs: Vec<u32> = sem_matches.iter().map(|&(_, idx)| idx).collect();
        println!("semantic search found {} matches", sem_idxs.len());

        let fused = reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0);

        for (idx, _score) in fused {
            let (_, body) = body_query.point((idx,), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                ))
            })?;

            println!("=============================");
            println!("{}", body);
            println!("");
        }

    } else if args.len() >= 4 && (args[1] == "querycsv" || args[1] == "hybridcsv" || args[1] == "fulltextcsv") {

        let use_fulltext = args[1] == "hybridcsv" || args[1] == "fulltextcsv";
        let use_semantic = args[1] != "fulltextcsv";

        let mut centers = vec![];
        for center in center_query.iter4()? {
            let t = Tensor::from_f32_bytes(&center?, 128, &Device::Cpu)?.flatten_all()?;
            centers.push(t);
        }
        let centers = Tensor::stack(&centers, 0)?;

        let csvname = &args[2];
        let file = File::open(csvname)?;
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(false)
            .from_reader(file);

        let file = File::create(&args[3]).unwrap();
        let mut writer = BufWriter::new(file);

        for result in rdr.deserialize() {
            let record: (String, String) = result?;
            let key = record.0;
            let question = record.1;

            let fts_idxs = if use_fulltext {
                println!("Doing full text search for: {}", question);
                fulltext_search(&db, &question)?
            } else {
                [].to_vec()
            };

            let sem_matches = if use_semantic {
                let qe = embedder.embed(&question)?.get(0)?;
                match_centroids(&mut bucket_sizes_query, &mut bucket_query, &qe, &centers, false).unwrap()
            } else {
                [].to_vec()
            };
            let sem_idxs: Vec<u32> = sem_matches.iter().map(|&(_, idx)| idx).collect();
            if use_semantic {
                println!("semantic search found {} matches", sem_idxs.len());
            }

            let fused = reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0);

            let mut filenames = vec![];
            for (idx, _score) in fused {
                let (filename, _) = body_query.point((idx,), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                    ))
                })?;
                filenames.push(filename);
            }

            write!(writer, "{}\t", key).unwrap();
            for filename in &filenames {
                write!(writer, "{},", filename).unwrap();
            }
            write!(writer, "\n").unwrap();
            writer.flush().unwrap();
        }

    } else {
        usage(&args[0]);
    }

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
