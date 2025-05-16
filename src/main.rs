use std::env;
use rand::prelude::*;
use rusqlite::{Connection, Statement, Result as SQLResult};
use scan_dir::ScanDir;
use sha2::{Sha256, Digest};
use min_heap::MinHeap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;

//use candle_transformers::models::t5;
mod t5;

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
        let device = Device::Cpu;
        let path = PathBuf::from(r"/Users/jhansen/src/xtr-warp/foo.safetensors");
        let weights_filename = vec![path];
        let config = std::fs::read_to_string("/Users/jhansen/src/xtr-warp/xtr-base-en/config.json")?;
        let config: t5::Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file("/Users/jhansen/src/xtr-warp/xtr-base-en/tokenizer.json").map_err(E::msg)?;
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

fn kmeans(data: &Tensor, k: usize, max_iter: u32, device: &Device) -> Result<(Tensor, Tensor)> {
    let (n, _) = data.dims2()?;
    //let mut rng = rand::rng();
    let mut rng = SmallRng::seed_from_u64(0);
    let mut indices = (0..n).collect::<Vec<_>>();
    indices.shuffle(&mut rng);

    let centroid_idx = indices[..k]
        .iter()
        .copied()
        .map(|x| x as u32)
        .collect::<Vec<_>>();

    let centroid_idx_tensor = Tensor::from_slice(centroid_idx.as_slice(), (k,), device)?;
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
            let cluster_data = data.index_select(&indices, 0)?;
            let sum = cluster_data.sum(0)?;
            let normalized = sum.broadcast_div(&sum.sqr()?.sum_keepdim(0)?.sqrt()?);
            centers_vec.push(normalized?);
        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }
    Ok((centers, cluster_assignments))
}

fn write_buckets(db: &DB,
        data: &Tensor,
        document_indices: &Tensor,
        centers: &Tensor,
        device: &Device) -> Result<()> {

    let (k, _) = centers.dims2()?;
    println!("k={}", k);

    let sim = data.matmul(&centers.transpose(D::Minus1, D::Minus2)?)?;
    let cluster_assignments = sim.argmax(D::Minus1)?;
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
        let cluster_data = data.index_select(&data_indices, 0)?;

        let vec = center.to_vec1::<f32>();
        let center_bytes = vec_f32_to_u8_vec(&vec?);

        println!("indices vec {}", data_indices);
        let document_subset = document_indices.index_select(&data_indices, 0)?;
        let vec = document_subset.to_vec1::<u32>();
        let indices_bytes = vec_u32_to_u8_vec(&vec?);

        let vec = cluster_data.to_vec2::<f32>();
        let embeddings_bytes = vecvec_f32_to_u8_vec(&vec?);

        db.add_bucket(i as u32, &center_bytes, &indices_bytes, &embeddings_bytes).unwrap();
    }
    Ok(())
}

fn topk(x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    // Sorted descending
    let sorted_indices = x.arg_sort_last_dim(false)?;
    let topk_indices = sorted_indices.narrow(D::Minus1, 0, k)?.contiguous()?;
    Ok(( x.gather(&topk_indices, D::Minus1)?, topk_indices))
}

fn match_centroids(bucket_query: &mut Query, query_embeddings: &Tensor, centers: &Tensor) -> Result<()> {
    println!("******************** LOOKUP **********************\n");
    let sim = query_embeddings.matmul(&centers.transpose(D::Minus1, D::Minus2)?).unwrap();

    let k = 2;
    let (topk_plus1_scores, topk_plus1_clusters) = topk(&sim, k + 1)?;
    let topk_clusters = topk_plus1_clusters.i((.., 0..k))?;
    let (m, n) = topk_clusters.dims2()?;

    let missing_similarities = topk_plus1_scores.i((.., k..k+1))?.reshape(m)?;
    println!("missing_similarities {}", missing_similarities);

    let topk_clusters = topk_clusters.reshape(m * n)?;
    let mut topk_clusters = topk_clusters.to_vec1::<u32>()?;
    topk_clusters.sort();
    topk_clusters.dedup();

    let mut heap = MinHeap::new();

    let mut all_document_embeddings = vec![];
    for i in topk_clusters {
        for result in bucket_query.iter3(i)? {
            let (document_indices, document_embeddings) = result?;
            let document_indices = u8_to_vec_u32(&document_indices);
            let document_embeddings = u8_to_tensor2d_f32(&document_embeddings, 128);
            let (m, _) = document_embeddings.dims2()?;
            for j in 0..m {
                heap.push((document_indices[j], all_document_embeddings.len()));
                all_document_embeddings.push(document_embeddings.get(j)?);
            }
        }
    }

    let all_document_embeddings = Tensor::stack(all_document_embeddings.as_slice(), 1)?;
    let sim = query_embeddings.matmul(&all_document_embeddings)?.transpose(0, 1)?;
    println!("all sim {}", sim);

    let mut last = std::u32::MAX;
    let mut current = Tensor::zeros((n,), DType::F32, &Device::Cpu)?;

    while let Some((idx, i)) = heap.pop() {

        if last != idx || heap.len() == 0 {
            println!("document={} scores={}", last, current.sum(0)?);
        }

        let row = sim.get(i)?;
        current = if idx == last {
            row.maximum(&current)?
        } else {
            row.maximum(&missing_similarities)?
        };

        last = idx;
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

fn stack_tensors(vectors: Vec<Tensor>) -> Tensor {
    Tensor::cat(&vectors, 0).unwrap() // `0` means stacking along rows (axis 0)
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
    ScanDir::files().read(dirname, |iter| {
        for (entry, _name) in iter {
            let path = entry.path();
            let filename = path.to_str().unwrap();
            let hash = compute_sha256(path.clone()).unwrap();
            println!("{} hash is {}", filename, hash);
            db.add_doc(&filename, &hash).unwrap();
        }
    }).unwrap();

    Ok(())
}

struct Document {
    filename: String,
    hash: String,
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
    documents: Box<dyn Iterator<Item = Document> + 'a>,
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
            Some(document) => {
                let filename = document.filename;
                let hash = document.hash;

                let path = PathBuf::from(filename);
                println!("read file {:?}", path);
                let file = File::open(path).unwrap();
                let mut reader = BufReader::new(file);
                let mut buffer = [0u8; 0x10000];

                loop {
                    let bytes_read = reader.read(&mut buffer).unwrap();
                    if bytes_read == 0 {
                        break;
                    }

                    let now = std::time::Instant::now();
                    let utf8 = std::str::from_utf8(&buffer[..bytes_read]).unwrap();
                    let embeddings = self.embedder.embed(utf8);
                    println!("embedder took {} ms.", now.elapsed().as_millis());
                    let normalized = normalize_l2(&embeddings.unwrap());

                    let split = split_tensor(&normalized.ok()?.get(0).ok()?);
                    doc_embedding.extend(split);
                }
                Some((hash, stack_tensors(doc_embedding)))
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
            state TEXT CHECK(state IN ('new', 'indexed')),
            hash TEXT)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS chunk(hash TEXT PRIMARY KEY, embedding BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();

        let query = "CREATE TABLE IF NOT EXISTS bucket(id INTEGER PRIMARY KEY,
            center BLOB NOT NULL, indices BLOB NOT NULL, embeddings BLOB NOT NULL)";
        connection.execute(query, ()).unwrap();
        Self { connection }
    }

    fn make_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT filename,hash FROM document WHERE state = 'new'
            ORDER BY filename")?;
        Ok(Query { stmt })
    }

    fn make_kmeans_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT document.filename,chunk.embedding FROM document,chunk
            WHERE document.hash == chunk.hash AND
            state = 'indexed'
            ORDER BY document.filename")?;
        Ok(Query { stmt })
    }

    fn make_bucket_center_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT center FROM bucket ORDER BY id")?;
        Ok(Query { stmt })
    }

    fn make_bucket_embeddings_query(self: &Self) -> SQLResult<Query> {
        let stmt = self.connection.prepare("SELECT indices,embeddings FROM bucket
            WHERE id = ?1")?;
        Ok(Query { stmt })
    }

    fn add_doc(self: &Self, filename: &str, hash: &str) -> SQLResult<()> {
        self.connection.execute("INSERT OR IGNORE INTO document VALUES(?1, 'new', ?2)", (&filename, &hash))?;
        Ok(())
    }

    fn set_embeddings(self: &Self, hash: &str, embeddings: &Vec<u8>) -> SQLResult<()> {
        let tx = self.connection.unchecked_transaction()?;
        tx.execute("INSERT OR REPLACE INTO chunk VALUES(?1, ?2)", (&hash, embeddings)).unwrap();
        tx.execute("UPDATE document set state = 'indexed' WHERE hash == ?1", (&hash, )).unwrap();
        tx.commit()
    }

    fn add_bucket(self: &Self, id: u32, center: &Vec<u8>, indices: &Vec<u8>, embeddings: &Vec<u8>) -> SQLResult<()> {
        self.connection.execute("INSERT OR REPLACE INTO bucket VALUES(?1, ?2, ?3, ?4)", (id, center, indices, embeddings))?;
        Ok(())
    }
}

impl<'connection> Query<'connection> {
    fn iter1(&mut self) -> SQLResult<impl Iterator<Item = SQLResult<Document>> + '_> {
        self.stmt.query_map([], |row| {
            Ok(Document {
                filename: row.get(0)?,
                hash: row.get(1)?,
            })
        })
    }

    fn iter2(&mut self) -> SQLResult<impl Iterator<Item = SQLResult<(String, Vec<u8>)>> + '_> {
        self.stmt.query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
            ))
        })
    }

    fn iter3(&mut self, id: u32) -> SQLResult<impl Iterator<Item = SQLResult<(Vec<u8>, Vec<u8>)>> + '_> {
        self.stmt.query_map([id], |row| {
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

fn vec_f32_to_u8_vec(data: &Vec<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * data.len() * std::mem::size_of::<f32>());
    for &val in data {
        bytes.extend(&val.to_ne_bytes()); // native-endian encoding
    }
    bytes
}

fn vecvec_f32_to_u8_vec(data: &[Vec<f32>]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * data[0].len() * std::mem::size_of::<f32>());
    for row in data {
        for &val in row {
            bytes.extend_from_slice(&val.to_ne_bytes()); // native-endian encoding
        }
    }
    bytes
}

fn vec_u32_to_u8_vec(data: &Vec<u32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * data.len() * std::mem::size_of::<u32>());
    for &val in data {
        bytes.extend_from_slice(&val.to_ne_bytes()); // native-endian encoding
    }
    bytes
}

fn vecvec_u32_to_u8_vec(data: &[Vec<u32>]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * data[0].len() * std::mem::size_of::<u32>());
    for row in data {
        for &val in row {
            bytes.extend_from_slice(&val.to_ne_bytes()); // native-endian encoding
        }
    }
    bytes
}

pub fn u8_to_tensor2d_f32(bytes: &[u8], cols: usize) -> Tensor {
    let f32_size = size_of::<f32>();

    assert!(bytes.len() % f32_size == 0);
    let total_f32s = bytes.len() / f32_size;

    let rows = total_f32s / cols;

    let mut f32s = Vec::with_capacity(total_f32s);
    for chunk in bytes.chunks_exact(f32_size) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        f32s.push(f32::from_ne_bytes(arr));
    }

    Tensor::from_vec(f32s, (rows, cols), &Device::Cpu).unwrap()
}

pub fn u8_to_tensor2d_u32(bytes: &[u8], cols: usize) -> Tensor {
    let u32_size = size_of::<u32>();

    assert!(bytes.len() % u32_size == 0);
    let total_u32s = bytes.len() / u32_size;

    let rows = total_u32s / cols;

    let mut u32s = Vec::with_capacity(total_u32s);
    for chunk in bytes.chunks_exact(u32_size) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        u32s.push(u32::from_ne_bytes(arr));
    }

    Tensor::from_vec(u32s, (rows, cols), &Device::Cpu).unwrap()
}

pub fn u8_to_tensor2d_u32_1d(bytes: &[u8]) -> Tensor {
    let u32_size = size_of::<u32>();

    assert!(bytes.len() % u32_size == 0);
    let total_u32s = bytes.len() / u32_size;

    let mut u32s = Vec::with_capacity(total_u32s);
    for chunk in bytes.chunks_exact(u32_size) {
        let arr: [u8; 4] = chunk.try_into().unwrap();
        u32s.push(u32::from_ne_bytes(arr));
    }

    Tensor::from_vec(u32s, total_u32s, &Device::Cpu).unwrap()
}

pub fn u8_to_vec_u32(bytes: &[u8]) -> Vec<u32> {
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

fn main() -> Result<()> {

    let args: Vec<String> = env::args().collect();
    let device = Device::Cpu;

    let embedder = Embedder::new();
    let db = DB::new();
    let mut query = db.make_query()?;
    let mut kmeans_query = db.make_kmeans_query()?;
    let mut bucket_query = db.make_bucket_embeddings_query()?;
    let mut center_query = db.make_bucket_center_query()?;

    register_documents(&db, "documents").unwrap();

    let embedding_iter = Gatherer::new(&mut query, &embedder);
    for (hash, embeddings) in embedding_iter {
        //println!("for hash {} {:?}", hash, embeddings.dims2().unwrap());
        //let (b, n) = embeddings.dims2().unwrap();
        let vec = embeddings.to_vec2::<f32>();
        let bytes = vecvec_f32_to_u8_vec(&vec?);
        db.set_embeddings(&hash, &bytes).unwrap();
    }

    let mut document_indices = Vec::<u32>::new();
    if args.len() == 2 && args[1] == "index" {
        let mut all_embeddings = vec![];
        let mut total = 0;
        for (document_idx, result) in kmeans_query.iter2()?.enumerate() {
            let (filename, embedding) = result?;
            println!("filename {} start at {}", filename, all_embeddings.len());
            let t = u8_to_tensor2d_f32(&embedding, 128);
            let split = split_tensor(&t);
            all_embeddings.extend(split);
            //println!("iter len {}", chunk?.embedding.len());
            let (m, n) = t.dims2()?;
            println!("shape is {}x{}", m, n);
            for _ in 0..m {
                document_indices.push(document_idx as u32);
            }
            total += m;
        }
        let matrix = stack_tensors(all_embeddings);
        println!("kmeans...");
        let now = std::time::Instant::now();
        //let (centers, idxs) = kmeans(&matrix, 1024, 5, &device)?;
        let k = 6;
        let (centers, idxs) = kmeans(&matrix, k, 5, &device)?;
        println!("kmeans took {} ms.", now.elapsed().as_millis());
        println!("idxs {}", idxs);

        println!("write buckets...");
        let document_indices = Tensor::from_vec(document_indices, total, &Device::Cpu).unwrap();
        let now = std::time::Instant::now();
        write_buckets(&db, &matrix, &document_indices, &centers, &device).unwrap();
        println!("write buckets took {} ms.", now.elapsed().as_millis());
    }

    let mut centers = vec![];
    for center in center_query.iter4()? {
        let t = u8_to_tensor2d_f32(&center?, 128);
        let split = split_tensor(&t);
        centers.extend(split);
    }
    let centers = stack_tensors(centers);

    let qe = embedder.embed("do buildings change size due to weather?")?.get(0)?;
    let _ = match_centroids(&mut bucket_query, &qe, &centers).unwrap();


    //println!("idxs {}", idxs);
    //println!("v {:?}", all_embeddings);

    //let matrix = gather_embeddings(&mut query, model, tokenizer)?;
    //let (_, idxs) = kmeans(&matrix, 16, 5, &device)?;

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}
