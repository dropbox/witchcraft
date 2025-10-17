use anyhow::Result;
use csv;
use log::{Level, LevelFilter, Metadata, Record};
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use text_splitter::TextSplitter;
use uuid::Uuid;

mod histogram;
mod warp;

use warp::DB;

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

#[derive(Debug, Deserialize)]
struct CSVRecord {
    name: String,
    body: String,
}

#[derive(Serialize, Deserialize)]
struct CorpusMetaData {
    key: String,
}

fn split_doc(body: String) -> Vec<String> {
    let max_characters = 300;
    let splitter = TextSplitter::new(max_characters);
    splitter
        .chunks(&body)
        .map(|body| format!("{body}\n").to_string())
        .collect()
}


pub fn read_csv(db: &mut DB, csvname: std::path::PathBuf) -> Result<()> {
    println!("register documents from CSV...");

    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    for result in rdr.deserialize() {
        let record: CSVRecord = result?;
        let metadata = CorpusMetaData { key: record.name };
        let metadata = serde_json::to_string(&metadata)?;
        let body = record.body;

        let bodies = split_doc(body.clone());
        let lens = bodies.iter().map(|b| b.chars().count()).collect();
        let body = bodies.join("");
        let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
        db.add_doc(&uuid, None, &metadata, &body, Some(lens)).unwrap();
    }

    Ok(())
}

pub fn bulk_search(
    db: &DB,
    embedder: &warp::Embedder,
    csvname: std::path::PathBuf,
    outputname: std::path::PathBuf,
    use_fulltext: bool,
    use_semantic: bool,
) -> Result<()> {
    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    let file = File::create(outputname).unwrap();
    let mut writer = BufWriter::new(file);

    let mut metadata_query = db.query("SELECT metadata FROM document WHERE rowid = ?1")?;
    let mut histogram = histogram::Histogram::new(10000);
    let mut embedder_histogram = histogram::Histogram::new(10000);

    for result in rdr.deserialize() {
        let record: (String, String) = result?;
        let key = record.0;
        let question = record.1;
        let top_k = 100;

        info!("searching for: {}", question);
        let now = std::time::Instant::now();
        let fts_matches = if use_fulltext {
            warp::fulltext_search(&db, &question, top_k, None)?
        } else {
            [].to_vec()
        };

        let sem_matches = if use_semantic {
            let now = std::time::Instant::now();
            let (qe, _offsets) = embedder.embed(&question)?;
            let qe = qe.get(0)?;
            qe.device().synchronize().unwrap();
            let embedder_latency_ms = now.elapsed().as_millis() as u32;
            embedder_histogram.record(embedder_latency_ms);
            debug!("embedder took {} ms.", now.elapsed().as_millis());
            warp::match_centroids(&db, &qe, 0.0, top_k, None).unwrap()
        } else {
            [].to_vec()
        };
        let sem_idxs: Vec<u32> = sem_matches.iter().map(|&(_, idx, _)| idx).collect();
        let mut fused = if use_fulltext {
            let fts_idxs: Vec<u32> = fts_matches.iter().map(|&(_, idx, _)| idx).collect();
            warp::reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
        } else {
            sem_idxs
        };
        fused.truncate(top_k);

        let mut metadatas = vec![];
        for idx in fused {
            let metadata = metadata_query.query_row((idx,), |row| Ok(row.get::<_, String>(0)?))?;
            metadatas.push(metadata);
        }
        let total_ms = now.elapsed().as_millis();
        histogram.record(total_ms.try_into().unwrap());
        debug!("search took {} ms in total", now.elapsed().as_millis());

        write!(writer, "{}\t", key).unwrap();
        for metadata in &metadatas {
            let data: CorpusMetaData = serde_json::from_str(metadata)?;
            write!(writer, "{},", data.key).unwrap();
        }
        write!(writer, "\n").unwrap();
        writer.flush().unwrap();
    }
    println!("p95 embedder latency = {} ms", embedder_histogram.p95());
    println!("p95 total search latency = {} ms", histogram.p95());
    Ok(())
}

fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Debug));

    let args: Vec<String> = env::args().collect();
    let device = warp::make_device();
    let assets = std::path::PathBuf::from("assets");
    let embedder = warp::Embedder::new(&device, &assets).unwrap();
    let mut cache = warp::EmbeddingsCache::new(1);
    let db_name = "mydb.sqlite";
    let db_name = std::path::PathBuf::from(db_name);

    if args.len() == 3 && args[1] == "readcsv" {
        let mut db = DB::new(db_name).unwrap();
        let csvname = &args[2];
        read_csv(&mut db, csvname.into()).unwrap();
    } else if args.len() == 2 && &args[1] == "embed" {
        let db = DB::new(db_name).unwrap();
        let _got = warp::embed_chunks(&db, &embedder, None).unwrap();
    } else if args.len() == 2 && &args[1] == "index" {
        let db = DB::new(db_name).unwrap();
        warp::index_chunks(&db, &device).unwrap();
    } else if args.len() >= 3 && (args[1] == "query" || args[1] == "hybrid") {
        let db = DB::new_reader(db_name).unwrap();
        let q = &args[2..].join(" ");
        let use_fulltext = args[1] == "hybrid";
        let results =
            warp::search(&db, &embedder, &mut cache, &q, 0.75, 10, use_fulltext, None).unwrap();
        for (score, _metadata, body, body_idx) in results {
            println!("{score}: {body} @ {body_idx}");
            println!("=============================================");
        }
    } else if args.len() >= 4
        && (args[1] == "querycsv" || args[1] == "hybridcsv" || args[1] == "fulltextcsv")
    {
        let db = DB::new_reader(db_name).unwrap();
        let use_fulltext = args[1] == "hybridcsv" || args[1] == "fulltextcsv";
        let use_semantic = args[1] != "fulltextcsv";
        let csvname = &args[2];
        let outputname = &args[3];
        bulk_search(
            &db,
            &embedder,
            csvname.into(),
            outputname.into(),
            use_fulltext,
            use_semantic,
        )
        .unwrap();
    } else if args.len() >= 4 && &args[1] == "score" {
        let sentences: Vec<String> = std::env::args().skip(3).collect();
        let scores =
            warp::score_query_sentences(&embedder, &mut cache, &args[2], &sentences).unwrap();
        for (i, score) in scores.iter().enumerate() {
            println!("`{}': score={}", args[3 + i], *score);
        }
    } else if args.len() == 2 && &args[1] == "clear" {
        let mut db = DB::new(db_name).unwrap();
        let _ = db.clear();
    } else {
        eprintln!("\n*** Usage: {} clear | readcsv <file> | embed | index | query <text> | hybrid <text> | querycsv <file> <results-file> ***\n", args[0]);
    };
    Ok(())
}
