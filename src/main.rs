use anyhow::Result;
use log::debug;
use log::{Level, LevelFilter, Metadata, Record};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use text_splitter::TextSplitter;
use uuid::Uuid;

mod histogram;

use witchcraft::DB;

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Trace
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
        db.add_doc(&uuid, None, &metadata, &body, Some(lens))
            .unwrap();
    }

    Ok(())
}

pub fn bulk_search(
    db: &DB,
    embedder: Option<&witchcraft::Embedder>,
    csvname: std::path::PathBuf,
    outputname: std::path::PathBuf,
    use_fulltext: bool,
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

        debug!("searching for: {}", question);
        let now = std::time::Instant::now();
        let fts_start = std::time::Instant::now();
        let fts_matches = if use_fulltext {
            witchcraft::fulltext_search(db, &question, top_k, None)?
        } else {
            vec![]
        };
        if use_fulltext {
            debug!(
                "fulltext search took {} ms.",
                fts_start.elapsed().as_millis()
            );
        }

        let sem_matches = if let Some(embedder) = embedder {
            let now = std::time::Instant::now();
            let (qe, _offsets) = embedder.embed(&question)?;
            let qe = qe.get(0)?;
            let embedder_latency_ms = now.elapsed().as_millis() as u32;
            embedder_histogram.record(embedder_latency_ms);

            let match_start = std::time::Instant::now();
            let matches = witchcraft::match_centroids(db, &qe, 0.0, top_k, None).unwrap();
            debug!(
                "match_centroids call took {} ms.",
                match_start.elapsed().as_millis()
            );
            matches
        } else {
            vec![]
        };
        let sem_idxs: Vec<witchcraft::DocPtr> = sem_matches
            .iter()
            .map(|&(_, idx, sub_idx)| (idx, sub_idx))
            .collect();

        let fusion_start = std::time::Instant::now();
        let mut fused = if use_fulltext {
            let fts_idxs: Vec<witchcraft::DocPtr> = fts_matches
                .iter()
                .map(|&(_, idx, sub_idx)| (idx, sub_idx))
                .collect();
            witchcraft::reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
        } else {
            sem_idxs
        };
        fused.truncate(top_k);
        debug!(
            "rank fusion took {} ms.",
            fusion_start.elapsed().as_millis()
        );

        let metadata_start = std::time::Instant::now();
        let mut metadatas = vec![];
        for (idx, _sub_idx) in &fused {
            let metadata = metadata_query.query_row((*idx,), |row| row.get::<_, String>(0))?;
            metadatas.push(metadata);
        }
        debug!(
            "fetching {} metadata took {} ms.",
            metadatas.len(),
            metadata_start.elapsed().as_millis()
        );
        let total_ms = now.elapsed().as_millis();
        histogram.record(total_ms.try_into().unwrap());
        debug!("search took {} ms in total", now.elapsed().as_millis());

        write!(writer, "{}\t", key).unwrap();
        for metadata in &metadatas {
            let data: CorpusMetaData = serde_json::from_str(metadata)?;
            write!(writer, "{},", data.key).unwrap();
        }
        writeln!(writer).unwrap();
        writer.flush().unwrap();
    }
    if embedder.is_some() {
        println!("p95 embedder latency = {} ms", embedder_histogram.p95());
    }
    println!("p95 total search latency = {} ms", histogram.p95());
    Ok(())
}

fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Info));

    let args: Vec<String> = env::args().collect();
    let assets = std::path::PathBuf::from("assets");
    let db_name = std::path::PathBuf::from("mydb.sqlite");

    if args.len() == 3 && args[1] == "readcsv" {
        let mut db = DB::new_fast(db_name).unwrap();
        let csvname = &args[2];
        read_csv(&mut db, csvname.into()).unwrap();
    } else if args.len() == 2 && &args[1] == "embed" {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let db = DB::new_fast(db_name).unwrap();
        let _got = witchcraft::embed_chunks(&db, &embedder, None).unwrap();
    } else if args.len() == 2 && &args[1] == "index" {
        let device = witchcraft::make_device();
        let db = DB::new_fast(db_name).unwrap();
        witchcraft::index_chunks(&db, &device).unwrap();
    } else if args.len() == 2 && &args[1] == "reindex" {
        let device = witchcraft::make_device();
        let db = DB::new_fast(db_name).unwrap();
        witchcraft::full_index(&db, &device).unwrap();
    } else if args.len() >= 3 && (args[1] == "query" || args[1] == "hybrid") {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let mut cache = witchcraft::EmbeddingsCache::new(1);
        let db = DB::new_reader(db_name).unwrap();
        let q = &args[2..].join(" ");
        let use_fulltext = args[1] == "hybrid";
        let results =
            witchcraft::search(&db, &embedder, &mut cache, q, 0.7, 10, use_fulltext, None).unwrap();
        for (score, _metadata, bodies, sub_idx, _date) in results {
            let idx = (sub_idx as usize).min(bodies.len().saturating_sub(1));
            let body = &bodies[idx];
            println!("{score}: {body} @ {sub_idx}");
            println!("=============================================");
        }
    } else if args.len() >= 4
        && (args[1] == "querycsv" || args[1] == "hybridcsv" || args[1] == "fulltextcsv")
    {
        let db = DB::new_reader(db_name).unwrap();
        let use_fulltext = args[1] == "hybridcsv" || args[1] == "fulltextcsv";
        let embedder = if args[1] != "fulltextcsv" {
            let device = witchcraft::make_device();
            Some(witchcraft::Embedder::new(&device, &assets).unwrap())
        } else {
            None
        };
        let csvname = &args[2];
        let outputname = &args[3];
        bulk_search(
            &db,
            embedder.as_ref(),
            csvname.into(),
            outputname.into(),
            use_fulltext,
        )
        .unwrap();
    } else if args.len() >= 4 && &args[1] == "score" {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let mut cache = witchcraft::EmbeddingsCache::new(1);
        let sentences: Vec<String> = std::env::args().skip(3).collect();
        let scores =
            witchcraft::score_query_sentences(&embedder, &mut cache, &args[2], &sentences).unwrap();
        for (i, score) in scores.iter().enumerate() {
            println!("`{}': score={}", args[3 + i], *score);
        }
    } else if args.len() == 2 && &args[1] == "clear" {
        let mut db = DB::new_fast(db_name).unwrap();
        db.clear();
    } else {
        eprintln!("\n*** Usage: {} clear | readcsv <file> | embed | index | reindex | query <text> | hybrid <text> | querycsv <file> <results-file> ***\n", args[0]);
    };
    Ok(())
}
