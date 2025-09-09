use anyhow::Result;
use csv;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

mod histogram;
mod warp;

use warp::DB;

#[derive(Debug, Deserialize)]
struct Record {
    name: String,
    body: String,
}

#[derive(Serialize, Deserialize)]
struct CorpusMetaData {
    key: String,
}

pub fn read_csv(db: &mut DB, csvname: std::path::PathBuf) -> Result<()> {
    println!("register documents from CSV...");

    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    for result in rdr.deserialize() {
        let record: Record = result?;
        let metadata = CorpusMetaData { key: record.name };
        let metadata = serde_json::to_string(&metadata)?;
        let body = record.body;
        db.add_doc(&metadata, &body).unwrap();
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

    let mut metadata_query = db.query("SELECT metadata FROM document WHERE rowid = ?1");
    let mut histogram = histogram::Histogram::new(10000);
    let mut embedder_histogram = histogram::Histogram::new(10000);

    for result in rdr.deserialize() {
        let record: (String, String) = result?;
        let key = record.0;
        let question = record.1;

        println!("\nSearching for: {}", question);
        let now = std::time::Instant::now();
        let fts_idxs = if use_fulltext {
            warp::fulltext_search(&db, &question, None)?
        } else {
            [].to_vec()
        };

        let sem_matches = if use_semantic {
            let now = std::time::Instant::now();
            let qe = embedder.embed(&question)?.get(0)?;
            qe.device().synchronize().unwrap();
            let embedder_latency_ms = now.elapsed().as_millis() as u32;
            embedder_histogram.record(embedder_latency_ms);
            println!("embedder took {} ms.", now.elapsed().as_millis());
            warp::match_centroids(&db, &qe, 0.0, 100, None).unwrap()
        } else {
            [].to_vec()
        };
        let sem_idxs: Vec<u32> = sem_matches.iter().map(|&(_, idx)| idx).collect();
        if use_semantic {
            println!("semantic search found {} matches", sem_idxs.len());
        }

        let fused = if use_fulltext && use_semantic {
            warp::reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
        } else if use_fulltext {
            fts_idxs
        } else {
            sem_idxs
        };

        let mut metadatas = vec![];
        for idx in fused {
            let metadata = metadata_query.query_row((idx,), |row| Ok(row.get::<_, String>(0)?))?;
            metadatas.push(metadata);
        }
        let total_ms = now.elapsed().as_millis();
        histogram.record(total_ms.try_into().unwrap());
        println!("search took {} ms in total", now.elapsed().as_millis());

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
    let args: Vec<String> = env::args().collect();
    let mut db = DB::new("mydb.sqlite");
    let device = warp::make_device();
    let embedder = warp::Embedder::new(&device);
    let mut cache = warp::EmbeddingsCache::new(1);

    if args.len() == 3 && args[1] == "readcsv" {
        let csvname = &args[2];
        read_csv(&mut db, csvname.into()).unwrap();
    } else if args.len() == 2 && &args[1] == "embed" {
        let _got = warp::embed_chunks(&db, &device, None).unwrap();
    } else if args.len() == 2 && &args[1] == "index" {
        warp::index_chunks(&db, &device).unwrap();
    } else if args.len() >= 3 && (args[1] == "query" || args[1] == "hybrid") {
        let q = &args[2..].join(" ");
        let use_fulltext = args[1] == "hybrid";
        let results =
            warp::search(&db, &embedder, &mut cache, &q, 0.75, 10, use_fulltext, None).unwrap();
        for (score, filename, body) in results {
            println!("{} : {} : {}", score, filename, body);
        }
    } else if args.len() >= 4
        && (args[1] == "querycsv" || args[1] == "hybridcsv" || args[1] == "fulltextcsv")
    {
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
    } else {
        eprintln!("\n*** Usage: {} scan | readcsv <file> | embed | index | query <text> | hybrid <text> | querycsv <file> <results-file> ***\n", args[0]);
    };
    Ok(())
}
