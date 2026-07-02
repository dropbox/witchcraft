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

#[derive(Clone)]
struct SaliencySpan {
    start: usize,
    end: usize,
    score: f32,
    token: String,
}

fn normalized_saliency(score: f32, min_score: f32, max_score: f32) -> f32 {
    let range = max_score - min_score;
    if range.abs() < 1e-6 {
        0.5
    } else {
        ((score - min_score) / range).clamp(0.0, 1.0)
    }
}

fn saliency_bg(norm: f32) -> (u8, u8, u8) {
    let norm = norm.clamp(0.0, 1.0);
    let r = 32.0 + 223.0 * norm;
    let g = 48.0 + 152.0 * norm;
    let b = 72.0 * (1.0 - norm);
    (r.round() as u8, g.round() as u8, b.round() as u8)
}

fn colorize_saliency_text(text: &str, spans: &[SaliencySpan]) -> String {
    if spans.is_empty() {
        return text.to_string();
    }

    let min_score = spans
        .iter()
        .map(|span| span.score)
        .fold(f32::INFINITY, f32::min);
    let max_score = spans
        .iter()
        .map(|span| span.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut spans = spans.to_vec();
    spans.sort_unstable_by(|a, b| (a.start, a.end).cmp(&(b.start, b.end)));

    let mut out = String::new();
    let mut cursor = 0usize;
    for span in spans {
        if span.start < cursor || span.start >= span.end || span.end > text.len() {
            continue;
        }
        if !text.is_char_boundary(span.start) || !text.is_char_boundary(span.end) {
            continue;
        }
        out.push_str(&text[cursor..span.start]);
        let norm = normalized_saliency(span.score, min_score, max_score);
        let (r, g, b) = saliency_bg(norm);
        let fg = if norm > 0.58 { "0;0;0" } else { "255;255;255" };
        out.push_str(&format!(
            "\x1b[48;2;{r};{g};{b}m\x1b[38;2;{fg}m{}\x1b[0m",
            &text[span.start..span.end]
        ));
        cursor = span.end;
    }
    out.push_str(&text[cursor..]);
    out
}

fn print_saliency(embedder: &witchcraft::Embedder, text: &str) -> Result<()> {
    let output = embedder.embed_with_gate_scores_and_tokens(text)?;
    let gate_scores = output.gate_scores.ok_or_else(|| {
        anyhow::anyhow!("encoder assets do not expose token gate scores")
    })?;
    let spans: Vec<SaliencySpan> = output.offsets
        .into_iter()
        .zip(output.tokens.into_iter())
        .zip(gate_scores.into_iter())
        .filter_map(|(((start, end), token), score)| {
            (start < end).then_some(SaliencySpan { start, end, score, token })
        })
        .collect();
    if spans.is_empty() {
        anyhow::bail!("no visible token spans to display");
    }

    let min_score = spans
        .iter()
        .map(|span| span.score)
        .fold(f32::INFINITY, f32::min);
    let max_score = spans
        .iter()
        .map(|span| span.score)
        .fold(f32::NEG_INFINITY, f32::max);

    println!(
        "gate saliency range: min={min_score:.4} max={max_score:.4} (colors are normalized per input)"
    );
    println!("{}", colorize_saliency_text(text, &spans));
    println!();
    println!("{:>3} {:>10} {:>8} {:>16} text", "#", "gate", "norm", "token");
    for (idx, span) in spans.iter().enumerate() {
        if span.end > text.len()
            || !text.is_char_boundary(span.start)
            || !text.is_char_boundary(span.end)
        {
            continue;
        }
        let text_span = text[span.start..span.end].replace('\n', "\\n");
        let norm = normalized_saliency(span.score, min_score, max_score);
        println!(
            "{idx:>3} {:>10.4} {:>7.1}% {:>16?} {:?}",
            span.score,
            100.0 * norm,
            span.token,
            text_span
        );
    }
    Ok(())
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
        db.add_doc(None, &uuid, None, &metadata, &body, Some(lens))
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
    validate_semantic_search(db, embedder)?;

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
    witchcraft::reset_bucket_io_counters();

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
            let qe = witchcraft::embed_query_for_search(embedder, &question)?;
            let embedder_latency_ms = now.elapsed().as_millis() as u32;
            embedder_histogram.record(embedder_latency_ms);

            let match_start = std::time::Instant::now();
            let matches = witchcraft::match_centroids_with_query_weights(
                db,
                &qe.embeddings,
                qe.weights.as_deref(),
                0.0,
                top_k,
                None,
            )?;
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
            witchcraft::hybrid_reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
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
    witchcraft::log_bucket_io_counters();
    Ok(())
}

pub fn bulk_exact_search(
    db: &DB,
    embedder: &witchcraft::Embedder,
    csvname: std::path::PathBuf,
    outputname: std::path::PathBuf,
) -> Result<()> {
    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    let mut records = Vec::new();
    let mut queries = Vec::new();
    let mut embedder_histogram = histogram::Histogram::new(10000);
    for result in rdr.deserialize() {
        let record: (String, String) = result?;
        let now = std::time::Instant::now();
        let (qe, _offsets) = embedder.embed(&record.1)?;
        let qe = qe.get(0)?;
        embedder_histogram.record(now.elapsed().as_millis() as u32);
        records.push(record);
        queries.push(qe);
    }

    let now = std::time::Instant::now();
    let results = witchcraft::exact_match_centroids_bulk(db, &queries, 100)?;
    println!("exact search took {} ms.", now.elapsed().as_millis());

    let file = File::create(outputname).unwrap();
    let mut writer = BufWriter::new(file);
    let mut metadata_query = db.query("SELECT metadata FROM document WHERE rowid = ?1")?;
    for ((key, _question), matches) in records.iter().zip(results.iter()) {
        write!(writer, "{}\t", key).unwrap();
        for (_score, idx, _sub_idx) in matches {
            let metadata = metadata_query.query_row((*idx,), |row| row.get::<_, String>(0))?;
            let data: CorpusMetaData = serde_json::from_str(&metadata)?;
            write!(writer, "{},", data.key).unwrap();
        }
        writeln!(writer).unwrap();
    }
    writer.flush().unwrap();
    println!("p95 embedder latency = {} ms", embedder_histogram.p95());
    Ok(())
}

fn validate_semantic_search(db: &DB, embedder: Option<&witchcraft::Embedder>) -> Result<()> {
    if embedder.is_none() {
        return Ok(());
    }
    if let Some(reason) = witchcraft::semantic_index_unavailable_reason(db)? {
        anyhow::bail!("semantic index is not ready: {reason}; run warp-cli index or warp-cli reindex");
    }
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
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let db = DB::new_fast(db_name).unwrap();
        witchcraft::index_chunks(&db, Some(&embedder), false).unwrap();
    } else if args.len() == 2 && &args[1] == "reindex" {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let db = DB::new_reader(db_name).unwrap();
        witchcraft::index_chunks(&db, Some(&embedder), true).unwrap();
    } else if args.len() >= 3 && &args[1] == "saliency" {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let text = args[2..].join(" ");
        print_saliency(&embedder, &text)?;
    } else if args.len() >= 3 && (args[1] == "query" || args[1] == "hybrid" || args[1] == "fulltext") {
        let embedder = if args[1] != "fulltext" {
            let device = witchcraft::make_device();
            Some(witchcraft::Embedder::new(&device, &assets).unwrap())
        } else {
            None
        };
        let mut cache = witchcraft::EmbeddingsCache::new(1);
        let db = DB::new_reader(db_name).unwrap();
        validate_semantic_search(&db, embedder.as_ref())?;
        let q = &args[2..].join(" ");
        let use_fulltext = args[1] == "hybrid" || args[1] == "fulltext";
        witchcraft::reset_bucket_io_counters();
        let results =
            witchcraft::search(&db, embedder.as_ref(), &mut cache, q, 0.0, 10, use_fulltext, None).unwrap();
        for (score, _metadata, bodies, sub_idx, _date) in results {
            let idx = (sub_idx as usize).min(bodies.len().saturating_sub(1));
            let body = &bodies[idx];
            println!("{score}: {body} @ {sub_idx}");
            println!("=============================================");
        }
        witchcraft::log_bucket_io_counters();
    } else if args.len() >= 4
        && (args[1] == "querycsv" || args[1] == "hybridcsv" || args[1] == "fulltextcsv" || args[1] == "exactcsv")
    {
        let use_fulltext = args[1] == "hybridcsv" || args[1] == "fulltextcsv";
        let embedder = if args[1] != "fulltextcsv" {
            let device = witchcraft::make_device();
            Some(witchcraft::Embedder::new(&device, &assets).unwrap())
        } else {
            None
        };
        let db = DB::new_reader(db_name).unwrap();
        let csvname = &args[2];
        let outputname = &args[3];
        if args[1] == "exactcsv" {
            bulk_exact_search(&db, embedder.as_ref().unwrap(), csvname.into(), outputname.into())?;
        } else {
            bulk_search(
                &db,
                embedder.as_ref(),
                csvname.into(),
                outputname.into(),
                use_fulltext,
            )?;
        }
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
        eprintln!("\n*** Usage: {} clear | readcsv <file> | embed | index | reindex | saliency <text> | query <text> | hybrid <text> | querycsv|exactcsv <file> <results-file> ***\n", args[0]);
    };
    Ok(())
}
