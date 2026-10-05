use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};

type Judgments = BTreeMap<String, BTreeMap<String, i64>>;
type DocumentMap = Option<BTreeMap<String, String>>;
type Run = BTreeMap<String, Vec<String>>;

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("parse {}", path.display()))
}

fn read_run(reader: impl BufRead, remap: &DocumentMap) -> Result<Run> {
    let mut run = Run::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read result line {}", line_index + 1))?;
        let (query, documents) = line.split_once('\t')
            .filter(|(query, documents)| !query.is_empty() && !documents.contains('\t'))
            .with_context(|| format!("result line {}: expected query<TAB>doc,doc,...", line_index + 1))?;
        // score.py assigns -rank and overwrites duplicate documents and queries.
        let mut ranks = BTreeMap::new();
        for (rank, document) in documents.split(',').enumerate() {
            if document.is_empty() {
                continue;
            }
            let document = match remap {
                Some(map) => map.get(document).with_context(||
                    format!("result line {}: no document mapping for {document}", line_index + 1))?,
                None => document,
            };
            ranks.insert(document.to_owned(), rank);
        }
        let mut ranked: Vec<_> = ranks.into_iter().collect();
        ranked.sort_unstable_by_key(|(_, rank)| *rank);
        run.insert(query.to_owned(), ranked.into_iter().map(|(document, _)| document).collect());
    }
    Ok(run)
}

fn ndcg(documents: &[String], judgments: &BTreeMap<String, i64>, cutoff: usize) -> f64 {
    let dcg: f64 = documents.iter().take(cutoff).enumerate()
        .map(|(rank, document)| {
            let gain = judgments.get(document).copied().unwrap_or(0).max(0) as f64;
            gain / ((rank + 2) as f64).log2()
        }).sum();
    let mut ideal: Vec<_> = judgments.values().copied().filter(|gain| *gain > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let ideal_dcg: f64 = ideal.iter().take(cutoff).enumerate()
        .map(|(rank, gain)| *gain as f64 / ((rank + 2) as f64).log2()).sum();
    if ideal_dcg == 0.0 { 0.0 } else { dcg / ideal_dcg }
}

fn evaluate(run: &Run, judgments: &Judgments, cutoff: usize) -> Result<f64> {
    if cutoff == 0 {
        bail!("cutoff must be positive");
    }
    // Match pytrec_eval: average submitted queries with qrels, including empty runs.
    let scores: Vec<_> = run.iter().filter_map(|(query, documents)| {
        judgments.get(query).map(|qrels| ndcg(documents, qrels, cutoff))
    }).collect();
    if scores.is_empty() {
        bail!("no result queries have relevance judgments");
    }
    Ok(scores.iter().sum::<f64>() / scores.len() as f64)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("Usage: trec-score RESULTS.tsv COLLECTION_MAP.json QRELS.json [CUTOFF]\n\
                  Print mean NDCG (default cutoff: 10). Use a JSON null map for original document IDs.");
        return Ok(());
    }
    if !(3..=4).contains(&args.len()) {
        bail!("Usage: trec-score RESULTS.tsv COLLECTION_MAP.json QRELS.json [CUTOFF]");
    }
    let cutoff = match args.get(3) {
        Some(value) => value.to_str().context("cutoff must be a positive integer")?
            .parse::<usize>().context("cutoff must be a positive integer")?,
        None => 10,
    };
    let remap: DocumentMap = read_json(Path::new(&args[1]))?;
    let judgments: Judgments = read_json(Path::new(&args[2]))?;
    let results_path = Path::new(&args[0]);
    let results = File::open(results_path).with_context(|| format!("open {}", results_path.display()))?;
    let run = read_run(BufReader::new(results), &remap)
        .with_context(|| format!("parse {}", results_path.display()))?;
    println!("{}", evaluate(&run, &judgments, cutoff)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judgments(json: &str) -> Judgments {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn graded_gain_and_cutoffs_match_trec_eval() {
        let qrels = judgments(r#"{"q":{"a":2,"b":1,"unjudged":-1}}"#);
        let run = read_run(&b"q\tb,a,unjudged"[..], &None).unwrap();
        assert!((evaluate(&run, &qrels, 10).unwrap() - 0.8597186998521972).abs() < 1e-14);
        assert_eq!(evaluate(&run, &qrels, 1).unwrap(), 0.5);
        let ideal = read_run(&b"q\ta,b"[..], &None).unwrap();
        assert_eq!(evaluate(&ideal, &qrels, 10).unwrap(), 1.0);
    }

    #[test]
    fn only_submitted_judged_queries_enter_the_average() {
        let qrels = judgments(r#"{"q":{"a":1},"empty":{"a":1},"zero":{"a":0},"missing":{"a":1}}"#);
        let run = read_run(&b"q\ta\r\nempty\t\nzero\ta\nunknown\ta"[..], &None).unwrap();
        assert_eq!(evaluate(&run, &qrels, 10).unwrap(), 1.0 / 3.0);
        assert!(evaluate(&Run::new(), &qrels, 10).is_err());
        assert!(evaluate(&run, &qrels, 0).is_err());
    }

    #[test]
    fn remapping_and_duplicate_overwrites_preserve_ranking() {
        let remap = serde_json::from_str(r#"{"1":"a","2":"b","3":"a"}"#).unwrap();
        let run = read_run(&b"q\t1\nq\t1,,2,3"[..], &remap).unwrap();
        assert_eq!(run["q"], ["b", "a"]);
        assert!(read_run(&b"q\t4"[..], &remap).is_err());
    }

    #[test]
    fn malformed_results_are_reported() {
        for input in ["q", "\ta", "q\ta\tb"] {
            assert!(read_run(input.as_bytes(), &None).is_err());
        }
    }
}
