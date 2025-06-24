use anyhow::Result;
use std::env;
use candle_core::Device;

mod warp;

fn main() -> Result<()> {

    let args: Vec<String> = env::args().collect();
    let db = warp::DB::new("mydb.sqlite");
    let device = Device::new_metal(0)?;
    let embedder = warp::Embedder::new(&device);

    let status = if args.len() == 3 && args[1] == "scan" {

        Ok( warp::scan_documents_dir(&db, &args[2]) )

    } else if args.len() == 3 && args[1] == "readcsv" {

        Ok( warp::read_csv(&db, &args[2]) )

    } else if args.len() == 2 && &args[1] == "embed" {

        Ok( warp::embed_chunks(&db, &device) )

    } else if args.len() == 2 && &args[1] == "index" {

        Ok( warp::index_chunks(&db, &device) )

    } else if args.len() >= 3 && (args[1] == "query" || args[1] == "hybrid") {

        let q = &args[2..].join(" ");
        let use_fulltext = args[1] == "hybrid";
        Ok( warp::search(&db, &embedder, &q, use_fulltext) )

    } else if args.len() >= 4 && (args[1] == "querycsv" || args[1] == "hybridcsv" || args[1] == "fulltextcsv") {

        let use_fulltext = args[1] == "hybridcsv" || args[1] == "fulltextcsv";
        let use_semantic = args[1] != "fulltextcsv";
        let csvname = &args[2];
        let outputname = &args[3];
        Ok( warp::bulk_search(&db, &embedder, &csvname, &outputname, use_fulltext, use_semantic) )

    } else {
       eprintln!("\n*** Usage: {} scan | readcsv <file> | embed | index | query <text> | hybrid <text> | querycsv <file> <results-file> ***\n", args[0]);
       Err(1)
    };

    status.unwrap()
}
