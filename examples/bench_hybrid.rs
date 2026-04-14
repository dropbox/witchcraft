use std::time::Instant;
use witchcraft::Embedder;
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let assets = std::path::PathBuf::from("assets");

    println!("Loading model...");
    let embedder = Embedder::new(&device, &assets)?;

    // Test texts of varying lengths
    let test_cases = vec![
        ("Short text for testing", 32),
        ("This is a medium length text that should result in approximately 64 tokens after tokenization by the T5 tokenizer. We need to make it a bit longer to reach that target.", 64),
        ("This is a longer text passage that we will use to test the performance of the T5 encoder at around 128 tokens. We need to include enough words to reach that token count. The quick brown fox jumps over the lazy dog. The rain in Spain stays mainly in the plain. To be or not to be, that is the question.", 128),
        ("This is an even longer text passage designed to reach approximately 256 tokens when tokenized. Climate change represents one of the most significant challenges facing humanity in the 21st century. Rising global temperatures are causing widespread environmental disruption, from melting polar ice caps to more frequent extreme weather events. Scientists have demonstrated through extensive research that human activities, particularly the burning of fossil fuels and deforestation, are the primary drivers of these changes. The consequences affect not only natural ecosystems but also human societies, threatening food security, water resources, and coastal communities worldwide.", 256),
    ];

    println!("\nBenchmarking hybrid-dequant performance:");
    println!("{:>6} {:>12} {:>10}", "Tokens", "Time (ms)", "tok/s");
    println!("{}", "-".repeat(32));

    for (text, _expected_tokens) in test_cases {
        // Warmup
        let _ = embedder.embed(text)?;

        // Benchmark (median of 7 runs)
        let mut times = Vec::new();
        let mut actual_tokens = 0;
        for _ in 0..7 {
            let start = Instant::now();
            let (embedding, _offsets) = embedder.embed(text)?;
            let elapsed = start.elapsed();
            times.push(elapsed.as_secs_f64());

            // Get actual token count from embedding shape
            let shape = embedding.dims();
            actual_tokens = shape[1]; // [batch, tokens, dim]
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_sec = times[3];
        let median_ms = median_sec * 1000.0;
        let tokens_per_sec = actual_tokens as f64 / median_sec;

        println!("{:>6} {:>12.1} {:>10.0}", actual_tokens, median_ms, tokens_per_sec);
    }

    Ok(())
}
