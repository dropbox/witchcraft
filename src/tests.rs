#[cfg(test)]
mod tests {
    use crate::{DB, EmbeddingCache};
    use candle_core::{Device, Tensor};
    use std::path::PathBuf;
    use tempfile::tempdir;
    use test_log::test;
    use uuid::Uuid;

    const FACTS : [&str; 33]= [
        "Bananas are berries, but strawberries aren't.",
        "Octopuses have three hearts and blue blood.",
        "A day on Venus is longer than a year on Venus.",
        "There are more trees on Earth than stars in the Milky Way.",
        "Water can boil and freeze at the same time, known as the triple point.",
        "A shrimp's heart is located in its head.",
        "Honey never spoils; archaeologists have found 3000-year-old edible honey.",
        "Wombat poop is cube-shaped to prevent it from rolling away.",
        "There's a species of jellyfish that is biologically immortal.",
        "Humans share about 60% of their DNA with bananas.",
        "The Eiffel Tower can grow taller in the summer due to heat expansion.",
        "Some turtles can breathe through their butts.",
        "The shortest war in history lasted 38 to 45 minutes (Anglo-Zanzibar War).",
        "There's a gas cloud in space that smells like rum and tastes like raspberries.",
        "Cows have best friends and get stressed when separated.",
        "A group of flamingos is called a 'flamboyance'.",
        "A single strand of spaghetti is called a spaghetto.",
        "There's a species of fungus that can turn ants into zombies.",
        "Sharks existed before trees.",
        "Scotland has 421 words for 'snow'.",
        "Hot water freezes faster than cold water, known as the Mpemba effect.",
        "The inventor of the frisbee was turned into a frisbee after he died.",
        "There's an island in Japan where bunnies outnumber people.",
        "Sloths can hold their breath longer than dolphins.",
        "You can hear a blue whale's heartbeat from over 2 miles away.",
        "Butterflies can taste with their feet.",
        "A day on Earth was once only 6 hours long in the distant past.",
        "Vatican City has the highest crime rate per capita due to its tiny population.",
        "There's an official Wizard of New Zealand, appointed by the government.",
        "A bolt of lightning is five times hotter than the surface of the sun.",
        "The letter 'E' is the most common letter in the English language.",
        "There's a lake in Australia that stays bright pink regardless of conditions.",
        "Cleopatra lived closer in time to the first moon landing than to the building of the Great Pyramid."
    ];
    const QUERIES: [(&str, u32); 3] = [
        ("a lake with funny colors", 31),
        ("A group of flamingos", 15),
        ("facts about fruits and berries", 0),
    ];

    const EASY_QUERIES: [(&str, u32); 3] = [
        ("a lake in Australia that stays bright pink", 31),
        ("A group of flamingos", 15),
        ("Bananas are berries", 0),
    ];
    const THRESHOLD: f32 = 0.7;

    fn index_chunks(db: &DB, _device: &candle_core::Device) -> anyhow::Result<()> {
        crate::index_chunks(db, None, false)
    }

    #[test]
    fn test_end_to_end() -> std::io::Result<()> {
        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("warp");
        let mut db = DB::new(path.clone()).unwrap();
        let mut reader_db = DB::new_reader(path.clone()).unwrap();

        let device = crate::make_device();
        let assets = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
        let embedder = crate::Embedder::new(&device, &assets).unwrap();
        let mut cache = crate::EmbeddingsCache::new(4);

        let mut uuids = vec![];
        for body in FACTS {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            uuids.push(uuid.clone());
            db.add_doc(None, &uuid, None, &uuid.to_string(), &body, None)
                .unwrap();
        }
        for round in 0..3 {
            crate::embed_chunks(&db, &embedder, None).unwrap();
            for (i, (q, pos)) in QUERIES.iter().enumerate() {
                let use_fulltext = round == 0;
                println!("searching for {q}");
                let results = crate::search(
                    &reader_db,
                    &embedder,
                    &mut cache,
                    &q.to_string(),
                    THRESHOLD,
                    10,
                    use_fulltext,
                    None,
                )
                .unwrap();
                if round == 0 {
                    assert!(results.len() == 1);
                } else {
                    if i < 2 {
                        assert!(results.len() == 1);
                    } else {
                        assert!(results.len() == 0);
                    }
                }
                for (score, metadata, body, body_idx, _date) in results {
                    let uuid = Uuid::parse_str(&metadata).unwrap();
                    let index = uuids.iter().position(|&u| u == uuid).unwrap();
                    println!("i={i} score={score} metadata={metadata} body={body:?} body_idx={body_idx} uuid-index {index}");
                    assert!(index == *pos as usize);
                }
            }
            db.remove_doc(&uuids[0].clone()).unwrap();
            index_chunks(&db, &device).unwrap();
        }
        let _ = crate::search(
            &reader_db,
            &embedder,
            &mut cache,
            &"".to_string(),
            THRESHOLD,
            10,
            true,
            None,
        )
        .unwrap();
        // Close reader_db before trying to delete database files
        reader_db.shutdown();
        db.clear();
        db.shutdown();

        match std::fs::metadata(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
            Ok(_) => panic!("temp file still exists: {}", path.display()),
        }

        Ok(())
    }

    #[test]
    fn test_index_materializes_embeddings_through_file_cache() {
        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("warp");
        let mut db = DB::new(path.clone()).unwrap();

        let device = crate::make_device();
        let assets = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
        let embedder = crate::Embedder::new(&device, &assets).unwrap();

        for body in [
            "Quartz lenses focus bright laboratory light for calibration.",
            "Careful indexing should compute cached vectors only when needed.",
            "The semantic index groups token embeddings by centroid proximity.",
            "A compact cache entry stores packed vectors and per-span counts.",
            "Chunk hashes make stable filenames for cached embedding records.",
        ] {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            db.add_doc(None, &uuid, None, &uuid.to_string(), &body, None)
                .unwrap();
        }

        let chunk_table_count: i64 = db
            .query("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'chunk'")
            .unwrap()
            .query_row((), |row| row.get(0))
            .unwrap();
        assert_eq!(chunk_table_count, 1);
        let chunk_rows: i64 = db
            .query("SELECT COUNT(*) FROM chunk")
            .unwrap()
            .query_row((), |row| row.get(0))
            .unwrap();
        assert_eq!(chunk_rows, 0);
        let generation_table_count: i64 = db
            .query("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'generation'")
            .unwrap()
            .query_row((), |row| row.get(0))
            .unwrap();
        assert_eq!(generation_table_count, 0);

        let embedding_cache =
            crate::FileEmbeddingCache::new(dir.path().join(crate::default_embedding_cache_dir()));
        assert!(!embedding_cache.root().exists());
        let options = crate::IndexOptions::new(3).unwrap();
        crate::index_chunks_with_cache_and_options(
            &db,
            &embedding_cache,
            Some(&embedder),
            false,
            options,
        )
        .unwrap();

        let cache_entries = std::fs::read_dir(embedding_cache.root()).unwrap().count();
        assert_eq!(cache_entries, 5);
    }

    #[test]
    fn test_sub_docs() -> std::io::Result<()> {
        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("warp");

        let mut db = DB::new(path.clone()).unwrap();
        let device = crate::make_device();
        let assets = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
        let embedder = crate::Embedder::new(&device, &assets).unwrap();
        let mut cache = crate::EmbeddingsCache::new(4);

        let mut lens = vec![];
        for fact in FACTS {
            lens.push(fact.chars().count());
        }
        let body = FACTS.join("");
        let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
        db.add_doc(None, &uuid, None, &uuid.to_string(), &body, Some(lens))
            .unwrap();

        for (q, pos) in QUERIES {
            let results = crate::search(
                &db,
                &embedder,
                &mut cache,
                &q.to_string(),
                THRESHOLD,
                10,
                false,
                None,
            )
            .unwrap();
            for (_score, _metadata, _body, body_idx, _date) in results {
                assert!(body_idx == pos);
            }
        }
        for (q, pos) in EASY_QUERIES {
            let results = crate::search(
                &db,
                &embedder,
                &mut cache,
                &q.to_string(),
                THRESHOLD,
                10,
                true,
                None,
            )
            .unwrap();
            for (_score, _metadata, _body, body_idx, _date) in results {
                assert!(body_idx == pos);
            }
        }
        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_sub_doc_scores_do_not_carry_between_subdocs() -> anyhow::Result<()> {
        let query = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 1.0],
            (2, 2),
            &Device::Cpu,
        )?;
        let embeddings = Tensor::from_vec(
            vec![
                0.9f32, 0.9, // subdoc 0 matches both query dimensions well.
                1.0, 0.0,    // subdoc 1 only improves the first dimension.
            ],
            (2, 2),
            &Device::Cpu,
        )?;
        let unindexed = vec![(vec![(1, 0), (1, 1)], embeddings)];

        let results = crate::match_centroids_raw(&[], &query, &unindexed, 0.0, 10)?;

        assert!(results[0].0 > 0.94);
        assert_eq!(results[0].1, 1);
        assert_eq!(results[0].2, 0);
        Ok(())
    }

    #[test]
    fn test_incremental_index() -> std::io::Result<()> {
        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("warp");
        let mut db = DB::new(path.clone()).unwrap();

        let device = crate::make_device();
        let assets = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
        let embedder = crate::Embedder::new(&device, &assets).unwrap();
        let mut cache = crate::EmbeddingsCache::new(4);

        // Phase 1: Insert initial documents, embed and full-index
        let mut uuids = vec![];
        for body in FACTS {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            uuids.push(uuid.clone());
            db.add_doc(None, &uuid, None, &uuid.to_string(), &body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        index_chunks(&db, &device).unwrap();

        // Verify search works after full index
        let results = crate::search(
            &db,
            &embedder,
            &mut cache,
            &"A group of flamingos".to_string(),
            THRESHOLD,
            10,
            false,
            None,
        )
        .unwrap();
        assert!(
            results.len() >= 1,
            "should find flamingo fact after full index"
        );

        // Phase 2: Add a few new documents (small batch triggers incremental)
        let new_facts = [
            "The Amazon rainforest produces about 20% of the world's oxygen.",
            "A teaspoon of neutron star material would weigh about 6 billion tons.",
            "Dolphins sleep with one eye open.",
        ];
        let mut new_uuids = vec![];
        for body in new_facts {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            new_uuids.push(uuid.clone());
            db.add_doc(None, &uuid, None, &uuid.to_string(), &body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        index_chunks(&db, &device).unwrap(); // should trigger incremental

        // Verify search finds both old and new documents
        let results = crate::search(
            &db,
            &embedder,
            &mut cache,
            &"A group of flamingos".to_string(),
            THRESHOLD,
            10,
            false,
            None,
        )
        .unwrap();
        assert!(
            results.len() >= 1,
            "should still find flamingo fact after incremental index"
        );

        let results = crate::search(
            &db,
            &embedder,
            &mut cache,
            &"dolphins sleeping habits".to_string(),
            THRESHOLD,
            10,
            false,
            None,
        )
        .unwrap();
        assert!(
            results.len() >= 1,
            "should find new dolphin fact after incremental index"
        );

        // Phase 3: Force compaction by calling full_index again
        // Add enough data to trigger full re-index (> 50% new)
        let more_facts = [
            "The human brain uses about 20% of the body's total energy.",
            "A group of owls is called a parliament.",
            "Cats have over 20 vocalizations, including the purr.",
            "The Great Wall of China is not visible from space with the naked eye.",
            "Polar bears have black skin underneath their white fur.",
            "Hummingbirds are the only birds that can fly backwards.",
            "An octopus has nine brains.",
            "The largest living organism is a honey fungus in Oregon.",
            "Seahorses are the only animals where the male gives birth.",
            "A bolt of lightning contains enough energy to toast 100,000 slices of bread.",
            "The fingerprints of koalas are virtually indistinguishable from human fingerprints.",
            "Trees can communicate with each other through underground fungal networks.",
            "A cockroach can live for a week without its head.",
            "The tongue of a blue whale weighs as much as an elephant.",
            "There are more possible iterations of a game of chess than atoms in the known universe.",
            "Bananas are radioactive due to their potassium content.",
            "The shortest complete sentence in the English language is 'Go.'",
            "An ant can carry 50 times its own body weight.",
            "Venus is the only planet that spins clockwise.",
            "A flock of crows is known as a murder.",
        ];
        for body in more_facts {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            db.add_doc(None, &uuid, None, &uuid.to_string(), &body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        index_chunks(&db, &device).unwrap(); // should trigger full re-index (compaction)

        // Verify search still works after compaction
        let results = crate::search(
            &db,
            &embedder,
            &mut cache,
            &"A group of flamingos".to_string(),
            THRESHOLD,
            10,
            false,
            None,
        )
        .unwrap();
        assert!(
            results.len() >= 1,
            "should still find flamingo fact after compaction"
        );

        let results = crate::search(
            &db,
            &embedder,
            &mut cache,
            &"dolphins sleeping habits".to_string(),
            THRESHOLD,
            10,
            false,
            None,
        )
        .unwrap();
        assert!(
            results.len() >= 1,
            "should still find dolphin fact after compaction"
        );

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_cascade() -> std::io::Result<()> {
        // With test constants L0_CAPACITY=4, LSM_FANOUT=2:
        // L0 cap=8, L1=16, L2=32, L3=64, ... L7=512
        // Phase 1: bulk insert all facts → cascades into one high level.
        // Phase 2: add a few more docs → creates a second, lower level.
        // This verifies search works across multiple generations at different levels.
        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("warp");
        let mut db = DB::new(path.clone()).unwrap();

        let device = crate::make_device();
        let assets = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
        let embedder = crate::Embedder::new(&device, &assets).unwrap();
        let mut cache = crate::EmbeddingsCache::new(4);

        // Phase 1: bulk insert
        for &body in &FACTS {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            db.add_doc(None, &uuid, None, &uuid.to_string(), body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        index_chunks(&db, &device).unwrap();

        // Phase 2: add more docs to create a second level
        let extra_facts = [
            "The Amazon rainforest produces about 20% of the world's oxygen.",
            "A teaspoon of neutron star material would weigh about 6 billion tons.",
            "Dolphins sleep with one eye open.",
        ];
        for body in extra_facts {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            db.add_doc(None, &uuid, None, &uuid.to_string(), body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        index_chunks(&db, &device).unwrap();

        // Verify file-backed generations span multiple levels
        let levels = crate::file_index::FileBackedIndex::new(path.clone())
            .level_embedding_counts()
            .unwrap();
        println!("cascade levels: {:?}", levels);
        assert!(
            levels.len() >= 2,
            "should have at least 2 levels after adding extra docs"
        );
        let rowid_sidecars = std::fs::read_dir(dir.path())?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("warp.rowids."))
            .count();
        assert_eq!(rowid_sidecars, 0);

        // Verify search finds results from both old and new data
        for (q, _pos) in EASY_QUERIES {
            let results = crate::search(
                &db,
                &embedder,
                &mut cache,
                &q.to_string(),
                THRESHOLD,
                10,
                true,
                None,
            )
            .unwrap();
            assert!(!results.is_empty(), "should find results for '{q}'");
        }

        let results = crate::search(
            &db,
            &embedder,
            &mut cache,
            &"dolphins sleeping habits".to_string(),
            THRESHOLD,
            10,
            false,
            None,
        )
        .unwrap();
        assert!(
            !results.is_empty(),
            "should find new dolphin fact across levels"
        );

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_scoring() -> std::io::Result<()> {
        let device = crate::make_device();
        let assets = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/assets"));
        let embedder = crate::Embedder::new(&device, &assets).unwrap();
        let mut cache = crate::EmbeddingsCache::new(4);
        let sentences = [
            "The inventor of the frisbee was turned into a frisbee after he died.",
            "There's an island in Japan where bunnies outnumber people.",
            "Sloths can hold their breath longer than dolphins.",
            "The shortest war in history lasted 38 to 45 minutes (Anglo-Zanzibar War).",
            "You can hear a blue whale's heartbeat from over 2 miles away.",
            "Butterflies can taste with their feet.",
            "A day on Earth was once only 6 hours long in the distant past.",
        ];
        let sentences = sentences.map(|s| s.to_string());

        let query = "what wash the shortest war ever?";
        for _ in 0..2 {
            let scores =
                crate::score_query_sentences(&embedder, &mut cache, &query.to_string(), &sentences)
                    .unwrap();
            let mut max = -1.0f32;
            let mut i_max = 0usize;
            for (i, score) in scores.iter().enumerate() {
                println!("score {score}");
                if *score > max {
                    max = *score;
                    i_max = i;
                }
            }
            assert!(i_max == 3);
        }
        Ok(())
    }

    #[test]
    fn test_embedder_without_assets() -> std::io::Result<()> {
        let device = crate::make_device();
        let assets = std::path::PathBuf::from("assets.notfound");
        match crate::Embedder::new(&device, &assets) {
            Ok(_embedder) => {
                unreachable!("should fail to create embedder without assets!");
            }
            Err(_v) => {}
        };
        Ok(())
    }

    #[test]
    fn test_open_bad_db_path() -> std::io::Result<()> {
        let badpath = std::path::PathBuf::from("/unknown/db/xxx");
        match DB::new_reader(badpath) {
            Ok(_db) => {
                unreachable!("should fail to create read-only db from bad path!");
            }
            Err(_v) => {}
        };
        Ok(())
    }

    #[test]
    fn test_open_corrupted_db() -> std::io::Result<()> {
        use std::io::Write;
        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("warp");
        {
            let foo: u32 = 0xfede_abe0;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .append(true)
                .create(true)
                .open(&path)?;
            file.write_all(&foo.to_le_bytes())?;
            file.flush()?;
        }
        let mut db = DB::new(path).unwrap();
        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_unindexed_embedding_count_uses_cache_metadata() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("counts.sqlite");
        let mut db = DB::new(path.clone())?;

        let uuid1 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"counts-1");
        let uuid2 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"counts-2");
        db.add_doc(None, &uuid1, None, "{}", "first count document", None)?;
        db.add_doc(None, &uuid2, None, "{}", "second count document", None)?;

        let rows = db
            .query("SELECT rowid, body, lens FROM document ORDER BY rowid")?
            .query_map((), |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let hash0 = crate::document_cache_hash(&rows[0].1, &rows[0].2);
        let hash1 = crate::document_cache_hash(&rows[1].1, &rows[1].2);
        let model = crate::model_id_for_dim(crate::DEFAULT_EMBEDDING_DIM);
        let cache =
            crate::FileEmbeddingCache::new(dir.path().join(crate::default_embedding_cache_dir()));
        cache.put(
            &hash0,
            &crate::CachedEmbeddings {
                model: model.clone(),
                counts: "not,a,count".to_string(),
                embedding_count: 7,
                embeddings: vec![],
            },
        )?;
        cache.put(
            &hash1,
            &crate::CachedEmbeddings {
                model,
                counts: "".to_string(),
                embedding_count: 11,
                embeddings: vec![],
            },
        )?;

        assert_eq!(crate::count_unindexed_cached_embeddings(&db, &cache)?, 18);
        let data_file = "counts.sqlite.buckets.0.test";
        let header_bytes = u32::try_from(crate::file_index::GENERATION_DATA_HEADER_BYTES)?;
        let mut sidecar = vec![];
        sidecar.extend_from_slice(&crate::file_index::GENERATION_DATA_APP_ID.to_le_bytes());
        sidecar.extend_from_slice(&crate::file_index::GENERATION_DATA_VERSION.to_le_bytes());
        sidecar.extend_from_slice(&0u32.to_le_bytes());
        sidecar.extend_from_slice(&0u32.to_le_bytes());
        sidecar.extend_from_slice(&(crate::DEFAULT_EMBEDDING_DIM as u32).to_le_bytes());
        sidecar.extend_from_slice(&header_bytes.to_le_bytes());
        sidecar.extend_from_slice(&header_bytes.to_le_bytes());
        sidecar.extend_from_slice(&u64::from(header_bytes).to_le_bytes());
        sidecar.extend_from_slice(&u64::try_from(rows[0].0)?.to_le_bytes());
        sidecar.extend_from_slice(&7u32.to_le_bytes());
        std::fs::write(dir.path().join(data_file), sidecar)?;
        std::fs::write(
            dir.path().join("counts.sqlite.index"),
            format!(
                "{}\t{}\n0\t7\t{data_file}\n",
                crate::file_index::GENERATION_DATA_APP_ID,
                crate::file_index::GENERATION_DATA_VERSION
            ),
        )?;
        assert_eq!(crate::count_unindexed_cached_embeddings(&db, &cache)?, 11);
        Ok(())
    }

    /// Regression test for scoring off-by-one: the last token vector was
    /// dropped because vmax_inplace was unreachable after the break.
    /// A single-document corpus exercises this: the one document is both
    /// the first and last element in the scoring loop.
    #[test]
    fn test_single_doc_search() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("single_doc.sqlite");
        let assets = PathBuf::from("assets");
        // Use CPU to avoid Metal buffer contention in parallel test runs
        let device = candle_core::Device::Cpu;
        let embedder = crate::Embedder::new(&device, &assets)?;
        let mut cache = crate::EmbeddingsCache::new(4);

        let mut db = DB::new(path.clone())?;
        let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"only-doc");
        db.add_doc(None, &uuid, None, &uuid.to_string(), "Honey never spoils", None)?;
        crate::embed_chunks(&db, &embedder, None)?;
        let chunk_rows: i64 = db.query("SELECT COUNT(*) FROM chunk")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(chunk_rows, 1);
        index_chunks(&db, &device)?;

        let results = crate::search(
            &db, &embedder, &mut cache,
            "honey preservation", 0.3, 10, false, None,
        )?;
        assert!(!results.is_empty(), "single-doc search must return the document");
        Ok(())
    }
      
    #[test]
    fn test_add_docs_batch() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("batch_test.sqlite");
        let mut db = DB::new(path)?;

        let docs: Vec<(
            Option<u64>,
            Uuid,
            Option<iso8601_timestamp::Timestamp>,
            &str,
            &str,
            Option<Vec<usize>>,
        )> = vec![
            (None, Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc1"), None, r#"{"title":"one"}"#, "first document body", None),
            (None, Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc2"), None, r#"{"title":"two"}"#, "second document body", None),
            (None, Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc3"), None, r#"{"title":"three"}"#, "third document body", None),
        ];

        let count = db.add_docs_batch(&docs)?;
        assert_eq!(count, 3);

        // Verify docs exist
        let row_count: i64 = db.query("SELECT COUNT(*) FROM document")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(row_count, 3);

        // Verify add_doc delegates to add_docs_batch correctly
        let uuid4 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc4");
        db.add_doc(None, &uuid4, None, r#"{"title":"four"}"#, "fourth body", None)?;
        let row_count: i64 = db.query("SELECT COUNT(*) FROM document")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(row_count, 4);

        // Verify upsert: re-add doc1 with different body
        let uuid1 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc1");
        db.add_doc(None, &uuid1, None, r#"{"title":"one-updated"}"#, "updated body", None)?;
        let row_count: i64 = db.query("SELECT COUNT(*) FROM document")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(row_count, 4); // still 4, not 5

        let metadata: String = db.query("SELECT metadata FROM document WHERE uuid = ?1")?
            .query_row((uuid1.to_string(),), |row| row.get(0))?;
        assert!(metadata.contains("one-updated"));

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_delete_tombstone_trigger_is_transactional() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("delete_tombstone.sqlite");
        let mut db = DB::new(path)?;
        let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"delete-tombstone");
        db.add_doc(None, &uuid, None, "{}", "delete me", None)?;

        db.begin_transaction()?;
        db.execute("DELETE FROM document WHERE rowid = 1")?;
        let queued: i64 = db.query("SELECT COUNT(*) FROM document_index_tombstone")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(queued, 1);
        db.rollback_transaction()?;

        let queued: i64 = db.query("SELECT COUNT(*) FROM document_index_tombstone")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(queued, 0);

        db.begin_transaction()?;
        db.execute("DELETE FROM document WHERE rowid = 1")?;
        db.commit_transaction()?;
        let queued: i64 = db.query("SELECT COUNT(*) FROM document_index_tombstone")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(queued, 1);

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_delete_tombstones_drain_into_file_index() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("delete_drain.sqlite");
        let mut db = DB::new(path.clone())?;
        let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"delete-drain");
        db.add_doc(None, &uuid, None, "{}", "delete me", None)?;
        db.remove_doc(&uuid)?;

        let queued: i64 = db.query("SELECT COUNT(*) FROM document_index_tombstone")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(queued, 1);

        let cache = crate::default_embedding_cache();
        crate::index_chunks_with_cache(&db, &cache, None, false)?;

        let queued: i64 = db.query("SELECT COUNT(*) FROM document_index_tombstone")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(queued, 0);

        let index = crate::file_index::FileBackedIndex::new(path.clone());
        let active = index.active_rowids()?;
        assert_eq!(active.get(&1), Some(&false));

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_chunk_cache_cleanup_triggers() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("chunk_cleanup.sqlite");
        let mut db = DB::new(path)?;
        let uuid1 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"chunk-cleanup-1");
        let uuid2 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"chunk-cleanup-2");
        let body = "same cached chunk body";
        db.add_doc(None, &uuid1, None, "{}", body, None)?;
        db.add_doc(None, &uuid2, None, "{}", body, None)?;

        let hash: String = db.query("SELECT hash FROM document WHERE uuid = ?1")?
            .query_row((uuid1.to_string(),), |row| row.get(0))?;
        db.query(
            "INSERT INTO chunk(hash, model, embeddings, counts, embedding_count)
             VALUES(?1, ?2, ?3, ?4, ?5)",
        )?
        .execute((&hash, "xtr-base-en", vec![1u8, 2, 3], "3", 3i64))?;

        db.remove_doc(&uuid1)?;
        let chunks: i64 = db.query("SELECT COUNT(*) FROM chunk WHERE hash = ?1")?
            .query_row((&hash,), |row| row.get(0))?;
        assert_eq!(chunks, 1);

        db.remove_doc(&uuid2)?;
        let chunks: i64 = db.query("SELECT COUNT(*) FROM chunk WHERE hash = ?1")?
            .query_row((&hash,), |row| row.get(0))?;
        assert_eq!(chunks, 0);

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_add_docs_with_explicit_rowids() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("rowid_test.sqlite");
        let mut db = DB::new(path)?;

        let uuid1 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"rowid-doc1");
        let uuid2 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"rowid-doc2");
        let docs = [
            (Some(10), uuid1, None, r#"{"title":"one"}"#, "first body", None),
            (Some(20), uuid2, None, r#"{"title":"two"}"#, "second body", None),
        ];

        assert_eq!(db.add_docs_batch(&docs)?, 2);
        let rowids = db
            .query("SELECT rowid FROM document ORDER BY rowid")?
            .query_map((), |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(rowids, vec![10, 20]);

        let uuid3 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"rowid-doc3");
        db.add_doc(None, &uuid3, None, "{}", "automatic rowid", None)?;
        let auto_rowid: i64 = db
            .query("SELECT rowid FROM document WHERE uuid = ?1")?
            .query_row((uuid3.to_string(),), |row| row.get(0))?;
        assert!(auto_rowid > 20);

        db.add_doc(
            Some(30),
            &uuid1,
            None,
            r#"{"title":"one-updated"}"#,
            "updated body",
            None,
        )?;
        let moved_rowid: i64 = db
            .query("SELECT rowid FROM document WHERE uuid = ?1")?
            .query_row((uuid1.to_string(),), |row| row.get(0))?;
        assert_eq!(moved_rowid, 30);

        let uuid4 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"rowid-doc4");
        assert!(db
            .add_doc(Some(30), &uuid4, None, "{}", "duplicate rowid", None)
            .is_err());
        assert!(db
            .add_doc(
                Some(i64::MAX as u64 + 1),
                &uuid4,
                None,
                "{}",
                "too large",
                None,
            )
            .is_err());

        let uuid5 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"rowid-doc5");
        let uuid6 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"rowid-doc6");
        let non_monotonic = [
            (Some(40), uuid5, None, "{}", "higher", None),
            (Some(39), uuid6, None, "{}", "lower", None),
        ];
        assert!(db.add_docs_batch(&non_monotonic).is_err());

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_search_with_uuid_filter() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("filter_test.sqlite");
        let assets = PathBuf::from("assets");
        let device = crate::make_device();
        let embedder = crate::Embedder::new(&device, &assets)?;
        let mut cache = crate::EmbeddingsCache::new(4);

        let mut db = DB::new(path.clone())?;

        // Insert two documents that would both match "flamingos"
        let uuid_a = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"filter-a");
        let uuid_b = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"filter-b");
        db.add_doc(None, &uuid_a, None, &uuid_a.to_string(), "A group of flamingos is called a flamboyance", None)?;
        db.add_doc(None, &uuid_b, None, &uuid_b.to_string(), "Flamingos are pink because of their diet of shrimp and algae", None)?;

        crate::embed_chunks(&db, &embedder, None)?;
        index_chunks(&db, &device)?;

        // Unfiltered search should return both
        let results = crate::search(
            &db, &embedder, &mut cache, "flamingos", THRESHOLD, 10, false, None,
        )?;
        assert!(results.len() == 2, "unfiltered search should find both flamingo docs, got {}", results.len());

        // Filtered search: only uuid_b
        let filter = crate::types::SqlStatementInternal {
            statement_type: crate::types::SqlStatementType::Condition,
            condition: Some(crate::types::SqlConditionInternal {
                key: "uuid".to_string(),
                operator: crate::types::SqlOperator::Equals,
                value: Some(crate::types::SqlValue::String(uuid_b.to_string())),
            }),
            logic: None,
            statements: None,
        };
        let results = crate::search(
            &db, &embedder, &mut cache, "flamingos", THRESHOLD, 10, false, Some(&filter),
        )?;
        assert!(results.len() == 1, "filtered search should find exactly one doc, got {}", results.len());
        assert_eq!(results[0].1, uuid_b.to_string(), "filtered result should be uuid_b");

        db.clear();
        db.shutdown();
        Ok(())
    }

    /// Searching an empty DB first cached an empty generation list, causing a subsequent
    /// search on a populated DB to skip all indexed embeddings.
    #[test]
    fn test_cross_db_cache_isolation() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let assets = PathBuf::from("assets");
        let device = crate::make_device();
        let embedder = crate::Embedder::new(&device, &assets)?;
        let mut cache = crate::EmbeddingsCache::new(4);

        // Populated DB with indexed data
        let baseline_path = dir.path().join("baseline.sqlite");
        let mut baseline = DB::new(baseline_path.clone())?;
        for &body in &FACTS {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            baseline.add_doc(None, &uuid, None, &uuid.to_string(), body, None)?;
        }
        crate::embed_chunks(&baseline, &embedder, None)?;
        index_chunks(&baseline, &device)?;

        // Empty DB (simulates an overlay with 0 generations)
        let overlay_path = dir.path().join("overlay.sqlite");
        let mut overlay = DB::new(overlay_path)?;

        // Search the empty DB first — this poisoned the global cache before the fix
        let overlay_results = crate::search(
            &overlay, &embedder, &mut cache,
            "a lake with funny colors", THRESHOLD, 10, false, None,
        )?;
        assert!(overlay_results.is_empty());

        // Search the populated DB — must still find indexed results
        let baseline_results = crate::search(
            &baseline, &embedder, &mut cache,
            "a lake with funny colors", THRESHOLD, 10, false, None,
        )?;
        assert!(
            !baseline_results.is_empty(),
            "baseline search must not be poisoned by prior empty-DB search"
        );

        baseline.clear();
        baseline.shutdown();
        overlay.shutdown();
        Ok(())
    }

    #[test]
    fn test_empty_body_not_embedded() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("empty_body_test.sqlite");
        let assets = PathBuf::from("assets");
        let mut db = DB::new(path.clone())?;
        let device = crate::make_device();
        let embedder = crate::Embedder::new(&device, &assets)?;

        // Add one doc with empty body and one with real content
        let uuid_empty = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"empty");
        let uuid_real = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"real");
        db.add_doc(None, &uuid_empty, None, "{}", "", None)?;
        db.add_doc(None, &uuid_real, None, "{}", "Octopuses have three hearts", None)?;

        let embedding_cache =
            crate::FileEmbeddingCache::new(dir.path().join(crate::default_embedding_cache_dir()));
        let count = crate::embed_chunks_with_cache(&db, &embedder, &embedding_cache, None)?;
        assert_eq!(count, 1, "only the non-empty doc should be embedded");

        let cache_entries = std::fs::read_dir(embedding_cache.root())?.count();
        assert_eq!(cache_entries, 1);

        let chunk_table_count: i64 = db.query("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'chunk'")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(chunk_table_count, 1);
        let chunk_rows: i64 = db.query("SELECT COUNT(*) FROM chunk")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(chunk_rows, 0);

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn fts5_query_splits_punctuation_and_uses_or_terms() {
        let (query, normalized) =
            super::super::fts5_query("what is the origin of COVID-19").unwrap();

        assert_eq!(
            query,
            "\"what\" OR \"is\" OR \"the\" OR \"origin\" OR \"of\" OR \"COVID\" OR \"19\"*"
        );
        assert_eq!(normalized, "what is the origin of COVID 19");
    }

    #[test]
    fn fulltext_search_matches_hyphenated_terms_without_requiring_every_word() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("warp.sqlite");
        let mut db = DB::new(path)?;

        let relevant = "The origin of COVID-19 was investigated in early pandemic research.";
        let distractor = "This paragraph says what is the origin of an unrelated weather report.";
        let relevant_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, relevant.as_bytes());
        let distractor_uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, distractor.as_bytes());
        db.add_doc(None, &relevant_uuid, None, "relevant", relevant, None)?;
        db.add_doc(None, &distractor_uuid, None, "distractor", distractor, None)?;

        let relevant_rowid: u32 = db
            .query("SELECT rowid FROM document WHERE metadata = 'relevant'")?
            .query_row((), |row| row.get(0))?;
        let results = crate::fulltext_search(&db, "what is the origin of COVID-19", 10, None)?;

        assert!(
            results.iter().any(|(_, rowid, _)| *rowid == relevant_rowid),
            "expected relevant COVID-19 document in {results:?}"
        );

        db.clear();
        db.shutdown();
        Ok(())
    }

    #[test]
    fn test_capi_add_search_results() -> anyhow::Result<()> {
        use std::ffi::{CStr, CString};
        use std::os::raw::c_void;

        struct EmbeddingStore {
            blobs: std::collections::HashMap<u64, Vec<u8>>,
            callback_calls: usize,
        }

        unsafe fn last_error(handle: *mut crate::capi::WitchcraftHandle) -> String {
            let ptr = crate::capi::witchcraft_last_error(handle);
            if ptr.is_null() {
                return "no error".to_string();
            }
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }

        unsafe extern "C" fn embedding_callback(
            rowid: u64,
            user_data: *mut c_void,
            dst: *mut u8,
            dst_cap: usize,
            out_len: *mut usize,
        ) -> i32 {
            let store = &mut *(user_data as *mut EmbeddingStore);
            store.callback_calls += 1;
            let Some(blob) = store.blobs.get(&rowid) else {
                return -1;
            };
            if out_len.is_null() {
                return -1;
            }
            *out_len = blob.len();
            if dst.is_null() {
                return 0;
            }
            if dst_cap < blob.len() {
                return -1;
            }
            std::ptr::copy_nonoverlapping(blob.as_ptr(), dst, blob.len());
            0
        }

        let dir = tempdir()?;
        let path = dir.path().join("capi.sqlite");
        let db_path = CString::new(path.to_string_lossy().as_bytes())?;
        let assets = CString::new("assets")?;
        let mut store = EmbeddingStore {
            blobs: std::collections::HashMap::new(),
            callback_calls: 0,
        };

        unsafe {
            let handle =
                crate::capi::witchcraft_open(db_path.as_ptr(), assets.as_ptr(), std::ptr::null());
            assert!(!handle.is_null(), "{}", last_error(std::ptr::null_mut()));

            let honey = "Honey never spoils";
            let blob = crate::capi::witchcraft_embed(
                handle,
                honey.as_ptr(),
                honey.len(),
            );
            assert_eq!(blob.status, 0, "{}", last_error(handle));
            store.blobs.insert(42, std::slice::from_raw_parts(blob.ptr, blob.len).to_vec());
            crate::capi::witchcraft_bytes_free(blob.ptr, blob.len);

            let octopus = "Octopuses have three hearts";
            let blob = crate::capi::witchcraft_embed(
                handle,
                octopus.as_ptr(),
                octopus.len(),
            );
            assert_eq!(blob.status, 0, "{}", last_error(handle));
            store.blobs.insert(43, std::slice::from_raw_parts(blob.ptr, blob.len).to_vec());
            crate::capi::witchcraft_bytes_free(blob.ptr, blob.len);

            let user_data = &mut store as *mut _ as *mut c_void;
            let honey_blob = &store.blobs[&42];
            assert_eq!(
                crate::capi::witchcraft_add(handle, 42, honey_blob.as_ptr(), honey_blob.len()),
                0,
                "{}",
                last_error(handle)
            );
            let octopus_blob = &store.blobs[&43];
            assert_eq!(
                crate::capi::witchcraft_add(handle, 43, octopus_blob.as_ptr(), octopus_blob.len()),
                0,
                "{}",
                last_error(handle)
            );
            assert_eq!(
                crate::capi::witchcraft_index(handle, Some(embedding_callback), user_data),
                0,
                "{}",
                last_error(handle)
            );

            let query = "honey never spoils";
            let result = crate::capi::witchcraft_search(
                handle,
                query.as_ptr(),
                query.len(),
                0.0,
                10,
            );
            assert_eq!(result.status, 0, "{}", last_error(handle));
            let hits = std::slice::from_raw_parts(result.ptr, result.len).to_vec();
            crate::capi::witchcraft_search_results_free(result.ptr, result.len);
            crate::capi::witchcraft_close(handle);

            assert!(store.callback_calls > 0);
            assert_eq!(hits.first().map(|hit| hit.rowid), Some(42));
            assert!(hits.first().map(|hit| hit.score.is_finite()).unwrap_or(false));
        }

        Ok(())
    }
}
