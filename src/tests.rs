#[cfg(test)]
mod tests {
    use crate::DB;
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
        "Bananas are berries, but strawberries aren't.",
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
            db.add_doc(&uuid, None, &uuid.to_string(), &body, None)
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
            crate::index_chunks(&db, &device).unwrap();
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
        db.add_doc(&uuid, None, &uuid.to_string(), &body, Some(lens))
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
            db.add_doc(&uuid, None, &uuid.to_string(), &body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        crate::index_chunks(&db, &device).unwrap();

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
            db.add_doc(&uuid, None, &uuid.to_string(), &body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        crate::index_chunks(&db, &device).unwrap(); // should trigger incremental

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
            db.add_doc(&uuid, None, &uuid.to_string(), &body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        crate::index_chunks(&db, &device).unwrap(); // should trigger full re-index (compaction)

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
            db.add_doc(&uuid, None, &uuid.to_string(), body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        crate::index_chunks(&db, &device).unwrap();

        // Phase 2: add more docs to create a second level
        let extra_facts = [
            "The Amazon rainforest produces about 20% of the world's oxygen.",
            "A teaspoon of neutron star material would weigh about 6 billion tons.",
            "Dolphins sleep with one eye open.",
        ];
        for body in extra_facts {
            let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
            db.add_doc(&uuid, None, &uuid.to_string(), body, None)
                .unwrap();
        }
        crate::embed_chunks(&db, &embedder, None).unwrap();
        crate::index_chunks(&db, &device).unwrap();

        // Verify generations span multiple levels
        let levels: Vec<(u32, usize)> = {
            let mut level_query = db
                .query("SELECT level, SUM(num_embeddings) FROM generation GROUP BY level ORDER BY level")
                .unwrap();
            level_query
                .query_map((), |row| Ok((row.get::<_, u32>(0)?, row.get::<_, usize>(1)?)))
                .unwrap()
                .map(Result::unwrap)
                .collect()
        };
        println!("cascade levels: {:?}", levels);
        assert!(
            levels.len() >= 2,
            "should have at least 2 levels after adding extra docs"
        );

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
        db.add_doc(&uuid, None, &uuid.to_string(), "Honey never spoils", None)?;
        crate::embed_chunks(&db, &embedder, None)?;
        crate::index_chunks(&db, &device)?;

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

        let docs: Vec<(Uuid, Option<iso8601_timestamp::Timestamp>, &str, &str, Option<Vec<usize>>)> = vec![
            (Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc1"), None, r#"{"title":"one"}"#, "first document body", None),
            (Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc2"), None, r#"{"title":"two"}"#, "second document body", None),
            (Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc3"), None, r#"{"title":"three"}"#, "third document body", None),
        ];

        let count = db.add_docs_batch(&docs)?;
        assert_eq!(count, 3);

        // Verify docs exist
        let row_count: usize = db.query("SELECT COUNT(*) FROM document")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(row_count, 3);

        // Verify add_doc delegates to add_docs_batch correctly
        let uuid4 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc4");
        db.add_doc(&uuid4, None, r#"{"title":"four"}"#, "fourth body", None)?;
        let row_count: usize = db.query("SELECT COUNT(*) FROM document")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(row_count, 4);

        // Verify upsert: re-add doc1 with different body
        let uuid1 = Uuid::new_v5(&Uuid::NAMESPACE_OID, b"doc1");
        db.add_doc(&uuid1, None, r#"{"title":"one-updated"}"#, "updated body", None)?;
        let row_count: usize = db.query("SELECT COUNT(*) FROM document")?
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
        db.add_doc(&uuid_a, None, &uuid_a.to_string(), "A group of flamingos is called a flamboyance", None)?;
        db.add_doc(&uuid_b, None, &uuid_b.to_string(), "Flamingos are pink because of their diet of shrimp and algae", None)?;

        crate::embed_chunks(&db, &embedder, None)?;
        crate::index_chunks(&db, &device)?;

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
            baseline.add_doc(&uuid, None, &uuid.to_string(), body, None)?;
        }
        crate::embed_chunks(&baseline, &embedder, None)?;
        crate::index_chunks(&baseline, &device)?;

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
        db.add_doc(&uuid_empty, None, "{}", "", None)?;
        db.add_doc(&uuid_real, None, "{}", "Octopuses have three hearts", None)?;

        let count = crate::embed_chunks(&db, &embedder, None)?;
        assert_eq!(count, 1, "only the non-empty doc should be embedded");

        // Verify the chunk table has exactly one entry
        let chunk_count: usize = db.query("SELECT COUNT(*) FROM chunk")?
            .query_row((), |row| row.get(0))?;
        assert_eq!(chunk_count, 1);

        db.clear();
        db.shutdown();
        Ok(())
    }
}
