#[cfg(test)]
mod layer_tests {
    use crate::layer::{LayerError, LayerStatus};
    use crate::DB;
    use std::path::PathBuf;
    use tempfile::tempdir;
    use test_log::test;

    fn make_db() -> (DB, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let path: PathBuf = dir.path().join("test.db");
        let db = DB::new(path).unwrap();
        (db, dir)
    }

    #[test]
    fn test_flat_topology_chain() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "overlay-a", None).unwrap();
        let b = db.create_layer(Some(root), "overlay-b", None).unwrap();

        let chain_a = db.layer_chain(a).unwrap();
        assert_eq!(chain_a.chain, vec![(a, 0), (root, 1)]);

        let chain_b = db.layer_chain(b).unwrap();
        assert_eq!(chain_b.chain, vec![(b, 0), (root, 1)]);

        let chain_root = db.layer_chain(root).unwrap();
        assert_eq!(chain_root.chain, vec![(root, 0)]);
    }

    #[test]
    fn test_deep_topology_chain() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "a", None).unwrap();
        let a1 = db.create_layer(Some(a), "a1", None).unwrap();

        let chain = db.layer_chain(a1).unwrap();
        assert_eq!(chain.chain, vec![(a1, 0), (a, 1), (root, 2)]);
    }

    #[test]
    fn test_seal_layer() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();

        db.seal_layer(root).unwrap();
        let layer = db.get_layer(root).unwrap();
        assert_eq!(layer.status, LayerStatus::Sealed);

        // Sealing again should fail with NotActive
        assert!(matches!(db.seal_layer(root), Err(LayerError::NotActive(_))));
    }

    #[test]
    fn test_seal_nonexistent() {
        let (db, _dir) = make_db();
        assert!(matches!(db.seal_layer(999), Err(LayerError::NotFound(999))));
    }

    #[test]
    fn test_layer_children() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "a", None).unwrap();
        let b = db.create_layer(Some(root), "b", None).unwrap();

        let children = db.layer_children(root).unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].id, a);
        assert_eq!(children[1].id, b);

        let leaf_children = db.layer_children(a).unwrap();
        assert!(leaf_children.is_empty());
    }

    #[test]
    fn test_layer_tree() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let _a = db.create_layer(Some(root), "a", None).unwrap();
        let _b = db.create_layer(Some(root), "b", None).unwrap();

        let tree = db.layer_tree().unwrap();
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_delete_leaf() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "a", None).unwrap();

        db.delete_layer(a).unwrap();
        let tree = db.layer_tree().unwrap();
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_delete_with_children_fails() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let _a = db.create_layer(Some(root), "a", None).unwrap();

        assert!(matches!(db.delete_layer(root), Err(LayerError::HasChildren(_))));
    }

    #[test]
    fn test_compact_reparents_via_compaction() {
        // reparent_layer is private; verify reparenting works through compact_layers
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "a", None).unwrap();
        let b = db.create_layer(Some(a), "b", None).unwrap();
        // b has a child that should be reparented to the compacted layer
        let c = db.create_layer(Some(b), "c", None).unwrap();

        let compacted = db.compact_layers(b, Some(root)).unwrap();

        // c was reparented to the compacted layer
        let child = db.get_layer(c).unwrap();
        assert_eq!(child.parent_id, Some(compacted));

        // Compacted layer's chain goes through root
        let chain = db.layer_chain(compacted).unwrap();
        assert_eq!(chain.chain, vec![(compacted, 0), (root, 1)]);
    }

    #[test]
    fn test_compact_sub_chain() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "a", None).unwrap();
        let a1 = db.create_layer(Some(a), "a1", None).unwrap();
        let a1_child = db.create_layer(Some(a1), "a1-child", None).unwrap();

        let compacted = db.compact_layers(a1, Some(root)).unwrap();

        assert_eq!(db.get_layer(a1).unwrap().status, LayerStatus::Compacted);
        assert_eq!(db.get_layer(a).unwrap().status, LayerStatus::Compacted);

        let compacted_layer = db.get_layer(compacted).unwrap();
        assert_eq!(compacted_layer.parent_id, Some(root));
        assert_eq!(compacted_layer.status, LayerStatus::Active);

        let child = db.get_layer(a1_child).unwrap();
        assert_eq!(child.parent_id, Some(compacted));
    }

    #[test]
    fn test_compact_full_chain_stop_at_none() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "a", None).unwrap();
        let a1 = db.create_layer(Some(a), "a1", None).unwrap();

        // Compact entire chain (stop_at = None): a1 → a → root
        let compacted = db.compact_layers(a1, None).unwrap();

        // All three original layers are marked compacted
        assert_eq!(db.get_layer(a1).unwrap().status, LayerStatus::Compacted);
        assert_eq!(db.get_layer(a).unwrap().status, LayerStatus::Compacted);
        assert_eq!(db.get_layer(root).unwrap().status, LayerStatus::Compacted);

        // New layer has no parent (it's the new root)
        let compacted_layer = db.get_layer(compacted).unwrap();
        assert_eq!(compacted_layer.parent_id, None);
    }

    #[test]
    fn test_compact_single_layer() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();

        // Compact root alone (stop_at = None, chain = [root])
        let compacted = db.compact_layers(root, None).unwrap();

        assert_eq!(db.get_layer(root).unwrap().status, LayerStatus::Compacted);

        let compacted_layer = db.get_layer(compacted).unwrap();
        assert_eq!(compacted_layer.parent_id, None);
        assert_eq!(compacted_layer.status, LayerStatus::Active);
    }

    #[test]
    fn test_clear_removes_layers() {
        let (mut db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let _a = db.create_layer(Some(root), "a", None).unwrap();

        db.clear();
        // Recreate after clear (clear sets remove_on_shutdown but also deletes rows)
        let tree = db.layer_tree().unwrap();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_metadata() {
        let (db, _dir) = make_db();
        let meta = r#"{"branch":"feature/foo","commit":"abc123"}"#;
        let id = db.create_layer(None, "baseline", Some(meta)).unwrap();
        let layer = db.get_layer(id).unwrap();
        assert_eq!(layer.metadata.as_deref(), Some(meta));
    }

    #[test]
    fn test_nonexistent_layer_chain() {
        let (db, _dir) = make_db();
        assert!(matches!(db.layer_chain(999), Err(LayerError::NotFound(999))));
    }

    #[test]
    fn test_delete_nonexistent() {
        let (db, _dir) = make_db();
        assert!(matches!(db.delete_layer(999), Err(LayerError::NotFound(999))));
    }

    #[test]
    fn test_compact_invalid_stop_at() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "a", None).unwrap();
        let b = db.create_layer(Some(root), "b", None).unwrap();

        // b is not an ancestor of a — should fail
        assert!(matches!(
            db.compact_layers(a, Some(b)),
            Err(LayerError::NotInChain { .. })
        ));
    }

    #[test]
    fn test_compact_self_stop_at() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let a = db.create_layer(Some(root), "a", None).unwrap();

        // compact(a, Some(a)) — zero layers to compact
        assert!(matches!(
            db.compact_layers(a, Some(a)),
            Err(LayerError::NotInChain { .. })
        ));
    }

    #[test]
    fn test_error_variants_are_distinguishable() {
        let (db, _dir) = make_db();
        let root = db.create_layer(None, "baseline", None).unwrap();
        let _child = db.create_layer(Some(root), "child", None).unwrap();

        // HasChildren vs NotFound on delete
        assert!(matches!(db.delete_layer(root), Err(LayerError::HasChildren(_))));
        assert!(matches!(db.delete_layer(999), Err(LayerError::NotFound(999))));

        // NotActive vs NotFound on seal
        let leaf = db.create_layer(Some(root), "leaf", None).unwrap();
        db.seal_layer(leaf).unwrap();
        assert!(matches!(db.seal_layer(leaf), Err(LayerError::NotActive(_))));
        assert!(matches!(db.seal_layer(999), Err(LayerError::NotFound(999))));
    }
}

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
            db.refresh_ft().unwrap();
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
                for (score, metadata, body, body_idx, _chunk) in results {
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
            for (_score, _metadata, _body, body_idx, _chunk) in results {
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
            for (_score, _metadata, _body, body_idx, _chunk) in results {
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
}
