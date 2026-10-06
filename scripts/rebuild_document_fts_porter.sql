-- Recreate the external-content FTS5 table with the Porter tokenizer and
-- Lucene/OpenSearch English stopwords.
--
-- Usage:
--   sqlite3 mydb.sqlite < scripts/rebuild_document_fts_porter.sql
--
-- This changes the tokenizer; a plain FTS5 'rebuild' command would only
-- refresh the existing index using whatever tokenizer the table already has.

BEGIN IMMEDIATE;

DROP TRIGGER IF EXISTS document_fts_insert;
DROP TRIGGER IF EXISTS document_fts_delete;
DROP TRIGGER IF EXISTS document_fts_update;

DROP TABLE IF EXISTS document_fts;
DROP TABLE IF EXISTS document_fts_stopword;

CREATE TABLE document_fts_stopword(
    term TEXT NOT NULL PRIMARY KEY
) WITHOUT ROWID;

INSERT INTO document_fts_stopword(term) VALUES
    ('a'),
    ('an'),
    ('and'),
    ('are'),
    ('as'),
    ('at'),
    ('be'),
    ('but'),
    ('by'),
    ('for'),
    ('if'),
    ('in'),
    ('into'),
    ('is'),
    ('it'),
    ('no'),
    ('not'),
    ('of'),
    ('on'),
    ('or'),
    ('such'),
    ('that'),
    ('the'),
    ('their'),
    ('then'),
    ('there'),
    ('these'),
    ('they'),
    ('this'),
    ('to'),
    ('was'),
    ('will'),
    ('with');

CREATE VIRTUAL TABLE document_fts
USING fts5(
    body,
    content='document',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

INSERT INTO document_fts(document_fts) VALUES('rebuild');

CREATE TRIGGER document_fts_insert AFTER INSERT ON document
BEGIN
    INSERT INTO document_fts(rowid, body) VALUES (new.rowid, new.body);
END;

CREATE TRIGGER document_fts_delete AFTER DELETE ON document
BEGIN
    INSERT INTO document_fts(document_fts, rowid, body)
        VALUES('delete', old.rowid, old.body);
END;

CREATE TRIGGER document_fts_update AFTER UPDATE ON document
BEGIN
    INSERT INTO document_fts(document_fts, rowid, body)
        VALUES('delete', old.rowid, old.body);
    INSERT INTO document_fts(rowid, body) VALUES (new.rowid, new.body);
END;

PRAGMA user_version = 15;

COMMIT;

PRAGMA optimize;
