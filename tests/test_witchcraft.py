"""
Smoke tests for the witchcraft Python extension module.

Run after building and installing the wheel:
    make python-wheel
    pip install target/wheels/witchcraft-*.whl
    pytest tests/test_witchcraft.py

Tests marked 'needs_assets' require model weights in assets/:
    make download
    pytest tests/test_witchcraft.py
"""

import os
import tempfile
import pytest
import witchcraft

ASSETS = os.path.join(os.path.dirname(__file__), '..', 'assets')

needs_assets = pytest.mark.skipif(
    not os.path.exists(os.path.join(ASSETS, 'config.json')),
    reason="model assets not downloaded — run 'make download'",
)

FACTS = [
    "Octopuses have three hearts and blue blood.",
    "A group of flamingos is called a flamboyance.",
    "Honey never spoils; archaeologists have found 3000-year-old edible honey.",
    "There is a lake in Australia that stays bright pink regardless of conditions.",
    "Sharks existed before trees.",
]


# --- Error propagation (Witchcraft::new fails fast on bad paths) ---

def test_bad_db_path_raises():
    with pytest.raises(RuntimeError):
        witchcraft.Witchcraft('/nonexistent/path/db.sqlite', ASSETS)


def test_bad_assets_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(RuntimeError):
            witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), '/nonexistent/assets')


# --- Drop / shutdown ---

@needs_assets
def test_drop_without_shutdown_does_not_hang():
    """Drop impl should join the indexer thread cleanly without explicit shutdown()."""
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        del wc  # triggers Drop


@needs_assets
def test_explicit_shutdown_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        wc.shutdown()
        # second shutdown (or Drop after shutdown) must not panic
        del wc


# --- Round-trip: add / index / search / score ---

@needs_assets
def test_search_returns_results():
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        for i, body in enumerate(FACTS):
            wc.add(
                f'00000000-0000-0000-0000-{i:012d}',
                '2024-01-01T00:00:00Z',
                '{}',
                body,
            )
        wc.index()
        wc.shutdown()

        # Re-open for search (new instance, same DB)
        wc2 = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        results = wc2.search('group of birds called a flamboyance', threshold=0.3, top_k=5)
        wc2.shutdown()

    assert isinstance(results, list)
    assert len(results) >= 1
    top = results[0]
    assert set(top.keys()) == {'score', 'metadata', 'body', 'idx', 'date'}
    assert isinstance(top['score'], float)
    assert 0.0 <= top['score'] <= 1.0


@needs_assets
def test_score_ranks_relevant_sentence_highest():
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        sentences = [
            "Flamingos are pink birds that live in flocks.",
            "Honey bees produce honey from flower nectar.",
            "Sharks are ancient fish that predate trees.",
        ]
        scores = wc.score('a group of flamingos', sentences)
        wc.shutdown()

    assert len(scores) == len(sentences)
    assert all(isinstance(s, float) for s in scores)
    assert scores[0] == max(scores), "flamingo sentence should score highest"


@needs_assets
def test_remove_doc():
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        uid = '00000000-0000-0000-0000-000000000001'
        wc.add(uid, '2024-01-01T00:00:00Z', '{}', FACTS[1])
        wc.remove(uid)
        wc.index()
        wc.shutdown()


@needs_assets
def test_clear():
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        for i, body in enumerate(FACTS):
            wc.add(f'00000000-0000-0000-0000-{i:012d}', '2024-01-01T00:00:00Z', '{}', body)
        wc.clear()
        wc.shutdown()


# --- Input validation ---

@needs_assets
def test_add_invalid_uuid_raises():
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        with pytest.raises(ValueError):
            wc.add('not-a-uuid', '2024-01-01T00:00:00Z', '{}', 'body')
        wc.shutdown()


@needs_assets
def test_add_invalid_date_raises():
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        with pytest.raises(ValueError):
            wc.add('550e8400-e29b-41d4-a716-446655440000', 'not-a-date', '{}', 'body')
        wc.shutdown()


@needs_assets
def test_remove_invalid_uuid_raises():
    with tempfile.TemporaryDirectory() as d:
        wc = witchcraft.Witchcraft(os.path.join(d, 'test.sqlite'), ASSETS)
        with pytest.raises(ValueError):
            wc.remove('bad-uuid')
        wc.shutdown()
