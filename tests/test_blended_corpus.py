"""Unit tests for format and sampling rules of the streamed corpus builder."""
from __future__ import annotations


def test_maintenance_rows_become_instruction_completion_examples():
    from scripts.prepare_blended_corpus import maintenance_to_text

    text = maintenance_to_text({
        "Equipment_ID": "P-101",
        "OrderType": "CM",
        "WorkOrderDescription": "Pump vibration is high.",
        "OperationDescription": "Isolate the pump and inspect the bearing.",
    })

    assert "<|user|>" in text and "<|assistant|>" in text
    assert "P-101" in text and "inspect the bearing" in text


def test_nemotron_chat_list_is_rendered():
    from scripts.prepare_blended_corpus import nemotron_to_text

    text = nemotron_to_text({
        "input": [{"role": "user", "content": "Solve 2 + 2."}],
        "output": "4",
    })

    assert text == "<|user|> Solve 2 + 2. <|assistant|> 4"


def test_sequential_jsonl_stream_uses_one_non_range_http_request(monkeypatch):
    from scripts.prepare_blended_corpus import stream_jsonl

    class _Response:
        def raise_for_status(self):
            return None
        def iter_lines(self, chunk_size):
            assert chunk_size == 128 * 1024
            return iter([b'{"input": "question", "output": "answer"}'])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    captured = {}
    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return _Response()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    assert list(stream_jsonl("https://example.test/source.jsonl")) == [{"input": "question", "output": "answer"}]
    assert captured["stream"] is True
    assert "Range" not in captured["headers"]


def test_local_source_paths_match_the_offline_downloader_layout(tmp_path):
    from scripts.prepare_blended_corpus import local_source_streams

    # Generators are lazy: constructing offline streams must not touch the
    # files before the corpus builder starts consuming a source.
    streams = local_source_streams(tmp_path)
    assert set(streams) == {"maintenance", "nemotron", "openorca"}


def test_default_blend_counts_sum_and_favour_maintenance():
    from scripts.prepare_blended_corpus import DEFAULT_WEIGHTS, _row_counts

    counts = _row_counts(300_000, DEFAULT_WEIGHTS)
    assert sum(counts.values()) == 300_000
    assert counts == {
        "maintenance": 120_000,
        "fineweb_edu": 75_000,
        "nemotron": 45_000,
        "openorca": 60_000,
    }


def test_weighted_interleave_does_not_concatenate_sources():
    from scripts.prepare_blended_corpus import interleave_weighted

    streams = {name: iter([{"source": name}] * count) for name, count in {
        "maintenance": 9, "nemotron": 7, "openorca": 4,
    }.items()}
    emitted = [name for name, _ in interleave_weighted(streams, {
        "maintenance": 9, "nemotron": 7, "openorca": 4,
    })]

    assert len(emitted) == 20
    assert emitted[:3] == ["maintenance", "nemotron", "openorca"]
    assert emitted.count("maintenance") == 9
