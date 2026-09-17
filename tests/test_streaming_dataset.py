from __future__ import annotations


class _Tokenizer:
    def encode(self, text, add_bos=True, add_eos=True):
        return ([1] if add_bos else []) + [ord(char) % 17 for char in text] + ([2] if add_eos else [])


def test_streaming_dataset_emits_fixed_token_windows_without_reading_all_rows(tmp_path):
    from training.dataset import StreamingTokenizedTextDataset

    path = tmp_path / "corpus.txt"
    path.write_text("abc\ndef\n", encoding="utf-8")
    samples = list(StreamingTokenizedTextDataset(path, _Tokenizer(), block_size=4))

    assert len(samples) == 2
    assert all(inputs.shape == targets.shape == (4,) for inputs, targets in samples)
