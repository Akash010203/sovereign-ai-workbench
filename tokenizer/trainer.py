"""
Byte-level BPE *training* -- i.e. learning the merge rules from a
corpus of raw text.

ALGORITHM (summary)
--------------------
1. Split the corpus into "words" using the same simple regex
   pre-tokenizer used at encode time (``PRETOKEN_PATTERN``).
2. Represent every word as a tuple of single bytes.
3. Count how often every adjacent pair of symbols occurs across the
   whole corpus (weighted by how often each word occurs).
4. Merge the single most frequent pair into one new symbol, everywhere
   it occurs.
5. Repeat steps 3-4 until the vocabulary reaches the target size, or
   the most frequent remaining pair is too rare to be worth learning.

Each merge is recorded, in the order it was learned. That order is
exactly what ``tokenizer/tokenizer.py`` replays (lowest rank = applied
first) when encoding new text.
"""
from __future__ import annotations

from collections import Counter

from tokenizer.tokenizer import (
    BASE_BYTE_VOCAB_SIZE,
    DEFAULT_SPECIAL_TOKENS,
    PRETOKEN_PATTERN,
    TokenizerVocab,
)

Symbol = bytes
Word = tuple[Symbol, ...]


def _pretokenize_corpus(lines: list[str]) -> Counter[Word]:
    """Turn raw text lines into word -> frequency counts.

    Each "word" is represented as a tuple of single bytes, e.g. the
    pre-token " the" (with its leading space) becomes
    ``(b' ', b't', b'h', b'e')``.
    """
    word_freqs: Counter[Word] = Counter()
    for line in lines:
        for piece in PRETOKEN_PATTERN.findall(line):
            piece_bytes = piece.encode("utf-8")
            word = tuple(bytes([b]) for b in piece_bytes)
            if word:
                word_freqs[word] += 1
    return word_freqs


def _count_pairs(word_freqs: Counter[Word]) -> Counter[tuple[Symbol, Symbol]]:
    """Count every adjacent symbol pair, weighted by word frequency."""
    pair_freqs: Counter[tuple[Symbol, Symbol]] = Counter()
    for word, freq in word_freqs.items():
        for left, right in zip(word[:-1], word[1:]):
            pair_freqs[(left, right)] += freq
    return pair_freqs


def _merge_word(word: Word, pair: tuple[Symbol, Symbol]) -> Word:
    """Replace every occurrence of ``pair`` inside ``word`` with one symbol."""
    merged_symbol = pair[0] + pair[1]
    new_word: list[Symbol] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(merged_symbol)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(
    corpus_lines: list[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    min_pair_frequency: int = 2,
) -> TokenizerVocab:
    """Learn a byte-level BPE vocabulary from ``corpus_lines``.

    Args:
        corpus_lines: raw training text, one string per line.
        vocab_size: target vocabulary size (special tokens + 256 base
            bytes + learned merges). Training stops early if no pair
            occurs at least ``min_pair_frequency`` times.
        special_tokens: reserved token names, e.g. ``["<pad>", "<unk>"]``.
        min_pair_frequency: stop merging once the most frequent
            remaining pair occurs fewer than this many times. This
            keeps the tokenizer from learning merges that only
            happened once, which would just memorize the toy corpus
            instead of learning genuine sub-word structure.

    Returns:
        A :class:`TokenizerVocab` containing the base-byte vocabulary
        and the ordered list of learned merges.
    """
    special_tokens = list(special_tokens) if special_tokens else list(
        DEFAULT_SPECIAL_TOKENS
    )
    num_special = len(special_tokens)

    if vocab_size < num_special + BASE_BYTE_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size ({vocab_size}) must be >= "
            f"len(special_tokens) ({num_special}) + 256 base bytes."
        )

    word_freqs = _pretokenize_corpus(corpus_lines)

    # The 256 raw byte values always occupy the first block of ids
    # after the special tokens.
    id_to_bytes: dict[int, bytes] = {
        num_special + byte_value: bytes([byte_value])
        for byte_value in range(BASE_BYTE_VOCAB_SIZE)
    }
    merges: list[tuple[bytes, bytes]] = []

    num_merges_needed = vocab_size - num_special - BASE_BYTE_VOCAB_SIZE

    for _ in range(num_merges_needed):
        pair_freqs = _count_pairs(word_freqs)
        if not pair_freqs:
            break  # nothing left to merge (every word is a single symbol)

        best_pair, best_freq = pair_freqs.most_common(1)[0]
        if best_freq < min_pair_frequency:
            break  # remaining pairs are too rare to be worth learning

        new_token_id = num_special + BASE_BYTE_VOCAB_SIZE + len(merges)
        id_to_bytes[new_token_id] = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        word_freqs = Counter(
            {
                _merge_word(word, best_pair): freq
                for word, freq in word_freqs.items()
            }
        )

    return TokenizerVocab(
        special_tokens=special_tokens,
        id_to_bytes=id_to_bytes,
        merges=merges,
    )
