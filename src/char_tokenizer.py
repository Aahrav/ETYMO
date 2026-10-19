"""
Character Tokenizer for Deep Learning Etymology Classifier
===========================================================
Encodes English and etymological root word strings into padded sequence
tensors of character IDs for PyTorch neural models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch


class CharTokenizer:
    """Character-level tokenizer supporting special tokens and sequence padding."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    PAD_IDX = 0
    UNK_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, char2idx: Optional[Dict[str, int]] = None, max_len: int = 24):
        self.max_len = max_len
        if char2idx is not None:
            self.char2idx = char2idx
            self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        else:
            self.char2idx = {
                self.PAD_TOKEN: self.PAD_IDX,
                self.UNK_TOKEN: self.UNK_IDX,
                self.SOS_TOKEN: self.SOS_IDX,
                self.EOS_TOKEN: self.EOS_IDX,
            }
            self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}

    def __len__(self) -> int:
        return len(self.char2idx)

    def fit(self, words: List[str]) -> "CharTokenizer":
        """Build character vocabulary from a list of words."""
        unique_chars = set()
        for word in words:
            if isinstance(word, str):
                for ch in word.strip().lower():
                    unique_chars.add(ch)

        # Sort characters for deterministic ID assignment
        for ch in sorted(unique_chars):
            if ch not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch

        return self

    def encode(
        self,
        word: str,
        max_len: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> Tuple[List[int], int]:
        """
        Encode a word into a fixed-length list of character indices and its true length.
        """
        if max_len is None:
            max_len = self.max_len

        clean_word = str(word).strip().lower()
        indices = []

        if add_special_tokens:
            indices.append(self.SOS_IDX)

        for ch in clean_word:
            indices.append(self.char2idx.get(ch, self.UNK_IDX))

        if add_special_tokens:
            indices.append(self.EOS_IDX)

        true_length = min(len(indices), max_len)

        # Truncate if longer than max_len
        if len(indices) > max_len:
            indices = indices[:max_len]
            if add_special_tokens:
                indices[-1] = self.EOS_IDX
        # Pad with PAD_IDX if shorter
        elif len(indices) < max_len:
            indices.extend([self.PAD_IDX] * (max_len - len(indices)))

        return indices, true_length

    def encode_batch(
        self,
        words: List[str],
        max_len: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of words into PyTorch tensors (input_ids, lengths)."""
        all_ids = []
        all_lengths = []
        for word in words:
            ids, length = self.encode(
                word, max_len=max_len, add_special_tokens=add_special_tokens
            )
            all_ids.append(ids)
            all_lengths.append(length)

        return (
            torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(all_lengths, dtype=torch.long),
        )

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Decode character indices back to string."""
        chars = []
        for idx in indices:
            if skip_special_tokens and idx in (
                self.PAD_IDX,
                self.SOS_IDX,
                self.EOS_IDX,
            ):
                continue
            chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save vocabulary to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_len": self.max_len,
            "char2idx": self.char2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CharTokenizer":
        """Load vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(char2idx=payload["char2idx"], max_len=payload.get("max_len", 24))


def build_and_save_vocab(
    dataset_csv: Union[str, Path] = "data/origin_dataset.csv",
    output_json: Union[str, Path] = "models/char_vocab.json",
    max_len: int = 24,
) -> CharTokenizer:
    """Build vocabulary from dataset and persist to models/char_vocab.json."""
    import pandas as pd

    df = pd.read_csv(dataset_csv)
    words = df["word"].dropna().astype(str).tolist()

    tokenizer = CharTokenizer(max_len=max_len)
    tokenizer.fit(words)
    tokenizer.save(output_json)

    print(f"✓ Character vocabulary created: {len(tokenizer)} tokens (max_len={max_len})")
    print(f"✓ Saved vocabulary to {output_json}")
    return tokenizer


if __name__ == "__main__":
    build_and_save_vocab()
