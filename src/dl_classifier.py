"""
Deep Learning Etymology Classifier (Phase 1)
=============================================
Neural sequence model utilizing Character Embeddings, Bidirectional LSTM,
and Self-Attention Pooling to identify historical linguistic origins.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.char_tokenizer import CharTokenizer
from src.config import ORIGIN_CLASSES, MODELS_DIR


class SelfAttentionPooling(nn.Module):
    """
    Learns an attention weight for each character timestep in the sequence,
    allowing the model to focus on morphologically salient prefixes, stems,
    and suffixes while ignoring padding.
    """

    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(
        self,
        lstm_outputs: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch_size, seq_len, input_dim)
            pad_mask: (batch_size, seq_len), True where token is padding (<pad>)

        Returns:
            context_vector: (batch_size, input_dim)
            attention_weights: (batch_size, seq_len)
        """
        # Calculate raw energy scores: (batch_size, seq_len, 1)
        energy = self.projection(lstm_outputs)
        energy = energy.squeeze(-1)  # (batch_size, seq_len)

        # Mask padding tokens with large negative value before softmax
        if pad_mask is not None:
            energy = energy.masked_fill(pad_mask, -1e9)

        # Softmax over sequence length
        attention_weights = F.softmax(energy, dim=-1)  # (batch_size, seq_len)

        # Compute weighted sum of LSTM hidden states
        # (batch_size, 1, seq_len) x (batch_size, seq_len, input_dim) -> (batch_size, 1, input_dim)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), lstm_outputs
        ).squeeze(1)

        return context_vector, attention_weights


class CharBiLSTMAttention(nn.Module):
    """
    Character-level Bidirectional LSTM with Self-Attention Pooling for
    morphology-aware historical root prediction.
    """

    def __init__(
        self,
        vocab_size: int = 50,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = len(ORIGIN_CLASSES),
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.padding_idx = padding_idx

        # 1. Learned character embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.emb_dropout = nn.Dropout(dropout)

        # 2. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3. Morphological Self-Attention Pooling
        # Bidirectional outputs have dimension hidden_dim * 2
        lstm_output_dim = hidden_dim * 2
        self.attention = SelfAttentionPooling(
            input_dim=lstm_output_dim,
            attention_dim=64,
        )

        # 4. Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            input_ids: (batch_size, seq_len) tensor of character IDs
            return_attention: If True, also returns character attention weights

        Returns:
            logits: (batch_size, num_classes)
            attention_weights (optional): (batch_size, seq_len)
        """
        # Create padding mask (True where token == padding_idx)
        pad_mask = input_ids == self.padding_idx

        # Character embedding: (batch_size, seq_len, embedding_dim)
        x = self.embedding(input_ids)
        x = self.emb_dropout(x)

        # LSTM sequence encoding: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Self-attention pooling
        context, attn_weights = self.attention(lstm_out, pad_mask=pad_mask)

        # Classification logits: (batch_size, num_classes)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits


def test_model_forward():
    """Unit test for CharBiLSTMAttention model instantiation and forward pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing CharBiLSTMAttention forward pass on: {device}")

    vocab_size = 50
    model = CharBiLSTMAttention(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_classes=6,
    ).to(device)

    # Dummy batch of 4 words, seq_len 24
    dummy_input = torch.randint(0, vocab_size, (4, 24)).to(device)
    # Simulate padding on the last tokens
    dummy_input[:, 18:] = 0

    logits, attn = model(dummy_input, return_attention=True)

    print("✓ Output logits shape:", logits.shape)
    print("✓ Attention weights shape:", attn.shape)
    print("✓ Softmax sum per row (should be 1.0):", attn.sum(dim=-1).tolist())
    print("✓ Model parameters count:", sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == "__main__":
    test_model_forward()
