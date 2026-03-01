"""
Unit tests for StockLSTM model architecture and forward pass.
Uses synthetic data — no real NEPSE CSV required.
"""

import sys
import os
import numpy as np
import pytest

# Allow import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


# ── Minimal StockLSTM copy for isolated testing ──────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ── Fixtures ─────────────────────────────────────────────────────────────────
INPUT_SIZE = 5       # open, high, low, close, volume
SEQ_LEN    = 30
BATCH_SIZE = 8


@pytest.fixture
def model() -> StockLSTM:
    return StockLSTM(input_size=INPUT_SIZE)


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_SIZE)


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestStockLSTM:
    def test_output_shape(self, model: StockLSTM, dummy_batch: torch.Tensor) -> None:
        """Output should be (batch, 3) — predicts 3-day, 5-day, 7-day prices."""
        out = model(dummy_batch)
        assert out.shape == (BATCH_SIZE, 3), f"Expected ({BATCH_SIZE}, 3), got {out.shape}"

    def test_no_nan_in_output(self, model: StockLSTM, dummy_batch: torch.Tensor) -> None:
        """Forward pass must not produce NaN values."""
        out = model(dummy_batch)
        assert not torch.isnan(out).any(), "Model output contains NaN"

    def test_no_inf_in_output(self, model: StockLSTM, dummy_batch: torch.Tensor) -> None:
        """Forward pass must not produce infinite values."""
        out = model(dummy_batch)
        assert not torch.isinf(out).any(), "Model output contains Inf"

    def test_parameters_count(self, model: StockLSTM) -> None:
        """Model should have a reasonable parameter count (>10k, <10M for defaults)."""
        n = sum(p.numel() for p in model.parameters())
        assert 10_000 < n < 10_000_000, f"Unexpected parameter count: {n}"

    def test_different_batch_sizes(self, model: StockLSTM) -> None:
        """Model should handle batch sizes 1, 4, and 32."""
        for bs in (1, 4, 32):
            x = torch.randn(bs, SEQ_LEN, INPUT_SIZE)
            out = model(x)
            assert out.shape == (bs, 3)

    def test_gradient_flows(self, model: StockLSTM, dummy_batch: torch.Tensor) -> None:
        """Gradients should flow through all parameters on backward pass."""
        out = model(dummy_batch)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_eval_mode_no_dropout(self, model: StockLSTM, dummy_batch: torch.Tensor) -> None:
        """eval() mode should produce deterministic output (dropout disabled)."""
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_batch)
            out2 = model(dummy_batch)
        assert torch.allclose(out1, out2), "Non-deterministic output in eval mode"