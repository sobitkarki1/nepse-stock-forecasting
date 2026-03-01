"""
Unit tests for data preparation logic.
Uses synthetic DataFrames — no real NEPSE CSV required.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants (must match train.py) ─────────────────────────────────────────
SEQUENCE_LENGTH = 30


# ── Inline prepare_data to isolate from train.py file-level side effects ─────
def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = ["open", "high", "low", "close", "volume"]
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for symbol in df["symbol"].unique():
        stock_df = df[df["symbol"] == symbol].sort_values("date").copy()
        if len(stock_df) < SEQUENCE_LENGTH + 7:
            continue

        stock_df["target_3d"] = stock_df["close"].shift(-3)
        stock_df["target_5d"] = stock_df["close"].shift(-5)
        stock_df["target_7d"] = stock_df["close"].shift(-7)
        stock_df = stock_df.dropna()

        if len(stock_df) < SEQUENCE_LENGTH:
            continue

        data = stock_df[features].values
        target_data = stock_df[["target_3d", "target_5d", "target_7d"]].values

        for i in range(SEQUENCE_LENGTH, len(data)):
            sequences.append(data[i - SEQUENCE_LENGTH : i])
            targets.append(target_data[i])

    return np.array(sequences), np.array(targets)


# ── Fixtures ─────────────────────────────────────────────────────────────────
def make_stock_df(n_days: int = 100, n_stocks: int = 3) -> pd.DataFrame:
    """Create a synthetic NEPSE-style DataFrame."""
    rng = np.random.default_rng(42)
    rows = []
    base_date = pd.Timestamp("2020-01-01")
    for i in range(n_stocks):
        symbol = f"STOCK{i:02d}"
        for d in range(n_days):
            close = 100 + rng.normal(0, 5)
            rows.append({
                "symbol": symbol,
                "date": base_date + pd.Timedelta(days=d),
                "open": close * (1 + rng.normal(0, 0.01)),
                "high": close * (1 + abs(rng.normal(0, 0.01))),
                "low": close * (1 - abs(rng.normal(0, 0.01))),
                "close": close,
                "volume": int(rng.integers(1000, 100_000)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return make_stock_df(n_days=100, n_stocks=3)


@pytest.fixture
def sequences_targets(sample_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return prepare_data(sample_df)


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestPrepareData:
    def test_returns_numpy_arrays(self, sequences_targets: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = sequences_targets
        assert isinstance(X, np.ndarray), "X should be ndarray"
        assert isinstance(y, np.ndarray), "y should be ndarray"

    def test_sequence_shape(self, sequences_targets: tuple[np.ndarray, np.ndarray]) -> None:
        X, _ = sequences_targets
        assert X.ndim == 3, f"Expected 3D array (samples, seq_len, features), got shape {X.shape}"
        assert X.shape[1] == SEQUENCE_LENGTH, f"seq_len mismatch: {X.shape[1]} != {SEQUENCE_LENGTH}"
        assert X.shape[2] == 5, f"Expected 5 features (OHLCV), got {X.shape[2]}"

    def test_target_shape(self, sequences_targets: tuple[np.ndarray, np.ndarray]) -> None:
        _, y = sequences_targets
        assert y.ndim == 2, f"Expected 2D target array, got {y.ndim}D"
        assert y.shape[1] == 3, f"Expected 3 targets (3/5/7-day), got {y.shape[1]}"

    def test_samples_match(self, sequences_targets: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = sequences_targets
        assert len(X) == len(y), f"Mismatch: {len(X)} sequences vs {len(y)} targets"

    def test_no_nan_in_sequences(self, sequences_targets: tuple[np.ndarray, np.ndarray]) -> None:
        X, _ = sequences_targets
        assert not np.isnan(X).any(), "NaN values found in sequences"

    def test_no_nan_in_targets(self, sequences_targets: tuple[np.ndarray, np.ndarray]) -> None:
        _, y = sequences_targets
        assert not np.isnan(y).any(), "NaN values found in targets"

    def test_prices_positive(self, sequences_targets: tuple[np.ndarray, np.ndarray]) -> None:
        _, y = sequences_targets
        assert (y > 0).all(), "Some target prices are non-positive"

    def test_skips_short_stocks(self) -> None:
        """Stocks with fewer than SEQUENCE_LENGTH + 7 days should be skipped."""
        df = make_stock_df(n_days=20, n_stocks=5)  # too short
        X, y = prepare_data(df)
        assert len(X) == 0, "Short stocks should produce zero sequences"