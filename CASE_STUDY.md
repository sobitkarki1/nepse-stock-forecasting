# NEPSE Stock Forecasting — Engineering Case Study

> **Project:** [`nepse-stock-forecasting`](https://github.com/sobitkarki1/nepse-stock-forecasting)
> **Stack:** Python · PyTorch · pandas · scikit-learn
> **Scale:** 550 stocks · 508,169 training sequences · 201,603 model parameters

---

## 1. Problem Statement

The Nepal Stock Exchange (NEPSE) lacks publicly available, open-source forecasting tools. Most retail investors in Nepal rely on intuition or delayed broker reports. This project set out to answer a concrete question: *can an LSTM trained purely on price/volume history give useful short-horizon forecasts across all NEPSE-listed equities?*

The goal was deliberately narrow and honest:
- Multi-horizon outputs: 3-day, 5-day, 7-day closing price
- Single model covering 550+ symbols (no per-stock fine-tuning)
- Reproducible pipeline from raw CSV to checkpoint

---

## 2. Data Engineering

### Source & Shape

The dataset is a merged, deduplicated CSV (`nepse_stock_data_merged_deduped.csv`) covering all NEPSE-listed stocks. Each row has five OHLCV columns:

| Column | Description |
|--------|-------------|
| Open   | Opening price (NPR) |
| High   | Intraday high (NPR) |
| Low    | Intraday low (NPR) |
| Close  | Closing price (NPR) |
| Volume | Traded volume |

### Key Preprocessing Decisions

**Rolling-window sequences.** Rather than feeding raw time series directly, the pipeline creates overlapping 30-day windows per stock. Each window becomes one training sample with a three-value target `[Close+3, Close+5, Close+7]`. This produced 508,169 sequences from 550 stocks — enough data for a single shared model.

**Per-feature StandardScaler normalization.** Raw NEPSE prices range from ~NPR 10 (penny stocks) to NPR 3,000+ (blue chips) in the same training batch. Un-normalized, gradient updates would be dominated by high-price stocks. Fitting a `StandardScaler` per feature on the training split before sliding the window prevents this bias.

**Time-based train/val/test split (70/15/15).** Shuffling time-series data leaks future information into the past. The split is applied chronologically: the earliest 70% of each stock's history is training data, the next 15% validation, the final 15% test. This mirrors real deployment conditions.

**Short-stock filtering.** Stocks with fewer than 31 trading days are silently skipped — there is not enough history to form even one sequence.

---

## 3. Model Architecture

### Why LSTM?

Transformer models require much more data and compute to outperform simpler recurrent networks on short financial time series. An LSTM with 2 layers and 128 hidden units struck the right trade-off: expressive enough to capture non-linear momentum patterns, small enough to train in 2–3 minutes on a GTX 1650 Ti.

### Architecture Summary

```
Input:    (batch, 30, 5)          # 30 days × 5 OHLCV features
LSTM-1:   hidden=128, layers=2, batch_first=True, dropout=0.2
                                   # dropout applied between layers
Final hidden state → Linear(128, 3)
Output:   (batch, 3)              # [price+3, price+5, price+7]
```

**Why take the final hidden state instead of all outputs?** The forecast target is a single future window, not a sequence. Using only `h_n[-1]` (last-layer, last-timestep hidden state) keeps the output head minimal and avoids temporal leakage from intermediate states.

**Parameter count: 201,603** — deliberately constrained to avoid overfitting on the noisiest penny stocks.

### Configuration

```python
SEQUENCE_LENGTH = 30    # days of look-back history
HIDDEN_SIZE     = 128   # LSTM hidden units per layer
NUM_LAYERS      = 2     # stacked LSTM layers
DROPOUT         = 0.2   # inter-layer dropout (disabled in eval mode)
LEARNING_RATE   = 0.001 # Adam optimizer
EPOCHS          = 50    # early stopping via best-val checkpoint
BATCH_SIZE      = 256   # GPU-optimized; 32 for CPU
```

---

## 4. Training Pipeline

### Loss Function

Mean Squared Error over the three output heads. Summing over all three horizons in a single backward pass forces the model to learn a shared representation of short-term momentum rather than three independent detached signals.

### Checkpointing

`best_model.pth` is saved whenever validation loss improves. This acts as implicit early stopping without a hard epoch count, and the saved weights are the ones actually used for inference.

### GPU Acceleration

```python
torch.backends.cudnn.benchmark = True  # auto-tune convolution paths
DataLoader(pin_memory=True, num_workers=4)  # reduce CPU→GPU transfer latency
```

On GTX 1650 Ti (4 GB VRAM): 50 epochs over 508K sequences completes in ~2–3 minutes. On CPU: ~15–20 minutes with batch size reduced to 32.

---

## 5. Evaluation Metrics

Results on the held-out 15% test set (time-based split), averaged over all 550 stocks:

| Horizon | MAE (NPR) | RMSE (NPR) | MAPE   |
|---------|-----------|------------|--------|
| 3-Day   | 23.59     | 50.50      | 9.18%  |
| 5-Day   | 27.44     | 60.04      | 11.10% |
| 7-Day   | 30.22     | 67.18      | 13.06% |

**Interpreting MAPE on NEPSE.** Because penny stocks (NPR 10–30) make up a large fraction of the 550-symbol universe, absolute errors are small but percentage errors are amplified even by small mis-predictions. The 9–13% MAPE should be read alongside MAE: a 3-day MAE of NPR 23.59 is negligible for a NPR 500+ stock and significant for a NPR 10 stock. Users focused on blue-chip equities will see materially better MAPE than the average.

**No directional accuracy metric was computed** in the initial release — this is called out as a known gap. A model that predicts a price NPR 5 below the actual is good on MAPE but useless if direction is random. Directional accuracy (% of predictions with correct sign) is listed as a future improvement.

---

## 6. Key Engineering Trade-offs

| Decision | Alternative Considered | Reason Chosen |
|----------|----------------------|---------------|
| Single shared model | One model per stock | 550 models × retraining overhead is impractical without a scheduler |
| 30-day look-back | 60 or 90 days | Shorter sequences → more training samples; NEPSE has many new-listed stocks |
| LSTM over GRU | GRU (fewer params) | LSTM's cell state gives the model a longer-range memory path at minimal extra cost |
| MSE loss | Huber / MAE loss | MSE penalises large outlier errors more heavily, which matters for gap-up/down events |
| StandardScaler | MinMaxScaler | StandardScaler is more robust to the extreme outliers common in thinly-traded stocks |
| Time split | Random shuffle | Prevents look-ahead bias — critical for any time-series evaluation |

---

## 7. Known Limitations

1. **No fundamental data.** The model sees only price and volume. Earnings announcements, rights issues, and dividend declarations are invisible to it.
2. **Single architecture for all sectors.** Banks, hydropower, and mutual funds behave differently. A sector-conditioned model or ensemble could improve results.
3. **Static dataset.** There is no live data ingestion pipeline yet. Predictions are as current as the last CSV export.
4. **No directional accuracy metric.** MAPE and MAE measure magnitude, not whether the predicted move direction is correct.
5. **Penny stock noise.** The 550-stock universe includes many illiquid stocks where the model's predictions are less reliable.

---

## 8. Results Snapshot

Top 10 predicted stocks (3-day horizon, as of October 22 2025 snapshot):

| Symbol  | Current (NPR) | Predicted (NPR) | Expected Return |
|---------|--------------|-----------------|-----------------|
| H8020   | 10.00        | 17.83           | +78.3%          |
| LEMF    | 9.47         | 16.11           | +70.1%          |
| NABILP  | 301.10       | 498.13          | +65.4%          |
| NABIL   | 507.90       | 514.56          | +1.3%           |
| SCB     | 629.00       | 637.53          | +1.4%           |

*High return estimates from penny stocks are the model's extrapolation of historical momentum — they should be cross-checked against fundamentals before any trading decision.*

---

## 9. Repository Structure

```
nepse-stock-forecasting/
├── train.py                        # Data processing + model training
├── predict.py                      # Single-stock inference
├── predict_top10.py                # Batch inference, top-10 ranking
├── tests/
│   ├── test_model.py               # 7 unit tests: shapes, gradients, NaN
│   └── test_data.py                # 8 unit tests: prepare_data() output
├── .github/workflows/ci.yml        # Ruff lint + pytest on every push
├── nepse_stock_data_merged_deduped.csv
├── best_model.pth
├── requirements.txt
└── README.md
```

---

## 10. Future Roadmap

- [ ] Technical indicators (RSI, MACD, Bollinger Bands) as additional features
- [ ] Directional accuracy metric in evaluation
- [ ] Attention mechanism over the 30-day window
- [ ] Live data ingestion from NEPSE API
- [ ] Sector-conditioned training (banks vs hydropower vs mutual funds)
- [ ] Streamlit or FastAPI dashboard for real-time predictions
- [ ] Ensemble: LSTM + GRU + Transformer with learned weighting

---

## 11. Reproducing the Results

```bash
git clone https://github.com/sobitkarki1/nepse-stock-forecasting
cd nepse-stock-forecasting
pip install -r requirements.txt

# Train
python train.py
# → best_model.pth, training_history.png, predictions.png

# Predict top 10
python predict_top10.py
# → top10_predictions.csv, all_predictions.csv

# Run tests
pytest tests/ -v
```

---

> **Disclaimer:** This project is for educational and research purposes only. Stock predictions are based solely on historical price patterns and do not constitute financial advice.