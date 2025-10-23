# 🚀 NEPSE Stock Price Forecasting with PyTorch LSTM

A lightweight deep learning model for predicting Nepal Stock Exchange (NEPSE) stock prices using LSTM neural networks. The model provides multi-horizon forecasts (3, 5, and 7 days) with GPU acceleration for fast training.

## 📊 Project Overview

This project uses a PyTorch LSTM model to analyze historical stock data from NEPSE and predict future price movements. The model is trained on 550+ stocks with 500K+ data sequences, achieving accurate predictions across multiple time horizons.

## ✨ Features

- **Multi-Horizon Forecasting**: Predicts stock prices for 3, 5, and 7 days ahead
- **GPU Accelerated**: Optimized for NVIDIA GPUs (tested on GTX 1650 Ti)
- **Batch Predictions**: Analyze all stocks and identify top opportunities
- **Lightweight Model**: Only 200K parameters for fast training and inference
- **High Accuracy**: 9-13% MAPE across all prediction horizons

## 🏗️ Model Architecture

- **Type**: LSTM (Long Short-Term Memory)
- **Layers**: 2 LSTM layers with 128 hidden units
- **Input Features**: Open, High, Low, Close, Volume
- **Output**: 3 predictions (3-day, 5-day, 7-day prices)
- **Parameters**: 201,603 trainable parameters
- **Framework**: PyTorch

## ⚡ Performance Metrics

| Horizon | MAE | RMSE | MAPE |
|---------|-----|------|------|
| 3-Day   | 23.59 | 50.50 | 9.18% |
| 5-Day   | 27.44 | 60.04 | 11.10% |
| 7-Day   | 30.22 | 67.18 | 13.06% |

## ⏱️ Training Time

- **GPU**: ~2-3 minutes (50 epochs on NVIDIA GTX 1650 Ti)
- **CPU**: ~15-20 minutes (50 epochs on modern CPU)
- **Dataset**: 508,169 sequences from 550 stocks
- **Batch Size**: 256 (GPU), 32 (CPU)

## 📈 Top 10 Predicted Stocks (Next 3 Days)

Based on the latest predictions (as of October 22, 2025):

| Rank | Symbol | Current Price | Predicted Price | Expected Return |
|------|--------|---------------|-----------------|-----------------|
| 1 | H8020 | NPR 10.00 | NPR 17.83 | +78.27% 🚀 |
| 2 | LEMF | NPR 9.47 | NPR 16.11 | +70.07% 🚀 |
| 3 | NABILP | NPR 301.10 | NPR 498.13 | +65.44% 🚀 |
| 4 | PRSF | NPR 10.05 | NPR 15.62 | +55.40% 🚀 |
| 5 | CMF1 | NPR 10.85 | NPR 15.66 | +44.32% 🚀 |
| 6 | PSF | NPR 10.25 | NPR 12.54 | +22.36% 📈 |
| 7 | HFL | NPR 18.00 | NPR 19.95 | +10.84% 📈 |
| 8 | CFL | NPR 117.00 | NPR 129.29 | +10.50% 📈 |
| 9 | KBLPO | NPR 109.00 | NPR 118.69 | +8.89% 📈 |
| 10 | GBIME | NPR 240.00 | NPR 257.42 | +7.26% 📈 |

*Note: High predicted returns should be verified with fundamental analysis. Past performance doesn't guarantee future results.*

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib

### Training the Model

```bash
python train.py
```

**Output:**
- `best_model.pth` - Trained model checkpoint
- `training_history.png` - Loss curves visualization
- `predictions.png` - Prediction accuracy plots

### Making Predictions

**Predict specific stocks:**
```bash
python predict.py
```

**Find top 10 opportunities:**
```bash
python predict_top10.py
```

**Output:**
- `all_predictions.csv` - Predictions for all stocks
- `top10_predictions.csv` - Top 10 stocks by expected return

## 📁 Project Structure

```
nepse-ml/
├── train.py                              # Main training script
├── predict.py                            # Predict specific stocks
├── predict_top10.py                      # Find top opportunities
├── requirements.txt                      # Dependencies
├── nepse_stock_data_merged_deduped.csv  # Dataset
├── best_model.pth                        # Trained model
├── training_history.png                  # Training visualization
├── predictions.png                       # Prediction plots
├── all_predictions.csv                   # All stock predictions
└── top10_predictions.csv                 # Top 10 stocks
```

## 🎯 Use Cases

1. **Day Trading**: Identify short-term price movements (3-day forecasts)
2. **Swing Trading**: Plan medium-term positions (5-7 day forecasts)
3. **Portfolio Analysis**: Evaluate multiple stocks simultaneously
4. **Risk Management**: Compare predictions across different horizons
5. **Market Screening**: Find stocks with highest growth potential

## 🔧 Technical Details

### Data Preprocessing
- Sequences of 30 days used for prediction
- StandardScaler normalization for features and targets
- Train/Val/Test split: 70%/15%/15% (time-based)

### GPU Optimization
- Batch size: 256 (optimized for GPU memory)
- DataLoader with 4 workers and pin_memory
- CUDNN benchmark mode enabled
- Mixed precision training ready

### Model Configuration
```python
SEQUENCE_LENGTH = 30      # Days of history
BATCH_SIZE = 256          # GPU optimized
HIDDEN_SIZE = 128         # LSTM hidden units
NUM_LAYERS = 2            # LSTM layers
LEARNING_RATE = 0.001     # Adam optimizer
EPOCHS = 50               # Training epochs
```

## 📊 Sample Predictions

**NABIL (Commercial Bank)**
- Current: NPR 507.90
- 3-day: NPR 514.56 (+1.31%)
- 5-day: NPR 514.60 (+1.32%)
- 7-day: NPR 513.58 (+1.12%)

**SCB (Standard Chartered Bank)**
- Current: NPR 629.00
- 3-day: NPR 637.53 (+1.36%)
- 5-day: NPR 637.89 (+1.41%)
- 7-day: NPR 636.95 (+1.26%)

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Stock predictions are based on historical patterns and should not be considered financial advice. Always:

- Conduct your own research
- Consult with financial advisors
- Consider fundamental analysis
- Be aware of market risks
- Never invest more than you can afford to lose

## 🛠️ Future Improvements

- [ ] Add technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Implement attention mechanisms
- [ ] Add sentiment analysis from news
- [ ] Real-time data integration
- [ ] Web dashboard for visualization
- [ ] Ensemble methods (LSTM + GRU + Transformer)
- [ ] Hyperparameter optimization
- [ ] Explainable AI features

## 📝 License

MIT License - Feel free to use for educational purposes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**

*Last Updated: October 23, 2025*
