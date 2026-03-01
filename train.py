"""
Minimal PyTorch LSTM for NEPSE Stock Price Prediction
Predicts 3, 5, and 7-day future prices
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Configuration
SEQUENCE_LENGTH = 30  # Use 30 days of history
BATCH_SIZE = 256  # Larger batch for GPU
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128  # Bigger for GPU
NUM_LAYERS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # Auto-optimize for GPU

# Simple LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 3)  # Predict 3, 5, 7 days
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Last time step

# Dataset
class StockDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prepare sequences and targets"""
    print("Preparing data...")
    
    # Use basic features
    features = ['open', 'high', 'low', 'close', 'volume']
    
    sequences = []
    targets = []
    
    for symbol in df['symbol'].unique():
        stock_df = df[df['symbol'] == symbol].sort_values('date').copy()
        
        if len(stock_df) < SEQUENCE_LENGTH + 7:
            continue
        
        # Create targets (future close prices)
        stock_df['target_3d'] = stock_df['close'].shift(-3)
        stock_df['target_5d'] = stock_df['close'].shift(-5)
        stock_df['target_7d'] = stock_df['close'].shift(-7)
        
        stock_df = stock_df.dropna()
        
        if len(stock_df) < SEQUENCE_LENGTH:
            continue
        
        data = stock_df[features].values
        target_data = stock_df[['target_3d', 'target_5d', 'target_7d']].values
        
        # Create sequences
        for i in range(SEQUENCE_LENGTH, len(data)):
            sequences.append(data[i-SEQUENCE_LENGTH:i])
            targets.append(target_data[i])
    
    return np.array(sequences), np.array(targets)

def train_model() -> None:
    """Main training function"""
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('nepse_stock_data_merged_deduped.csv')
    df.columns = df.columns.str.lower()  # Convert to lowercase
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df)} records for {df['symbol'].nunique()} stocks")
    
    # Prepare sequences
    X, y = prepare_data(df)
    print(f"\nCreated {len(X)} sequences")
    print(f"Sequence shape: {X.shape}, Target shape: {y.shape}")
    
    # Train/val/test split (80/10/10)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.11, shuffle=False)
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize data
    print("\nNormalizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)
    
    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True)  # GPU optimization
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Create model
    input_size = X_train.shape[2]
    model = StockLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'input_size': input_size
            }, 'best_model.pth')
            print(f"  ✓ Best model saved!")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    print("\n✓ Training history saved to training_history.png")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch).cpu().numpy()
            predictions.append(outputs)
            actuals.append(y_batch.numpy())
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Inverse transform
    predictions = scaler_y.inverse_transform(predictions)
    actuals = scaler_y.inverse_transform(actuals)
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    for i, days in enumerate([3, 5, 7]):
        mae = mean_absolute_error(actuals[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(actuals[:, i], predictions[:, i]))
        mape = np.mean(np.abs((actuals[:, i] - predictions[:, i]) / actuals[:, i])) * 100
        
        print(f"\n{days}-Day Forecast:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    # Plot predictions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, days in enumerate([3, 5, 7]):
        axes[i].scatter(actuals[:, i], predictions[:, i], alpha=0.5, s=10)
        min_val = min(actuals[:, i].min(), predictions[:, i].min())
        max_val = max(actuals[:, i].max(), predictions[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[i].set_xlabel('Actual Price')
        axes[i].set_ylabel('Predicted Price')
        axes[i].set_title(f'{days}-Day Forecast')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("\n✓ Predictions saved to predictions.png")
    
    print("\n" + "="*60)
    print("DONE! Model saved to best_model.pth")
    print("="*60)

if __name__ == "__main__":
    train_model()

