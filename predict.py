"""
Make predictions on new data using the trained model
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

SEQUENCE_LENGTH = 30

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 3)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def predict_stock(symbol, model_path='best_model.pth', data_path='nepse_stock_data_merged_deduped.csv'):
    """Predict future prices for a specific stock"""
    
    # Load model and scalers
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = StockLSTM(checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    # Load data
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()  # Convert to lowercase
    df['date'] = pd.to_datetime(df['date'])
    
    # Get stock data
    stock_df = df[df['symbol'] == symbol].sort_values('date')
    
    if len(stock_df) < SEQUENCE_LENGTH:
        print(f"Not enough data for {symbol}")
        return
    
    # Prepare features
    features = ['open', 'high', 'low', 'close', 'volume']
    last_sequence = stock_df[features].iloc[-SEQUENCE_LENGTH:].values
    
    # Normalize
    last_sequence_scaled = scaler_X.transform(last_sequence)
    
    # Predict
    with torch.no_grad():
        X = torch.FloatTensor(last_sequence_scaled).unsqueeze(0)
        pred_scaled = model(X).numpy()[0]
    
    # Inverse transform
    dummy = np.zeros((1, 3))
    dummy[0] = pred_scaled
    predictions = scaler_y.inverse_transform(dummy)[0]
    
    # Display results
    current_price = stock_df['close'].iloc[-1]
    last_date = stock_df['date'].iloc[-1]
    
    print(f"\n{'='*60}")
    print(f"PREDICTIONS FOR {symbol}")
    print(f"{'='*60}")
    print(f"Current Price ({last_date.date()}): NPR {current_price:.2f}")
    print(f"\nForecasted Prices:")
    
    for i, days in enumerate([3, 5, 7]):
        pred_price = predictions[i]
        change = ((pred_price - current_price) / current_price) * 100
        direction = "📈" if change > 0 else "📉"
        print(f"  {days}-day: NPR {pred_price:.2f} ({change:+.2f}%) {direction}")
    
    return predictions

if __name__ == "__main__":
    # Example predictions
    stocks = ['NABIL', 'BOKL', 'SCB', 'NICA', 'EBL']
    
    print("Loading model and making predictions...\n")
    
    for stock in stocks:
        try:
            predict_stock(stock)
        except Exception as e:
            print(f"Error predicting {stock}: {e}")
