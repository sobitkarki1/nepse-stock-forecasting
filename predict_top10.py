"""
Predict and rank top 10 stocks for next available date
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

def predict_all_stocks(model_path='best_model.pth', data_path='nepse_stock_data_merged_deduped.csv'):
    """Predict for all stocks and return top 10 by expected return"""
    
    # Load model and scalers
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = StockLSTM(checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    # Load data
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()  # Normalize column names
    df['date'] = pd.to_datetime(df['date'])
    
    # Get latest date
    latest_date = df['date'].max()
    print(f"Latest data date: {latest_date.date()}")
    print(f"Predicting for next trading day...\n")
    
    features = ['open', 'high', 'low', 'close', 'volume']
    predictions_list = []
    
    # Predict for each stock
    for symbol in df['symbol'].unique():
        stock_df = df[df['symbol'] == symbol].sort_values('date')
        
        if len(stock_df) < SEQUENCE_LENGTH:
            continue
        
        # Get last sequence
        last_sequence = stock_df[features].iloc[-SEQUENCE_LENGTH:].values
        last_sequence_scaled = scaler_X.transform(last_sequence)
        
        # Make prediction
        with torch.no_grad():
            X = torch.FloatTensor(last_sequence_scaled).unsqueeze(0)
            pred_scaled = model(X).numpy()[0]
        
        # Inverse transform
        dummy = np.zeros((1, 3))
        dummy[0] = pred_scaled
        predictions = scaler_y.inverse_transform(dummy)[0]
        
        # Get current price
        current_price = stock_df['close'].iloc[-1]
        current_date = stock_df['date'].iloc[-1]
        
        # Calculate returns
        return_3d = ((predictions[0] - current_price) / current_price) * 100
        return_5d = ((predictions[1] - current_price) / current_price) * 100
        return_7d = ((predictions[2] - current_price) / current_price) * 100
        
        predictions_list.append({
            'Symbol': symbol,
            'Current Price': current_price,
            'Data Date': current_date,
            '3-Day Pred': predictions[0],
            '3-Day Return %': return_3d,
            '5-Day Pred': predictions[1],
            '5-Day Return %': return_5d,
            '7-Day Pred': predictions[2],
            '7-Day Return %': return_7d
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(predictions_list)
    
    # Sort by 3-day return
    top_10 = results_df.nlargest(10, '3-Day Return %')
    
    print("="*80)
    print("TOP 10 STOCKS - HIGHEST PREDICTED 3-DAY RETURNS")
    print("="*80)
    print(f"\n{'Rank':<6}{'Symbol':<10}{'Current':<12}{'3-Day':<12}{'Return':<10}{'Signal'}")
    print("-"*80)
    
    for idx, row in enumerate(top_10.itertuples(), 1):
        signal = "🚀" if row._5 > 5 else "📈" if row._5 > 2 else "↗️" if row._5 > 0 else "→"
        print(f"{idx:<6}{row.Symbol:<10}NPR {row._2:<8.2f}NPR {row._4:<8.2f}{row._5:>7.2f}%  {signal}")
    
    # Save to CSV
    results_df.to_csv('all_predictions.csv', index=False)
    top_10.to_csv('top10_predictions.csv', index=False)
    
    print("\n" + "="*80)
    print("DETAILED TOP 10 ANALYSIS")
    print("="*80)
    
    for idx, row in enumerate(top_10.itertuples(), 1):
        print(f"\n#{idx} - {row.Symbol}")
        print(f"  Current Price: NPR {row._2:.2f} (as of {row._3.date()})")
        print(f"  3-Day Forecast: NPR {row._4:.2f} ({row._5:+.2f}%)")
        print(f"  5-Day Forecast: NPR {row._6:.2f} ({row._7:+.2f}%)")
        print(f"  7-Day Forecast: NPR {row._8:.2f} ({row._9:+.2f}%)")
    
    print(f"\n✓ Full predictions saved to: all_predictions.csv")
    print(f"✓ Top 10 saved to: top10_predictions.csv")
    
    return top_10

if __name__ == "__main__":
    predict_all_stocks()
