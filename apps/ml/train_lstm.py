"""
LSTM Price Prediction Training Pipeline

Trains LSTM models on historical price data for multiple assets.
Uses MLflow for experiment tracking and model versioning.

Features:
- Downloads 5 years of historical data from yFinance
- Trains LSTM model with proper train/validation/test split
- Logs metrics and artifacts to MLflow
- Saves model weights for production serving

Usage:
    python train_lstm.py --symbol BTCUSDT --epochs 50
    
    # Or train all symbols:
    python train_lstm.py --all

Requirements:
    pip install torch numpy pandas yfinance mlflow scikit-learn
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import json

# ML imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. Run: pip install torch")

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not installed. Run: pip install mlflow")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. Run: pip install yfinance")


# ============================================
# CONFIGURATION
# ============================================

CRYPTO_SYMBOLS = {
    "BTCUSDT": "BTC-USD",  # yfinance symbol
    "ETHUSDT": "ETH-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
}

INDEX_SYMBOLS = {
    "^GSPC": "^GSPC",  # S&P 500
    "^NSEI": "^NSEI",  # Nifty 50
    "^FTSE": "^FTSE",  # FTSE 100
    "^N225": "^N225",  # Nikkei 225
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================
# LSTM MODEL ARCHITECTURE
# ============================================

class LSTMPredictor(nn.Module):
    """
    LSTM model for price prediction.
    
    Architecture:
    - 2 LSTM layers with dropout
    - Fully connected output layer
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        prediction = self.fc(last_output)
        
        return prediction


# ============================================
# DATA PREPARATION
# ============================================

def download_historical_data(
    symbol: str,
    yf_symbol: str,
    years: int = 5
) -> pd.DataFrame:
    """Download historical price data from yFinance."""
    
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required for data download")
    
    print(f"📥 Downloading {years} years of data for {symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(start=start_date, end=end_date, interval="1h")
    
    if df.empty:
        # Try daily data if hourly not available
        df = ticker.history(start=start_date, end=end_date, interval="1d")
    
    print(f"   Downloaded {len(df)} data points")
    
    return df


def prepare_sequences(
    prices: np.ndarray,
    lookback: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training.
    
    Args:
        prices: 1D array of prices
        lookback: Number of timesteps to look back
    
    Returns:
        X: (samples, lookback, 1) array of sequences
        y: (samples, 1) array of targets
    """
    X, y = [], []
    
    for i in range(lookback, len(prices)):
        X.append(prices[i - lookback:i])
        y.append(prices[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y


def normalize_data(
    data: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """Min-max normalization."""
    
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    
    normalized = (data - min_val) / (max_val - min_val + 1e-8)
    
    return normalized, min_val, max_val


def denormalize_data(
    data: np.ndarray,
    min_val: float,
    max_val: float
) -> np.ndarray:
    """Reverse normalization."""
    return data * (max_val - min_val) + min_val


# ============================================
# TRAINING LOOP
# ============================================

def train_model(
    symbol: str,
    yf_symbol: str,
    epochs: int = 50,
    batch_size: int = 32,
    lookback: int = 60,
    hidden_size: int = 128,
    learning_rate: float = 0.001,
    train_split: float = 0.7,
    val_split: float = 0.15,
) -> dict:
    """
    Train LSTM model for a single symbol.
    
    Returns:
        dict with training results and model path
    """
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training")
    
    # ============================================
    # 1. DOWNLOAD AND PREPARE DATA
    # ============================================
    
    df = download_historical_data(symbol, yf_symbol)
    prices = df['Close'].values.astype(np.float32)
    
    if len(prices) < lookback + 100:
        raise ValueError(f"Insufficient data for {symbol}: {len(prices)} points")
    
    # Normalize
    prices_norm, min_price, max_price = normalize_data(prices)
    
    # Create sequences
    X, y = prepare_sequences(prices_norm, lookback)
    
    # Split data
    n = len(X)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # ============================================
    # 2. INITIALIZE MODEL
    # ============================================
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    model = LSTMPredictor(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # ============================================
    # 3. TRAINING LOOP
    # ============================================
    
    # MLflow tracking
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
        mlflow.set_experiment(f"lstm_{symbol}")
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    print(f"\n🚀 Training LSTM for {symbol}...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_d = X_val_t.to(device)
            y_val_d = y_val_t.to(device)
            val_outputs = model(X_val_d)
            val_loss = criterion(val_outputs, y_val_d).item()
        
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # ============================================
    # 4. EVALUATION ON TEST SET
    # ============================================
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        X_test_d = X_test_t.to(device)
        y_test_d = y_test_t.to(device)
        test_outputs = model(X_test_d)
        test_loss = criterion(test_outputs, y_test_d).item()
        
        # Calculate MAE in original scale
        predictions = test_outputs.cpu().numpy()
        actuals = y_test_t.numpy()
        
        pred_denorm = denormalize_data(predictions, min_price, max_price)
        actual_denorm = denormalize_data(actuals, min_price, max_price)
        
        mae = np.mean(np.abs(pred_denorm - actual_denorm))
        mape = np.mean(np.abs((actual_denorm - pred_denorm) / actual_denorm)) * 100
    
    print(f"\n✅ Test Results for {symbol}:")
    print(f"   MSE Loss: {test_loss:.6f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # ============================================
    # 5. SAVE MODEL
    # ============================================
    
    model_path = os.path.join(MODEL_DIR, f"lstm_{symbol}.pt")
    
    # Save model and metadata
    torch.save({
        'model_state_dict': best_model_state,
        'config': {
            'hidden_size': hidden_size,
            'num_layers': 2,
            'lookback': lookback,
            'input_size': 1,
            'output_size': 1,
        },
        'normalization': {
            'min_price': float(min_price),
            'max_price': float(max_price),
        },
        'metrics': {
            'test_mse': test_loss,
            'test_mae': mae,
            'test_mape': mape,
        },
        'symbol': symbol,
        'trained_at': datetime.now().isoformat(),
    }, model_path)
    
    print(f"💾 Model saved to: {model_path}")
    
    # ============================================
    # 6. LOG TO MLFLOW
    # ============================================
    
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"lstm_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log parameters
            mlflow.log_param("symbol", symbol)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("lookback", lookback)
            mlflow.log_param("hidden_size", hidden_size)
            mlflow.log_param("learning_rate", learning_rate)
            
            # Log metrics
            mlflow.log_metric("test_mse", test_loss)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_mape", mape)
            mlflow.log_metric("best_val_loss", best_val_loss)
            
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            print(f"📊 Logged to MLflow")
    
    return {
        "symbol": symbol,
        "model_path": model_path,
        "test_mse": test_loss,
        "test_mae": mae,
        "test_mape": mape,
    }


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Train LSTM price prediction models")
    parser.add_argument("--symbol", type=str, help="Symbol to train (e.g., BTCUSDT)")
    parser.add_argument("--all", action="store_true", help="Train all symbols")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback period")
    
    args = parser.parse_args()
    
    if args.all:
        symbols_to_train = {**CRYPTO_SYMBOLS, **INDEX_SYMBOLS}
    elif args.symbol:
        if args.symbol in CRYPTO_SYMBOLS:
            symbols_to_train = {args.symbol: CRYPTO_SYMBOLS[args.symbol]}
        elif args.symbol in INDEX_SYMBOLS:
            symbols_to_train = {args.symbol: INDEX_SYMBOLS[args.symbol]}
        else:
            print(f"❌ Unknown symbol: {args.symbol}")
            return
    else:
        # Default: train BTC only
        symbols_to_train = {"BTCUSDT": CRYPTO_SYMBOLS["BTCUSDT"]}
    
    print(f"\n{'='*60}")
    print("SENTINANCE LSTM TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Symbols: {list(symbols_to_train.keys())}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Lookback: {args.lookback}")
    print(f"{'='*60}\n")
    
    results = []
    
    for symbol, yf_symbol in symbols_to_train.items():
        try:
            result = train_model(
                symbol=symbol,
                yf_symbol=yf_symbol,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lookback=args.lookback,
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error training {symbol}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['symbol']}: MAE=${r['test_mae']:.2f}, MAPE={r['test_mape']:.2f}%")


if __name__ == "__main__":
    main()
