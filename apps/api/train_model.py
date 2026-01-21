"""
LSTM Model Training Pipeline

This script:
1. Downloads real historical crypto data
2. Preprocesses and creates sequences
3. Trains an LSTM neural network
4. Saves model weights to disk
5. Provides inference functions

Run: python train_model.py
"""

import os
import json
import pickle
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional
import structlog

log = structlog.get_logger()

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CONFIG = {
    "symbol": "BTCUSDT",
    "sequence_length": 60,      # Use 60 time steps to predict next
    "prediction_horizon": 1,    # Predict 1 step ahead
    "train_split": 0.8,         # 80% train, 20% test
    "epochs": 50,
    "batch_size": 32,
    "hidden_size": 128,
    "num_layers": 2,
    "learning_rate": 0.001,
    "model_path": os.path.join(MODEL_DIR, "lstm_btc.pt"),
    "scaler_path": os.path.join(MODEL_DIR, "scaler.pkl"),
}

# ============================================
# DATA DOWNLOAD
# ============================================

def download_historical_data(symbol: str = "BTCUSDT", days: int = 365) -> List[dict]:
    """
    Download historical kline data from Binance API.
    Returns list of OHLCV candles.
    """
    import httpx
    
    print(f"📥 Downloading {days} days of {symbol} data from Binance...")
    
    # Binance klines endpoint
    url = "https://api.binance.com/api/v3/klines"
    
    all_candles = []
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    interval = "1h"  # 1 hour candles
    limit = 1000  # Max per request
    
    # Calculate how many requests we need
    hours_needed = days * 24
    requests_needed = (hours_needed // limit) + 1
    
    for i in range(requests_needed):
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": end_time,
        }
        
        try:
            response = httpx.get(url, params=params, timeout=30)
            response.raise_for_status()
            candles = response.json()
            
            if not candles:
                break
            
            # Parse candles
            for c in candles:
                all_candles.append({
                    "timestamp": c[0],
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                })
            
            # Update end_time for next request
            end_time = candles[0][0] - 1
            
            print(f"   Downloaded batch {i+1}/{requests_needed} ({len(all_candles)} candles total)")
            
        except Exception as e:
            print(f"   Error downloading: {e}")
            break
    
    # Sort by timestamp
    all_candles.sort(key=lambda x: x["timestamp"])
    
    print(f"✅ Downloaded {len(all_candles)} hourly candles")
    return all_candles


# ============================================
# DATA PREPROCESSING
# ============================================

def preprocess_data(candles: List[dict]) -> Tuple:
    """
    Preprocess candle data for LSTM training.
    
    IMPORTANT: This correctly handles train/val/test split to avoid data leakage:
    1. Split data FIRST (80% train, 10% val, 10% test)
    2. Fit scaler ONLY on training data
    3. Transform val/test using the fitted scaler
    
    Returns: (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    try:
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("❌ Missing dependencies. Run: pip install numpy scikit-learn")
        return None
    
    print("🔧 Preprocessing data...")
    
    # Extract close prices
    prices = np.array([c["close"] for c in candles]).reshape(-1, 1)
    
    # --- CRITICAL FIX: Split data BEFORE scaling ---
    # Split: 80% train, 10% validation, 10% test
    n = len(prices)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_prices = prices[:train_end]
    val_prices = prices[train_end:val_end]
    test_prices = prices[val_end:]
    
    print(f"   Raw split: {len(train_prices)} train, {len(val_prices)} val, {len(test_prices)} test")
    
    # --- Fit scaler ONLY on training data ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_prices)  # Only fit on train!
    
    # Transform all splits using the fitted scaler
    scaled_train = scaler.transform(train_prices)
    scaled_val = scaler.transform(val_prices)
    scaled_test = scaler.transform(test_prices)
    
    # Helper function to create sequences
    seq_len = CONFIG["sequence_length"]
    
    def create_sequences(scaled_data):
        X, y = [], []
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i-seq_len:i, 0])
            y.append(scaled_data[i, 0])
        X = np.array(X)
        y = np.array(y)
        # Reshape for LSTM: (samples, sequence_length, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y
    
    X_train, y_train = create_sequences(scaled_train)
    X_val, y_val = create_sequences(scaled_val)
    X_test, y_test = create_sequences(scaled_test)
    
    print(f"   Sequence samples: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ============================================
# LSTM MODEL (PyTorch)
# ============================================

def create_model():
    """Create LSTM model using PyTorch."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("❌ PyTorch not installed. Run: pip install torch")
        return None
    
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
            super(LSTMPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2
            )
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_size)
            )
        
        def forward(self, x):
            # LSTM output
            lstm_out, _ = self.lstm(x)
            # Take last time step
            last_output = lstm_out[:, -1, :]
            # Fully connected
            prediction = self.fc(last_output)
            return prediction
    
    model = LSTMPredictor(
        input_size=1,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=1
    )
    
    return model


# ============================================
# TRAINING LOOP
# ============================================

def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train the LSTM model with proper validation.
    
    IMPORTANT: Uses validation set for per-epoch metrics and early stopping.
    Test set is LOCKED and only evaluated once after training completes.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        import numpy as np
    except ImportError:
        print("❌ Missing PyTorch. Run: pip install torch")
        return None
    
    print("\n🚀 Starting model training...")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Hidden size: {CONFIG['hidden_size']}")
    print(f"   LSTM layers: {CONFIG['num_layers']}")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # Create model
    model = create_model()
    if model is None:
        return None
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Training loop - now uses VALIDATION set for metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Evaluate on VALIDATION set (not test!) ---
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_losses.append(val_loss)
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{CONFIG['epochs']} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"   ⚠️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"   📌 Restored best model from validation (loss: {best_val_loss:.6f})")
    
    # --- FINAL: Evaluate on TEST set (only once!) ---
    print("\n📊 Final evaluation on held-out TEST set...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_loss = criterion(test_pred, y_test_t).item()
    
    print(f"   Final Train Loss: {train_losses[-1]:.6f}")
    print(f"   Final Val Loss: {best_val_loss:.6f}")
    print(f"   🔒 TEST Loss (holdout): {test_loss:.6f}")
    
    return model, train_losses, val_losses, test_loss


# ============================================
# SAVE MODEL
# ============================================

def save_model(model, scaler):
    """Save model weights and scaler."""
    import torch
    
    # Create models directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save PyTorch model
    torch.save(model.state_dict(), CONFIG["model_path"])
    print(f"💾 Model saved to: {CONFIG['model_path']}")
    
    # Save scaler
    with open(CONFIG["scaler_path"], "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler saved to: {CONFIG['scaler_path']}")
    
    # Save config
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Config saved to: {config_path}")


# ============================================
# INFERENCE
# ============================================

def load_model():
    """Load trained model for inference."""
    import torch
    
    if not os.path.exists(CONFIG["model_path"]):
        print("❌ No trained model found. Run training first.")
        return None, None
    
    # Load model
    model = create_model()
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    model.eval()
    
    # Load scaler
    with open(CONFIG["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler


def predict(prices: List[float], model=None, scaler=None) -> dict:
    """
    Make a prediction using the trained model.
    
    Args:
        prices: List of recent prices (at least 60)
        model: Trained model (loads from disk if None)
        scaler: Fitted scaler (loads from disk if None)
    
    Returns:
        Prediction dict with price and direction
    """
    import torch
    import numpy as np
    
    if model is None or scaler is None:
        model, scaler = load_model()
        if model is None:
            return {"error": "No trained model available"}
    
    # Need at least sequence_length prices
    seq_len = CONFIG["sequence_length"]
    if len(prices) < seq_len:
        return {"error": f"Need at least {seq_len} prices, got {len(prices)}"}
    
    # Take last sequence_length prices
    recent_prices = np.array(prices[-seq_len:]).reshape(-1, 1)
    
    # Scale
    scaled = scaler.transform(recent_prices)
    
    # Reshape for LSTM
    X = scaled.reshape(1, seq_len, 1)
    X_tensor = torch.FloatTensor(X)
    
    # Predict
    model.eval()
    with torch.no_grad():
        scaled_pred = model(X_tensor).numpy()[0, 0]
    
    # Inverse scale
    predicted_price = scaler.inverse_transform([[scaled_pred]])[0, 0]
    current_price = prices[-1]
    
    # Calculate change
    change_percent = ((predicted_price - current_price) / current_price) * 100
    
    # Determine direction
    if change_percent > 0.5:
        direction = "bullish"
    elif change_percent < -0.5:
        direction = "bearish"
    else:
        direction = "neutral"
    
    return {
        "current_price": current_price,
        "predicted_price": round(predicted_price, 2),
        "change_percent": round(change_percent, 2),
        "direction": direction,
        "model": "lstm_btc",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================
# MAIN
# ============================================

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🧠 SENTINANCE LSTM TRAINING PIPELINE")
    print("=" * 60)
    print()
    
    # Step 1: Download data
    candles = download_historical_data(CONFIG["symbol"], days=180)
    
    if len(candles) < 1000:
        print("❌ Not enough data for training")
        return
    
    # Step 2: Preprocess (now returns train/val/test splits)
    result = preprocess_data(candles)
    if result is None:
        return
    
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = result
    
    # Step 3: Train (now uses validation set for metrics, test locked)
    result = train_model(X_train, y_train, X_val, y_val, X_test, y_test)
    if result is None:
        return
    
    model, train_losses, val_losses, test_loss = result
    
    # Step 4: Save
    save_model(model, scaler)
    
    # Step 5: Test inference
    print("\n🔮 Testing inference...")
    test_prices = [c["close"] for c in candles[-100:]]
    prediction = predict(test_prices, model, scaler)
    print(f"   Current: ${prediction['current_price']:,.2f}")
    print(f"   Predicted: ${prediction['predicted_price']:,.2f}")
    print(f"   Change: {prediction['change_percent']:+.2f}%")
    print(f"   Direction: {prediction['direction'].upper()}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel files saved to {MODEL_DIR}/")
    print("You can now use the trained model for predictions.")


if __name__ == "__main__":
    main()
