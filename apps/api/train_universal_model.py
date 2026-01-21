"""
Universal Multi-Asset LSTM Training

Trains LSTM models on ALL available data:
- Crypto (BTC, ETH, SOL, BNB)
- US Indices (S&P 500, NASDAQ, Dow Jones)
- Indian Indices (NIFTY 50, SENSEX)
- Asian Indices (NIKKEI 225, Shanghai)
- European Indices (FTSE, DAX, CAC 40)

Creates one universal model and per-asset models.

Run: python train_universal_model.py
"""

import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Tuple
import structlog

log = structlog.get_logger()

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Assets to train on
ASSETS = {
    # Crypto (hourly data)
    "btcusdt": {"name": "Bitcoin", "type": "crypto", "interval": "hourly"},
    "ethusdt": {"name": "Ethereum", "type": "crypto", "interval": "hourly"},
    "solusdt": {"name": "Solana", "type": "crypto", "interval": "hourly"},
    "bnbusdt": {"name": "BNB", "type": "crypto", "interval": "hourly"},
    
    # USA (daily data)
    "gspc": {"name": "S&P 500", "type": "index", "interval": "daily"},
    "ixic": {"name": "NASDAQ", "type": "index", "interval": "daily"},
    "dji": {"name": "Dow Jones", "type": "index", "interval": "daily"},
    
    # India
    "nsei": {"name": "NIFTY 50", "type": "index", "interval": "daily"},
    "bsesn": {"name": "SENSEX", "type": "index", "interval": "daily"},
    
    # Japan
    "n225": {"name": "NIKKEI 225", "type": "index", "interval": "daily"},
    
    # UK
    "ftse": {"name": "FTSE 100", "type": "index", "interval": "daily"},
    
    # Europe
    "gdaxi": {"name": "DAX", "type": "index", "interval": "daily"},
    "fchi": {"name": "CAC 40", "type": "index", "interval": "daily"},
    
    # China
    "000001_ss": {"name": "Shanghai", "type": "index", "interval": "daily"},
}

CONFIG = {
    "sequence_length": 60,
    "sequence_length_daily": 30,
    "train_split": 0.8,
    "epochs": 30,
    "batch_size": 32,
    "hidden_size": 128,
    "num_layers": 2,
    "learning_rate": 0.001,
}


# ============================================
# DATA LOADING
# ============================================

def load_asset_data(symbol: str) -> List[float]:
    """Load closing prices from CSV."""
    filepath = os.path.join(DATA_DIR, f"{symbol}_historical.csv")
    
    if not os.path.exists(filepath):
        print(f"   ⚠️ {filepath} not found")
        return []
    
    prices = []
    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                try:
                    prices.append(float(parts[5]))  # Close price
                except ValueError:
                    continue
    
    return prices


def prepare_sequences(prices: List[float], seq_len: int = 60) -> Tuple:
    """
    Prepare training sequences from prices.
    
    FIXED: Now splits data FIRST, then fits scaler only on training portion
    to avoid data leakage. Returns train/val/test splits.
    """
    try:
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("Missing dependencies. Run: pip install numpy scikit-learn")
        return None
    
    if len(prices) < seq_len + 100:
        return None
    
    prices_array = np.array(prices).reshape(-1, 1)
    
    # --- CRITICAL FIX: Split data BEFORE scaling ---
    # Split: 80% train, 10% val, 10% test
    n = len(prices_array)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_prices = prices_array[:train_end]
    val_prices = prices_array[train_end:val_end]
    test_prices = prices_array[val_end:]
    
    # --- Fit scaler ONLY on training data ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_prices)  # Only fit on train!
    
    # Transform all splits using the fitted scaler
    scaled_train = scaler.transform(train_prices)
    scaled_val = scaler.transform(val_prices)
    scaled_test = scaler.transform(test_prices)
    
    # Helper to create sequences
    def create_seqs(scaled_data):
        X, y = [], []
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i-seq_len:i, 0])
            y.append(scaled_data[i, 0])
        X = np.array(X).reshape(-1, seq_len, 1) if len(X) > 0 else np.array([])
        y = np.array(y)
        return X, y
    
    X_train, y_train = create_seqs(scaled_train)
    X_val, y_val = create_seqs(scaled_val)
    X_test, y_test = create_seqs(scaled_test)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }


# ============================================
# MODEL
# ============================================

def create_model():
    """Create LSTM model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    
    class MultiAssetLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=128, num_layers=2):
            super().__init__()
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
                nn.Linear(64, 1)
            )
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    return MultiAssetLSTM(
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"]
    )


def train_model(data: Dict, model_name: str = "universal"):
    """Train LSTM model on prepared data."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("PyTorch not installed")
        return None
    
    X_train = torch.FloatTensor(data["X_train"])
    y_train = torch.FloatTensor(data["y_train"]).reshape(-1, 1)
    X_val = torch.FloatTensor(data["X_val"])
    y_val = torch.FloatTensor(data["y_val"]).reshape(-1, 1)
    X_test = torch.FloatTensor(data["X_test"])
    y_test = torch.FloatTensor(data["y_test"]).reshape(-1, 1)

    if len(X_val) == 0:
        X_val = X_test
        y_val = y_test
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )
    
    model = create_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    print(f"\n   Training {model_name}...")
    print(f"   Samples: {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test")

    best_val_loss = float("inf")
    best_state = None
    
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
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"   Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
                f"Train: {epoch_loss/len(train_loader):.6f} | Val: {val_loss:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test).item()

    print(f"   Final Val Loss: {best_val_loss:.6f} | Test: {test_loss:.6f}")

    return model, test_loss


def save_model(model, scaler, symbol: str):
    """Save trained model and scaler."""
    import torch
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pt")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
    
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"   💾 Saved: {model_path}")


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("🧠 SENTINANCE UNIVERSAL MODEL TRAINING")
    print("=" * 70)
    print(f"\n📊 Training on {len(ASSETS)} assets:")
    
    for symbol, info in ASSETS.items():
        print(f"   • {info['name']} ({symbol})")
    
    training_results = []
    
    # Train individual models for each asset
    print("\n" + "=" * 70)
    print("🔄 TRAINING INDIVIDUAL MODELS")
    print("=" * 70)
    
    for symbol, info in ASSETS.items():
        print(f"\n📈 {info['name']} ({symbol.upper()})")
        
        # Load data
        prices = load_asset_data(symbol)
        if not prices:
            continue
        
        print(f"   Loaded {len(prices):,} price points")
        
        # Prepare sequences
        seq_len = CONFIG["sequence_length"] if info["interval"] == "hourly" else CONFIG["sequence_length_daily"]
        data = prepare_sequences(prices, seq_len)
        if data is None:
            print(f"   ⚠️ Insufficient data for training")
            continue
        
        # Train
        result = train_model(data, symbol)
        if result is None:
            continue
        model, test_loss = result
        
        # Save
        save_model(model, data["scaler"], symbol)
        
        training_results.append({
            "symbol": symbol,
            "name": info["name"],
            "type": info["type"],
            "sequence_length": seq_len,
            "data_points": len(prices),
            "train_samples": len(data["X_train"]),
            "val_samples": len(data["X_val"]),
            "test_samples": len(data["X_test"]),
            "test_loss": test_loss,
        })
    
    # Save training summary
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "trained_at": datetime.utcnow().isoformat(),
            "config": CONFIG,
            "models": training_results,
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 Models trained: {len(training_results)}")
    for r in training_results:
        print(f"   ✓ {r['name']}: {r['data_points']:,} points → {r['train_samples']:,} samples")
    
    print(f"\n💾 Models saved to: {os.path.abspath(MODEL_DIR)}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
