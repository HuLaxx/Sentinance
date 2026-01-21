"""
Production-Grade Multi-Asset LSTM Training

Features:
- Early stopping to prevent overfitting
- Learning rate scheduling
- Dropout regularization
- Batch normalization
- Validation-based model selection
- Saves best model only

Run: python train_production_model.py
"""

import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional
import structlog

log = structlog.get_logger()

# ============================================
# CONFIGURATION
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

ASSETS = {
    # Crypto - most important
    "btcusdt": {"name": "Bitcoin", "priority": 1},
    "ethusdt": {"name": "Ethereum", "priority": 1},
    
    # Major indices
    "gspc": {"name": "S&P 500", "priority": 1},
    "nsei": {"name": "NIFTY 50", "priority": 1},
    "n225": {"name": "NIKKEI 225", "priority": 2},
    "ftse": {"name": "FTSE 100", "priority": 2},
    
    # Secondary
    "solusdt": {"name": "Solana", "priority": 2},
    "bnbusdt": {"name": "BNB", "priority": 2},
    "ixic": {"name": "NASDAQ", "priority": 2},
    "bsesn": {"name": "SENSEX", "priority": 2},
}

# Hyperparameters optimized for financial time series
HYPERPARAMS = {
    "sequence_length": 60,       # Look back 60 periods
    "hidden_size": 64,           # Smaller = less overfitting
    "num_layers": 2,             # 2 layers is usually optimal
    "dropout": 0.3,              # 30% dropout for regularization
    "learning_rate": 0.001,      # Adam default
    "batch_size": 64,            # Larger batch = more stable
    "max_epochs": 100,           # Max epochs
    "patience": 10,              # Early stopping patience
    "min_delta": 0.0001,         # Min improvement to count
    "train_split": 0.7,          # 70% train
    "val_split": 0.15,           # 15% validation, 15% test
}


# ============================================
# DATA LOADING
# ============================================

def load_prices(symbol: str) -> List[float]:
    """Load closing prices from CSV."""
    filepath = os.path.join(DATA_DIR, f"{symbol}_historical.csv")
    
    if not os.path.exists(filepath):
        return []
    
    prices = []
    with open(filepath, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                try:
                    prices.append(float(parts[5]))
                except:
                    continue
    return prices


def prepare_data(prices: List[float]) -> Optional[Dict]:
    """
    Prepare train/val/test splits with proper scaling.
    
    FIXED: Now splits data FIRST, then fits scaler only on training portion
    to avoid data leakage.
    """
    try:
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("Run: pip install numpy scikit-learn")
        return None
    
    seq_len = HYPERPARAMS["sequence_length"]
    
    if len(prices) < seq_len + 100:
        return None
    
    # Convert to numpy
    data = np.array(prices).reshape(-1, 1)
    
    # --- CRITICAL FIX: Split data FIRST, BEFORE scaling ---
    n = len(data)
    train_end = int(n * HYPERPARAMS["train_split"])
    val_end = int(n * (HYPERPARAMS["train_split"] + HYPERPARAMS["val_split"]))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # --- Fit scaler ONLY on training data ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)  # Only fit on train!
    
    # Transform all splits using the fitted scaler
    scaled_train = scaler.transform(train_data)
    scaled_val = scaler.transform(val_data)
    scaled_test = scaler.transform(test_data)
    
    # Helper to create sequences
    def create_seqs(scaled):
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i, 0])
            y.append(scaled[i, 0])
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
        "total_samples": len(X_train) + len(X_val) + len(X_test),
    }


# ============================================
# MODEL WITH REGULARIZATION
# ============================================

def create_model():
    """Create LSTM with dropout and proper initialization."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    
    class RegularizedLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=HYPERPARAMS["hidden_size"],
                num_layers=HYPERPARAMS["num_layers"],
                batch_first=True,
                dropout=HYPERPARAMS["dropout"] if HYPERPARAMS["num_layers"] > 1 else 0,
            )
            
            self.bn = nn.BatchNorm1d(HYPERPARAMS["hidden_size"])
            
            self.fc = nn.Sequential(
                nn.Linear(HYPERPARAMS["hidden_size"], 32),
                nn.ReLU(),
                nn.Dropout(HYPERPARAMS["dropout"]),
                nn.Linear(32, 1)
            )
            
            # Initialize weights
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last = lstm_out[:, -1, :]
            normalized = self.bn(last)
            return self.fc(normalized)
    
    return RegularizedLSTM()


# ============================================
# TRAINING WITH EARLY STOPPING
# ============================================

class EarlyStopping:
    """Stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        import torch
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save best model weights
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def train_asset(symbol: str, info: Dict) -> Optional[Dict]:
    """Train model for a single asset with early stopping."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        return None
    
    print(f"\n{'='*50}")
    print(f"📈 Training: {info['name']} ({symbol.upper()})")
    print(f"{'='*50}")
    
    # Load data
    prices = load_prices(symbol)
    if not prices:
        print(f"   ⚠️ No data found")
        return None
    
    print(f"   📊 Loaded {len(prices):,} price points")
    
    # Prepare data
    data = prepare_data(prices)
    if data is None:
        print(f"   ⚠️ Insufficient data")
        return None
    
    print(f"   📦 Train: {len(data['X_train']):,} | Val: {len(data['X_val']):,} | Test: {len(data['X_test']):,}")
    
    # Convert to tensors
    X_train = torch.FloatTensor(data["X_train"])
    y_train = torch.FloatTensor(data["y_train"]).reshape(-1, 1)
    X_val = torch.FloatTensor(data["X_val"])
    y_val = torch.FloatTensor(data["y_val"]).reshape(-1, 1)
    X_test = torch.FloatTensor(data["X_test"])
    y_test = torch.FloatTensor(data["y_test"]).reshape(-1, 1)
    
    # Data loader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=HYPERPARAMS["batch_size"],
        shuffle=True
    )
    
    # Model, loss, optimizer
    model = create_model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=HYPERPARAMS["patience"],
        min_delta=HYPERPARAMS["min_delta"]
    )
    
    print(f"\n   🚀 Training (max {HYPERPARAMS['max_epochs']} epochs, patience={HYPERPARAMS['patience']})")
    
    best_val_loss = float('inf')
    
    for epoch in range(HYPERPARAMS["max_epochs"]):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"   ⏹️  Early stopping at epoch {epoch+1}")
            break
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Load best model
    if early_stopping.best_model:
        model.load_state_dict(early_stopping.best_model)
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        test_loss = criterion(test_pred, y_test).item()
    
    print(f"\n   ✅ Final Test Loss: {test_loss:.6f}")
    print(f"   ✅ Best Val Loss: {best_val_loss:.6f}")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{symbol}_model.pt")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
    
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(data["scaler"], f)
    
    print(f"   💾 Saved: {model_path}")
    
    return {
        "symbol": symbol,
        "name": info["name"],
        "data_points": len(prices),
        "train_samples": len(data["X_train"]),
        "val_samples": len(data["X_val"]),
        "test_samples": len(data["X_test"]),
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "epochs_trained": epoch + 1,
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("🧠 SENTINANCE PRODUCTION MODEL TRAINING")
    print("=" * 60)
    print("\n📋 Hyperparameters:")
    for k, v in HYPERPARAMS.items():
        print(f"   {k}: {v}")
    
    print(f"\n📊 Assets to train: {len(ASSETS)}")
    
    results = []
    
    # Train priority 1 first, then priority 2
    for priority in [1, 2]:
        priority_assets = {k: v for k, v in ASSETS.items() if v.get("priority", 2) == priority}
        
        if priority_assets:
            print(f"\n{'='*60}")
            print(f"🎯 PRIORITY {priority} ASSETS")
            print(f"{'='*60}")
            
            for symbol, info in priority_assets.items():
                result = train_asset(symbol, info)
                if result:
                    results.append(result)
    
    # Save summary
    summary = {
        "trained_at": datetime.utcnow().isoformat(),
        "hyperparameters": HYPERPARAMS,
        "models": results,
    }
    
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    
    print(f"\n📊 Models trained: {len(results)}/{len(ASSETS)}")
    print("\n📈 Results:")
    for r in sorted(results, key=lambda x: x["test_loss"]):
        print(f"   {r['name']:15} | Test Loss: {r['test_loss']:.6f} | Epochs: {r['epochs_trained']}")
    
    print(f"\n💾 Models saved to: {os.path.abspath(MODEL_DIR)}/")


if __name__ == "__main__":
    main()
