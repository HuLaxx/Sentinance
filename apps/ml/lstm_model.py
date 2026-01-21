"""
LSTM Price Prediction Model

Deep learning model for time-series price prediction.
Uses PyTorch for implementation.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ModelConfig:
    """LSTM model configuration."""
    input_size: int = 5  # OHLCV
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 1  # Predict close price
    dropout: float = 0.2
    sequence_length: int = 60  # 60 time steps
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100


if TORCH_AVAILABLE:
    
    class PriceDataset(Dataset):
        """Dataset for price prediction."""
        
        def __init__(self, data: np.ndarray, sequence_length: int = 60):
            self.data = data
            self.sequence_length = sequence_length
        
        def __len__(self):
            return len(self.data) - self.sequence_length
        
        def __getitem__(self, idx):
            x = self.data[idx:idx + self.sequence_length]
            y = self.data[idx + self.sequence_length, 3]  # Close price
            return torch.FloatTensor(x), torch.FloatTensor([y])
    
    
    class LSTMModel(nn.Module):
        """LSTM model for price prediction."""
        
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            
            self.lstm = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
            )
            
            self.fc = nn.Sequential(
                nn.Linear(config.hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(32, config.output_size),
            )
        
        def forward(self, x):
            # x: (batch, seq_len, input_size)
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use last hidden state
            out = self.fc(lstm_out[:, -1, :])
            return out
    
    
    class Trainer:
        """Training utilities for LSTM model."""
        
        def __init__(self, model: LSTMModel, config: ModelConfig):
            self.model = model
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            self.optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.learning_rate
            )
            self.criterion = nn.MSELoss()
        
        def train_epoch(self, dataloader: DataLoader) -> float:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0.0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            return total_loss / len(dataloader)
        
        def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
            """Evaluate model."""
            self.model.eval()
            total_loss = 0.0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    
                    total_loss += loss.item()
                    predictions.extend(outputs.cpu().numpy().flatten())
                    actuals.extend(batch_y.cpu().numpy().flatten())
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
            
            return total_loss / len(dataloader), rmse
        
        def train(
            self, 
            train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None
        ):
            """Full training loop."""
            best_val_loss = float('inf')
            
            for epoch in range(self.config.epochs):
                train_loss = self.train_epoch(train_loader)
                
                if val_loader:
                    val_loss, val_rmse = self.evaluate(val_loader)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model("best_model.pt")
                    
                    if (epoch + 1) % 10 == 0:
                        print(f"Epoch {epoch+1}/{self.config.epochs} - "
                              f"Train Loss: {train_loss:.6f} - "
                              f"Val Loss: {val_loss:.6f} - "
                              f"Val RMSE: {val_rmse:.2f}")
                else:
                    if (epoch + 1) % 10 == 0:
                        print(f"Epoch {epoch+1}/{self.config.epochs} - "
                              f"Train Loss: {train_loss:.6f}")
        
        def save_model(self, path: str):
            """Save model checkpoint."""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
            }, path)
        
        def load_model(self, path: str):
            """Load model checkpoint."""
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    def predict(model: LSTMModel, data: np.ndarray, device: torch.device) -> float:
        """Make prediction with trained model."""
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(data).unsqueeze(0).to(device)
            prediction = model(x)
            return prediction.cpu().item()


# Fallback for when torch is not available
else:
    class ModelConfig:
        pass
    
    class LSTMModel:
        pass
    
    class Trainer:
        pass
    
    def predict(*args, **kwargs):
        raise ImportError("PyTorch not installed")
