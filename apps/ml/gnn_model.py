"""
GNN (Graph Neural Network) for Market Analysis

Uses PyTorch Geometric for:
- Asset correlation networks
- Market structure analysis
- Cross-asset prediction
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

# Check if torch_geometric is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


@dataclass
class GNNConfig:
    """GNN model configuration."""
    num_node_features: int = 10  # Features per asset
    hidden_channels: int = 64
    num_classes: int = 3  # up, down, neutral
    num_heads: int = 4  # For GAT
    dropout: float = 0.3


if TORCH_GEOMETRIC_AVAILABLE:
    
    class MarketGNN(nn.Module):
        """
        Graph Neural Network for market analysis.
        
        Nodes: Assets (BTC, ETH, etc.)
        Edges: Correlations between assets
        """
        
        def __init__(self, config: GNNConfig):
            super().__init__()
            self.config = config
            
            # GCN layers
            self.conv1 = GCNConv(config.num_node_features, config.hidden_channels)
            self.conv2 = GCNConv(config.hidden_channels, config.hidden_channels)
            self.conv3 = GCNConv(config.hidden_channels, config.hidden_channels)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_channels, 32),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(32, config.num_classes),
            )
        
        def forward(self, x, edge_index, batch=None):
            # x: node features, edge_index: adjacency
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.config.dropout, training=self.training)
            
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=self.config.dropout, training=self.training)
            
            x = self.conv3(x, edge_index)
            
            # If batch provided, pool for graph-level prediction
            if batch is not None:
                x = global_mean_pool(x, batch)
            
            return self.classifier(x)
    
    
    class MarketGAT(nn.Module):
        """
        Graph Attention Network for market analysis.
        Uses attention to weight asset relationships.
        """
        
        def __init__(self, config: GNNConfig):
            super().__init__()
            self.config = config
            
            # GAT layers
            self.conv1 = GATConv(
                config.num_node_features, 
                config.hidden_channels, 
                heads=config.num_heads,
                dropout=config.dropout
            )
            self.conv2 = GATConv(
                config.hidden_channels * config.num_heads, 
                config.hidden_channels,
                heads=1,
                dropout=config.dropout
            )
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_channels, config.num_classes),
            )
        
        def forward(self, x, edge_index, batch=None):
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.config.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            
            if batch is not None:
                x = global_mean_pool(x, batch)
            
            return self.classifier(x)
    
    
    def build_correlation_graph(
        prices: dict,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a correlation graph from price data.
        
        Args:
            prices: Dict of symbol -> list of prices
            threshold: Minimum correlation for edge
        
        Returns:
            node_features, edge_index
        """
        symbols = list(prices.keys())
        n = len(symbols)
        
        # Calculate correlation matrix
        price_matrix = np.array([prices[s] for s in symbols])
        corr_matrix = np.corrcoef(price_matrix)
        
        # Build edge index (COO format)
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Node features (basic stats per asset)
        features = []
        for symbol in symbols:
            p = prices[symbol]
            features.append([
                np.mean(p),
                np.std(p),
                np.min(p),
                np.max(p),
                p[-1] / p[0] - 1,  # Return
                (p[-1] - np.mean(p)) / np.std(p),  # Z-score
                np.percentile(p, 25),
                np.percentile(p, 50),
                np.percentile(p, 75),
                len(p),
            ])
        
        node_features = torch.tensor(features, dtype=torch.float)
        
        return node_features, edge_index
    
    
    def predict_market_direction(
        model: MarketGNN,
        prices: dict,
        device: torch.device
    ) -> dict:
        """
        Predict market direction for assets.
        
        Returns dict of symbol -> predicted direction
        """
        model.eval()
        
        x, edge_index = build_correlation_graph(prices)
        x = x.to(device)
        edge_index = edge_index.to(device)
        
        with torch.no_grad():
            logits = model(x, edge_index)
            predictions = torch.argmax(logits, dim=1)
        
        symbols = list(prices.keys())
        directions = ['bearish', 'neutral', 'bullish']
        
        return {
            symbols[i]: directions[predictions[i].item()]
            for i in range(len(symbols))
        }


else:
    # Fallback stubs
    class MarketGNN:
        pass
    
    class MarketGAT:
        pass
    
    def build_correlation_graph(*args, **kwargs):
        raise ImportError("PyTorch Geometric not installed")
    
    def predict_market_direction(*args, **kwargs):
        raise ImportError("PyTorch Geometric not installed")
