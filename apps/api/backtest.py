"""
Backtesting Service

Simple backtesting for prediction strategies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import structlog

log = structlog.get_logger()


@dataclass
class BacktestResult:
    """Backtest results."""
    strategy: str
    symbol: str
    period_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return_percent: float
    max_drawdown_percent: float
    sharpe_ratio: Optional[float]
    
    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "period_days": self.period_days,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate * 100, 2),
            "total_return_percent": round(self.total_return_percent, 2),
            "max_drawdown_percent": round(self.max_drawdown_percent, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2) if self.sharpe_ratio else None,
        }


def backtest_momentum(
    prices: list[float],
    lookback: int = 5,
    holding_period: int = 1
) -> BacktestResult:
    """
    Backtest momentum strategy.
    Buy if momentum positive, sell if negative.
    """
    if len(prices) < lookback + holding_period + 10:
        return BacktestResult(
            strategy="momentum",
            symbol="",
            period_days=len(prices),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_return_percent=0,
            max_drawdown_percent=0,
            sharpe_ratio=None,
        )
    
    trades = []
    returns = []
    equity = [100.0]  # Start with $100
    
    for i in range(lookback, len(prices) - holding_period):
        # Calculate momentum
        momentum = (prices[i] - prices[i - lookback]) / prices[i - lookback]
        
        # Signal: positive momentum = long
        if momentum > 0:
            entry = prices[i]
            exit_price = prices[i + holding_period]
            trade_return = (exit_price - entry) / entry
            returns.append(trade_return)
            trades.append({
                "entry": entry,
                "exit": exit_price,
                "return": trade_return,
                "win": trade_return > 0,
            })
            equity.append(equity[-1] * (1 + trade_return))
        else:
            equity.append(equity[-1])
    
    # Calculate metrics
    winning = sum(1 for t in trades if t["win"])
    total = len(trades)
    win_rate = winning / total if total > 0 else 0
    
    total_return = (equity[-1] - 100) / 100 * 100 if equity else 0
    
    # Max drawdown
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Sharpe ratio (simplified)
    if returns:
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std = variance ** 0.5
        sharpe = (avg_return * 252) / (std * (252 ** 0.5)) if std > 0 else 0
    else:
        sharpe = None
    
    return BacktestResult(
        strategy="momentum",
        symbol="",
        period_days=len(prices),
        total_trades=total,
        winning_trades=winning,
        losing_trades=total - winning,
        win_rate=win_rate,
        total_return_percent=total_return,
        max_drawdown_percent=max_dd * 100,
        sharpe_ratio=sharpe,
    )


def backtest_mean_reversion(
    prices: list[float],
    period: int = 20,
    threshold: float = 2.0
) -> BacktestResult:
    """
    Backtest mean reversion strategy.
    Buy when price is 2 std below mean, sell when above.
    """
    if len(prices) < period + 10:
        return BacktestResult(
            strategy="mean_reversion",
            symbol="",
            period_days=len(prices),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_return_percent=0,
            max_drawdown_percent=0,
            sharpe_ratio=None,
        )
    
    trades = []
    returns = []
    equity = [100.0]
    
    for i in range(period, len(prices) - 1):
        window = prices[i - period:i]
        mean = sum(window) / period
        std = (sum((p - mean) ** 2 for p in window) / period) ** 0.5
        
        if std == 0:
            equity.append(equity[-1])
            continue
        
        z_score = (prices[i] - mean) / std
        
        # Buy if oversold, sell if overbought
        if z_score < -threshold:
            entry = prices[i]
            exit_price = prices[i + 1]
            trade_return = (exit_price - entry) / entry
            returns.append(trade_return)
            trades.append({"return": trade_return, "win": trade_return > 0})
            equity.append(equity[-1] * (1 + trade_return))
        elif z_score > threshold:
            # Short (simplified as negative return expectation)
            entry = prices[i]
            exit_price = prices[i + 1]
            trade_return = (entry - exit_price) / entry
            returns.append(trade_return)
            trades.append({"return": trade_return, "win": trade_return > 0})
            equity.append(equity[-1] * (1 + trade_return))
        else:
            equity.append(equity[-1])
    
    # Metrics
    winning = sum(1 for t in trades if t["win"])
    total = len(trades)
    win_rate = winning / total if total > 0 else 0
    total_return = (equity[-1] - 100) / 100 * 100 if equity else 0
    
    peak = max(equity) if equity else 100
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd
    
    return BacktestResult(
        strategy="mean_reversion",
        symbol="",
        period_days=len(prices),
        total_trades=total,
        winning_trades=winning,
        losing_trades=total - winning,
        win_rate=win_rate,
        total_return_percent=total_return,
        max_drawdown_percent=max_dd * 100,
        sharpe_ratio=None,
    )
