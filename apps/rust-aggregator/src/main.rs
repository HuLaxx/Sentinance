//! Sentinance Price Aggregator
//! 
//! High-performance WebSocket price aggregator in Rust.
//! Connects to multiple exchanges and calculates VWAP.

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_tungstenite::{connect_async, tungstenite::Message};

// ===========================================
// DATA STRUCTURES
// ===========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceUpdate {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub exchange: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedPrice {
    pub symbol: String,
    pub vwap: f64,
    pub prices: HashMap<String, f64>,
    pub total_volume: f64,
    pub timestamp: u64,
}

// ===========================================
// EXCHANGE CONNECTORS
// ===========================================

pub struct ExchangeConnector {
    pub name: String,
    pub ws_url: String,
    pub subscribe_msg: String,
}

impl ExchangeConnector {
    pub fn binance(symbols: &[&str]) -> Self {
        let streams: Vec<String> = symbols
            .iter()
            .map(|s| format!("{}@trade", s.to_lowercase()))
            .collect();
        
        Self {
            name: "binance".to_string(),
            ws_url: format!(
                "wss://stream.binance.com:9443/stream?streams={}",
                streams.join("/")
            ),
            subscribe_msg: "".to_string(), // Binance uses URL-based subscription
        }
    }
    
    pub fn coinbase(symbols: &[&str]) -> Self {
        let product_ids: Vec<String> = symbols
            .iter()
            .map(|s| s.replace("USDT", "-USD"))
            .collect();
        
        let subscribe = serde_json::json!({
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": ["ticker"]
        });
        
        Self {
            name: "coinbase".to_string(),
            ws_url: "wss://ws-feed.exchange.coinbase.com".to_string(),
            subscribe_msg: subscribe.to_string(),
        }
    }
}

// ===========================================
// PRICE AGGREGATOR
// ===========================================

pub struct PriceAggregator {
    prices: Arc<RwLock<HashMap<String, HashMap<String, PriceUpdate>>>>,
}

impl PriceAggregator {
    pub fn new() -> Self {
        Self {
            prices: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn update_price(&self, update: PriceUpdate) {
        let mut prices = self.prices.write().await;
        
        let symbol_prices = prices
            .entry(update.symbol.clone())
            .or_insert_with(HashMap::new);
        
        symbol_prices.insert(update.exchange.clone(), update);
    }
    
    pub async fn get_vwap(&self, symbol: &str) -> Option<AggregatedPrice> {
        let prices = self.prices.read().await;
        let symbol_prices = prices.get(symbol)?;
        
        if symbol_prices.is_empty() {
            return None;
        }
        
        let mut total_volume = 0.0;
        let mut weighted_sum = 0.0;
        let mut price_map = HashMap::new();
        let mut latest_timestamp = 0u64;
        
        for (exchange, update) in symbol_prices {
            weighted_sum += update.price * update.volume;
            total_volume += update.volume;
            price_map.insert(exchange.clone(), update.price);
            latest_timestamp = latest_timestamp.max(update.timestamp);
        }
        
        let vwap = if total_volume > 0.0 {
            weighted_sum / total_volume
        } else {
            0.0
        };
        
        Some(AggregatedPrice {
            symbol: symbol.to_string(),
            vwap,
            prices: price_map,
            total_volume,
            timestamp: latest_timestamp,
        })
    }
}

// ===========================================
// MAIN
// ===========================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting Sentinance Price Aggregator...");
    
    let aggregator = Arc::new(PriceAggregator::new());
    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    
    // Connect to Binance
    let binance = ExchangeConnector::binance(&symbols);
    let agg_clone = aggregator.clone();
    
    tokio::spawn(async move {
        if let Err(e) = connect_binance(&binance, agg_clone).await {
            eprintln!("Binance error: {}", e);
        }
    });
    
    // Print VWAP every second
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        
        for symbol in &symbols {
            if let Some(agg) = aggregator.get_vwap(symbol).await {
                println!(
                    "{}: VWAP=${:.2} (exchanges: {})",
                    symbol,
                    agg.vwap,
                    agg.prices.len()
                );
            }
        }
    }
}

async fn connect_binance(
    connector: &ExchangeConnector,
    aggregator: Arc<PriceAggregator>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (ws_stream, _) = connect_async(&connector.ws_url).await?;
    let (mut _write, mut read) = ws_stream.split();
    
    println!("Connected to {}", connector.name);
    
    while let Some(msg) = read.next().await {
        if let Ok(Message::Text(text)) = msg {
            // Parse Binance trade message
            if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(trade) = data.get("data") {
                    let symbol = trade["s"].as_str().unwrap_or("").to_string();
                    let price = trade["p"].as_str()
                        .and_then(|p| p.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let volume = trade["q"].as_str()
                        .and_then(|q| q.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    
                    if price > 0.0 {
                        let update = PriceUpdate {
                            symbol,
                            price,
                            volume,
                            exchange: "binance".to_string(),
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_millis() as u64,
                        };
                        
                        aggregator.update_price(update).await;
                    }
                }
            }
        }
    }
    
    Ok(())
}
