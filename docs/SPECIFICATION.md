# **PROJECT 1: SENTINANCE - ENTERPRISE MARKET INTELLIGENCE TERMINAL**

## **I. Executive Summary**

### **Project Title**
**Sentinance: Real-Time Crypto Market Intelligence Platform with Agentic AI**

### **Real-World Problem Statement**
Cryptocurrency traders and institutional investors lose billions annually due to information asymmetry and delayed decision-making. Current solutions suffer from:
- **Fragmented data sources**: Prices, news, social sentiment, and on-chain metrics exist in silos
- **Manual synthesis**: Analysts spend hours correlating signals across 10+ platforms
- **No predictive edge**: Reactive dashboards show what happened, not what's coming
- **Alert fatigue**: 100+ notifications daily with 95% false positives
- **Compliance blind spots**: No audit trail for investment decisions

**Business Value**: Sentinance provides institutional-grade market intelligence that:
- Reduces research time from 4 hours to 15 minutes per asset
- Increases alpha generation through predictive multi-signal analysis
- Provides explainable AI recommendations for regulatory compliance
- Detects market manipulation and rug-pull events 12-48 hours early
- Creates auditable decision trails for SEC/FINRA requirements

### **High-Level Solution**
Sentinance is an enterprise-grade, event-driven market intelligence terminal that unifies crypto market data through a sophisticated data engineering pipeline processing 50K+ events/second from heterogeneous sources (exchanges, news APIs, blockchain nodes, Twitter). The platform employs a dual-brain architecture: a **real-time analytics engine** for low-latency pattern detection (sub-second) and an **agentic AI layer** powered by LangGraph for synthesis, reasoning, and proactive alerting. 

The system implements a Lambda architecture with hot path (Kafka streaming for millisecond-latency alerts) and cold path (Spark batch jobs for historical backtesting). Machine learning models—fine-tuned LLMs for sentiment, GNNs for wallet network analysis, and LSTM ensembles for price forecasting—are continuously retrained through MLflow-managed pipelines with automated A/B testing. The front-end is a Next.js 15 application with generative UI that dynamically renders investigation tools (charts, network graphs, risk reports) based on AI agent tool calls, providing Goldman Sachs-level UX for crypto markets.

---

## **II. Technical Stack & Granular Details**

### **Languages**
- **Python 3.11+**: Data engineering, ML training, FastAPI services
- **TypeScript 5.2+**: Full-stack (Next.js, API types)
- **Rust**: High-performance price aggregation service (optional for performance)
- **SQL**: dbt transformations, analytical queries
- **HCL**: Terraform infrastructure definitions

### **Frameworks**

**Backend:**
- **FastAPI 0.104+**: Main API gateway, WebSocket streaming
- **LangGraph 0.0.40+**: Multi-agent orchestration
- **DSPy 2.0+**: Prompt optimization and versioning
- **TorchServe 0.9+**: Model serving for GNN and LSTM
- **Ray 2.7+**: Distributed model training and hyperparameter tuning

**Frontend:**
- **Next.js 15.0**: App router, server components, streaming
- **Vercel AI SDK 3.0**: Generative UI, tool invocations
- **shadcn/ui**: Component library
- **TanStack Query**: State management
- **Recharts + D3.js**: Charting and network visualizations

**Data Engineering:**
- **Apache Kafka 3.6 / Redpanda**: Event streaming backbone
- **Apache Spark 3.5**: Batch processing, feature engineering
- **dbt 1.7**: SQL transformations, data modeling
- **Apache Airflow 2.7**: Workflow orchestration
- **Debezium 2.4**: Change Data Capture (CDC)

**ML/DS:**
- **PyTorch 2.1**: Deep learning (GNN, LSTM)
- **PyTorch Geometric 2.4**: Graph Neural Networks
- **Transformers 4.35 (HuggingFace)**: Fine-tuning LLMs
- **PEFT (LoRA)**: Efficient fine-tuning
- **Prophet**: Time-series decomposition
- **scikit-learn**: Feature engineering, baselines

### **Infrastructure**

**Container Orchestration:**
- **Kubernetes 1.28** (EKS): Microservices deployment
- **Helm 3.13**: Package management
- **Istio 1.19**: Service mesh (mTLS, traffic management)
- **cert-manager**: TLS certificate automation

**Infrastructure as Code:**
- **Terraform 1.6**: AWS resource provisioning
- **Terragrunt**: DRY Terraform configurations

**Cloud Services (AWS):**
- **EKS**: Kubernetes cluster
- **ECS Fargate**: Serverless containers (price aggregator)
- **S3**: Data lake (raw/processed data, model artifacts)
- **RDS Aurora PostgreSQL**: Transactional database
- **DocumentDB**: MongoDB-compatible (user preferences)
- **ElastiCache Redis**: Caching, semantic cache
- **MSK (Managed Kafka)**: Production Kafka cluster
- **Lambda**: Serverless functions (webhooks, alerts)
- **SageMaker**: Model training (alternative to Ray)
- **CloudWatch + X-Ray**: Metrics and distributed tracing
- **Secrets Manager**: API keys, database credentials
- **ALB**: Application load balancing
- **Route 53**: DNS management

### **Data Stores**

**Relational:**
- **PostgreSQL 16 (Aurora)**: Main transactional DB
  - Tables: users, portfolios, alerts, audit_logs
  - Extensions: pgvector (embeddings), TimescaleDB (time-series)

**NoSQL:**
- **MongoDB (DocumentDB)**: User profiles, dashboard configs
- **Redis (ElastiCache)**: Session cache, semantic cache (vector search), rate limiting

**Vector:**
- **Qdrant 1.7** (self-hosted on K8s): News embeddings, semantic search
- Alternative: **Pinecone** (managed)

**Data Lake:**
- **S3**: Parquet files (partitioned by date/symbol)
- **Delta Lake 3.0**: ACID transactions on S3
- **Iceberg** (alternative): Better schema evolution

**Graph:**
- **Neo4j 5.13** (AuraDB): Wallet relationships, transaction graphs

### **DevOps/MLOps Tools**

**CI/CD:**
- **GitHub Actions**: Build, test, deploy pipelines
- **ArgoCD**: GitOps Kubernetes deployments
- **Flux** (alternative): GitOps

**ML Lifecycle:**
- **MLflow 2.8**: Experiment tracking, model registry
- **Weights & Biases** (alternative): Advanced experiment tracking
- **DVC 3.30**: Data versioning
- **Great Expectations 0.18**: Data quality validation
- **Evidently AI 0.4**: ML monitoring, drift detection

**Orchestration:**
- **Airflow 2.7**: DAGs for batch jobs, model retraining
- **Prefect** (alternative): Modern workflow engine

**Observability:**
- **Prometheus 2.48**: Metrics collection
- **Grafana 10.2**: Dashboards
- **Loki**: Log aggregation
- **Tempo**: Distributed tracing (alternative to X-Ray)
- **OpenTelemetry**: Instrumentation SDK
- **Arize Phoenix**: LLM observability

**Security:**
- **Vault (HashiCorp)**: Secrets management (alternative to AWS Secrets Manager)
- **Falco**: Runtime security monitoring
- **Trivy**: Container vulnerability scanning

### **Libraries (The "Small Details")**

**Python Backend:**
```python
# API & Validation
fastapi==0.104.1
pydantic==2.5.0
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.23
alembic==1.13.0  # Migrations
asyncpg==0.29.0  # Async Postgres
psycopg2-binary==2.9.9

# Kafka
confluent-kafka==2.3.0
kafka-python==2.0.2

# Redis
redis==5.0.1
redis-py-cluster==2.1.3

# Authentication
python-jose[cryptography]==3.3.0  # JWT
passlib[bcrypt]==1.7.4

# Observability
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-instrumentation-fastapi==0.42b0
prometheus-client==0.19.0

# LLM/AI
langchain==0.1.0
langgraph==0.0.40
langsmith==0.0.70  # Tracing
dspy-ai==2.0.0
openai==1.6.0
anthropic==0.7.0

# ML
torch==2.1.2
torch-geometric==2.4.0
transformers==4.36.0
peft==0.7.1  # LoRA
bitsandbytes==0.41.0  # Quantization
shap==0.44.0
scikit-learn==1.3.2

# Data
pandas==2.1.4
polars==0.19.19  # Fast dataframes
pyarrow==14.0.2
duckdb==0.9.2  # In-process OLAP

# Utils
tenacity==8.2.3  # Retries
pydantic-yaml==1.2.0  # Config files
python-dotenv==1.0.0
structlog==23.2.0  # Structured logging
```

**TypeScript Frontend:**
```json
{
  "dependencies": {
    "next": "15.0.0",
    "react": "19.0.0",
    "ai": "3.0.0",
    "@tanstack/react-query": "5.17.0",
    "zustand": "4.4.7",
    "zod": "3.22.4",
    "@t3-oss/env-nextjs": "0.7.1",
    "recharts": "2.10.3",
    "d3": "7.8.5",
    "@auth/core": "0.18.0",
    "next-auth": "5.0.0-beta.4",
    "framer-motion": "10.16.16",
    "tailwindcss": "3.4.0",
    "sharp": "0.33.1"
  },
  "devDependencies": {
    "typescript": "5.3.3",
    "@types/react": "18.2.45",
    "eslint": "8.56.0",
    "prettier": "3.1.1",
    "vitest": "1.0.4",
    "@playwright/test": "1.40.1"
  }
}
```

---

## **III. System Architecture & Design**

### **High-Level Architecture**

**Architecture Style**: **Event-Driven Microservices with CQRS**

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  Next.js 15 (SSR/SSG) + Vercel AI SDK (Generative UI)          │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTPS/WSS
┌──────────────────────▼──────────────────────────────────────────┐
│                     API GATEWAY (Kong/AWS ALB)                   │
│  Rate Limiting │ Auth │ WAF │ API Versioning                    │
└─────┬─────────────────┬──────────────────────┬──────────────────┘
      │                 │                      │
      ▼                 ▼                      ▼
┌─────────────┐  ┌─────────────┐      ┌──────────────────┐
│ Auth Service│  │ Query API   │      │ Command API      │
│ (FastAPI)   │  │ (FastAPI)   │      │ (FastAPI)        │
│ JWT + OAuth │  │ Read Models │      │ Write Models     │
└─────────────┘  └──────┬──────┘      └────────┬─────────┘
                        │ Read                  │ Write
                        ▼                       ▼
              ┌──────────────────┐    ┌──────────────────┐
              │  PostgreSQL      │    │  Event Store     │
              │  (Read Replicas) │    │  (Kafka Topics)  │
              └──────────────────┘    └────────┬─────────┘
                                               │
                    ┌──────────────────────────┼──────────┐
                    ▼                          ▼          ▼
          ┌──────────────────┐      ┌──────────────┐  ┌────────────┐
          │ Stream Processor │      │ Batch Jobs   │  │ ML Service │
          │ (Kafka Streams/  │      │ (Spark/dbt)  │  │ (TorchServe│
          │  Flink)          │      │ via Airflow  │  │  + Ray)    │
          └─────────┬────────┘      └──────┬───────┘  └─────┬──────┘
                    │                      │                │
                    ▼                      ▼                ▼
          ┌─────────────────────────────────────────────────┐
          │           UNIFIED DATA PLATFORM                  │
          │  S3 Data Lake (Delta/Iceberg) + Feature Store   │
          │  Redis (Cache) + Qdrant (Vectors) + Neo4j      │
          └─────────────────────────────────────────────────┘
                              ▲
                              │ Ingest
          ┌───────────────────┴────────────────────┐
          │      DATA INGESTION SERVICES           │
          │  Price Stream │ News │ Social │ Chain  │
          └────────────────────────────────────────┘
```

**Key Components:**

1. **Ingestion Layer** (Data Engineering):
   - **Price Aggregator**: Rust service connecting to 10+ exchanges via WebSocket
   - **News Scraper**: Scrapy spiders + Kafka producers
   - **Social Listener**: Twitter API → Kafka (sentiment analysis)
   - **On-Chain Monitor**: Ethereum/Bitcoin node + Debezium CDC

2. **Event Backbone** (Data Engineering):
   - **Kafka/MSK** with topics: `prices`, `news`, `social`, `transactions`, `alerts`
   - **Schema Registry** (Confluent): Avro schemas for type safety

3. **Stream Processing** (Data Engineering + ML):
   - **Kafka Streams**: Real-time aggregations (OHLCV candles, moving averages)
   - **Flink** (optional): Complex event processing (pattern detection)

4. **Batch Processing** (Data Engineering + DS):
   - **Spark Jobs**: Daily feature engineering, backfills
   - **dbt Models**: SQL transformations (facts, dimensions, aggregates)
   - **Airflow DAGs**: Orchestration, scheduling

5. **ML Services** (MLE):
   - **Training Pipeline**: Ray + MLflow for distributed training
   - **Model Registry**: MLflow with S3 backend
   - **Inference**: TorchServe for low-latency predictions (<50ms p99)
   - **A/B Testing**: Custom framework with statistical significance tests

6. **Agentic AI Layer** (MLE + SE):
   - **LangGraph Agents**: Multi-agent system (Researcher, Analyst, Risk Manager)
   - **Tool Registry**: MCP protocol for tool discovery
   - **Semantic Cache**: Redis with vector search (90% cache hit rate)

7. **API Layer** (SE):
   - **Query API** (CQRS Read): Optimized for fast reads, materialized views
   - **Command API** (CQRS Write): Handles mutations, publishes events
   - **WebSocket Server**: Real-time price/alert streaming

8. **Frontend** (SE):
   - **Next.js 15**: Server components, streaming SSR
   - **Generative UI**: AI agents render React components (charts, reports)
   - **State Management**: Zustand for client state, React Query for server state

### **Data Flow & Orchestration**

**Scenario: User asks "Is Bitcoin showing manipulation signals?"**

```
1. USER ACTION (Frontend)
   └─> Next.js chat interface sends query via POST /api/analyze

2. API GATEWAY
   └─> Validates JWT, checks rate limits (Redis)
   └─> Routes to Query API

3. QUERY API (FastAPI)
   └─> Publishes command to Kafka topic: `commands.analyze`
   └─> Returns 202 Accepted with correlation_id
   └─> Opens WebSocket for streaming response

4. AGENTIC AI SERVICE (LangGraph)
   └─> Consumes `commands.analyze` event
   └─> LangGraph StateGraph executes:
       
   a) PLANNER NODE
      └─> LLM breaks query into sub-tasks:
          - "Check order book depth anomalies"
          - "Analyze whale wallet movements"
          - "Scan news for pump-and-dump mentions"
   
   b) TOOL EXECUTOR NODES (Parallel)
      └─> Calls MCP tools:
          - get_order_book_depth(symbol="BTC")
            └─> Redis cache hit? → Return cached
            └─> Cache miss? → Query PostgreSQL → Cache → Return
          
          - analyze_wallet_network(address="0x...")
            └─> Queries Neo4j for connected wallets
            └─> Runs GNN inference (TorchServe)
            └─> Returns risk score + subgraph
          
          - search_news_sentiment(query="Bitcoin manipulation")
            └─> Qdrant vector search on news embeddings
            └─> Returns top 5 articles + sentiment scores
   
   c) SYNTHESIZER NODE
      └─> LLM (fine-tuned Mistral) reasons over tool outputs
      └─> Generates structured report (JSON):
          {
            "manipulation_likelihood": 0.78,
            "evidence": [...],
            "recommendation": "High caution - 3 whale wallets moved 10K BTC to exchanges"
          }
   
   d) OBSERVABILITY
      └─> OpenTelemetry traces entire flow
      └─> Phoenix logs LLM inputs/outputs, token usage
      └─> Langfuse tracks agent reasoning path

5. RESPONSE STREAMING
   └─> Agent publishes results to Kafka: `results.analyze.{correlation_id}`
   └─> WebSocket service consumes event
   └─> Streams JSON chunks to client via SSE

6. GENERATIVE UI (Frontend)
   └─> Vercel AI SDK receives tool invocations:
       - RiskGauge component for manipulation_likelihood
       - WalletNetworkGraph for Neo4j subgraph
       - NewsCarousel for sentiment articles
   └─> React renders dynamically generated UI

7. AUDIT TRAIL (Compliance)
   └─> Every decision logged to `audit_logs` table (Postgres)
   └─> Immutable record: user_id, query, tools_called, model_used, timestamp
```

**Critical Data Paths:**

**Hot Path (Real-Time):**
```
Exchange WebSocket → Price Aggregator → Kafka → Kafka Streams → Redis
                                                          ↓
User WebSocket ←─────────────────────────────────────── Redis
(Sub-second latency)
```

**Cold Path (Batch):**
```
S3 Data Lake → Spark → Feature Engineering → dbt → PostgreSQL
                                                   ↓
                                             ML Training (Daily)
```

### **Design Patterns**

1. **CQRS (Command Query Responsibility Segregation)**
   - **Write Model**: Command API writes to event store (Kafka)
   - **Read Model**: Query API reads from materialized views (PostgreSQL read replicas)
   - **Why**: Optimize reads and writes independently, scalability

2. **Event Sourcing**
   - All state changes are events in Kafka (immutable log)
   - Events: `PriceUpdated`, `AlertTriggered`, `ModelRetrained`
   - **Why**: Audit trail, time-travel debugging, replay capability

3. **Saga Pattern**
   - Long-running workflows (e.g., model retraining pipeline)
   - Orchestration: Airflow DAG coordinates Spark → MLflow → TorchServe
   - **Why**: Manage distributed transactions, failure recovery

4. **Repository Pattern**
   - Abstract data access: `UserRepository`, `AlertRepository`
   - Interface: `Repository[T]` with `get()`, `save()`, `query()`
   - **Why**: Testability, database agnostic

5. **Factory Pattern**
   - `ModelFactory.create(model_type="gnn")` → Returns trained model
   - **Why**: Encapsulate model instantiation logic

6. **Circuit Breaker**
   - External API calls (CoinGecko, Twitter) wrapped in Tenacity circuit breaker
   - Fail fast if 50% of requests fail in 60s
   - **Why**: Prevent cascade failures

7. **Bulkhead Pattern**
   - Separate thread pools for price streaming vs news scraping
   - **Why**: Isolate failures, prevent resource starvation

8. **Strangler Fig**
   - Gradually migrate monolith → microservices
   - API gateway routes new features to new services
   - **Why**: Low-risk incremental migration

### **Design Decisions (Trade-offs)**

| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| **Streaming** | Kafka | RabbitMQ | Kafka: Better for event sourcing, higher throughput (1M msg/s), log-based. RabbitMQ: Better for RPC patterns. |
| **Batch Processing** | Spark | Pandas | Spark: Handles TB-scale data, distributed. Pandas: Simpler but memory-limited. |
| **Orchestration** | Airflow | Prefect | Airflow: Mature, large community, task dependencies. Prefect: Modern, better DX but smaller ecosystem. |
| **API Framework** | FastAPI | Flask | FastAPI: Native async, auto-docs, Pydantic validation. Flask: Simpler but requires manual validation. |
| **Frontend** | Next.js 15 | Remix | Next.js: RSC, better DX, Vercel AI SDK. Remix: Nested routes, better forms. |
| **Database** | PostgreSQL | MongoDB | PostgreSQL: ACID, SQL, pgvector for embeddings. MongoDB: Flexible schema but weaker transactions. |
| **Container Orchestration** | Kubernetes | ECS | Kubernetes: Portable, rich ecosystem (Helm, Istio). ECS: Simpler but AWS lock-in. |
| **Vector DB** | Qdrant | Pinecone | Qdrant: Open-source, self-hosted, no costs. Pinecone: Managed but expensive at scale. |
| **LLM Serving** | vLLM | TGI | vLLM: Higher throughput (PagedAttention), better for production. TGI: HuggingFace integration. |
| **Model Training** | Ray | SageMaker | Ray: Open-source, Kubernetes-native, no vendor lock-in. SageMaker: Managed but more expensive. |
| **Graph DB** | Neo4j | DGraph | Neo4j: Mature, Cypher query language, strong community. DGraph: Faster but less mature. |

---

## **IV. Implementation Roadmap (The "Quad-Core" Integration)**

### **Phase 0: Foundation (Week 1-2)**

**All Tracks:**
- Set up monorepo: `pnpm workspaces` or `turborepo`
- Configure Terraform: VPC, subnets, EKS cluster, RDS, S3
- Deploy base infrastructure: Kafka (MSK), PostgreSQL (RDS), Redis (ElastiCache)
- Set up GitHub Actions: lint, test, build, deploy
- Initialize observability: Prometheus, Grafana, OpenTelemetry

---

### **1. Data Engineering Track (Week 3-8)**

#### **Week 3-4: Data Ingestion**

**Task 1.1: Price Aggregator Service**
```bash
# Technology: Rust + tokio for async WebSocket handling
# Goal: Aggregate prices from 10 exchanges with <10ms latency

crypto-price-aggregator/
├── src/
│   ├── main.rs
│   ├── exchanges/
│   │   ├── binance.rs  # WebSocket client
│   │   ├── coinbase.rs
│   │   └── kraken.rs
│   ├── aggregator.rs  # Merge streams, calculate VWAP
│   └── kafka_producer.rs
└── Cargo.toml

# Key Implementation:
- Connect to exchange WebSockets simultaneously
- Handle reconnection with exponential backoff
- Normalize price formats (Binance uses strings, Coinbase uses floats)
- Produce to Kafka `prices` topic with Avro schema
- Handle rate limits: Binance 1200 req/min, Coinbase 10 req/s

# Gotcha: Binance WebSocket drops after 24h - implement ping/pong heartbeat
```

**Task 1.2: News Scraper Pipeline**
```python
# Technology: Scrapy + Kafka + Schema Registry
# Goal: Scrape CoinDesk, Cointelegraph, CryptoSlate every 5 minutes

# File: scrapy_project/spiders/news_spider.py
class NewsSpider(scrapy.Spider):
    name = 'crypto_news'
    
    def parse(self, response):
        # Extract: title, body, publish_date, author
        # NLP preprocessing: spaCy tokenization, entity extraction
        # Produce to Kafka `news.raw` topic
        
    # Gotcha: CoinDesk uses infinite scroll (requires Selenium)
    # Gotcha: Rate limiting - respect robots.txt, add delays

# Deployment: Run as Kubernetes CronJob every 5 minutes
```

**Task 1.3: Change Data Capture (CDC) for On-Chain Data**
```yaml
# Technology: Debezium + Ethereum Node
# Goal: Capture wallet transactions in real-time

# Deploy Ethereum full node (Geth) on EKS
# Configure Debezium connector to tail transaction logs
# Stream to Kafka `blockchain.transactions` topic

# Debezium connector config:
apiVersion: kafka.strimzi.io/v1beta2
kind: KafkaConnector
metadata:
  name: ethereum-cdc
spec:
  class: io.debezium.connector.ethereum.EthereumConnector
  config:
    ethereum.node.url: "http://geth:8545"
    topic.prefix: "blockchain"
    
# Gotcha: Geth sync takes 7 days - use checkpoint sync or Infura API
```

#### **Week 5-6: Stream Processing**

**Task 1.4: Real-Time Aggregations**
```python
# Technology: Kafka Streams (Python: faust-streaming)
# Goal: Calculate 1min, 5min, 15min OHLCV candles

from faust import App, Stream, windowing

app = App('price-aggregator', broker='kafka:9092')

price_topic = app.topic('prices', value_type=PriceEvent)

@app.agent(price_topic)
async def calculate_candles(stream: Stream[PriceEvent]):
    async for window in stream.tumbling(60.0, key='symbol').items():
        symbol, events = window
        ohlcv = {
            'open': events[0].price,
            'high': max(e.price for e in events),
            'low': min(e.price for e in events),
            'close': events[-1].price,
            'volume': sum(e.volume for e in events)
        }
        await app.topic('candles.1m').send(key=symbol, value=ohlcv)

# Gotcha: Faust doesn't handle late arrivals well - add 30s grace period
# Gotcha: State store needs persistent volume (K8s PVC)
```

**Task 1.5: Event Schema Evolution**
```python
# Technology: Confluent Schema Registry + Avro
# Goal: Version schemas, maintain backward compatibility

# schemas/price_event_v1.avsc
{
  "type": "record",
  "name": "PriceEvent",
  "namespace": "sentinance.events",
  "fields": [
    {"name": "symbol", "type": "string"},
    {"name": "price", "type": "double"},
    {"name": "timestamp", "type": "long"}
  ]
}

# schemas/price_event_v2.avsc (add exchange field)
{
  "fields": [
    {"name": "symbol", "type": "string"},
    {"name": "price", "type": "double"},
    {"name": "timestamp", "type": "long"},
    {"name": "exchange", "type": "string", "default": "unknown"}  # Backward compat
  ]
}

# Gotcha: Always add default values for new fields
# Gotcha: Never remove required fields (use tombstones instead)
```

#### **Week 7-8: Batch Processing & dbt**

**Task 1.6: Spark Feature Engineering**
```python
# Technology: PySpark + Delta Lake
# Goal: Daily batch job to compute 200+ features

# File: spark_jobs/feature_engineering.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, stddev, col

spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .getOrCreate()

# Read from S3 data lake
df = spark.read.format("delta").load("s3://sentinance/prices/")

# Technical indicators
df = df.withColumn("rsi_14", calculate_rsi(col("close"), 14))
df = df.withColumn("macd", calculate_macd(col("close")))
df = df.withColumn("volatility_30d", stddev("close").over(Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-30, 0)))

# Write to Feature Store (Feast)
df.write.format("delta").mode("overwrite").save("s3://sentinance/features/")

# Gotcha: Spark default partitioning (200) is too high - set to 10-20 for small datasets
# Gotcha: Broadcast joins fail for DataFrames > 10MB - increase spark.sql.autoBroadcastJoinThreshold
```

**Task 1.7: dbt Data Modeling**
```sql
-- File: dbt/models/marts/fact_daily_prices.sql
-- Goal: Create dimensional model for analytics

{{ config(materialized='incremental', unique_key='price_date_symbol') }}

WITH daily_agg AS (
  SELECT
    symbol,
    DATE(timestamp) AS price_date,
    FIRST_VALUE(price) OVER (PARTITION BY symbol, DATE(timestamp) ORDER BY timestamp) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST_VALUE(price) OVER (PARTITION BY symbol, DATE(timestamp) ORDER BY timestamp) AS close,
    SUM(volume) AS volume
  FROM {{ source('raw', 'prices') }}
  WHERE timestamp >= '{{ var("start_date") }}'
  GROUP BY symbol, DATE(timestamp)
)

SELECT * FROM daily_agg

-- Gotcha: Incremental models need to handle late-arriving data
-- Use is_incremental() macro to filter
```

**Task 1.8: Airflow DAG**
```python
# File: airflow/dags/daily_pipeline.py
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.dbt.operators.dbt import DbtRunOperator

with DAG('daily_pipeline', schedule_interval='@daily') as dag:
    
    feature_engineering = SparkSubmitOperator(
        task_id='spark_features',
        application='spark_jobs/feature_engineering.py',
        conn_id='spark_k8s'
    )
    
    dbt_transform = DbtRunOperator(
        task_id='dbt_run',
        project_dir='/opt/airflow/dbt/',
        profiles_dir='/opt/airflow/dbt/',
        select='marts+'  # Run marts models
    )
    
    validate_data = PythonOperator(
        task_id='validate_features',
        python_callable=run_great_expectations
    )
    
    feature_engineering >> dbt_transform >> validate_data

# Gotcha: Airflow scheduler needs celery executor for parallel tasks
# Gotcha: Set DAG catchup=False to avoid backfilling on first deploy
```

---

### **2. Data Science Track (Week 9-12)**

#### **Week 9-10: Exploratory Data Analysis & Feature Engineering**

**Task 2.1: EDA Notebook**
```python
# File: notebooks/01_price_eda.ipynb
# Goal: Understand distributions, correlations, seasonality

import pandas as pd
import plotly.express as px
from scipy import stats

# Load data from S3 via DuckDB (fast!)
import duckdb
con = duckdb.connect()
df = con.execute("SELECT * FROM read_parquet('s3://sentinance/prices/*.parquet')").df()

# Analysis:
# 1. Price distribution (log-normal?)
# 2. Correlation matrix (BTC vs ETH = 0.87)
# 3. Seasonality (Prophet decomposition)
# 4. Outlier detection (Z-score > 3)

# Gotcha: Don't plot 10M points directly - downsample with .resample('1H')
# Gotcha: S3 reads are slow - cache to local Parquet with pyarrow
```

**Task 2.2: Advanced Feature Engineering**
```python
# File: ds/feature_engineering/graph_features.py
# Goal: Create network features for wallet analysis

import networkx as nx
from node2vec import Node2Vec

# Build transaction graph from Neo4j
G = nx.Graph()
# Add edges: (wallet_a, wallet_b, weight=transaction_amount)

# Compute centrality features
df['pagerank'] = nx.pagerank(G)
df['betweenness'] = nx.betweenness_centrality(G)
df['clustering_coef'] = nx.clustering(G)

# Node2Vec embeddings (128-dim)
node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=200)
model = node2vec.fit()
embeddings = model.wv

# Gotcha: NetworkX is slow for graphs >1M nodes - use graph-tool or igraph
# Gotcha: Node2Vec training takes hours - cache embeddings
```

**Task 2.3: Data Quality Validation**
```python
# File: ds/validation/great_expectations_suite.py
# Technology: Great Expectations
# Goal: Automated data quality checks

import great_expectations as gx

context = gx.get_context()

# Define expectations
suite = context.add_expectation_suite("price_data_quality")

# Expectations:
context.add_expectation(
    suite_name="price_data_quality",
    expectation_type="expect_column_values_to_be_between",
    kwargs={"column": "price", "min_value": 0, "max_value": 1000000}
)
context.add_expectation(
    suite_name="price_data_quality",
    expectation_type="expect_column_values_to_not_be_null",
    kwargs={"column": "timestamp"}
)

# Run validation
results = context.run_checkpoint("daily_validation")

# Gotcha: Great Expectations generates huge HTML reports - disable in prod
# Gotcha: Failed expectations should fail the Airflow DAG - set raise_on_error=True
```

#### **Week 11-12: Model Experimentation**

**Task 2.4: Baseline Models**
```python
# File: ds/modeling/baseline.py
# Goal: Establish baseline performance

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

# Problem: Predict if price will increase >5% in next 24h (binary classification)

X_train, y_train = load_features('train')
X_test, y_test = load_features('test')

# Baseline 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print(f"Baseline RF: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Baseline 2: XGBoost
import xgboost as xgb
xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Log to MLflow
import mlflow
with mlflow.start_run():
    mlflow.log_params({"model": "xgboost", "max_depth": 5})
    mlflow.log_metrics({"f1": f1, "precision": precision})
    mlflow.sklearn.log_model(xgb_model, "model")

# Gotcha: Class imbalance (10% positive) - use SMOTE or class_weight='balanced'
```

**Task 2.5: Hyperparameter Tuning**
```python
# File: ds/modeling/hyperparameter_tuning.py
# Technology: Ray Tune + Optuna
# Goal: Find optimal hyperparameters

from ray import tune
from ray.tune.search.optuna import OptunaSearch

def train_fn(config):
    model = xgb.XGBClassifier(**config)
    model.fit(X_train, y_train)
    score = f1_score(y_test, model.predict(X_test))
    return {"f1": score}

search_space = {
    "max_depth": tune.randint(3, 10),
    "learning_rate": tune.loguniform(0.001, 0.1),
    "n_estimators": tune.randint(50, 300)
}

tuner = tune.Tuner(
    train_fn,
    tune_config=tune.TuneConfig(
        metric="f1",
        mode="max",
        search_alg=OptunaSearch(),
        num_samples=100  # 100 trials
    ),
    param_space=search_space
)

results = tuner.fit()
best_config = results.get_best_result().config

# Gotcha: Ray Tune uses a lot of memory - limit concurrent trials with max_concurrent
# Gotcha: Save checkpoints to S3 to avoid losing progress
```

---

### **3. ML Engineering Track (Week 13-18)**

#### **Week 13-14: Advanced Model Development**

**Task 3.1: Fine-Tune LLM for Sentiment**
```python
# File: ml/training/finetune_llm.py
# Technology: HuggingFace Transformers + PEFT (LoRA)
# Goal: Fine-tune Mistral-7B for crypto sentiment classification

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    num_labels=3,  # Bullish, Bearish, Neutral
    load_in_8bit=True  # Quantization for memory efficiency
)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Only adapt attention layers
)
model = get_peft_model(model, lora_config)

# Load training data (10K labeled crypto news articles)
dataset = load_dataset("csv", data_files={"train": "s3://sentinance/labeled_news.csv"})

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral-sentiment",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True  # Mixed precision
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)
trainer.train()

# Save to MLflow
import mlflow
mlflow.transformers.log_model(model, "mistral-sentiment-lora")

# Gotcha: Mistral-7B needs 24GB VRAM - use 8-bit quantization + gradient checkpointing
# Gotcha: LoRA adapters are only 10MB - store base model separately
```

**Task 3.2: Graph Neural Network for Wallet Risk**
```python
# File: ml/training/train_gnn.py
# Technology: PyTorch Geometric + Ray
# Goal: Train GAT model to predict wallet risk score

import torch
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader

class WalletGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Load graph from Neo4j
from torch_geometric.data import Data
# x: Node features (wallet balance, transaction count, etc.)
# edge_index: Transaction graph edges
# y: Risk labels (0=safe, 1=risky)

data = Data(x=features, edge_index=edges, y=labels)
loader = DataLoader([data], batch_size=1)

# Training loop
model = WalletGNN(in_channels=128, hidden_channels=256, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(200):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

# Gotcha: GAT requires full graph in memory - use NeighborLoader for large graphs
# Gotcha: Graph data augmentation (edge dropout) prevents overfitting
```

**Task 3.3: LSTM Ensemble for Price Forecasting**
```python
# File: ml/training/train_lstm.py
# Goal: Ensemble of LSTMs for 1h, 4h, 24h price predictions

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class PriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 3)  # Predict [1h, 4h, 24h] ahead
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Last timestep

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = create_sequences(price_data)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32)

# Train 5 models with different seeds (ensemble)
models = [PriceLSTM(input_size=10, hidden_size=128, num_layers=2) for _ in range(5)]

for model in models:
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = nn.MSELoss()(pred, y_batch)
            loss.backward()
            optimizer.step()

# Ensemble prediction: Average of 5 models
def predict_ensemble(X):
    preds = [model(X).detach() for model in models]
    return torch.mean(torch.stack(preds), dim=0)

# Gotcha: LSTM forgets long-term dependencies - add attention mechanism
# Gotcha: Normalize prices (MinMaxScaler) - neural networks don't handle large values well
```

#### **Week 15-16: Model Deployment & Serving**

**Task 3.4: Package Models with TorchServe**
```bash
# File: ml/serving/package_model.sh
# Goal: Create .mar file for TorchServe deployment

# Create model archive for GNN
torch-model-archiver \
  --model-name wallet_gnn \
  --version 1.0 \
  --model-file ml/models/gnn.py \
  --serialized-file ml/checkpoints/gnn_best.pth \
  --handler ml/serving/gnn_handler.py \
  --export-path ml/serving/model-store/

# Custom handler for preprocessing
# File: ml/serving/gnn_handler.py
class GNNHandler(BaseHandler):
    def preprocess(self, data):
        # Convert JSON request to PyG Data object
        wallet_address = data[0]['body']['wallet']
        # Fetch subgraph from Neo4j
        # Convert to torch_geometric.data.Data
        return graph_data
    
    def inference(self, data):
        with torch.no_grad():
            return self.model(data.x, data.edge_index)
    
    def postprocess(self, inference_output):
        risk_score = torch.softmax(inference_output, dim=1)[0][1].item()
        return [{"risk_score": risk_score}]

# Gotcha: TorchServe default timeout is 120s - increase for graph queries
# Gotcha: Pre-load Neo4j connection pool to avoid cold starts
```

**Task 3.5: Deploy to Kubernetes**
```yaml
# File: k8s/torchserve-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve-gnn
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: torchserve
        image: pytorch/torchserve:0.9.0-gpu
        ports:
        - containerPort: 8080  # Inference API
        - containerPort: 8081  # Management API
        volumeMounts:
        - name: model-store
          mountPath: /home/model-server/model-store
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
        env:
        - name: TS_CONFIG_FILE
          value: /home/model-server/config.properties
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: model-store-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: torchserve-svc
spec:
  selector:
    app: torchserve-gnn
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer

# Gotcha: GPU nodes are expensive - use node affinity to schedule on spot instances
# Gotcha: Add liveness/readiness probes to prevent serving stale models
```

**Task 3.6: Model Versioning & Registry**
```python
# File: ml/mlops/model_registry.py
# Technology: MLflow Model Registry
# Goal: Version models, promote to production

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = "runs:/<run_id>/model"
mlflow.register_model(model_uri, "wallet_gnn")

# Transition to production
client.transition_model_version_stage(
    name="wallet_gnn",
    version=3,
    stage="Production",
    archive_existing_versions=True
)

# Load production model in serving code
model = mlflow.pytorch.load_model("models:/wallet_gnn/Production")

# Gotcha: MLflow uses S3 for storage - set MLFLOW_S3_ENDPOINT_URL
# Gotcha: Model versioning breaks if run_id is reused - use unique experiment names
```

#### **Week 17-18: MLOps - Monitoring & Retraining**

**Task 3.7: Model Monitoring with Evidently**
```python
# File: ml/monitoring/drift_detection.py
# Technology: Evidently AI
# Goal: Detect data drift, trigger retraining

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

# Load production inference logs
production_data = pd.read_parquet("s3://sentinance/logs/inference/2024-01-15.parquet")
reference_data = pd.read_parquet("s3://sentinance/training/features.parquet")

# Create report
report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=reference_data, current_data=production_data)

# Check for drift
drift_score = report.as_dict()['metrics'][0]['result']['drift_score']
if drift_score > 0.3:
    # Trigger retraining DAG
    from airflow.api.common.trigger_dag import trigger_dag
    trigger_dag("model_retraining", run_id=f"drift_detected_{datetime.now()}")

# Gotcha: Evidently reports are huge HTML files - store to S3, don't email
# Gotcha: Drift detection on high-cardinality features (wallet addresses) is noisy - use embeddings
```

**Task 3.8: A/B Testing Framework**
```python
# File: ml/experimentation/ab_test.py
# Goal: Test new model (challenger) vs production (champion)

from scipy import stats

class ABTestFramework:
    def __init__(self, champion_model, challenger_model):
        self.champion = champion_model
        self.challenger = challenger_model
        self.champion_results = []
        self.challenger_results = []
    
    def route_traffic(self, request):
        # 90% traffic to champion, 10% to challenger
        if random.random() < 0.9:
            result = self.champion.predict(request)
            self.champion_results.append(result['accuracy'])
            return result
        else:
            result = self.challenger.predict(request)
            self.challenger_results.append(result['accuracy'])
            return result
    
    def analyze(self):
        # After 10K requests, run t-test
        if len(self.challenger_results) < 1000:
            return "Insufficient data"
        
        t_stat, p_value = stats.ttest_ind(self.champion_results, self.challenger_results)
        if p_value < 0.05 and mean(self.challenger_results) > mean(self.champion_results):
            return "Challenger wins! Promote to production."
        else:
            return "Champion still better."

# Gotcha: Don't test during market volatility - results will be skewed
# Gotcha: Log every prediction with model version for rollback
```

**Task 3.9: Automated Retraining Pipeline**
```python
# File: airflow/dags/model_retraining.py
from airflow import DAG
from airflow.operators.python import PythonOperator

def retrain_gnn():
    # Fetch latest data from feature store
    # Train GNN with new data
    # Evaluate on hold-out test set
    # If performance > current production model:
    #   - Register to MLflow
    #   - Update TorchServe model store
    #   - Reload TorchServe (POST /models/{model}/reload)

with DAG('model_retraining', schedule_interval='@weekly') as dag:
    retrain = PythonOperator(task_id='retrain_gnn', python_callable=retrain_gnn)
    evaluate = PythonOperator(task_id='evaluate', python_callable=evaluate_model)
    deploy = PythonOperator(task_id='deploy', python_callable=deploy_if_better)
    
    retrain >> evaluate >> deploy

# Gotcha: Retraining can degrade performance - always compare to baseline
# Gotcha: Store training data hash in MLflow to reproduce experiments
```

---

### **4. Full-Stack Engineering Track (Week 19-24)**

#### **Week 19-20: Backend API Development**

**Task 4.1: FastAPI with CQRS**
```python
# File: backend/api/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Sentinance API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sentinance.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Command Model (Writes)
class CreateAlertCommand(BaseModel):
    user_id: str
    symbol: str
    condition: str  # "price > 50000"
    
@app.post("/commands/alerts")
async def create_alert(cmd: CreateAlertCommand, user=Depends(get_current_user)):
    # Publish to Kafka
    producer.send('commands.create_alert', value=cmd.dict())
    return {"status": "pending", "correlation_id": str(uuid4())}

# Query Model (Reads)
@app.get("/queries/prices/{symbol}")
async def get_price(symbol: str, cache=Depends(get_redis)):
    # Check cache first
    cached = cache.get(f"price:{symbol}")
    if cached:
        return json.loads(cached)
    
    # Query PostgreSQL read replica
    result = await db.fetch_one("SELECT * FROM prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 1", symbol)
    
    # Cache for 10 seconds
    cache.setex(f"price:{symbol}", 10, json.dumps(dict(result)))
    return result

# Gotcha: CORS wildcard (*) is insecure - whitelist specific domains
# Gotcha: Use Pydantic v2 for 2x faster validation
```

**Task 4.2: Authentication with JWT**
```python
# File: backend/auth/jwt_handler.py
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Refresh token pattern
@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    # Validate refresh token (stored in Redis with expiry)
    user_id = redis.get(f"refresh:{refresh_token}")
    if not user_id:
        raise HTTPException(401, "Invalid refresh token")
    
    new_access = create_access_token({"sub": user_id})
    return {"access_token": new_access}

# Gotcha: Store JWT secret in AWS Secrets Manager, not .env
# Gotcha: Refresh tokens must be revocable - store in Redis with TTL
```

**Task 4.3: WebSocket for Real-Time Streaming**
```python
# File: backend/websockets/price_stream.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/prices")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Subscribe to Kafka topic and stream to WebSocket
        consumer = KafkaConsumer('prices', bootstrap_servers='kafka:9092')
        for message in consumer:
            price_event = json.loads(message.value)
            await websocket.send_json(price_event)
    except WebSocketDisconnect:
        manager.active_connections.remove(websocket)

# Gotcha: WebSocket connections timeout after 10 minutes - send ping frames
# Gotcha: Don't broadcast to disconnected clients - check is_connected
```

**Task 4.4: Rate Limiting Middleware**
```python
# File: backend/middleware/rate_limiter.py
from fastapi import Request, HTTPException
from redis import Redis
import time

redis = Redis(host='redis', port=6379)

async def rate_limit_middleware(request: Request, call_next):
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return await call_next(request)
    
    # Sliding window: 100 requests per minute
    key = f"rate_limit:{user_id}"
    current = time.time()
    
    # Remove old requests (>60s ago)
    redis.zremrangebyscore(key, 0, current - 60)
    
    # Count requests in last 60s
    request_count = redis.zcard(key)
    if request_count >= 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Add current request
    redis.zadd(key, {current: current})
    redis.expire(key, 60)
    
    return await call_next(request)

app.middleware("http")(rate_limit_middleware)

# Gotcha: Use sorted sets (ZADD) for sliding window, not counters
# Gotcha: Set Redis key expiry to avoid memory leak
```

#### **Week 21-22: Frontend Development**

**Task 4.5: Next.js 15 with App Router**
```typescript
// File: web/app/dashboard/page.tsx
import { Suspense } from 'react';
import PriceChart from '@/components/PriceChart';
import { getPrices } from '@/lib/api';

export default async function DashboardPage() {
  // Server-side data fetching
  const prices = await getPrices('BTC');
  
  return (
    <div className="grid grid-cols-2 gap-4">
      <Suspense fallback={<ChartSkeleton />}>
        <PriceChart data={prices} />
      </Suspense>
      
      <Suspense fallback={<div>Loading...</div>}>
        <NewsPanel />
      </Suspense>
    </div>
  );
}

// Gotcha: Suspense boundaries prevent waterfall requests
// Gotcha: Use React.cache() to deduplicate API calls
```

**Task 4.6: Generative UI with Vercel AI SDK**
```typescript
// File: web/app/chat/page.tsx
'use client';

import { useChat } from 'ai/react';

export default function ChatPage() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: '/api/chat',
  });
  
  return (
    <div>
      {messages.map((msg) => (
        <div key={msg.id}>
          {msg.role === 'assistant' && msg.toolInvocations?.map((tool) => {
            // Render React components based on tool
            if (tool.toolName === 'show_price_chart') {
              return <PriceChart data={tool.result} />;
            }
            if (tool.toolName === 'show_risk_gauge') {
              return <RiskGauge score={tool.result.risk_score} />;
            }
          })}
          <p>{msg.content}</p>
        </div>
      ))}
      
      <form onSubmit={handleSubmit}>
        <input value={input} onChange={handleInputChange} />
      </form>
    </div>
  );
}

// Backend: File: web/app/api/chat/route.ts
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

export async function POST(req: Request) {
  const { messages } = await req.json();
  
  const result = await streamText({
    model: openai('gpt-4'),
    messages,
    tools: {
      show_price_chart: {
        description: 'Show a price chart for a symbol',
        parameters: z.object({ symbol: z.string() }),
        execute: async ({ symbol }) => {
          const prices = await fetch(`/api/prices/${symbol}`).then(r => r.json());
          return prices;
        },
      },
      show_risk_gauge: {
        description: 'Show wallet risk gauge',
        parameters: z.object({ wallet: z.string() }),
        execute: async ({ wallet }) => {
          const risk = await fetch(`/api/risk/${wallet}`).then(r => r.json());
          return risk;
        },
      },
    },
  });
  
  return result.toAIStreamResponse();
}

// Gotcha: Tool execution happens server-side - secure with auth
// Gotcha: Stream tokens to client for better UX (avoid blank screen)
```

**Task 4.7: State Management with Zustand**
```typescript
// File: web/lib/store.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface DashboardState {
  selectedSymbol: string;
  timeframe: '1h' | '24h' | '7d';
  setSymbol: (symbol: string) => void;
  setTimeframe: (tf: '1h' | '24h' | '7d') => void;
}

export const useDashboardStore = create<DashboardState>()(
  persist(
    (set) => ({
      selectedSymbol: 'BTC',
      timeframe: '24h',
      setSymbol: (symbol) => set({ selectedSymbol: symbol }),
      setTimeframe: (tf) => set({ timeframe: tf }),
    }),
    {
      name: 'dashboard-storage',
    }
  )
);

// Usage:
const Dashboard = () => {
  const { selectedSymbol, setSymbol } = useDashboardStore();
  // ...
}

// Gotcha: Persist middleware uses localStorage - check for SSR
// Gotcha: Zustand is simpler than Redux - use for most apps
```

**Task 4.8: Real-Time Updates with WebSocket**
```typescript
// File: web/hooks/useWebSocket.ts
import { useEffect, useState } from 'react';

export function useWebSocket(url: string) {
  const [data, setData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => setData(JSON.parse(event.data));
    ws.onerror = () => setIsConnected(false);
    ws.onclose = () => {
      setIsConnected(false);
      // Reconnect after 3 seconds
      setTimeout(() => {
        useWebSocket(url);
      }, 3000);
    };
    
    return () => ws.close();
  }, [url]);
  
  return { data, isConnected };
}

// Usage:
const PriceTicker = () => {
  const { data, isConnected } = useWebSocket('wss://api.sentinance.com/ws/prices');
  return <div>{data?.price} {isConnected ? '🟢' : '🔴'}</div>;
}

// Gotcha: WebSocket reconnection causes memory leaks - cleanup in useEffect
// Gotcha: Show connection status to user - red dot for disconnected
```

#### **Week 23-24: DevOps & Production**

**Task 4.9: Docker Multi-Stage Builds**
```dockerfile
# File: backend/Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/dependencies -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/dependencies /app/dependencies
COPY . .

ENV PYTHONPATH=/app/dependencies
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Gotcha: Multi-stage builds reduce image size by 60%
# Gotcha: Use .dockerignore to exclude node_modules, __pycache__
```

**Task 4.10: CI/CD with GitHub Actions**
```yaml
# File: .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: |
          pip install pytest
          pytest tests/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t sentinance-api:${{ github.sha }} .
      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker push sentinance-api:${{ github.sha }}
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Update Kubernetes
        run: |
          kubectl set image deployment/api api=sentinance-api:${{ github.sha }}
          kubectl rollout status deployment/api

# Gotcha: Use GitHub secrets for AWS credentials, not hardcoded
# Gotcha: Add rollback step if deployment fails
```

**Task 4.11: Observability Stack**
```yaml
# File: k8s/observability/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'fastapi'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: fastapi
            action: keep

---
# Grafana dashboard for API latency
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard
data:
  api-latency.json: |
    {
      "panels": [
        {
          "title": "API Latency p95",
          "targets": [
            {
              "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket{job='fastapi'})"
            }
          ]
        }
      ]
    }

# Gotcha: Scrape interval too low (1s) overloads Prometheus
# Gotcha: Use recording rules for expensive queries
```

---

## **V. Small Technical Details & "Gotchas"**

### **Critical Implementation Details**

#### **Data Engineering**

1. **Kafka Idempotency**
   ```python
   # Use deterministic message keys
   key = f"{symbol}:{exchange}:{timestamp_ms}"
   producer.send('prices', key=key.encode(), value=price_event)
   
   # Gotcha: If timestamp precision is seconds, not milliseconds, you get duplicates
   ```

2. **Schema Registry Compatibility**
   ```bash
   # Always test schema compatibility before deploying
   curl -X POST http://schema-registry:8081/compatibility/subjects/prices-value/versions/latest \
     -H "Content-Type: application/vnd.schemaregistry.v1+json" \
     -d '{"schema": "..."}'
   
   # Gotcha: BACKWARD compatibility allows adding optional fields, FORWARD allows removing
   ```

3. **Spark Shuffle Partitions**
   ```python
   # Default 200 partitions is too high for small datasets
   spark.conf.set("spark.sql.shuffle.partitions", "20")
   
   # Gotcha: Set based on data size: 1 partition per 128MB
   ```

4. **dbt Incremental Models**
   ```sql
   -- Handle late-arriving data
   {% if is_incremental() %}
     WHERE timestamp >= (SELECT MAX(timestamp) FROM {{ this }}) - INTERVAL '1 hour'
   {% endif %}
   
   -- Gotcha: Without lookback window, late data is missed
   ```

5. **Airflow Connection Pools**
   ```python
   # Increase pool size for parallel tasks
   AIRFLOW__CORE__SQL_ALCHEMY_POOL_SIZE = 20
   
   # Gotcha: Default pool size (5) causes "too many connections" errors
   ```

#### **Data Science & ML**

6. **Feature Leakage Prevention**
   ```python
   # WRONG: Fit scaler on entire dataset
   scaler = StandardScaler().fit(X)  # Leaks test data!
   
   # RIGHT: Fit only on training data
   scaler = StandardScaler().fit(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # Gotcha: Leakage inflates performance by 10-20%
   ```

7. **Class Imbalance Handling**
   ```python
   # For 1% positive class, use class weights
   class_weight = {0: 1, 1: 99}
   model = XGBClassifier(scale_pos_weight=99)  # 99:1 ratio
   
   # Gotcha: SMOTE oversampling can introduce unrealistic examples
   ```

8. **Cross-Validation for Time-Series**
   ```python
   # WRONG: Random KFold shuffles time order
   from sklearn.model_selection import TimeSeriesSplit
   
   tscv = TimeSeriesSplit(n_splits=5)
   for train_idx, test_idx in tscv.split(X):
       # Train on past, test on future
   
   # Gotcha: Random CV allows future data to leak into training
   ```

9. **GPU Memory Management**
   ```python
   # PyTorch doesn't free GPU memory automatically
   del model
   torch.cuda.empty_cache()
   
   # Gotcha: Running multiple experiments crashes due to OOM
   ```

10. **LoRA Adapter Merging**
    ```python
    # Merge LoRA back into base model for deployment
    from peft import PeftModel
    
    base_model = AutoModel.from_pretrained("mistral-7b")
    model = PeftModel.from_pretrained(base_model, "lora_adapter")
    merged_model = model.merge_and_unload()
    
    # Gotcha: Without merging, inference requires PEFT library
    ```

#### **MLOps**

11. **MLflow Model URI Formats**
    ```python
    # Different URI formats for different sources
    "runs:/<run_id>/model"  # Specific run
    "models:/model_name/Production"  # Registry stage
    "s3://bucket/path/model"  # Direct S3
    
    # Gotcha: Wrong URI format causes cryptic "model not found" errors
    ```

12. **TorchServe Batch Inference**
    ```python
    # config.properties
    batch_size=8
    max_batch_delay=50  # milliseconds
    
    # Gotcha: Without batching, throughput is 10x lower
    ```

13. **Model Drift Detection Cadence**
    ```python
    # Run drift detection DAILY, not after every prediction
    # Cost: $0.01 per run vs $1000/day for all predictions
    
    # Gotcha: Too frequent drift checks waste money
    ```

14. **A/B Test Statistical Power**
    ```python
    # Need at least 1000 samples per variant for p-value < 0.05
    from scipy.stats import power
    
    required_n = power.tt_ind_solve_power(
        effect_size=0.2,  # Small effect
        alpha=0.05,
        power=0.8,
        ratio=1.0
    )
    # Result: ~788 samples per group
    
    # Gotcha: Testing with 100 samples gives false positives
    ```

#### **Backend**

15. **Database Connection Pooling**
    ```python
    # SQLAlchemy connection pool
    engine = create_engine(
        "postgresql://...",
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True  # Check connection before use
    )
    
    # Gotcha: Without pool_pre_ping, stale connections cause errors
    ```

16. **JWT Secret Rotation**
    ```python
    # Support multiple secrets for zero-downtime rotation
    SECRETS = [os.getenv("JWT_SECRET"), os.getenv("JWT_SECRET_OLD")]
    
    def verify_token(token):
        for secret in SECRETS:
            try:
                return jwt.decode(token, secret, algorithms=["HS256"])
            except JWTError:
                continue
        raise HTTPException(401)
    
    # Gotcha: Single secret rotation invalidates all tokens immediately
    ```

17. **CORS Preflight Caching**
    ```python
    # Cache preflight responses for 1 hour
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://app.sentinance.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600  # This line!
    )
    
    # Gotcha: Without max_age, browser sends OPTIONS request for every API call
    ```

18. **WebSocket Ping/Pong**
    ```python
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        
        async def ping_task():
            while True:
                await asyncio.sleep(30)
                await websocket.send_text("ping")
        
        asyncio.create_task(ping_task())
    
    # Gotcha: Idle WebSockets timeout after 60s - send pings
    ```

#### **Frontend**

19. **Next.js Dynamic Imports**
    ```typescript
    // Lazy load heavy components
    const PriceChart = dynamic(() => import('@/components/PriceChart'), {
      loading: () => <Skeleton />,
      ssr: false  // Disable SSR for client-only code
    });
    
    // Gotcha: Without dynamic import, bundle size is 2MB
    ```

20. **React Query Stale Time**
    ```typescript
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          staleTime: 5 * 60 * 1000,  // 5 minutes
          cacheTime: 10 * 60 * 1000,  // 10 minutes
        },
      },
    });
    
    // Gotcha: staleTime=0 (default) refetches on every mount
    ```

21. **Framer Motion Performance**
    ```typescript
    // Use layout animations sparingly
    <motion.div layout layoutId="card">  // Triggers reflow
    
    // Gotcha: layout animations on many elements cause 30fps drops
    ```

22. **TypeScript Strict Mode**
    ```json
    {
      "compilerOptions": {
        "strict": true,
        "noUncheckedIndexedAccess": true,  // This catches array bugs!
        "exactOptionalPropertyTypes": true
      }
    }
    
    // Gotcha: Without noUncheckedIndexedAccess, array[i] is assumed defined
    ```

#### **DevOps**

23. **Kubernetes Resource Limits**
    ```yaml
    resources:
      requests:
        cpu: "500m"
        memory: "512Mi"
      limits:
        cpu: "1000m"  # 2x request
        memory: "1Gi"  # 2x request
    
    # Gotcha: No limits = pods consume all node resources
    # Gotcha: limits < requests causes eviction
    ```

24. **Helm Chart Secrets**
    ```yaml
    # values.yaml
    secretRef:
      name: sentinance-secrets  # External secret, not in chart
    
    # Gotcha: NEVER put secrets in values.yaml - use external-secrets-operator
    ```

25. **Docker Layer Caching**
    ```dockerfile
    # Put least-changed files first
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    # Put most-changed files last
    COPY . .
    
    # Gotcha: COPY . . at the top invalidates all layers
    ```

26. **Liveness vs Readiness Probes**
    ```yaml
    livenessProbe:
      httpGet:
        path: /health/live  # Is process alive?
      initialDelaySeconds: 30
    
    readinessProbe:
      httpGet:
        path: /health/ready  # Is it ready for traffic?
      initialDelaySeconds: 5
      periodSeconds: 5
    
    # Gotcha: liveness=readiness causes restart loops during warm-up
    ```

27. **Pre-commit Hooks**
    ```yaml
    # .pre-commit-config.yaml
    repos:
      - repo: https://github.com/psf/black
        rev: 23.11.0
        hooks:
          - id: black
      - repo: https://github.com/pycqa/flake8
        rev: 6.1.0
        hooks:
          - id: flake8
      - repo: https://github.com/pre-commit/mirrors-mypy
        rev: v1.7.0
        hooks:
          - id: mypy
    
    # Run: pre-commit install
    # Gotcha: Without pre-commit, linting errors reach production
    ```

28. **Environment Variable Injection**
    ```bash
    # WRONG: Hardcoded in Dockerfile
    ENV DATABASE_URL=postgresql://...
    
    # RIGHT: Injected at runtime
    docker run -e DATABASE_URL=$DATABASE_URL ...
    
    # Kubernetes: Use ConfigMaps + Secrets
    envFrom:
      - configMapRef:
          name: app-config
      - secretRef:
          name: app-secrets
    
    # Gotcha: Hardcoded secrets leak to Docker Hub
    ```

29. **Log Structured Logging**
    ```python
    import structlog
    
    log = structlog.get_logger()
    log.info("user_login", user_id="123", ip="1.2.3.4")
    
    # Output: {"event": "user_login", "user_id": "123", "ip": "1.2.3.4", "timestamp": "..."}
    
    # Gotcha: String logs are hard to query in CloudWatch/Loki
    ```

30. **Rate Limit Headers**
    ```python
    @app.middleware("http")
    async def add_rate_limit_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = "100"
        response.headers["X-RateLimit-Remaining"] = str(100 - request_count)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        return response
    
    # Gotcha: Without headers, users don't know when to retry
    ```

---

## **Summary: Integration Points**

**How the tracks integrate:**

1. **DE → DS**: Spark feature engineering → Feature store → DS notebooks
2. **DS → MLE**: Jupyter experiments → MLflow registry → TorchServe deployment
3. **MLE → SE**: TorchServe API → FastAPI backend → Frontend UI
4. **SE → DE**: User actions → Kafka events → Stream processing
5. **All → Observability**: OTEL spans → Prometheus metrics → Grafana dashboards

**The result**: A single, cohesive system where data flows from raw ingestion → transformed features → trained models → user-facing predictions, with every component integrated and observable.

---

**Ready to build Project 1: Sentinance?**

This specification is comprehensive, production-grade, and ready for implementation. All 4 tracks integrate into ONE cohesive system.

**Next**: Would you like me to generate the same level of detail for:
- **Project 2: GraphGuard** (Graph-based fraud detection)
- **Project 3: FleetFlow** (Real-time delivery optimization)

Or start implementing Sentinance Phase 1 right now?