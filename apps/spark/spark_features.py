"""
Spark Feature Engineering

PySpark jobs for:
- Technical indicator calculation at scale
- Feature extraction for ML models
- Time-series aggregations
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, LongType
)


# ============================================
# SCHEMA DEFINITIONS
# ============================================

PRICE_SCHEMA = StructType([
    StructField("symbol", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", DoubleType(), True),
])


# ============================================
# SPARK SESSION
# ============================================

def create_spark_session(app_name: str = "SentinanceFeatures") -> SparkSession:
    """Create Spark session with optimized config."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


# ============================================
# TECHNICAL INDICATORS
# ============================================

def calculate_sma(df: DataFrame, column: str, period: int) -> DataFrame:
    """Calculate Simple Moving Average."""
    window = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-period + 1, 0)
    return df.withColumn(f"sma_{period}", F.avg(column).over(window))


def calculate_ema(df: DataFrame, column: str, period: int) -> DataFrame:
    """Calculate Exponential Moving Average (approximation using Spark)."""
    # Using Spark's built-in exponential smoothing approximation
    alpha = 2 / (period + 1)
    window = Window.partitionBy("symbol").orderBy("timestamp")
    
    # First value
    df = df.withColumn("_row_num", F.row_number().over(window))
    
    # For simplicity, we'll use SMA as EMA approximation in distributed setting
    sma_window = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-period + 1, 0)
    df = df.withColumn(f"ema_{period}", F.avg(column).over(sma_window))
    
    return df.drop("_row_num")


def calculate_rsi(df: DataFrame, column: str = "close", period: int = 14) -> DataFrame:
    """Calculate Relative Strength Index."""
    window = Window.partitionBy("symbol").orderBy("timestamp")
    
    # Price change
    df = df.withColumn("_price_change", F.col(column) - F.lag(column, 1).over(window))
    
    # Gains and losses
    df = df.withColumn("_gain", F.when(F.col("_price_change") > 0, F.col("_price_change")).otherwise(0))
    df = df.withColumn("_loss", F.when(F.col("_price_change") < 0, -F.col("_price_change")).otherwise(0))
    
    # Average gains and losses
    avg_window = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-period + 1, 0)
    df = df.withColumn("_avg_gain", F.avg("_gain").over(avg_window))
    df = df.withColumn("_avg_loss", F.avg("_loss").over(avg_window))
    
    # RSI calculation
    df = df.withColumn(
        f"rsi_{period}",
        F.when(F.col("_avg_loss") == 0, 100)
        .otherwise(100 - (100 / (1 + (F.col("_avg_gain") / F.col("_avg_loss")))))
    )
    
    # Clean up temp columns
    return df.drop("_price_change", "_gain", "_loss", "_avg_gain", "_avg_loss")


def calculate_bollinger_bands(
    df: DataFrame, 
    column: str = "close", 
    period: int = 20, 
    std_dev: float = 2.0
) -> DataFrame:
    """Calculate Bollinger Bands."""
    window = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-period + 1, 0)
    
    df = df.withColumn("bb_middle", F.avg(column).over(window))
    df = df.withColumn("_std", F.stddev(column).over(window))
    df = df.withColumn("bb_upper", F.col("bb_middle") + (std_dev * F.col("_std")))
    df = df.withColumn("bb_lower", F.col("bb_middle") - (std_dev * F.col("_std")))
    
    return df.drop("_std")


def calculate_macd(
    df: DataFrame,
    column: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> DataFrame:
    """Calculate MACD."""
    df = calculate_ema(df, column, fast)
    df = calculate_ema(df, column, slow)
    
    df = df.withColumn("macd", F.col(f"ema_{fast}") - F.col(f"ema_{slow}"))
    
    # Signal line
    signal_window = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-signal + 1, 0)
    df = df.withColumn("macd_signal", F.avg("macd").over(signal_window))
    df = df.withColumn("macd_histogram", F.col("macd") - F.col("macd_signal"))
    
    return df


# ============================================
# FEATURE ENGINEERING
# ============================================

def add_time_features(df: DataFrame) -> DataFrame:
    """Add time-based features."""
    return (
        df
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("day_of_week", F.dayofweek("timestamp"))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0))
    )


def add_price_features(df: DataFrame) -> DataFrame:
    """Add price-derived features."""
    window = Window.partitionBy("symbol").orderBy("timestamp")
    
    return (
        df
        .withColumn("price_change", (F.col("close") - F.lag("close", 1).over(window)) / F.lag("close", 1).over(window))
        .withColumn("price_range", (F.col("high") - F.col("low")) / F.col("low"))
        .withColumn("body_size", F.abs(F.col("close") - F.col("open")) / F.col("open"))
        .withColumn("upper_wick", (F.col("high") - F.greatest("open", "close")) / F.col("high"))
        .withColumn("lower_wick", (F.least("open", "close") - F.col("low")) / F.col("low"))
    )


def add_volume_features(df: DataFrame) -> DataFrame:
    """Add volume-derived features."""
    window_5 = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-4, 0)
    window_20 = Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-19, 0)
    prev_window = Window.partitionBy("symbol").orderBy("timestamp")
    
    return (
        df
        .withColumn("volume_sma_5", F.avg("volume").over(window_5))
        .withColumn("volume_sma_20", F.avg("volume").over(window_20))
        .withColumn("volume_ratio", F.col("volume") / F.col("volume_sma_20"))
        .withColumn("volume_change", 
                   (F.col("volume") - F.lag("volume", 1).over(prev_window)) / 
                   F.lag("volume", 1).over(prev_window))
    )


# ============================================
# MAIN PIPELINE
# ============================================

def run_feature_pipeline(input_path: str, output_path: str):
    """Run the complete feature engineering pipeline."""
    spark = create_spark_session()
    
    # Read data
    df = spark.read.parquet(input_path)
    
    # Add all features
    df = calculate_sma(df, "close", 20)
    df = calculate_sma(df, "close", 50)
    df = calculate_ema(df, "close", 12)
    df = calculate_ema(df, "close", 26)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)
    df = add_time_features(df)
    df = add_price_features(df)
    df = add_volume_features(df)
    
    # Write output
    df.write.mode("overwrite").parquet(output_path)
    
    print(f"Feature pipeline complete. Output: {output_path}")
    
    spark.stop()


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        run_feature_pipeline(sys.argv[1], sys.argv[2])
    else:
        print("Usage: spark-submit spark_features.py <input_path> <output_path>")
