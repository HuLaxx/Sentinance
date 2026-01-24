"""
Great Expectations Configuration

Data quality validation for price data.

Setup:
    great_expectations init
    
Run validation:
    great_expectations checkpoint run price_checkpoint
"""

import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration

def create_price_expectations():
    """Create expectation suite for price data."""
    
    context = gx.get_context()
    
    # Create expectation suite
    suite_name = "price_data_suite"
    suite = context.add_or_update_expectation_suite(
        expectation_suite_name=suite_name
    )
    
    # Define expectations
    expectations = [
        # Symbol should not be null
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "symbol"}
        ),
        # Price should be positive
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "price", "min_value": 0, "strict_min": True}
        ),
        # Volume should be non-negative
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "volume", "min_value": 0}
        ),
        # 24h change should be within reasonable bounds
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "change_24h", "min_value": -50, "max_value": 100}
        ),
        # Timestamp should not be null
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "timestamp"}
        ),
        # Symbol should be in valid set
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "symbol",
                "value_set": [
                    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", 
                    "XRPUSDT", "ADAUSDT", "DOGEUSDT"
                ]
            }
        ),
    ]
    
    for expectation in expectations:
        suite.add_expectation(expectation_configuration=expectation)
    
    context.save_expectation_suite(suite)
    print(f"Created expectation suite: {suite_name}")
    
    return suite


def run_validation(df):
    """Run validation on a dataframe."""
    context = gx.get_context()
    
    # Get datasource
    datasource = context.sources.add_pandas("price_source")
    data_asset = datasource.add_dataframe_asset("price_data")
    
    # Build batch request
    batch_request = data_asset.build_batch_request(dataframe=df)
    
    # Get checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name="price_checkpoint",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "price_data_suite"
            }
        ]
    )
    
    # Run validation
    results = checkpoint.run()
    
    return results


if __name__ == "__main__":
    create_price_expectations()
    print("Great Expectations suite created!")
