"""Verify FastAPI dependency wiring and model schema survive P0 lint repairs."""

from fastapi.testclient import TestClient

from model_router import QueryClassifier, QueryComplexity
from structured_output import MarketAnalysis


def test_anonymous_user_cannot_list_others_alerts(client: TestClient):
    response = client.get("/api/alerts")
    assert response.status_code == 401


def test_anonymous_user_cannot_delete_others_alerts(client: TestClient):
    response = client.delete("/api/alerts/a-nonexistent-alert")
    assert response.status_code == 401


def test_market_analysis_keeps_json_schema_example():
    schema = MarketAnalysis.model_json_schema()
    assert schema["example"]["symbol"] == "BTCUSDT"
    assert schema["example"]["confidence"] == 0.75


def test_query_classifier_keyword_contract_is_preserved():
    simple, _ = QueryClassifier.classify("What is bitcoin?")
    expert, _ = QueryClassifier.classify(
        "Explain arbitrage and derivatives in digital asset markets"
    )
    assert simple is QueryComplexity.SIMPLE
    assert expert is QueryComplexity.EXPERT
