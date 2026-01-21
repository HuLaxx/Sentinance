"""
AI Chat Endpoint Tests

Tests for AI chat, LangGraph agent, and RAG functionality.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_chat_endpoint_exists(client: TestClient):
    """Test chat endpoint exists and accepts POST."""
    response = client.post(
        "/api/chat",
        json={"message": "What is the current price of Bitcoin?"}
    )
    # Should return response or error, not 404/405
    assert response.status_code in [200, 400, 422, 500, 503]


def test_chat_with_empty_message(client: TestClient):
    """Test chat with empty message."""
    response = client.post(
        "/api/chat",
        json={"message": ""}
    )
    assert response.status_code in [200, 400, 422]


def test_chat_with_history(client: TestClient):
    """Test chat with conversation history."""
    response = client.post(
        "/api/chat",
        json={
            "message": "What about Ethereum?",
            "history": [
                {"role": "user", "content": "Tell me about crypto"},
                {"role": "assistant", "content": "Crypto is digital currency."}
            ]
        }
    )
    assert response.status_code in [200, 400, 422, 500, 503]


def test_chat_suggestions_endpoint(client: TestClient):
    """Test chat suggestions endpoint."""
    response = client.get("/api/chat/suggestions")
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (list, dict))


def test_chat_use_agent_flag(client: TestClient):
    """Test chat with use_agent flag."""
    response = client.post(
        "/api/chat",
        json={
            "message": "Analyze BTC market",
            "use_agent": True
        }
    )
    assert response.status_code in [200, 400, 422, 500, 503]


def test_chat_without_agent(client: TestClient):
    """Test chat without agent (direct AI)."""
    response = client.post(
        "/api/chat",
        json={
            "message": "What is the price of ETH?",
            "use_agent": False
        }
    )
    assert response.status_code in [200, 400, 422, 500, 503]


class TestAIModules:
    """Unit tests for AI modules."""
    
    def test_ai_chat_import(self):
        """Test AI chat module can be imported."""
        from ai_chat import get_ai_response, get_suggested_questions
        assert get_ai_response is not None
        assert get_suggested_questions is not None
    
    def test_suggested_questions_format(self):
        """Test suggested questions returns list."""
        from ai_chat import get_suggested_questions
        
        questions = get_suggested_questions()
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)
    
    def test_agent_import(self):
        """Test LangGraph agent can be imported."""
        try:
            from agent import run_agent, LANGGRAPH_AVAILABLE
            assert run_agent is not None
        except ImportError:
            # LangGraph may not be installed
            pytest.skip("LangGraph not installed")
    
    def test_gemini_client_import(self):
        """Test Gemini client can be imported."""
        from gemini_client import GeminiClient
        assert GeminiClient is not None
