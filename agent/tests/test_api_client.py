import requests
import pytest

class TestConfig:
    SERVICE_URL = "http://localhost:4241/search"

@pytest.fixture
def api_client():
    return TestConfig.SERVICE_URL

def test_search_valid_query(api_client):
    query_fact = "What are the legal implications of contract breach?"
    payload = {
        "fact": query_fact,
        "top_k": 5
    }
    
    response = requests.post(api_client, json=payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) <= 5
    for item in results:
        assert 'fact_id' in item
        assert 'score' in item
        assert 'rank' in item

def test_search_invalid_query(api_client):
    query_fact = ""
    payload = {
        "fact": query_fact,
        "top_k": 5
    }
    
    response = requests.post(api_client, json=payload)
    assert response.status_code == 400  # Assuming the API returns 400 for bad requests

def test_search_service_unavailable(api_client):
    # Simulate service down by using an incorrect URL
    invalid_url = "http://localhost:4241/invalid_endpoint"
    payload = {
        "fact": "What is the law regarding theft?",
        "top_k": 5
    }
    
    response = requests.post(invalid_url, json=payload)
    assert response.status_code == 404  # Assuming the API returns 404 for not found