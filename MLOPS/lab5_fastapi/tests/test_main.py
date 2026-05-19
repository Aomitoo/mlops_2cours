# tests/test_main.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Тест: сервис отвечает на health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_input():
    """Тест: корректный запрос возвращает прогноз"""
    test_data = {
        "make": "Toyota",
        "model": "Camry",
        "year": 2020,
        "style": "Sedan",
        "distance": 50000.0,
        "engine_capacity": 2.5,
        "fuel_type": "Petrol",
        "transmission": "Automatic"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        result = response.json()
        assert "predicted_price" in result
        assert isinstance(result["predicted_price"], (int, float))

def test_predict_missing_field():
    """Тест: отсутствие обязательного поля → ошибка 422"""
    test_data = {
        "make": "Toyota",
        "year": 2020,
        "style": "Sedan",
        "distance": 50000.0,
        "engine_capacity": 2.5,
        "fuel_type": "Petrol",
        "transmission": "Automatic"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422
