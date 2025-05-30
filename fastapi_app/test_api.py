from fastapi.testclient import TestClient
import pytest
from api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_input():
    test_input = {
        "age": 30,
        "gender": 1,
        "years_of_experience": 5,
        "work_location": 1,
        "hours_worked_per_week": 40,
        "number_of_virtual_meetings": 10,
        "work_life_balance_rating": 3,
        "stress_level": 3,
        "access_to_mental_health_resources": 1,
        "productivity_change": 0,
        "social_isolation_rating": 3,
        "satisfaction_with_remote_work": 4,
        "company_support_for_remote_work": 4,
        "physical_activity": 5,
        "sleep_quality": 4
    }
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_input():
    # Test with missing fields
    invalid_input = {
        "age": 30,
        "gender": 1
    }
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

def test_predict_out_of_range():
    # Test with out of range values
    invalid_input = {
        "age": 150,  # age > 100
        "gender": 1,
        "years_of_experience": 5,
        "work_location": 1,
        "hours_worked_per_week": 40,
        "number_of_virtual_meetings": 10,
        "work_life_balance_rating": 3,
        "stress_level": 3,
        "access_to_mental_health_resources": 1,
        "productivity_change": 0,
        "social_isolation_rating": 3,
        "satisfaction_with_remote_work": 4,
        "company_support_for_remote_work": 4,
        "physical_activity": 5,
        "sleep_quality": 4
    }
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__])