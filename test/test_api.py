
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["message"] == "Predictive Maintenance API is running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "product_type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 100,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["machine_failure_prediction"] in [0, 1]
    assert data["prediction_label"] in [
        "Failure predicted",
        "No failure predicted",
    ]


def test_invalid_product_type():
    payload = {
        "product_type": "X",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 100,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_temperature():
    payload = {
        "product_type": "M",
        "air_temperature_k": 100.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 100,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
