# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from main import app  # Import your FastAPI app

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client
