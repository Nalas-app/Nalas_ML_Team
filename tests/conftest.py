import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_service'))

from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c