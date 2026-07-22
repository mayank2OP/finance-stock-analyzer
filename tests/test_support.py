import uuid

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
import models  # noqa: F401 - registers all tables


test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
Base.metadata.create_all(bind=test_engine)


def _test_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


def isolated_client(app) -> TestClient:
    app.dependency_overrides[get_db] = _test_db
    return TestClient(app)


def create_auth_headers(client: TestClient) -> dict[str, str]:
    username = f"test_{uuid.uuid4().hex[:12]}"
    password = "correct-horse-42"
    registered = client.post("/auth/register", json={"username": username, "password": password})
    if registered.status_code != 201:
        raise AssertionError(registered.text)
    logged_in = client.post("/auth/token", data={"username": username, "password": password})
    if logged_in.status_code != 200:
        raise AssertionError(logged_in.text)
    return {"Authorization": f"Bearer {logged_in.json()['access_token']}"}
