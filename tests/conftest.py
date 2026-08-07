import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app, get_db
from api.database import Base 

# Define an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

# Create the SQLAlchemy engine for the test database 
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=StaticPool)

# Create a session factory for generating database sessions
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function", autouse=True)
def setup_db():
    ''' Create a fresh database schema for every test '''
    
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
 
@pytest.fixture
def client():
    ''' Override database dependency and yield TestClient '''
    
    # Creates a new session using the test database
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Replace the original get_db dependency with the test version
    app.dependency_overrides[get_db] = override_get_db
    
    # Prevent SQLAlchemy from creating database tables when the app starts
    with patch("api.main.Base.metadata.create_all"):
        with TestClient(app) as test_client:
            yield test_client
            
    app.dependency_overrides.clear()