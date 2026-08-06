import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from api.database import Base

class PredictionLog(Base):
    ''' The database ORM definition in sqlalchemy '''
    
    __tablename__ = "prediction_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_name = Column(String(255), nullable=False)
    predicted_class = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    model_version = Column(String(50), default="v1.0.0", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))