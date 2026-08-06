from pydantic import BaseModel, ConfigDict
from uuid import UUID
from datetime import datetime

class PredictionResponse(BaseModel):
    ''' Schema for response '''
    
    id: UUID
    original_name: str
    predicted_class: str
    confidence: float
    latency_ms: float
    model_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)