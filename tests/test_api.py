import io
import numpy as np
from unittest.mock import patch

def test_health_check(client) -> None:
    ''' Test the root endpoint to ensure the API is running '''
    
    response = client.get("/")
    assert response.status_code == 200
 
@patch("api.main.extract_features")
def test_audio_upload_success(mock_extract, client) -> None:
    ''' Test the API response while mocking ONLY the audio extraction '''
    
    # Mock the feature extraction function so it doesn't process real audio
    mock_extract.return_value = np.zeros(40)
    
    # Create an in-memory fake WAV file to simulate an uploaded audio file
    fake_audio_data = io.BytesIO(b"fake wav bytes")
    
    # Send a POST request with the fake audio file
    response = client.post("/predict", files={"file": ("test.wav", fake_audio_data, "audio/wav")})
        
    assert response.status_code == 200
    
    data = response.json()
    
    # Check that the expected fields are present in the response
    assert "prediction" in data
    assert "confidence" in data
    
    mock_extract.assert_called_once()