import os
import time
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.database import engine, get_db
from api.models_db import Base, PredictionLog

from pipeline.audio_processor import extract_features, CLASS_MAPPING
from engine.layers import forward_propagation_with_dropout

# Try to get the base directory from a relative position
BASE_DIR = Path(__file__).resolve().parent.parent

# Model parameters and scaler paths
MODEL_PATH = BASE_DIR / "trained_model_params.npz"
SCALER_PATH = BASE_DIR / "artifacts" / "scaler.npz"

MODEL_VERSION = "v1.0.0"

INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Features and parameters initialization for the warn-up
parameters = {}
layers = []
X_mean = None
X_std = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    ''' On initialization load model and try a model warn-up before making it avaliable for use '''
    global parameters, layers, X_mean, X_std
    
    # Create Postgres table if not exists
    Base.metadata.create_all(bind=engine)
    
    # Check if the parameters path exists and load them
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model parameters not found at '{MODEL_PATH}'")
    
    data = np.load(MODEL_PATH)
    parameters = {key: data[key] for key in data.files}
    
    # Shape the number of layers according to the numbers of parameters in each layer
    num_layers = len([k for k in parameters.keys() if k.startswith("W")])
    layers = [parameters["W1"].shape[1]] + [parameters[f"W{l}"].shape[0] for l in range(1, num_layers + 1)]
    
    # Load Normalization Statistics
    if SCALER_PATH.exists():
        scaler = np.load(SCALER_PATH)
        X_mean = scaler["mean"]
        X_std = scaler["std"]
    else:
        print("'artifacts/scaler.npz' not found. Using raw unstandardized features")

    print("Executing API warn-up")
    try:
        # Warn-up Postgresql connection Pool
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Librosa dummy values for warn-up
        dummy_signal = np.zeros(22050, dtype=np.float32)
        dummy_mfcc = librosa.feature.mfcc(y=dummy_signal, sr=22050, n_mfcc=40)
        dummy_features = np.mean(dummy_mfcc.T, axis=0).reshape(-1, 1)

        # Dummy features for the Neural Network warn-up
        if X_mean is not None and X_std is not None:
            dummy_features = (dummy_features - X_mean) / X_std

        # Do one round of forward propagation for warn-up
        _ = forward_propagation_with_dropout(dummy_features, parameters, layers, keep_prob=1.0, is_training=False)

    except Exception as e:
        print(f"Error in warn-up: {e}")

    yield 

    # Clear the parameters
    parameters.clear()


app = FastAPI(
    title="Audio Sound Classifier API",
    description="Custom NumPy Deep Learning API for Audio Sound Classification",
    version="1.0",
    lifespan=lifespan
)


@app.get("/")
def health_check():
    ''' Simple response on trying to acess the main URL '''
    
    return {"status": "online", "model_layers": layers}


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ''' Accept an audio file, make the prediction and save the response '''
    
    # Start the counter for inference time
    start_time = time.perf_counter()
    
    # Valid file extensions
    valid_extensions = ('.wav', '.mp3', '.ogg', '.flac')
    
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format. Please upload one of: {valid_extensions}"
        )
    
    # Read the file content and stores it in a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Extraction and preprocessing of the features
        features = extract_features(tmp_path)
        features = features.reshape(-1, 1)

        if X_mean is not None and X_std is not None:
            features = (features - X_mean) / X_std

        # Make inference according to the loaded values
        probabilities, _ = forward_propagation_with_dropout(features, parameters, layers, keep_prob=1.0, is_training=False)

        # Stores the results in variables
        predicted_class_id = int(np.argmax(probabilities, axis=0)[0])
        predicted_label = INV_CLASS_MAPPING.get(predicted_class_id, "Unknown")
        confidence_val = float(probabilities[predicted_class_id, 0])
        
        # Calculates the confidence score of the model in the prediction
        confidence_scores = {
            INV_CLASS_MAPPING[i]: round(float(probabilities[i, 0]) * 100, 2) 
            for i in range(len(INV_CLASS_MAPPING))
        }

        # Latency calculation
        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)

        # Log the results and saves in the database
        log_entry = PredictionLog(
            original_name=file.filename,
            predicted_class=predicted_label,
            confidence=confidence_val,
            latency_ms=latency_ms,
            model_version=MODEL_VERSION
        )
        
        db.add(log_entry)
        db.commit()
        
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": f"{confidence_val * 100:.2f}%",
            "latency_ms": latency_ms,
            "probabilities_pct": confidence_scores
        })

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    finally:
        # Delete the temporary file created
        if os.path.exists(tmp_path):
            os.remove(tmp_path)