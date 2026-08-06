import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Gets the base directory based on relative path
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Loads the .env
load_dotenv(dotenv_path=ENV_PATH)

# Gets the CSV and audio folder paths
CSV_PATH = os.getenv("CSV_PATH")
AUDIO_DIR = os.getenv("AUDIO_FOLDER")

# The classes encondings
CLASS_MAPPING = {
    "crow": 0,
    "footsteps": 1,
    "washing_machine": 2,
    "engine": 3,
    "wind": 4
}

def extract_features(file_path, sr=22050, duration=5.0, n_mfcc=20):
    ''' Preprocess audio and extract 40 summary MFCC features '''
    
    # Calculate the target value of samples
    target_samples = int(sr * duration)
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    
    # Normalizes the audio amplitude so the loudest samples has amplitude of 1.0
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # If the audio contains fewer samples than required
    if len(audio) < target_samples:
        # append zeros to the end until it reaches the target length
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    else:
        audio = audio[:target_samples]
    
    # Extract the mfccs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Reduce the number of features to the mean and standard deviation
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    return np.hstack([mfcc_mean, mfcc_std])

def process_audio_and_extract_mfcc(save_npy=True):
    ''' Process audio files, normalize, save/load and return X and Y '''
    
    # Load cached .npy files if present
    if os.path.exists("X_data.npy") and os.path.exists("Y_data.npy"):
        X = np.load("X_data.npy")
        Y = np.load("Y_data.npy")
        
        return X, Y

    # Extract the features from the audios dataset
    df = pd.read_csv(CSV_PATH)
    target_df = df[df["category"].isin(CLASS_MAPPING.keys())].reset_index(drop=True)

    X_list, Y_list = [], []

    # Iterate through the ESC-50 metadata, load each existing audio file, extract its features, and store its label
    for idx, row in target_df.iterrows():
        filename = row["filename"]
        category = row["category"]
        file_path = os.path.join(AUDIO_DIR, filename)
        
        if os.path.exists(file_path):
            features = extract_features(file_path)
            X_list.append(features)
            Y_list.append(CLASS_MAPPING[category])
    
    # Get the features and the label from the X_list and Y_list
    X = np.array(X_list).T 
    Y_labels = np.array(Y_list)

    # Get the number of classes and training examples,
    # then initialize a one-hot encoded label matrix
    num_classes = len(CLASS_MAPPING)
    m = Y_labels.shape[0]
    Y = np.zeros((num_classes, m))
    
    # Set the corresponding class index to 1 for each example
    Y[Y_labels, np.arange(m)] = 1

    # Compute the mean and standard deviation of each feature
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True) + 1e-8
    
    # Standardize the feature values
    X = (X - X_mean) / X_std
    
    # Create the artifacts directory and save the feature scaling parameters
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    np.savez(artifacts_dir / "scaler.npz", mean=X_mean, std=X_std)

    return X, Y

if __name__ == "__main__":
    process_audio_and_extract_mfcc()