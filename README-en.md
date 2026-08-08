# Audio Classification Neural Network From Scratch

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Deep_Learning-013243.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-336791.svg?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED.svg?logo=docker&logoColor=white)

This project's goal was to build a Multi-Layer Perceptron (MLP) Neural Network from scratch and apply it to audio classification. Instead of relying on high-level frameworks like PyTorch or TensorFlow, the Neural Network was built purely in NumPy, implementing algorithms such as Forward Propagation (with and without Dropout), Backpropagation (standard and with optimization algorithms), and learning optimization algorithms like Adam and learning rate decay.

The model was initially trained to recognize 5 classes: wind, crow, footsteps, washing machine, and engine. However, it can be adapted to recognize additional classes or replace the existing ones, by redoing the training and generating new parameters. The number of neurons and layers can also be defined dynamically, by changing the corresponding values in the `train.py` script.

Model inference is exposed via a **REST API built with FastAPI**, which extracts audio features (MFCCs) in real time, runs the prediction, and **logs each inference (latency and confidence level) to a PostgreSQL database**, all orchestrated via **Docker and Docker Compose**.

---

## Features

- **Custom Engine (NumPy):** Deep neural network, loss functions, optimizers, and layers built entirely from scratch.
- **Retrainable and Adaptable Model:** The library's architecture was designed so the model can be retrained or, with minor adaptations to `train.py`, repurposed for other tasks.
- **Audio Processing:** Extraction of 40 MFCC (Mel-frequency cepstral) coefficients per audio file (mean and standard deviation of the original MFCCs for dimensionality reduction) using the `librosa` library.
- **Fast, Documented API:** FastAPI with automatic Swagger UI documentation.
- **Database Integration:** Real-time logging of inferences and metrics using SQLAlchemy and PostgreSQL.
- **Containerized Environment:** Production-ready architecture with Docker and multi-stage builds.

---

## Project Structure

```text
audio-ml-from-scratch/
├── api/                    # FastAPI code and database integration
│   ├── main.py              # API endpoints (/predict, /)
│   ├── database.py          # PostgreSQL connection
│   ├── models_db.py         # SQLAlchemy schemas (PredictionLog)
│   └── schemas.py           # Pydantic schemas for validation
├── artifacts/               # Saved artifacts (.npz, scalers)
├── data/                    # Datasets (e.g. esc50.csv) and raw audio files
│   ├── raw/                 # Where audio files should be placed
│   └── esc50.csv            # Mapping of audio files to their labels
├── docs/                    # Theoretical documentation (Backprop, Forwardprop, MFCCs)
├── engine/                  # Neural network math modules (NumPy)
│   ├── layers.py            # Forward, Backward, Dropout
│   ├── loss.py               # Loss and activation functions
│   ├── model.py              # Model architecture
│   └── optimizers.py         # Weight updates (Adam, SGD, etc.)
├── pipeline/                 # Data processing
│   └── audio_processor.py    # MFCC extraction and class mapping
├── requirements/              # Folder with dependency files
│   ├── dev.txt                # Development dependencies
│   ├── prod.txt               # Production dependencies
│   └── test.txt                # Testing dependencies
├── tests/
│   ├── conftest.py            # Centralizes test configuration
│   ├── test_api.py            # Tests for API endpoints and functions
│   └── test_engine.py         # Tests for the model's core components
├── .dockerignore               # Files ignored by Docker
├── .env.example                 # Example of required environment variables
├── .gitignore                   # Files ignored by git
├── train.py                     # Main script to train the model
├── Dockerfile                   # API build configuration
└── docker-compose.yml           # PostgreSQL + API orchestration
```

---

## Prerequisites

To run this project on your machine, you need:

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.
- (Optional, for development) Python 3.11+ and `virtualenv`.

---

## How to Run

### Via Docker (Recommended)

The `docker-compose.yml` will launch the PostgreSQL database and the FastAPI API simultaneously.

> **Note:** Make sure **Docker Desktop** is open and running before executing the commands.

**1. Configure Environment Variables**

Create a `.env` file in the project root (use `.env.example` as a template) with the database credentials:

```env
POSTGRES_USER=audio_admin
POSTGRES_PASSWORD=secure_password_98765
POSTGRES_DB=audio_classifier_db
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}
```

**2. Launch the Application**

In the terminal, run:

```bash
docker-compose up -d --build
```

**3. Access the API**

- **Swagger UI Documentation:** [http://localhost:8080/docs](http://localhost:8080/docs)
- The API will automatically attempt to load the saved model weights located in `artifacts/trained_model_params.npz`. If the file does not exist, run the training script first.

---

### Local Execution (Development Environment)

If you want to run the project directly on your local machine without Docker:

**1. Create and activate the virtual environment:**

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate
```

**2. Install development dependencies:**

```bash
pip install -r requirements/dev.txt
```

**3. Start the development server with Uvicorn:**

```bash
uvicorn api.main:app --reload --port 8080
```

---

**4. Access the API:**

- **Swagger UI Documentation:** [http://localhost:8080/docs](http://localhost:8080/docs)
- The API will automatically attempt to load the saved model weights located in `artifacts/trained_model_params.npz`. If the file does not exist, run the training script first.

---

## Training the Model

If you'd like to retrain the model on different classes or make modifications, follow these steps:

**1. Install local dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements/dev.txt
```

**2. Download and place the raw data**

Before running training, you **need to download the audio files from the dataset** (ESC-50) and place them inside the `raw` folder (e.g. `data/raw/audio`, as mapped in your `.env`). The script will fail if it can't find the audio files to extract features from.

**3. Customize the classes (Optional)**

If you want to train the model to recognize different classes, define your target classes in `audio_processor.py` before running training.

**4. Run the training pipeline:**

```bash
python train.py
```

*This command will read the audio files in the `raw` folder, extract the MFCCs, run the training epochs in NumPy, and save the weights to `trained_model_params.npz` at the project root and the normalization metadata to `scaler.npz` inside `artifacts/`.*

---

## API Endpoints

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Returns the health check status and the model's loaded layer dimensions. |
| `POST` | `/predict` | Accepts a `.wav` or `.mp3` file, extracts MFCCs, runs Forward Propagation through the neural network, logs the metrics to **Postgres**, and returns the prediction. |

**Example `/predict` Response:**

```json
{
  "filename": "strong_wind.wav",
  "prediction": "wind",
  "confidence": "96.40%",
  "latency_ms": 12.35,
  "probabilities_pct": {
    "wind": 96.40,
    "crow": 2.15,
    "engine": 1.45
  }
}
```

---

## Author

Built by **Leandro Carvalho**. This project is open source and was created for educational purposes, as a deep dive into the math behind Deep Learning and ML Engineering.
