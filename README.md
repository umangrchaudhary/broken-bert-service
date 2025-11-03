# Text Classification with BERT

This project implements a text classification model using DistilBERT for sentiment analysis on review data.

## Features

- Uses pre-trained DistilBERT model (`distilbert-base-uncased`)
- Fine-tunes for binary sentiment classification (positive/negative)
- Includes data preprocessing and tokenization
- Supports both training and evaluation
- Saves trained model and tokenizer for inference

## Project Structure

```
├── app/                 # FastAPI application
│   ├── main.py         # FastAPI app setup
│   ├── endpoints.py    # API route handlers
│   └── schemas.py      # Pydantic models
├── ml/                  # ML module
│   ├── data.py         # Data loading and preprocessing
│   ├── model.py        # ReviewClassifier for inference
│   └── train.py        # Model definition and training logic
├── assets/             # Data and model storage
│   ├── reviews.csv     # Sample dataset
│   ├── model.pth       # Trained model (generated after training)
│   └── tokenizer/      # Saved tokenizer (generated after training)
├── tests/              # Test suite
│   └── test_api.py     # API tests
├── requirements.txt    # Dependencies
└── README.md          # Documentation
```

## Dataset Format

The dataset should be a CSV file with two columns:
- `review`: Text reviews (string)
- `label`: Sentiment labels ("positive" or "negative")

Example:
```csv
review,label
"This movie was fantastic!",positive
"Terrible film, waste of time.",negative
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training script with default parameters:
```bash
python -m ml.train
```

Or with custom parameters:
```bash
python -m ml.train --epochs 3 --batch_size 32 --learning_rate 1e-5
```

### Command Line Arguments

- `--csv_path`: Path to dataset CSV file (default: `assets/reviews.csv`)
- `--model_path`: Path to save trained model (default: `assets/model.pth`)
- `--tokenizer_path`: Path to save tokenizer (default: `assets/tokenizer/`)
- `--epochs`: Number of training epochs (default: 2)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--max_length`: Maximum sequence length (default: 128)

## Model Architecture

- **Base Model**: DistilBERT (`distilbert-base-uncased`)
- **Classification Head**: Linear layer with dropout
- **Loss Function**: Cross-entropy loss with class weighting
- **Optimizer**: AdamW

## Training Details

- The model is trained for 2 epochs by default
- Uses stratified train-test split (80-20)
- Implements class weighting for handling imbalanced data
- Tracks both training and validation accuracy
- Automatically detects and uses GPU if available

## Output

The script outputs:
- Training progress with loss and accuracy per epoch
- Final training and test accuracies
- Saved model location (`assets/model.pth`)
- Saved tokenizer location (`assets/tokenizer/`)

## Testing

Run the API tests:
```bash
# Install test dependencies (included in requirements.txt)
pip install pytest httpx

# Run tests
pytest tests/test_api.py

# Run with verbose output
pytest tests/test_api.py -v
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Transformers 4.30+
- FastAPI 0.104+
- Additional dependencies listed in `requirements.txt`

## API Usage

After training the model, you can serve it via a REST API using FastAPI:

### Starting the API Server

```bash
# Run the FastAPI server (recommended)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Or run the script directly
python app/main.py

# For production (without reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- Main API: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`
- Alternative docs: `http://127.0.0.1:8000/redoc`

### API Endpoints

#### Single Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was fantastic!"}'
```

Response:
```json
{
  "label": "positive",
  "confidence": 0.93
}
```

#### Batch Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great movie!", "Terrible film."]}'
```

#### Health Check
```bash
curl "http://127.0.0.1:8000/health"
```

#### Model Information
```bash
curl "http://127.0.0.1:8000/model/info"
```

### API Structure

```
app/
├── main.py             # FastAPI application setup
├── endpoints.py        # API route handlers
└── schemas.py          # Pydantic models for request/response
```

## Sample Dataset

The project includes a sample `assets/reviews.csv` file with 20 movie reviews for testing purposes.