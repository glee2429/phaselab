# Deduplication Project

A document deduplication system that uses supervised learning with hard negative mining to identify duplicate documents.

## Project Structure

The project has been refactored to follow object-oriented programming principles:

```
dedup_project/
├── app/
│   ├── data/
│   │   ├── __init__.py
│   │   └── processor.py        # DataProcessor for loading and processing documents
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── encoder.py          # EmbeddingEncoder for document embeddings and ANN search
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py       # DuplicateClassifier for training and evaluating models
│   ├── services/
│   │   ├── __init__.py
│   │   └── deduplication_service.py  # Orchestration service for the pipeline
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration handling
│   └── __init__.py
├── main.py                      # Entry point
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```

## Features

- Document embedding using SentenceTransformers
- Fast approximate nearest neighbor search with Annoy
- Hard negative mining for better model training
- Evaluation with precision, recall, F1, and ROC curve

## Usage

Run the deduplication pipeline with:

```bash
python main.py --data path/to/your/data.csv
```

### Using UV

This project uses `uv` for dependency management and running commands. These are more efficient alternatives to traditional tools:

```bash
# Run the application using uv (no virtual environment activation needed)
uv run python main.py --data dataset.csv

# Check code style
uv run ruff check .

# Fix code style issues automatically
uv run ruff check --fix .

# Format code with black
uv run black .
```

### Command Line Arguments

- `--data`: Path to the CSV file with document data (required)
- `--model`: SentenceTransformer model name/path (default: "sentence-transformers/all-MiniLM-L6-v2")
- `--negative-ratio`: How many random negative pairs to sample (default: 1.0)
- `--test-size`: Fraction of pairs to use as test set (default: 0.2)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--similarity-threshold`: Cosine similarity threshold for hard negatives (default: 0.80)
- `--neighbors`: Number of neighbors to search for hard negatives (default: 20)
- `--num-trees`: Number of trees in Annoy index (default: 10)

## Data Format

The input CSV should contain the following columns:
- `core_id`: Unique document ID
- `processed_title`: Document title
- `processed_abstract`: Document abstract
- `labelled_duplicates`: List of IDs of duplicate documents (string format: "['123','456']")

## Requirements

Python 3.12+ and the dependencies listed in pyproject.toml
