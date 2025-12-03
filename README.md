# Information Retrieval System
# CS 516: Information Retrieval and Text Mining - Fall 2025
# Information Technology University (ITU)

## Overview

A complete, locally-running Information Retrieval (IR) system implementing multiple retrieval strategies including Boolean Retrieval, TF-IDF, and BM25 ranking algorithms.

**Dataset:** [Kaggle News Articles Dataset](https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles) - News articles from thenews.com.pk (2015-present) covering business and sports.

## Features

- **Multiple Retrieval Methods:**
  - Boolean Retrieval (AND, OR, NOT operators)
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - BM25 (Best Matching 25) - probabilistic ranking
  - Hybrid scoring (weighted combination of TF-IDF and BM25)

- **Text Preprocessing:**
  - Case normalization
  - Punctuation removal
  - Tokenization
  - Stop word removal
  - Porter Stemming

- **Evaluation Metrics:**
  - Precision, Recall, F1 Score
  - Mean Average Precision (MAP)
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)

- **Efficiency Features:**
  - Inverted index for fast retrieval
  - Index persistence (save/load)
  - Query caching

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone or download this repository:
```bash
cd "/Users/apple/Desktop/IR assignment"
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (done automatically on first run):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Dataset Setup

### Download the Kaggle News Articles Dataset

1. Go to: https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles
2. Click "Download" (requires Kaggle account)
3. Extract `Articles.csv` to the `data/` folder:

```bash
mkdir -p data
# Move the downloaded Articles.csv to data/
mv ~/Downloads/archive/Articles.csv ./data/
```

The dataset contains:
- **~2,700+ news articles** from thenews.com.pk
- **Columns:** Article, Heading, Date, NewsType (business/sports)
- **Size:** ~5 MB

## Quick Start

### 1. Build the Index (with Kaggle dataset)
```bash
python main.py --build-index --data-path ./data/Articles.csv --save-index
```

### 2. Interactive Search
```bash
python main.py --interactive --index-path ./index.pkl
```

### 3. Single Query
```bash
python main.py --query "Pakistan cricket match" --method bm25 --index-path ./index.pkl
```

### 4. Run Evaluation
```bash
python main.py --evaluate --queries ./data/queries.json --qrels ./data/qrels.json --index-path ./index.pkl
```

## Usage Examples

### Single Query
```bash
python main.py --query "information retrieval systems" --method bm25 --top-k 10 --index-path ./index.pkl
```

### Boolean Query
```bash
python main.py --query "information AND retrieval NOT machine" --method boolean --index-path ./index.pkl
```

### Using Different Scoring Methods
```bash
# BM25 (default, best for most cases)
python main.py --query "text mining" --method bm25

# TF-IDF
python main.py --query "text mining" --method tfidf

# Hybrid (combines BM25 and TF-IDF)
python main.py --query "text mining" --method hybrid
```

## Project Structure

```
IR assignment/
├── main.py                 # Main entry point and CLI
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # Text preprocessing module
│   ├── indexing.py        # Inverted index implementation
│   ├── retrieval.py       # Retrieval engine (TF-IDF, BM25, Boolean)
│   ├── evaluation.py      # Evaluation metrics (Precision, Recall, MAP, etc.)
│   └── data_loader.py     # Document loading utilities
├── data/
│   └── sample/            # Sample dataset
│       ├── documents.json
│       ├── queries.json
│       └── qrels.json
└── docs/
    └── technical_report.md # Technical report
```

## Data Formats

### Documents (JSON)
```json
[
    {"id": "doc1", "title": "Title", "content": "Document content..."},
    {"id": "doc2", "title": "Title 2", "content": "More content..."}
]
```

### Documents (CSV)
```csv
id,title,content
doc1,Title,Document content...
doc2,Title 2,More content...
```

### Documents (Directory)
Place `.txt` files in a directory. Each file is treated as a document with the filename as the document ID.

### Queries (JSON)
```json
{
    "q1": "information retrieval",
    "q2": "text mining techniques"
}
```

### Relevance Judgments (JSON)
```json
{
    "q1": ["doc1", "doc3", "doc5"],
    "q2": ["doc2", "doc4"]
}
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--create-sample` | Create sample dataset |
| `--build-index` | Build index from documents |
| `--interactive` | Run interactive search mode |
| `--query` | Run a single query |
| `--evaluate` | Evaluate the system |
| `--data-dir` | Directory for sample data (default: ./data/sample) |
| `--data-path` | Path to documents (file or directory) |
| `--index-path` | Path to save/load index (default: ./index.pkl) |
| `--queries` | Path to queries file for evaluation |
| `--qrels` | Path to relevance judgments file |
| `--method` | Scoring method: bm25, tfidf, boolean, hybrid |
| `--top-k` | Number of results to return (default: 10) |
| `--save-index` | Save index after building |

## Configuration

The system can be configured by modifying parameters in `main.py`:

### Preprocessing Options
- `lowercase`: Convert to lowercase (default: True)
- `remove_punctuation`: Remove punctuation (default: True)
- `remove_stopwords`: Remove stop words (default: True)
- `use_stemming`: Apply Porter stemming (default: True)
- `min_word_length`: Minimum token length (default: 2)

### BM25 Parameters
- `k1`: Term frequency saturation (default: 1.5)
- `b`: Length normalization (default: 0.75)

### Hybrid Retrieval Weights
- `tfidf_weight`: Weight for TF-IDF scores (default: 0.3)
- `bm25_weight`: Weight for BM25 scores (default: 0.7)

## API Usage

```python
from src.indexing import InvertedIndex
from src.retrieval import RetrievalEngine, ScoringMethod
from src.preprocessing import TextPreprocessor

# Create and populate index
preprocessor = TextPreprocessor()
index = InvertedIndex(preprocessor)
index.add_document("doc1", "Title", "Content text here...")

# Create retrieval engine
engine = RetrievalEngine(index, ScoringMethod.BM25)

# Search
results = engine.search("query text", top_k=10)
for result in results.results:
    print(f"{result.rank}. {result.title} (score: {result.score:.4f})")
```

## Troubleshooting

### NLTK Data Not Found
```bash
python -c "import nltk; nltk.download('all')"
```

### Module Not Found
Ensure you're in the project directory and the virtual environment is activated.

### Memory Issues with Large Datasets
Use streaming loading:
```python
loader = DocumentLoader(path)
for batch in loader.load_streaming(batch_size=100):
    for doc_id, title, content in batch:
        index.add_document(doc_id, title, content)
```

## License

This project is for educational purposes as part of CS 516 coursework at ITU.

## Author

Student - Information Technology University (ITU)
Fall 2025
