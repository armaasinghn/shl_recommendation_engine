# SHL Assessment Recommendation Engine

A production-ready recommendation system that suggests relevant SHL assessments based on natural language queries or job descriptions.

## 🎯 Overview

This system helps hiring managers and recruiters find the right assessments for their hiring needs. Given a job description or query, it returns the most relevant SHL assessments using:

- **TF-IDF based semantic search** for relevance matching
- **Query analysis** for extracting constraints (duration, test types, job levels)
- **Balanced recommendations** that include both technical and behavioral assessments when needed

## 📁 Project Structure

```
shl_recommendation_engine/
├── src/
│   ├── __init__.py          # Package initializer
│   ├── config.py            # Configuration constants
│   ├── models.py            # Data models (Assessment, QueryInfo)
│   ├── text_processor.py    # Text preprocessing & TF-IDF
│   ├── query_analyzer.py    # Query understanding & extraction
│   ├── search_engine.py     # Search & ranking logic
│   ├── recommender.py       # Main recommendation interface
│   └── evaluator.py         # Evaluation metrics (Recall@K)
├── data/
│   └── shl_assessments.json # Assessment catalog (scraped)
├── templates/
│   └── index.html           # Web frontend
├── app.py                   # Flask API application
├── run_evaluation.py        # Evaluation & prediction script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd shl_recommendation_engine

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
python app.py
```

The server starts at `http://localhost:5000`

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Get recommendations
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Java developer who collaborates with business teams"}'
```

### 4. Open Web Interface

Visit `http://localhost:5000` in your browser for the interactive UI.

## 📡 API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{"status": "healthy"}
```

### POST /recommend
Get assessment recommendations.

**Request:**
```json
{
  "query": "Job description or natural language query"
}
```

**Response:**
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/...",
      "name": "Python (New)",
      "adaptive_support": "No",
      "description": "Multi-choice test...",
      "duration": 11,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    }
  ]
}
```

## 📊 Evaluation

### Run Evaluation on Train Set

```bash
python run_evaluation.py --data path/to/Gen_AI_Dataset.xlsx
```

### Generate Test Predictions

```bash
python run_evaluation.py --data path/to/Gen_AI_Dataset.xlsx --output predictions.csv
```

### Quick Test with Sample Queries

```bash
python run_evaluation.py --quick
```

## 🧠 How It Works

### 1. Query Analysis (`query_analyzer.py`)
- Extracts duration constraints (e.g., "30-40 mins")
- Identifies required test types (technical, behavioral, cognitive)
- Detects job level (entry, mid, senior, manager, executive)
- Determines if balanced results are needed

### 2. Text Processing (`text_processor.py`)
- Tokenization and stopword removal
- Synonym expansion for better recall
- TF-IDF vectorization for semantic similarity

### 3. Search Engine (`search_engine.py`)
- Computes relevance scores using:
  - TF-IDF cosine similarity (40%)
  - Keyword matching (40%)
  - Test type match (10%)
  - Job level match (10%)
- Filters by duration constraints
- Balances technical/behavioral assessments when needed

### 4. Recommender (`recommender.py`)
- Main interface tying everything together
- Formats responses for API consumption

## 📈 Metrics

The system is evaluated using **Mean Recall@K**:

```
Recall@K = (# relevant in top K) / (total relevant)
Mean Recall@K = (1/N) × Σ Recall@Ki
```

## 🛠️ Configuration

Edit `src/config.py` to customize:
- API host/port
- Number of recommendations (1-10)
- Test type mappings
- Keyword lists for detection

## 📦 Deployment

### Using Gunicorn (Production)

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 4
```

### Using Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

### Environment Variables

- `PORT`: Server port (default: 5000)
- `DEBUG`: Enable debug mode (default: False)

## 📝 Sample Queries

```
1. "Java developer who collaborates with business teams"
2. "Entry level sales position for graduates, 30 min test"
3. "Python, SQL and JavaScript developer, max 60 minutes"
4. "Customer support executive with English communication"
5. "Senior data analyst with Excel and SQL, cognitive tests"
6. "Product manager with SDLC, Jira and Confluence expertise"
```

## 🔗 Links

- [SHL Product Catalog](https://www.shl.com/solutions/products/product-catalog/)
- [API Documentation](http://localhost:5000/)

## 📄 License

MIT License

## 👤 Author

Data Science Expert - SHL AI Internship Assessment
