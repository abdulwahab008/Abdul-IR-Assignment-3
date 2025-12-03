# Assignment Requirements Checklist
## CS 516: Information Retrieval and Text Mining - Homework 3

Use this checklist to verify all requirements are met before submission.

---

## ✅ REQUIREMENTS VERIFICATION

### 1. Local Implementation
| Requirement | Status | Evidence |
|-------------|--------|----------|
| Runs end-to-end on local machine | ✅ DONE | Python-based, no cloud dependencies |
| No cloud-hosted vector databases | ✅ DONE | Uses local inverted index (pickle file) |
| Uses local libraries only | ✅ DONE | NLTK, NumPy (all local) |

**Files:** `main.py`, `src/*.py`

---

### 2. Reproducible Pipeline
| Requirement | Status | Evidence |
|-------------|--------|----------|
| Source code included | ✅ DONE | `src/` folder with all modules |
| README with instructions | ✅ DONE | `README.md` with setup & usage |
| Configuration files | ✅ DONE | `requirements.txt` |

**Commands to verify:**
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build & Run
python main.py --build-index --data-path ./data/Articles.csv --save-index
python main.py --interactive --index-path ./index.pkl
```

---

### 3. Technical Report Sections

#### 3.1 System Architecture
| Requirement | Status | Location in Report |
|-------------|--------|-------------------|
| System Diagram | ✅ DONE | Section 1.1 - ASCII block diagram |
| Figure Caption | ✅ DONE | Section 1.2 - 2-3 sentence description |

#### 3.2 Description of Retrieval System
| Requirement | Status | Location in Report |
|-------------|--------|-------------------|
| Data preprocessing steps | ✅ DONE | Section 2.1 |
| - Normalization | ✅ DONE | Section 2.1.2 |
| - Capitalization handling | ✅ DONE | Section 2.1.2 (lowercase) |
| - Tokenization | ✅ DONE | Section 2.1.3 |
| Indexing techniques | ✅ DONE | Section 2.2 |
| - Boolean | ✅ DONE | Section 2.3.1 |
| - TF-IDF | ✅ DONE | Section 2.3.2 |
| - BM25 | ✅ DONE | Section 2.3.3 |
| Scoring and ranking criteria | ✅ DONE | Section 2.3 |
| Justifications provided | ✅ DONE | Throughout Section 2 |

#### 3.3 Evaluation
| Requirement | Status | Location in Report |
|-------------|--------|-------------------|
| Qualitative evaluation | ✅ DONE | Section 3.5 |
| Quantitative evaluation | ✅ DONE | Section 3.2-3.3 |
| - Precision, Recall, F1 | ✅ DONE | Section 3.2.1 |
| - MAP | ✅ DONE | Section 3.2.2 |
| - MRR | ✅ DONE | Section 3.2.3 |
| - NDCG | ✅ DONE | Section 3.2.4 |
| Memory footprint | ✅ DONE | Section 3.4.1 |
| Querying speed | ✅ DONE | Section 3.4.2 |

#### 3.4 Discussion
| Requirement | Status | Location in Report |
|-------------|--------|-------------------|
| Major findings | ✅ DONE | Section 4.1 |
| Shortcomings | ✅ DONE | Section 4.2 |
| Improvement plans | ✅ DONE | Section 4.3 |

#### 3.5 References
| Requirement | Status | Location in Report |
|-------------|--------|-------------------|
| Academic citations | ✅ DONE | Section 5 (6 references) |
| Consistent format | ✅ DONE | Author-Date format |

#### 3.6 AI Disclosure
| Requirement | Status | Location in Report |
|-------------|--------|-------------------|
| Summary of AI tools used | ✅ DONE | Section 6.1 |
| Screenshots of prompts | ✅ DONE | Section 6.2 (17 screenshots) |
| Location in code indicated | ✅ DONE | Section 6.3 |
| Modifications explained | ✅ DONE | Section 6.4 |

---

### 4. Submission Requirements
| Requirement | Status | Notes |
|-------------|--------|-------|
| Single PDF report | ✅ DONE | `docs/technical_report.pdf` (7.4 MB) |
| GitHub repository | ✅ DONE | https://github.com/abdulwahab008/Abdul-IR-Assignment-3 |
| README with instructions | ✅ DONE | `README.md` |
| Configuration files | ✅ DONE | `requirements.txt` |
| AI screenshots in PDF | ✅ DONE | 17 screenshots in Section 6.2 |

---

## 📁 PROJECT STRUCTURE

```
IR assignment/
├── main.py                    ✅ Main CLI entry point
├── requirements.txt           ✅ Dependencies
├── README.md                  ✅ Setup instructions
├── create_qrels.py            ✅ Relevance judgment helper
├── index.pkl                  ✅ Saved index
├── src/
│   ├── __init__.py           ✅ Package init
│   ├── preprocessing.py       ✅ Text preprocessing
│   ├── indexing.py           ✅ Inverted index + TF-IDF + BM25
│   ├── retrieval.py          ✅ Search engine
│   ├── evaluation.py         ✅ Evaluation metrics
│   └── data_loader.py        ✅ Dataset loading
├── data/
│   ├── Articles.csv          ✅ Kaggle News Articles (2,692)
│   ├── queries.json          ✅ Test queries (10)
│   └── qrels.json            ✅ Relevance judgments
└── docs/
    ├── technical_report.md   ✅ Report (Markdown)
    ├── technical_report.pdf  ✅ Report (PDF with images)
    ├── architecture.png      ✅ System architecture diagram
    └── ai_screenshots/       ✅ 17 AI usage screenshots
```

---

## ✅ ALL ACTIONS COMPLETED

### 1. Dataset ✅
- Downloaded from Kaggle: https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles
- Located at: `data/Articles.csv`
- 2,692 articles indexed

### 2. Index Built ✅
```bash
python main.py --build-index --data-path ./data/Articles.csv --save-index
# Result: index.pkl created (19,411 vocabulary terms)
```

### 3. Testing Complete ✅
```bash
python main.py --interactive --index-path ./index.pkl
# All methods tested: BM25, TF-IDF, Boolean, Hybrid
```

### 4. Evaluation Complete ✅
```bash
python main.py --evaluate --queries ./data/queries.json --qrels ./data/qrels.json
# BM25: Precision=0.96, Recall=1.00, MAP=1.00, NDCG=1.00
```

### 5. AI Disclosure Screenshots ✅
- 17 screenshots added to `docs/ai_screenshots/`
- All referenced in Section 6.2 of technical report

### 6. PDF Report Generated ✅
```bash
pandoc docs/technical_report.md -o docs/technical_report.pdf --pdf-engine=pdflatex
# Result: technical_report.pdf (7.4 MB with all images)
```

### 7. GitHub Repository ✅
```bash
git push origin main
# Repository: https://github.com/abdulwahab008/Abdul-IR-Assignment-3
```

---

## 🎯 IMPLEMENTATION COVERAGE

| Feature | Implemented | File |
|---------|-------------|------|
| **Retrieval Methods** | | |
| Boolean (AND, OR, NOT) | ✅ | `src/indexing.py`, `src/retrieval.py` |
| TF-IDF | ✅ | `src/indexing.py` |
| BM25 | ✅ | `src/indexing.py` |
| Hybrid (TF-IDF + BM25) | ✅ | `src/retrieval.py` |
| **Preprocessing** | | |
| Lowercase conversion | ✅ | `src/preprocessing.py` |
| Punctuation removal | ✅ | `src/preprocessing.py` |
| Tokenization (NLTK) | ✅ | `src/preprocessing.py` |
| Stopword removal | ✅ | `src/preprocessing.py` |
| Porter Stemming | ✅ | `src/preprocessing.py` |
| **Indexing** | | |
| Inverted index | ✅ | `src/indexing.py` |
| Positional index | ✅ | `src/indexing.py` |
| Index persistence | ✅ | `src/indexing.py` |
| **Evaluation** | | |
| Precision@K | ✅ | `src/evaluation.py` |
| Recall@K | ✅ | `src/evaluation.py` |
| F1 Score | ✅ | `src/evaluation.py` |
| MAP | ✅ | `src/evaluation.py` |
| MRR | ✅ | `src/evaluation.py` |
| NDCG | ✅ | `src/evaluation.py` |
| Efficiency metrics | ✅ | `src/evaluation.py` |

---

**All core requirements are implemented. Complete the action items above before submission.**
