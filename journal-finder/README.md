# CS Journal Finder

A data mining project that recommends the **top-5 most relevant Computer Science journals** for a given article abstract, and generates topic clusters for CS subject areas.

## Dataset
- Raw Web of Science export: **23,801 article records** from **466 publications**
- Final modelling corpus after CS/abstract filtering: **20,944 articles** from **410 journals**
- SQLite database: `../CompSciencePub.sqlite`

## Project Structure
```
journal-finder/
|-- notebooks/
|   |-- 01_data_extraction_eda.ipynb       # DB -> Parquet, EDA
|   |-- 02_recommender_models.ipynb        # TF-IDF, SBERT, LDA, Hybrid
|   |-- 03_evaluation.ipynb                # Metrics and comparison table
|   `-- 04_topic_clustering.ipynb          # LDA + UMAP/HDBSCAN clustering
|-- src/
|   |-- io_db.py          # SQLite -> Pandas -> Parquet
|   |-- preprocessing.py  # Text cleaning pipeline
|   |-- recommender.py    # TF-IDF / SBERT / LDA / Hybrid recommenders
|   `-- clustering.py     # Topic clustering utilities
|-- app/
|   `-- streamlit_app.py  # Interactive demo
|-- data/
|   |-- raw/              # Parquet exports from SQLite
|   `-- processed/        # Cleaned master DataFrame + SBERT embeddings
|-- models/               # Saved TF-IDF, SBERT index, LDA artifacts
|-- report/
|   |-- main.tex          # IEEE conference paper (LaTeX)
|   `-- refs.bib          # BibTeX references
`-- requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage
### Jupyter Notebooks
Run notebooks **01 -> 02 -> 03 -> 04** in order. Notebook 01 extracts data from the SQLite and saves Parquet files; all subsequent notebooks load from Parquet.

### Streamlit Demo App
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501`, paste an abstract, and get top-5 journal recommendations with explanations.

## Methods
| Method | Description | Strengths |
|--------|-------------|-----------|
| **TF-IDF + Cosine** | Bag-of-words bigram TF-IDF; per-journal centroid matching | Fast, explainable, strongest current result |
| **SBERT + Cosine** | `all-MiniLM-L6-v2` sentence embeddings; semantic matching | Captures related phrasing |
| **LDA + JS-Divergence** | Topic distribution matching via Jensen-Shannon divergence | Probabilistic, interpretable topics |
| **Hybrid (RRF)** | Reciprocal Rank Fusion of all three methods | Ensemble baseline; improves over SBERT/LDA |

## Evaluation
Current held-out 20% test results:

| Method | Top-1 | Top-3 | Top-5 | MRR@5 |
|--------|------:|------:|------:|------:|
| TF-IDF + Cosine | 0.3136 | 0.5143 | 0.6185 | 0.4246 |
| SBERT + Cosine | 0.2273 | 0.3982 | 0.5065 | 0.3254 |
| LDA + JS-Div | 0.1171 | 0.2498 | 0.3344 | 0.1935 |
| Hybrid (RRF) | 0.2357 | 0.4343 | 0.5344 | 0.3447 |

## Topic Clustering
Notebook 04 generates:
- Global LDA topic visualization with pyLDAvis
- Per-subject UMAP + HDBSCAN sub-topic clusters
- Subject-topic heatmap
- Exported cluster summaries in `data/outputs/subject_topics.json`
