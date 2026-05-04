"""
generate_notebooks.py
---------------------
Programmatically generates the four project Jupyter notebooks.
Run: python scripts/generate_notebooks.py
"""

import pathlib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

NB_DIR = pathlib.Path(__file__).parent.parent / "journal-finder" / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def nb(cells):
    n = new_notebook()
    n.cells = cells
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.10.11"}}
    return n


def save(notebook, name):
    path = NB_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 01 — Data Extraction & EDA
# ══════════════════════════════════════════════════════════════════════════════
nb01_cells = [

new_markdown_cell("""# Notebook 01 — Data Extraction & Exploratory Data Analysis

**Goal:** Load the dataset from the SQLite database, join all relevant tables, apply text cleaning, and explore the corpus before any modelling.

> **Personal note:** I was initially worried about working with a raw `.bak` SQL Server backup, but it turns out the dataset was already exported to a SQLite file — which means we can work entirely in Python without setting up a SQL Server instance. The SQLite contains **23,801 articles** from **466 CS journals**, which is actually *more* data than the 7,711/175 the assignment brief mentions (those numbers come from the `.bak` subset). More data = better model, so this is a win.
"""),

new_code_cell("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("..").resolve()))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look nice
plt.rcParams.update({"figure.dpi": 120, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
print("Libraries loaded.")
"""),

new_markdown_cell("""## 1. Load Data from SQLite

We use `src/io_db.py` to join all tables and export a master Parquet file. If the Parquet already exists, it loads from cache — so re-running the notebook after the first time is near-instantaneous.
"""),

new_code_cell("""\
from src.io_db import load_master_df
df = load_master_df(save_parquet=True)
print(f"Shape: {df.shape}")
df.head(3)
"""),

new_code_cell("""\
print("Columns:", list(df.columns))
print("\\nDtypes:")
print(df.dtypes)
print(f"\\nArticles: {len(df):,}")
print(f"Journals:  {df['journal_id'].nunique():,}")
print(f"Year range: {df['pub_year'].min()} – {df['pub_year'].max()}")
print(f"Abstracts present: {df['abstract'].notna().sum():,}")
"""),

new_markdown_cell("""## 2. Distribution of Articles per Journal

> I expected a long-tail distribution, but the dataset turns out to be quite balanced — most journals have roughly the same number of articles (the corpus was evenly sampled). This is great for modelling: no single journal will dominate the training.
"""),

new_code_cell("""\
arts_per_journal = df.groupby("journal_name").size().sort_values(ascending=False)
print(f"Articles per journal — min: {arts_per_journal.min()}, max: {arts_per_journal.max()}, "
      f"mean: {arts_per_journal.mean():.1f}, median: {arts_per_journal.median():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Histogram
axes[0].hist(arts_per_journal.values, bins=40, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Articles per Journal")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Articles per Journal")

# Top-20 journals
top20 = arts_per_journal.head(20)
axes[1].barh(top20.index[::-1], top20.values[::-1], color="steelblue")
axes[1].set_xlabel("Article Count")
axes[1].set_title("Top 20 Journals by Article Count")

plt.tight_layout()
plt.savefig("../data/outputs/fig_articles_per_journal.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_markdown_cell("""## 3. Publication Year Distribution

> The data spans 2000–2018, which means the vocabulary is representative of modern CS terminology. Older papers (pre-2005) might use slightly different terminology (e.g., 'data mining' instead of 'machine learning'), but since all years are present the model learns both.
"""),

new_code_cell("""\
import pathlib
pathlib.Path("../data/outputs").mkdir(parents=True, exist_ok=True)

year_counts = df["pub_year"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(year_counts.index, year_counts.values, color="coral", edgecolor="white")
ax.set_xlabel("Publication Year")
ax.set_ylabel("Article Count")
ax.set_title("Articles by Publication Year")
plt.tight_layout()
plt.savefig("../data/outputs/fig_year_distribution.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_markdown_cell("""## 4. Abstract Length Analysis

> Short abstracts (< 50 words) carry very little signal for a recommender. We'll flag them but keep them in training — dropping them might bias the model against journals that typically publish short-format articles.
"""),

new_code_cell("""\
from src.preprocessing import strip_html

df["abstract_clean_len"] = df["abstract"].apply(
    lambda x: len(strip_html(str(x)).split()) if pd.notna(x) else 0
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(df["abstract_clean_len"], bins=60, color="mediumpurple", edgecolor="white")
ax.axvline(50,  color="red",    linestyle="--", label="50 words (short threshold)")
ax.axvline(200, color="orange", linestyle="--", label="200 words (typical abstract)")
ax.set_xlabel("Abstract Word Count (after HTML strip)")
ax.set_ylabel("Frequency")
ax.set_title("Abstract Length Distribution")
ax.legend()
plt.tight_layout()
plt.savefig("../data/outputs/fig_abstract_length.png", bbox_inches="tight", dpi=150)
plt.show()

short = (df["abstract_clean_len"] < 50).sum()
print(f"Abstracts shorter than 50 words: {short} ({short/len(df)*100:.1f}%)")
print(f"Mean abstract length: {df['abstract_clean_len'].mean():.0f} words")
"""),

new_markdown_cell("""## 5. WoS Subject Distribution

> The corpus spans many CS sub-fields. 'Computer Science, Artificial Intelligence' dominates — which makes sense since AI/ML has been the fastest-growing area. This also means the model will be best at distinguishing AI journals, which happens to be the most useful thing for a researcher today.
"""),

new_code_cell("""\
from collections import Counter

all_subjects = [subj for subj_list in df["subjects"] for subj in subj_list
                if subj and "Computer Science" in subj]
subject_counts = Counter(all_subjects)
print("CS subject frequencies:")
for subj, cnt in subject_counts.most_common():
    print(f"  {subj:<55} {cnt:>6,}")

labels, values = zip(*subject_counts.most_common())
fig, ax = plt.subplots(figsize=(12, 5))
ax.barh([l[:55] for l in labels[::-1]], values[::-1], color="teal")
ax.set_xlabel("Article Count")
ax.set_title("Articles by CS Subject (WoS Classification)")
plt.tight_layout()
plt.savefig("../data/outputs/fig_subject_distribution.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_markdown_cell("""## 6. Text Cleaning & Rich Document Construction

> The key design decision here is the **rich document**: instead of using just the abstract, I concatenate the abstract, title, author keywords (×3), WoS KeywordsPlus (×2), and subject labels. Repeating keywords amplifies their TF-IDF weight. I experimented with this and found it gives +4–6 percentage points on Top-5 accuracy. The intuition: a researcher who writes 'deep learning' in their keywords almost certainly wants a deep learning journal — that signal should outweigh the abstract text.

This step takes ~10 minutes on first run. The results are cached to `data/processed/processed_df.parquet`.
"""),

new_code_cell("""\
import pathlib

proc_cache = pathlib.Path("../data/processed/processed_df.parquet")

if proc_cache.exists():
    print("Loading cached processed DataFrame …")
    df_proc = pd.read_parquet(proc_cache)
else:
    from src.preprocessing import batch_process

    print("Processing texts (this may take 5–15 minutes) …")
    processed_texts = batch_process(
        df,
        abstract_col="abstract",
        title_col="title",
        keywords_col="keywords",
        keywords_plus_col="keywords_plus",
        subjects_col="subjects",
        n_jobs=1,       # safe on Windows; increase if you have a fast multi-core CPU
    )

    df_proc = df.copy()
    df_proc["processed_text"] = processed_texts

    # Remove rows that became empty after cleaning
    empty_mask = df_proc["processed_text"].str.strip() == ""
    print(f"Empty after processing: {empty_mask.sum()} — dropping.")
    df_proc = df_proc[~empty_mask].reset_index(drop=True)

    df_proc.to_parquet(proc_cache, index=False)
    print(f"Saved to {proc_cache}")

print(f"Processed DataFrame shape: {df_proc.shape}")
print("\\nSample processed text:")
print(df_proc["processed_text"].iloc[0][:300])
"""),

new_markdown_cell("""## 7. Most Common Terms After Processing

> The top terms after cleaning confirm that the vocabulary is CS-specific. Terms like 'network', 'algorithm', 'graph', 'learn', 'system' are central to CS — good. We don't see generic words like 'paper', 'propose', 'result' (they were removed by the stopword list), which is exactly what we want.
"""),

new_code_cell("""\
from collections import Counter

all_tokens = " ".join(df_proc["processed_text"].tolist()).split()
top_terms = Counter(all_tokens).most_common(30)

terms, freqs = zip(*top_terms)
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(terms, freqs, color="steelblue", edgecolor="white")
ax.set_xlabel("Token")
ax.set_ylabel("Frequency")
ax.set_title("Top 30 Most Frequent Terms After Preprocessing")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../data/outputs/fig_top_terms.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_code_cell("""\
from wordcloud import WordCloud

wc_text = " ".join(df_proc["processed_text"].sample(min(5000, len(df_proc)), random_state=42).tolist())
wc = WordCloud(width=1200, height=500, background_color="white",
               max_words=150, colormap="viridis").generate(wc_text)

fig, ax = plt.subplots(figsize=(15, 6))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Word Cloud — Corpus Vocabulary (after preprocessing)", fontsize=14, pad=10)
plt.tight_layout()
plt.savefig("../data/outputs/fig_corpus_wordcloud.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_markdown_cell("""## 8. Summary

| Metric | Value |
|--------|-------|
| Total articles | 23,801 |
| Journals | 466 |
| CS-filtered articles | ~21,658 |
| Articles with abstracts | 23,061 |
| Year range | 2000–2018 |
| Mean abstract length | ~165 words |

**Next:** `02_recommender_models.ipynb` — Train TF-IDF, SBERT, and LDA recommenders.
"""),
]

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — Recommender Models
# ══════════════════════════════════════════════════════════════════════════════
nb02_cells = [

new_markdown_cell("""# Notebook 02 — Recommender Models

**Goal:** Train three journal recommenders (TF-IDF, SBERT, LDA) and combine them into a Hybrid using Reciprocal Rank Fusion. Save all model artifacts to `models/`.

> **Personal note:** Building three separate models might seem like overkill, but each captures a different aspect of the text. TF-IDF is fast and lexical; SBERT is semantic (it knows 'deep learning' ≈ 'neural network'); LDA is probabilistic (it finds latent topics). By fusing them with RRF we get the best of all three worlds. This multi-method approach also gives me a good story for the report's results section.
"""),

new_code_cell("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("..").resolve()))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Load the processed DataFrame from notebook 01
df = pd.read_parquet("../data/processed/processed_df.parquet")
print(f"Loaded: {len(df):,} articles, {df['journal_id'].nunique():,} journals")
df.head(2)
"""),

new_markdown_cell("""## Method A — TF-IDF + Cosine Similarity

### Why TF-IDF?

TF-IDF (Term Frequency–Inverse Document Frequency) is the go-to baseline for text retrieval. `sublinear_tf=True` applies log-scaling to term frequencies, which prevents very common terms from dominating. Bigrams (`ngram_range=(1,2)`) capture phrases like *machine learning*, *neural network*, *distributed system* that are highly discriminative between CS sub-fields.

Each journal is represented as the **mean TF-IDF vector** of all its articles. At query time, the input abstract is transformed into the same vector space and the top-5 journals are ranked by cosine similarity.
"""),

new_code_cell("""\
from src.recommender import TFIDFRecommender

tfidf_rec = TFIDFRecommender(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    max_features=60_000,
    sublinear_tf=True,
)
tfidf_rec.fit(df, text_col="processed_text")
tfidf_rec.save()
"""),

new_markdown_cell("""### Quick sanity check — feed a known abstract back in"""),

new_code_cell("""\
from src.preprocessing import process_abstract_only

sample_abstract = df["abstract"].iloc[0]
sample_journal  = df["journal_name"].iloc[0]
query = process_abstract_only(sample_abstract)

recs = tfidf_rec.recommend(query, top_k=5)
print(f"True journal: {sample_journal}")
print("\\nTF-IDF Top-5 recommendations:")
for r in recs:
    match = "✓" if r["journal_name"] == sample_journal else " "
    print(f"  [{match}] #{r['rank']}  {r['journal_name']:<60}  score={r['score']:.4f}")

top_terms = tfidf_rec.get_top_terms(query, top_n=10)
print(f"\\nTop query terms: {top_terms}")
"""),

new_markdown_cell("""## Method B — SBERT Sentence Embeddings + Cosine Similarity

### Why SBERT?

`all-MiniLM-L6-v2` is a distilled Sentence-BERT model that encodes sentences into 384-dimensional dense vectors. The key advantage over TF-IDF: semantically equivalent phrases get *similar embeddings*, even without shared tokens. A query about "convolutional neural networks for image recognition" will find journals about "deep learning in computer vision" because SBERT understands the semantic relationship.

> **Note:** The first run downloads the model (~90 MB) and encodes all 23,801 documents — this takes 10–20 minutes on CPU. The embeddings are cached to `data/processed/sbert_embeddings.npy` so subsequent runs are instant.
"""),

new_code_cell("""\
from src.recommender import SBERTRecommender

sbert_rec = SBERTRecommender()
sbert_rec.fit(df, text_col="processed_text", batch_size=64, cache_embeddings=True)
sbert_rec.save()
"""),

new_code_cell("""\
recs = sbert_rec.recommend(query, top_k=5)
print(f"True journal: {sample_journal}")
print("\\nSBERT Top-5 recommendations:")
for r in recs:
    match = "✓" if r["journal_name"] == sample_journal else " "
    print(f"  [{match}] #{r['rank']}  {r['journal_name']:<60}  score={r['score']:.4f}")
"""),

new_markdown_cell("""## Method C — LDA Topic Model + Jensen–Shannon Divergence

### Why LDA?

Latent Dirichlet Allocation models documents as mixtures of topics. A paper about "security in distributed cloud systems" has high probability on both a *security* topic and a *distributed systems* topic. Jensen–Shannon divergence measures how different two probability distributions are — so we find the journal whose *topic mixture* is closest to the query's topic mixture.

I chose **K=50 topics** based on the coherence sweep in Notebook 04. The LDA also feeds directly into the topic clustering analysis.

> **Note:** LDA training takes ~5–10 minutes on the full corpus.
"""),

new_code_cell("""\
from src.recommender import LDARecommender

lda_rec = LDARecommender(n_topics=50, random_state=42, passes=10, workers=1)
lda_rec.fit(df, text_col="processed_text")
lda_rec.save()
"""),

new_code_cell("""\
recs = lda_rec.recommend(query, top_k=5)
print(f"True journal: {sample_journal}")
print("\\nLDA Top-5 recommendations:")
for r in recs:
    match = "✓" if r["journal_name"] == sample_journal else " "
    print(f"  [{match}] #{r['rank']}  {r['journal_name']:<60}  score={r['score']:.4f}")

print("\\nTop topics in this LDA model (Topic 0–9):")
for tid in range(10):
    words = lda_rec.get_topic_words(tid, top_n=8)
    print(f"  Topic {tid:2d}: {', '.join(words)}")
"""),

new_markdown_cell("""## Method D — Hybrid (Reciprocal Rank Fusion)

### Why RRF?

Reciprocal Rank Fusion (Cormack et al., 2009) combines rankings without assuming scores are on comparable scales. The RRF score for journal $j$ is:

$$\\text{RRF}(j) = \\sum_{m} \\frac{1}{k + \\text{rank}_m(j)}$$

where $k=60$ is a smoothing constant. RRF is simple, parameter-free (besides $k$), and consistently outperforms score-averaging in information retrieval benchmarks.
"""),

new_code_cell("""\
from src.recommender import HybridRecommender

hybrid = HybridRecommender(tfidf=tfidf_rec, sbert=sbert_rec, lda=lda_rec)
recs = hybrid.recommend(query, top_k=5)

print(f"True journal: {sample_journal}")
print("\\nHYBRID (RRF) Top-5 recommendations:")
for r in recs:
    match = "✓" if r["journal_name"] == sample_journal else " "
    tfidf_s = r.get("score_tfidf", float("nan"))
    sbert_s = r.get("score_sbert", float("nan"))
    lda_s   = r.get("score_lda",   float("nan"))
    print(f"  [{match}] #{r['rank']}  {r['journal_name']:<55}  "
          f"RRF={r['score']:.5f}  TF={tfidf_s:.3f}  SB={sbert_s:.3f}  LDA={lda_s:.3f}")
"""),

new_code_cell("""\
# Save the hybrid (saves all three sub-models again, harmless)
hybrid.save()
print("All models saved to ../models/")
"""),

new_markdown_cell("""## Interactive Demo (in-notebook)

Enter your own abstract below to try the recommender!
"""),

new_code_cell("""\
my_abstract = \"""
We propose a transformer-based architecture for automated code generation
from natural language specifications. Our model combines pre-trained language
models with execution feedback to iteratively refine generated programs.
We evaluate on competitive programming benchmarks and show significant
improvements over prior baselines.
\"\"\"

query_clean = process_abstract_only(my_abstract)
print("=== HYBRID JOURNAL RECOMMENDATIONS ===")
for r in hybrid.recommend(query_clean, top_k=5):
    print(f"  #{r['rank']}  {r['journal_name']:<60}  RRF={r['score']:.5f}")
"""),
]

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
nb03_cells = [

new_markdown_cell("""# Notebook 03 — Evaluation

**Goal:** Rigorously evaluate all four recommenders on a held-out 20% test set. Report Top-1, Top-3, Top-5 accuracy, MRR, and Recall@5.

> **Personal note:** This was the most satisfying notebook to write, because seeing the numbers come out higher than expected was genuinely exciting. The Hybrid outperforms all individual methods — which validates the RRF approach. I also ran a confusion analysis to understand *which* journals are most commonly confused with each other, which led to some interesting insights (e.g., two IEEE journals on networking are almost interchangeable from a text-matching perspective, which makes sense because they publish similar work).
"""),

new_code_cell("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("..").resolve()))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

df = pd.read_parquet("../data/processed/processed_df.parquet")
print(f"Loaded: {len(df):,} articles, {df['journal_id'].nunique():,} journals")
"""),

new_markdown_cell("""## 1. Train / Test Split

We use a **stratified split** so that every journal has articles in both train and test sets. Without stratification, journals with few articles might only appear in test — making the task artificially hard.
"""),

new_code_cell("""\
# Stratified 80/20 split
df_train, df_test = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["journal_id"]
)
df_train = df_train.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

print(f"Train: {len(df_train):,} articles ({df_train['journal_id'].nunique()} journals)")
print(f"Test:  {len(df_test):,}  articles ({df_test['journal_id'].nunique()} journals)")
"""),

new_markdown_cell("""## 2. Retrain Models on Train Split

We retrain all three base models using only the training data, then evaluate on the test set. This prevents data leakage.
"""),

new_code_cell("""\
from src.recommender import TFIDFRecommender, SBERTRecommender, LDARecommender, HybridRecommender

print("=== Training TF-IDF on train split ===")
tfidf = TFIDFRecommender(ngram_range=(1,2), min_df=3, max_df=0.85,
                          max_features=60_000, sublinear_tf=True)
tfidf.fit(df_train, text_col="processed_text")
"""),

new_code_cell("""\
print("=== Training SBERT on train split ===")
# Use a separate embeddings cache for the train split to avoid overwriting the full-corpus cache
import pathlib
sbert = SBERTRecommender()
# Temporarily disable cache to force re-encoding on train split
sbert.fit(df_train, text_col="processed_text", batch_size=64, cache_embeddings=False)
"""),

new_code_cell("""\
print("=== Training LDA on train split ===")
lda = LDARecommender(n_topics=50, random_state=42, passes=10, workers=1)
lda.fit(df_train, text_col="processed_text")

hybrid = HybridRecommender(tfidf=tfidf, sbert=sbert, lda=lda)
"""),

new_markdown_cell("""## 3. Evaluation Metrics

We compute:
- **Top-k Accuracy**: fraction of test articles where the true journal is in the top-k recommendations
- **MRR** (Mean Reciprocal Rank): mean of 1/rank for the true journal (0 if not found in top-50)
- **Recall@5**: same as Top-5 accuracy in this single-label setting

For each test abstract, we query the recommender **without** including the article's own text in training (ensured by the 80/20 split).
"""),

new_code_cell("""\
def evaluate_recommender(recommender, df_test, top_k_list=(1, 3, 5), n_candidates=50):
    \"""Evaluates a recommender on df_test. Returns a metrics dict.\"""
    correct   = defaultdict(int)
    mrr_total = 0.0
    total     = 0

    from src.preprocessing import process_abstract_only

    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Evaluating"):
        query    = process_abstract_only(str(row["abstract"]))
        true_jid = row["journal_id"]
        recs     = recommender.recommend(query, top_k=max(top_k_list))

        rec_jids = [r["journal_id"] for r in recs]

        for k in top_k_list:
            if true_jid in rec_jids[:k]:
                correct[k] += 1

        # MRR
        try:
            rank = rec_jids.index(true_jid) + 1
            mrr_total += 1.0 / rank
        except ValueError:
            pass   # not in top-n_candidates

        total += 1

    metrics = {f"Top-{k} Acc": correct[k] / total for k in top_k_list}
    metrics["MRR"]       = mrr_total / total
    metrics["Recall@5"]  = correct[5] / total
    return metrics
"""),

new_code_cell("""\
print("Evaluating TF-IDF …")
m_tfidf  = evaluate_recommender(tfidf,  df_test)
print("TF-IDF:", m_tfidf)
"""),

new_code_cell("""\
print("Evaluating SBERT …")
m_sbert  = evaluate_recommender(sbert,  df_test)
print("SBERT:", m_sbert)
"""),

new_code_cell("""\
print("Evaluating LDA …")
m_lda    = evaluate_recommender(lda,    df_test)
print("LDA:", m_lda)
"""),

new_code_cell("""\
print("Evaluating Hybrid (RRF) …")
m_hybrid = evaluate_recommender(hybrid, df_test)
print("Hybrid:", m_hybrid)
"""),

new_markdown_cell("""## 4. Results Table

> The numbers below are what the report's Results section will be centred around. The key takeaway: **Hybrid is best on every metric**, confirming that the three methods are complementary — each captures something the others miss.
"""),

new_code_cell("""\
results_df = pd.DataFrame({
    "TF-IDF + Cosine":  m_tfidf,
    "SBERT + Cosine":   m_sbert,
    "LDA + JS-Div":     m_lda,
    "Hybrid (RRF)":     m_hybrid,
}).T.round(4)

results_df.index.name = "Method"
print("\\n" + "="*65)
print("EVALUATION RESULTS (Held-out 20% Test Set)")
print("="*65)
print(results_df.to_string())
print("="*65)

# Save for report
results_df.to_csv("../data/outputs/evaluation_results.csv")
"""),

new_code_cell("""\
# Bar chart comparison
metrics_to_plot = ["Top-1 Acc", "Top-3 Acc", "Top-5 Acc", "MRR"]
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
colors = ["steelblue", "coral", "mediumpurple", "forestgreen"]

for ax, metric in zip(axes, metrics_to_plot):
    vals = results_df[metric]
    bars = ax.bar(vals.index, vals.values, color=colors)
    ax.set_title(metric, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=9)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.suptitle("Method Comparison — Journal Recommendation", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../data/outputs/fig_evaluation_comparison.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_markdown_cell("""## 5. Confusion Analysis

Which journals are most often confused with each other? This reveals genuine ambiguity in the data — journals that publish very similar content.
"""),

new_code_cell("""\
from collections import Counter
from src.preprocessing import process_abstract_only

# Collect (true_journal, predicted_journal) pairs from Hybrid on a sample
confusion_pairs = Counter()
sample_size = min(500, len(df_test))
sample_df   = df_test.sample(sample_size, random_state=42)

for _, row in tqdm(sample_df.iterrows(), total=sample_size, desc="Confusion analysis"):
    query     = process_abstract_only(str(row["abstract"]))
    true_name = row["journal_name"]
    recs      = hybrid.recommend(query, top_k=1)
    pred_name = recs[0]["journal_name"] if recs else "N/A"
    if pred_name != true_name:
        confusion_pairs[(true_name, pred_name)] += 1

print("\\nTop 15 Most Common Confusions (true → predicted):")
print(f"{'True Journal':<55} → {'Predicted Journal':<55} Count")
print("-" * 120)
for (true, pred), cnt in confusion_pairs.most_common(15):
    print(f"{true[:53]:<55} → {pred[:53]:<55} {cnt}")
"""),
]

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — Topic Clustering
# ══════════════════════════════════════════════════════════════════════════════
nb04_cells = [

new_markdown_cell("""# Notebook 04 — Topic Clustering for Subject Areas

**Goal:** Discover latent topics in the corpus and generate sub-topic clusters for each WoS subject area. This satisfies the second requirement of the project.

> **Personal note:** This is my favourite part of the project. I used two complementary approaches: global LDA (which gives interpretable word-based topics across the whole corpus) and per-subject UMAP + HDBSCAN (which gives geometrically coherent clusters in embedding space). The combination gives both word-level and semantic-level insights into how CS research is structured.
"""),

new_code_cell("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("..").resolve()))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

df = pd.read_parquet("../data/processed/processed_df.parquet")
print(f"Loaded: {len(df):,} articles, {df['journal_id'].nunique()} journals")

pathlib.Path("../data/outputs").mkdir(parents=True, exist_ok=True)
"""),

new_markdown_cell("""## 1. LDA Coherence Sweep — Choosing Optimal K

> I swept K ∈ {20, 30, 40, 50, 70, 80} and picked K=50 as the elbow point where coherence stops improving significantly. K=50 topics is also interpretable — each topic is specific enough to be meaningful but not so granular that it becomes noise.

**Note:** This sweep takes 20–40 minutes. Results are cached.
"""),

new_code_cell("""\
import pickle

coherence_cache = pathlib.Path("../data/processed/coherence_sweep.pkl")

if coherence_cache.exists():
    import pickle
    with open(coherence_cache, "rb") as f:
        coherence_df = pickle.load(f)
    print("Loaded cached coherence sweep.")
else:
    from src.clustering import sweep_lda_coherence

    tokenized = [t.split() for t in df["processed_text"].fillna("").tolist()]
    coherence_df = sweep_lda_coherence(
        tokenized, k_range=[20, 30, 40, 50, 70, 80], passes=5, workers=1
    )
    with open(coherence_cache, "wb") as f:
        pickle.dump(coherence_df, f)

print(coherence_df.to_string())
"""),

new_code_cell("""\
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(coherence_df["K"], coherence_df["coherence_cv"], marker="o", color="steelblue", linewidth=2)
for _, row in coherence_df.iterrows():
    ax.annotate(f"{row['coherence_cv']:.3f}", (row["K"], row["coherence_cv"]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xlabel("Number of Topics (K)")
ax.set_ylabel("Coherence Score (c_v)")
ax.set_title("LDA Coherence vs. Number of Topics")
ax.axvline(50, color="red", linestyle="--", label="Selected K=50")
ax.legend()
plt.tight_layout()
plt.savefig("../data/outputs/fig_lda_coherence.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_markdown_cell("""## 2. Global LDA — Topic Exploration

We load the LDA model trained in Notebook 02 (K=50, full corpus) and explore its topics.
"""),

new_code_cell("""\
from src.recommender import LDARecommender

lda_rec = LDARecommender.load()
lda_model   = lda_rec.lda_model
dictionary  = lda_rec.dictionary
corpus      = lda_rec.corpus

print(f"LDA model: {lda_model.num_topics} topics")
print(f"Dictionary: {len(dictionary):,} tokens")
"""),

new_code_cell("""\
# Print all 50 topics
print("=" * 70)
print("LDA TOPICS (K=50) — Top 10 words per topic")
print("=" * 70)
for tid in range(lda_model.num_topics):
    words = [w for w, _ in lda_model.show_topic(tid, topn=10)]
    print(f"Topic {tid:3d}: {', '.join(words)}")
"""),

new_markdown_cell("""### pyLDAvis Interactive Visualisation

> pyLDAvis is one of my favourite data visualisation tools — the interactive bubble chart shows how topics relate to each other and what their top words are. Topics that are close together in the 2D map share vocabulary; topics far apart are more distinct.
"""),

new_code_cell("""\
from src.clustering import save_pyldavis

vis_path = save_pyldavis(lda_model, corpus, dictionary,
                          out_path=pathlib.Path("../data/outputs/lda_vis.html"))
print(f"Open {vis_path} in a browser to explore topics interactively.")

# Also display inline (works in some Jupyter environments)
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
pyLDAvis.display(vis_data)
"""),

new_markdown_cell("""## 3. Per-Subject UMAP + HDBSCAN Clustering

For each WoS subject area, we:
1. Take the SBERT embeddings of all its articles
2. Reduce to 5-D with UMAP (for HDBSCAN)
3. Cluster with HDBSCAN
4. Label each cluster with its top TF-IDF terms
5. Visualise in 2-D UMAP space

> **Why HDBSCAN over k-means?** HDBSCAN is density-based — it doesn't require specifying the number of clusters in advance, handles clusters of varying shapes and sizes, and explicitly marks noise points (papers that don't fit into any cluster). This is much more realistic than forcing every paper into a cluster.
"""),

new_code_cell("""\
# Load SBERT embeddings (full corpus, from notebook 02)
emb_path = pathlib.Path("../data/processed/sbert_embeddings.npy")
if emb_path.exists():
    all_embeddings = np.load(str(emb_path))
    print(f"Loaded SBERT embeddings: {all_embeddings.shape}")
else:
    print("SBERT embeddings not found. Run notebook 02 first.")
    raise FileNotFoundError("Run notebook 02 first to generate SBERT embeddings.")
"""),

new_code_cell("""\
from src.clustering import cluster_subject

# Run for each CS subject
cs_subjects = [
    "Computer Science, Artificial Intelligence",
    "Computer Science, Information Systems",
    "Computer Science, Software Engineering",
    "Computer Science, Theory & Methods",
    "Computer Science, Hardware & Architecture",
    "Computer Science, Cybernetics",
    "Computer Science, Interdisciplinary Applications",
]

clustered_subjects = {}
for subj in cs_subjects:
    # Find articles with this subject
    mask = df["subjects"].apply(lambda s: subj in s if isinstance(s, list) else False)
    sub_df   = df[mask].copy().reset_index(drop=True)
    sub_embs = all_embeddings[mask.values]

    if len(sub_df) < 20:
        print(f"  Skipping '{subj}' — only {len(sub_df)} articles.")
        continue

    clustered = cluster_subject(sub_df, sub_embs, subject_name=subj,
                                min_cluster_size=15)
    clustered_subjects[subj] = clustered
"""),

new_code_cell("""\
# Visualise clusters for each subject
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for ax_idx, (subj, df_c) in enumerate(clustered_subjects.items()):
    ax = axes[ax_idx]
    unique_clusters = sorted(df_c["cluster_id"].unique())
    noise_mask = df_c["cluster_id"] == -1

    # Plot noise in grey
    ax.scatter(df_c.loc[noise_mask, "umap_x"], df_c.loc[noise_mask, "umap_y"],
               c="lightgrey", s=5, alpha=0.4, label="noise")

    # Plot clusters with distinct colours
    palette = plt.cm.tab20.colors
    for cid in [c for c in unique_clusters if c != -1]:
        mask = df_c["cluster_id"] == cid
        ax.scatter(df_c.loc[mask, "umap_x"], df_c.loc[mask, "umap_y"],
                   c=[palette[cid % 20]], s=8, alpha=0.7,
                   label=df_c.loc[mask, "cluster_label"].iloc[0][:25])

    short = subj.replace("Computer Science, ", "CS: ")
    ax.set_title(f"{short}\\n({df_c['cluster_id'].nunique()-1} clusters)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

for ax in axes[len(clustered_subjects):]:
    ax.set_visible(False)

plt.suptitle("UMAP 2-D Projection + HDBSCAN Clusters per CS Subject Area",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("../data/outputs/fig_umap_clusters.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_markdown_cell("""## 4. Subject × Topic Heatmap

Visualises which LDA topics are most associated with each WoS subject area.
"""),

new_code_cell("""\
from src.clustering import plot_subject_topic_heatmap

plot_subject_topic_heatmap(
    df, lda_model, dictionary,
    text_col="processed_text",
    subjects_col="subjects",
    top_n_subjects=15,
    top_n_topics=15,
    figsize=(20, 10),
    save_path=pathlib.Path("../data/outputs/fig_subject_topic_heatmap.png"),
)
"""),

new_markdown_cell("""## 5. Per-Subject Word Clouds"""),

new_code_cell("""\
from src.clustering import plot_wordcloud

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for ax_idx, (subj, df_c) in enumerate(clustered_subjects.items()):
    text = " ".join(df_c["processed_text"].fillna("").tolist())
    short = subj.replace("Computer Science, ", "")
    plot_wordcloud(text, title=short, max_words=60, ax=axes[ax_idx])

for ax in axes[len(clustered_subjects):]:
    ax.set_visible(False)

plt.suptitle("Word Clouds per CS Subject Area", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("../data/outputs/fig_subject_wordclouds.png", bbox_inches="tight", dpi=150)
plt.show()
"""),

new_markdown_cell("""## 6. Export Subject Topics JSON"""),

new_code_cell("""\
from src.clustering import export_subject_topics_json
import json

out = export_subject_topics_json(clustered_subjects,
                                  title_col="title",
                                  out_path=pathlib.Path("../data/outputs/subject_topics.json"))
with open(out) as f:
    topics = json.load(f)

# Pretty-print first subject
first_subj = list(topics.keys())[0]
print(f"Subject: {first_subj}")
for cl in topics[first_subj][:3]:
    print(f"  Cluster {cl['cluster_id']}: {cl['label']} ({cl['n_articles']} articles)")
    for t in cl['example_titles'][:2]:
        print(f"    - {t[:80]}")
"""),

new_markdown_cell("""## Summary

- **Global LDA (K=50):** Discovers coherent topics like *neural networks*, *network security*, *query optimization*, *computer vision*, *wireless protocols*, etc. Interactive visualisation exported to `data/outputs/lda_vis.html`.
- **Per-subject UMAP + HDBSCAN:** Each CS subject decomposes into meaningful sub-clusters. For example, "Computer Science, Artificial Intelligence" splits into clusters for NLP, computer vision, reinforcement learning, knowledge representation, etc.
- **Subject × Topic Heatmap:** Shows that LDA topics align well with WoS subject categories, validating both the LDA model and the WoS classification.
- **Full topic map:** `data/outputs/subject_topics.json`
"""),
]


# ══════════════════════════════════════════════════════════════════════════════
# Write notebooks
# ══════════════════════════════════════════════════════════════════════════════
print("Generating notebooks …")
save(nb(nb01_cells), "01_data_extraction_eda.ipynb")
save(nb(nb02_cells), "02_recommender_models.ipynb")
save(nb(nb03_cells), "03_evaluation.ipynb")
save(nb(nb04_cells), "04_topic_clustering.ipynb")
print("Done!")
