"""
recommender.py
--------------
Four journal recommenders:
  A) TFIDFRecommender    — TF-IDF bigram + cosine similarity
  B) SBERTRecommender    — Sentence-BERT embeddings + cosine similarity
  C) LDARecommender      — LDA topic distributions + Jensen–Shannon divergence
  D) HybridRecommender   — Reciprocal Rank Fusion (RRF) of A + B + C

Each recommender exposes:
    .fit(df)           — trains on the master DataFrame
    .recommend(query, top_k) → list of dicts with journal info + score
    .save(path) / .load(path)
"""

from __future__ import annotations

import pathlib
import json
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# A) TF-IDF Recommender
# ══════════════════════════════════════════════════════════════════════════════
class TFIDFRecommender:
    """
    Represents each journal as the mean TF-IDF vector of all its articles.
    At query time, the input abstract is vectorised with the same vocabulary
    and the top-k journals are returned by cosine similarity.

    Why 1–2 grams? Unigrams catch single keywords; bigrams capture phrases like
    'machine learning', 'neural network', 'distributed system', etc. that are
    highly discriminative between CS sub-fields.
    """

    def __init__(self,
                 ngram_range=(1, 2),
                 min_df: int = 3,
                 max_df: float = 0.85,
                 max_features: int = 60_000,
                 sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.journal_vectors: Optional[np.ndarray] = None
        self.journal_index: Optional[pd.DataFrame] = None  # journal_id, journal_name, …

    def fit(self, df: pd.DataFrame,
            text_col: str = "processed_text",
            journal_id_col: str = "journal_id",
            journal_name_col: str = "journal_name",
            journal_issn_col: str = "journal_issn") -> "TFIDFRecommender":
        print("[TF-IDF] Fitting vectorizer …")
        doc_matrix = self.vectorizer.fit_transform(df[text_col].fillna(""))

        journal_ids   = df[journal_id_col].values
        unique_jids   = sorted(df[journal_id_col].unique())

        # Per-journal centroid (mean of article TF-IDF vectors)
        print("[TF-IDF] Computing per-journal centroids …")
        rows = []
        centroids = []
        for jid in unique_jids:
            mask = journal_ids == jid
            centroid = doc_matrix[mask].mean(axis=0)           # (1 × vocab)
            centroids.append(np.asarray(centroid).flatten())
            row = df.loc[df[journal_id_col] == jid, [journal_name_col, journal_issn_col]].iloc[0]
            rows.append({"journal_id": jid,
                         "journal_name": row[journal_name_col],
                         "journal_issn": row[journal_issn_col]})

        self.journal_vectors = np.vstack(centroids)           # (n_journals × vocab)
        self.journal_index   = pd.DataFrame(rows)
        print(f"[TF-IDF] Ready — {len(unique_jids)} journals, vocab={len(self.vectorizer.vocabulary_):,}")
        return self

    def recommend(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Returns top_k journal dicts with keys: journal_id, journal_name, journal_issn, score, rank."""
        qvec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(qvec, self.journal_vectors).flatten()
        idx  = np.argsort(sims)[::-1][:top_k]
        results = []
        for rank, i in enumerate(idx, 1):
            rec = self.journal_index.iloc[i].to_dict()
            rec["score"] = float(sims[i])
            rec["rank"]  = rank
            results.append(rec)
        return results

    def get_top_terms(self, query_text: str, top_n: int = 10) -> list[str]:
        """Returns the top TF-IDF terms from the query (for explanation)."""
        qvec   = self.vectorizer.transform([query_text])
        feat   = self.vectorizer.get_feature_names_out()
        scores = np.asarray(qvec.todense()).flatten()
        idx    = np.argsort(scores)[::-1][:top_n]
        return [feat[i] for i in idx if scores[i] > 0]

    def save(self, path: Optional[pathlib.Path] = None) -> None:
        path = path or MODELS_DIR / "tfidf_recommender.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[TF-IDF] Saved to {path}")

    @classmethod
    def load(cls, path: Optional[pathlib.Path] = None) -> "TFIDFRecommender":
        path = path or MODELS_DIR / "tfidf_recommender.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[TF-IDF] Loaded from {path}")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# B) SBERT Recommender
# ══════════════════════════════════════════════════════════════════════════════
class SBERTRecommender:
    """
    Encodes each article's rich document with a Sentence-BERT model and stores
    per-journal centroid embeddings. Query abstracts are encoded at runtime.

    Model choice — 'all-MiniLM-L6-v2':
        - 384-dimensional embeddings
        - ~22 M parameters, very fast on CPU
        - State-of-the-art performance on semantic similarity benchmarks
        - No GPU required (though GPU accelerates bulk encoding significantly)

    The key advantage over TF-IDF: semantically similar phrases (e.g.,
    'deep learning' and 'neural network', or 'data stream' and 'real-time
    processing') get similar embeddings even if they share no tokens.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name   = model_name or self.MODEL_NAME
        self._model       = None
        self.centroids:   Optional[np.ndarray] = None
        self.journal_index: Optional[pd.DataFrame] = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[SBERT] Loading model '{self.model_name}' …")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, df: pd.DataFrame,
            text_col: str = "processed_text",
            journal_id_col: str = "journal_id",
            journal_name_col: str = "journal_name",
            journal_issn_col: str = "journal_issn",
            batch_size: int = 64,
            cache_embeddings: bool = True) -> "SBERTRecommender":
        processed_dir = pathlib.Path(__file__).parent.parent / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        emb_cache = processed_dir / "sbert_embeddings.npy"

        model = self._get_model()

        if cache_embeddings and emb_cache.exists():
            print(f"[SBERT] Loading cached embeddings from {emb_cache}")
            all_embeddings = np.load(str(emb_cache))
        else:
            print(f"[SBERT] Encoding {len(df):,} documents (batch_size={batch_size}) …")
            texts = df[text_col].fillna("").tolist()
            all_embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2-norm → dot product = cosine
            )
            if cache_embeddings:
                np.save(str(emb_cache), all_embeddings)
                print(f"[SBERT] Embeddings saved to {emb_cache}")

        # Per-journal centroid
        journal_ids  = df[journal_id_col].values
        unique_jids  = sorted(df[journal_id_col].unique())
        rows, centroids = [], []
        for jid in unique_jids:
            mask     = journal_ids == jid
            centroid = all_embeddings[mask].mean(axis=0)
            norm     = np.linalg.norm(centroid)
            centroids.append(centroid / norm if norm > 0 else centroid)
            row = df.loc[df[journal_id_col] == jid,
                         [journal_name_col, journal_issn_col]].iloc[0]
            rows.append({"journal_id": jid,
                         "journal_name": row[journal_name_col],
                         "journal_issn": row[journal_issn_col]})

        self.centroids     = np.vstack(centroids)
        self.journal_index = pd.DataFrame(rows)
        print(f"[SBERT] Ready — {len(unique_jids)} journals, embed_dim={self.centroids.shape[1]}")
        return self

    def _encode_query(self, query_text: str) -> np.ndarray:
        model = self._get_model()
        vec   = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0]

    def recommend(self, query_text: str, top_k: int = 5) -> list[dict]:
        qvec = self._encode_query(query_text)
        sims = self.centroids @ qvec                  # dot product = cosine (normalised)
        idx  = np.argsort(sims)[::-1][:top_k]
        results = []
        for rank, i in enumerate(idx, 1):
            rec = self.journal_index.iloc[i].to_dict()
            rec["score"] = float(sims[i])
            rec["rank"]  = rank
            results.append(rec)
        return results

    def save(self, path: Optional[pathlib.Path] = None) -> None:
        # Don't pickle the live model (large); save centroids + index separately
        path = path or MODELS_DIR / "sbert_recommender.pkl"
        model_ref = self._model
        self._model = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._model = model_ref
        print(f"[SBERT] Saved to {path}")

    @classmethod
    def load(cls, path: Optional[pathlib.Path] = None) -> "SBERTRecommender":
        path = path or MODELS_DIR / "sbert_recommender.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[SBERT] Loaded from {path}")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# C) LDA Recommender
# ══════════════════════════════════════════════════════════════════════════════
class LDARecommender:
    """
    Fits an LDA topic model on the corpus. Each journal is represented as the
    mean topic distribution across its articles. Recommendation is by minimum
    Jensen–Shannon divergence between the query's topic distribution and each
    journal's mean distribution.

    Why LDA in addition to TF-IDF and SBERT?
        LDA captures *latent* topics — a document about 'security vulnerabilities
        in distributed cloud systems' will have high probability on both a
        'security' topic and a 'distributed systems' topic, which helps find
        journals covering either angle.

    The number of topics K is tuned by coherence score c_v.
    """

    def __init__(self, n_topics: int = 50, random_state: int = 42,
                 passes: int = 10, workers: int = 1):
        self.n_topics    = n_topics
        self.random_state = random_state
        self.passes      = passes
        self.workers     = workers
        self.lda_model   = None
        self.dictionary  = None
        self.corpus      = None
        self.journal_dists: Optional[np.ndarray] = None
        self.journal_index: Optional[pd.DataFrame] = None

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        return [t.split() for t in texts]

    def fit(self, df: pd.DataFrame,
            text_col: str = "processed_text",
            journal_id_col: str = "journal_id",
            journal_name_col: str = "journal_name",
            journal_issn_col: str = "journal_issn") -> "LDARecommender":
        import gensim
        from gensim import corpora, models

        print(f"[LDA] Building dictionary …")
        tokenized = self._tokenize(df[text_col].fillna("").tolist())
        self.dictionary = corpora.Dictionary(tokenized)
        self.dictionary.filter_extremes(no_below=5, no_above=0.85, keep_n=40_000)

        self.corpus = [self.dictionary.doc2bow(tokens) for tokens in tokenized]

        print(f"[LDA] Training LDA (K={self.n_topics}, passes={self.passes}) …")
        self.lda_model = models.LdaMulticore(
            self.corpus,
            num_topics=self.n_topics,
            id2word=self.dictionary,
            passes=self.passes,
            workers=self.workers,
            random_state=self.random_state,
            alpha="symmetric",
            eta="auto",
        )

        # Per-document topic distributions
        print("[LDA] Computing document topic distributions …")
        doc_dists = np.zeros((len(df), self.n_topics), dtype=np.float32)
        for i, bow in enumerate(self.corpus):
            for tid, prob in self.lda_model.get_document_topics(bow, minimum_probability=0.0):
                doc_dists[i, tid] = prob

        # Per-journal mean topic distribution
        journal_ids = df[journal_id_col].values
        unique_jids = sorted(df[journal_id_col].unique())
        rows, dists = [], []
        for jid in unique_jids:
            mask = journal_ids == jid
            dist = doc_dists[mask].mean(axis=0)
            dist = dist / (dist.sum() + 1e-12)   # renormalise
            dists.append(dist)
            row = df.loc[df[journal_id_col] == jid,
                         [journal_name_col, journal_issn_col]].iloc[0]
            rows.append({"journal_id": jid,
                         "journal_name": row[journal_name_col],
                         "journal_issn": row[journal_issn_col]})

        self.journal_dists = np.vstack(dists)
        self.journal_index = pd.DataFrame(rows)
        print(f"[LDA] Ready — {len(unique_jids)} journals, {self.n_topics} topics.")
        return self

    def _query_dist(self, query_text: str) -> np.ndarray:
        tokens = query_text.split()
        bow    = self.dictionary.doc2bow(tokens)
        topic_probs = self.lda_model.get_document_topics(bow, minimum_probability=0.0)
        dist = np.zeros(self.n_topics, dtype=np.float32)
        for tid, prob in topic_probs:
            dist[tid] = prob
        s = dist.sum()
        return dist / (s + 1e-12)

    def recommend(self, query_text: str, top_k: int = 5) -> list[dict]:
        qdist = self._query_dist(query_text)
        # JS divergence: lower = more similar → negate for ranking
        js_divs = np.array([
            jensenshannon(qdist, self.journal_dists[i])
            for i in range(len(self.journal_index))
        ])
        idx = np.argsort(js_divs)[:top_k]
        results = []
        for rank, i in enumerate(idx, 1):
            rec = self.journal_index.iloc[i].to_dict()
            rec["score"] = float(1.0 - js_divs[i])   # convert to similarity
            rec["rank"]  = rank
            results.append(rec)
        return results

    def get_topic_words(self, topic_id: int, top_n: int = 10) -> list[str]:
        return [w for w, _ in self.lda_model.show_topic(topic_id, topn=top_n)]

    def save(self, path: Optional[pathlib.Path] = None) -> None:
        path = path or MODELS_DIR / "lda_recommender.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[LDA] Saved to {path}")

    @classmethod
    def load(cls, path: Optional[pathlib.Path] = None) -> "LDARecommender":
        path = path or MODELS_DIR / "lda_recommender.pkl"
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[LDA] Loaded from {path}")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# D) Hybrid Recommender (Reciprocal Rank Fusion)
# ══════════════════════════════════════════════════════════════════════════════
class HybridRecommender:
    """
    Combines TF-IDF, SBERT, and LDA rankings using Reciprocal Rank Fusion (RRF).

    RRF score for journal j:
        RRF(j) = Σ  1 / (k + rank_m(j))
                  m ∈ {tfidf, sbert, lda}

    where k=60 is a smoothing constant (Cormack et al., 2009).

    Why RRF instead of score averaging?
        Scores from different models are on different scales (cosine vs.
        1 − JS-divergence). Rank-based fusion is scale-invariant and robust
        to one method returning a very confident but wrong answer.
    """

    RRF_K = 60

    def __init__(self,
                 tfidf: Optional[TFIDFRecommender] = None,
                 sbert: Optional[SBERTRecommender] = None,
                 lda:   Optional[LDARecommender]   = None):
        self.tfidf = tfidf
        self.sbert = sbert
        self.lda   = lda

    def _load_components(self) -> None:
        if self.tfidf is None:
            self.tfidf = TFIDFRecommender.load()
        if self.sbert is None:
            self.sbert = SBERTRecommender.load()
        if self.lda is None:
            self.lda = LDARecommender.load()

    def recommend(self, query_text: str, top_k: int = 5,
                  n_candidates: int = 50) -> list[dict]:
        """
        Fetches top-n_candidates from each sub-recommender, fuses by RRF,
        and returns the final top-k results.
        """
        self._load_components()

        # Get candidates from each method
        results_by_method = {
            "tfidf": self.tfidf.recommend(query_text, top_k=n_candidates),
            "sbert": self.sbert.recommend(query_text, top_k=n_candidates),
            "lda":   self.lda.recommend(  query_text, top_k=n_candidates),
        }

        # RRF fusion
        rrf_scores: dict[int, float] = {}
        journal_info: dict[int, dict] = {}

        for method, results in results_by_method.items():
            for item in results:
                jid = item["journal_id"]
                rrf_scores[jid] = rrf_scores.get(jid, 0.0) + 1.0 / (self.RRF_K + item["rank"])
                if jid not in journal_info:
                    journal_info[jid] = {k: v for k, v in item.items()
                                         if k not in ("score", "rank")}
                # Store individual method scores
                journal_info[jid][f"score_{method}"] = item["score"]

        # Sort by RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final = []
        for rank, (jid, rrf_score) in enumerate(ranked, 1):
            rec = journal_info[jid].copy()
            rec["score"]     = rrf_score
            rec["rank"]      = rank
            final.append(rec)
        return final

    def save(self, path: Optional[pathlib.Path] = None) -> None:
        # Save components separately (SBERT model weights are not pickleable)
        if self.tfidf: self.tfidf.save()
        if self.sbert: self.sbert.save()
        if self.lda:   self.lda.save()
        print("[Hybrid] All components saved.")

    @classmethod
    def load(cls) -> "HybridRecommender":
        return cls(
            tfidf=TFIDFRecommender.load(),
            sbert=SBERTRecommender.load(),
            lda=LDARecommender.load(),
        )


# ── Convenience function ───────────────────────────────────────────────────────
def load_hybrid() -> HybridRecommender:
    """Loads the pre-trained hybrid recommender. Run notebooks 01–02 first."""
    return HybridRecommender.load()
