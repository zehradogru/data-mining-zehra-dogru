"""
clustering.py
-------------
Topic clustering utilities:
  1. LDA coherence sweep (to find optimal K)
  2. pyLDAvis visualisation export
  3. Per-subject UMAP + HDBSCAN sub-topic clustering
  4. Word-cloud generation
  5. Subject × topic heatmap
"""

from __future__ import annotations

import json
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

MODELS_DIR   = pathlib.Path(__file__).parent.parent / "models"
PROC_DIR     = pathlib.Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR   = pathlib.Path(__file__).parent.parent / "data" / "outputs"
for _d in (MODELS_DIR, PROC_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LDA coherence sweep
# ══════════════════════════════════════════════════════════════════════════════
def sweep_lda_coherence(tokenized_texts: list[list[str]],
                        k_range=(20, 30, 40, 50, 70, 80),
                        passes: int = 5,
                        workers: int = 1,
                        random_state: int = 42) -> pd.DataFrame:
    """
    Trains LDA for each K in k_range and returns a DataFrame with
    (K, c_v_coherence) so we can pick the best K by the elbow.
    """
    from gensim import corpora, models
    from gensim.models.coherencemodel import CoherenceModel

    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.85, keep_n=40_000)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

    rows = []
    for k in k_range:
        print(f"  [Coherence] K={k} …")
        lda = models.LdaMulticore(
            corpus,
            num_topics=k,
            id2word=dictionary,
            passes=passes,
            workers=workers,
            random_state=random_state,
        )
        cm = CoherenceModel(model=lda, texts=tokenized_texts,
                            dictionary=dictionary, coherence="c_v")
        coh = cm.get_coherence()
        rows.append({"K": k, "coherence_cv": coh})
        print(f"  [Coherence] K={k}  c_v={coh:.4f}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 2. pyLDAvis export
# ══════════════════════════════════════════════════════════════════════════════
def save_pyldavis(lda_model, corpus, dictionary,
                  out_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Exports an interactive pyLDAvis HTML to out_path."""
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis

    out_path = out_path or OUTPUT_DIR / "lda_vis.html"
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(vis_data, str(out_path))
    print(f"[pyLDAvis] Saved to {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 3. Per-subject UMAP + HDBSCAN clustering
# ══════════════════════════════════════════════════════════════════════════════
def cluster_subject(df_subject: pd.DataFrame,
                    embeddings: np.ndarray,
                    subject_name: str,
                    min_cluster_size: int = 10,
                    umap_n_components: int = 5,
                    umap_n_neighbors: int = 15,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Reduces SBERT embeddings with UMAP and clusters with HDBSCAN for a
    single subject's articles.

    Returns the input DataFrame with columns added:
        umap_x, umap_y  (2-D for plotting)
        cluster_id      (-1 = noise)
        cluster_label   (top TF-IDF terms of the cluster)
    """
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(df_subject) < min_cluster_size * 2:
        df_subject = df_subject.copy()
        df_subject["umap_x"] = 0.0
        df_subject["umap_y"] = 0.0
        df_subject["cluster_id"]    = -1
        df_subject["cluster_label"] = "too few articles"
        return df_subject

    print(f"  [UMAP+HDBSCAN] '{subject_name}': {len(df_subject)} articles …")

    # Reduce to 5-D for clustering, 2-D for plotting
    reducer_5d = umap.UMAP(n_components=umap_n_components,
                           n_neighbors=umap_n_neighbors,
                           metric="cosine",
                           random_state=random_state,
                           low_memory=False)
    emb_5d = reducer_5d.fit_transform(embeddings)

    reducer_2d = umap.UMAP(n_components=2,
                           n_neighbors=umap_n_neighbors,
                           metric="cosine",
                           random_state=random_state,
                           low_memory=False)
    emb_2d = reducer_2d.fit_transform(embeddings)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(emb_5d)

    df_out = df_subject.copy()
    df_out["umap_x"]     = emb_2d[:, 0]
    df_out["umap_y"]     = emb_2d[:, 1]
    df_out["cluster_id"] = labels

    # Label each cluster with its top TF-IDF terms
    cluster_labels = {-1: "noise"}
    unique_clusters = sorted(set(labels) - {-1})
    if unique_clusters and "processed_text" in df_out.columns:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5_000)
        for cid in unique_clusters:
            mask  = labels == cid
            texts = df_out.loc[mask, "processed_text"].fillna("").tolist()
            if not texts:
                cluster_labels[cid] = f"cluster_{cid}"
                continue
            try:
                X      = vec.fit_transform(texts)
                scores = np.asarray(X.mean(axis=0)).flatten()
                top_i  = np.argsort(scores)[::-1][:5]
                terms  = vec.get_feature_names_out()[top_i]
                cluster_labels[cid] = ", ".join(terms)
            except Exception:
                cluster_labels[cid] = f"cluster_{cid}"

    df_out["cluster_label"] = df_out["cluster_id"].map(cluster_labels).fillna("unknown")
    n_clusters = len(unique_clusters)
    noise_pct  = (labels == -1).mean() * 100
    print(f"  [UMAP+HDBSCAN] '{subject_name}': {n_clusters} clusters, noise={noise_pct:.1f}%")
    return df_out


def cluster_all_subjects(df: pd.DataFrame,
                         embeddings: np.ndarray,
                         subjects_col: str = "subjects",
                         **kwargs) -> dict[str, pd.DataFrame]:
    """
    Runs cluster_subject for each unique WoS subject in the corpus.
    Returns a dict: subject_name → clustered DataFrame.
    """
    # Explode subjects
    df_exp = df.copy()
    df_exp["subject_single"] = df_exp[subjects_col].apply(
        lambda x: x if isinstance(x, (list, np.ndarray)) else []
    )
    df_exp = df_exp.explode("subject_single").dropna(subset=["subject_single"])
    df_exp = df_exp[df_exp["subject_single"].str.strip() != ""]

    unique_subjects = df_exp["subject_single"].unique()
    print(f"[Clustering] Found {len(unique_subjects)} subjects.")

    results = {}
    for subj in sorted(unique_subjects):
        idx  = df_exp[df_exp["subject_single"] == subj].index
        # Map back to original df positions
        orig_mask = df.index.isin(idx)
        sub_df    = df[orig_mask].copy()
        sub_embs  = embeddings[orig_mask]
        clustered = cluster_subject(sub_df, sub_embs, subj, **kwargs)
        results[subj] = clustered

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. Word-cloud generation
# ══════════════════════════════════════════════════════════════════════════════
def plot_wordcloud(text: str,
                  title: str = "",
                  max_words: int = 80,
                  ax: Optional[plt.Axes] = None,
                  save_path: Optional[pathlib.Path] = None) -> None:
    from wordcloud import WordCloud
    wc = WordCloud(width=800, height=400, background_color="white",
                   max_words=max_words, colormap="viridis").generate(text)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Subject × topic heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_subject_topic_heatmap(df: pd.DataFrame,
                               lda_model,
                               dictionary,
                               text_col: str = "processed_text",
                               subjects_col: str = "subjects",
                               top_n_subjects: int = 20,
                               top_n_topics: int = 15,
                               figsize=(18, 10),
                               save_path: Optional[pathlib.Path] = None) -> None:
    """
    Computes per-subject mean topic distributions and plots a heatmap.
    Rows = top subjects by article count; columns = top active topics.
    """
    # Build doc-topic matrix
    tokenized = [t.split() for t in df[text_col].fillna("").tolist()]
    corpus    = [dictionary.doc2bow(tok) for tok in tokenized]
    n_topics  = lda_model.num_topics
    doc_dists = np.zeros((len(df), n_topics), dtype=np.float32)
    for i, bow in enumerate(corpus):
        for tid, prob in lda_model.get_document_topics(bow, minimum_probability=0.0):
            doc_dists[i, tid] = prob

    # Explode subjects and compute means
    df2 = df.copy()
    df2["_docidx"] = range(len(df2))
    df2 = df2.explode(subjects_col).dropna(subset=[subjects_col])
    df2 = df2[df2[subjects_col].str.strip() != ""]

    subject_counts = df2[subjects_col].value_counts()
    top_subjects   = subject_counts.head(top_n_subjects).index.tolist()

    subj_topic_matrix = []
    for subj in top_subjects:
        idx  = df2[df2[subjects_col] == subj]["_docidx"].values
        mean = doc_dists[idx].mean(axis=0)
        subj_topic_matrix.append(mean)

    heat = np.vstack(subj_topic_matrix)   # (n_subjects × n_topics)

    # Select top_n_topics by max activation across subjects
    top_topic_idx = np.argsort(heat.max(axis=0))[::-1][:top_n_topics]
    heat_sub      = heat[:, top_topic_idx]

    # Topic labels = top 3 words
    topic_labels = []
    for tid in top_topic_idx:
        words = [w for w, _ in lda_model.show_topic(tid, topn=3)]
        topic_labels.append(f"T{tid}: {', '.join(words)}")

    # Shorten subject names
    short_subjects = [s[:40] for s in top_subjects]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heat_sub, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(top_n_topics))
    ax.set_xticklabels(topic_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(top_n_subjects))
    ax.set_yticklabels(short_subjects, fontsize=9)
    ax.set_title("Subject × Topic Heatmap (mean LDA topic probability)", fontsize=13, pad=12)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[Heatmap] Saved to {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Export subject topics JSON
# ══════════════════════════════════════════════════════════════════════════════
def export_subject_topics_json(clustered_subjects: dict[str, pd.DataFrame],
                               title_col: str = "title",
                               out_path: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Saves a JSON mapping: subject → list of {cluster_id, label, n_articles, example_titles}.
    """
    out_path = out_path or OUTPUT_DIR / "subject_topics.json"
    result = {}
    for subj, df_c in clustered_subjects.items():
        clusters = []
        for cid, grp in df_c.groupby("cluster_id"):
            if cid == -1:
                continue
            examples = grp[title_col].dropna().head(3).tolist()
            clusters.append({
                "cluster_id":    int(cid),
                "label":         grp["cluster_label"].iloc[0],
                "n_articles":    len(grp),
                "example_titles": examples,
            })
        result[subj] = sorted(clusters, key=lambda x: x["n_articles"], reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[JSON] Subject topics saved to {out_path}")
    return out_path
