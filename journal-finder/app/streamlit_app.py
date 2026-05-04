"""
streamlit_app.py
----------------
Interactive demo for the CS Journal Finder.

Run:
    streamlit run app/streamlit_app.py

Requires model artifacts from notebooks 01-02 to be present in models/.
"""

import json
import pathlib
import sys
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


st.set_page_config(
    page_title="CS Journal Finder",
    page_icon="CS",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title { font-size: 2.2rem; font-weight: 750; color: #1a1a2e; }
    .sub-title { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .rec-card {
        background: #f8f9fa;
        border-left: 4px solid #4a90d9;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .rec-rank { font-size: 1.2rem; font-weight: 800; color: #4a90d9; }
    .rec-name { font-size: 1.05rem; font-weight: 650; color: #1a1a2e; }
    .rec-issn { font-size: 0.82rem; color: #888; }
    .rec-score { font-size: 0.9rem; color: #444; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading recommender models...")
def load_models():
    from src.recommender import (
        HybridRecommender,
        LDARecommender,
        SBERTRecommender,
        TFIDFRecommender,
    )

    models = {}
    errors = []

    try:
        models["tfidf"] = TFIDFRecommender.load()
    except Exception as exc:
        errors.append(f"TF-IDF: {exc}")

    try:
        models["sbert"] = SBERTRecommender.load()
    except Exception as exc:
        errors.append(f"SBERT: {exc}")

    try:
        models["lda"] = LDARecommender.load()
    except Exception as exc:
        errors.append(f"LDA: {exc}")

    if {"tfidf", "sbert", "lda"}.issubset(models):
        models["hybrid"] = HybridRecommender(
            tfidf=models["tfidf"],
            sbert=models["sbert"],
            lda=models["lda"],
        )

    return models, errors


@st.cache_data(show_spinner=False)
def load_subject_topics():
    path = ROOT / "data" / "outputs" / "subject_topics.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_master_df():
    path = ROOT / "data" / "processed" / "processed_df.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


with st.sidebar:
    st.markdown("## Settings")
    method = st.selectbox(
        "Recommender method",
        [
            "TF-IDF + Cosine (Best in evaluation)",
            "Hybrid (RRF)",
            "SBERT + Cosine",
            "LDA + JS-Divergence",
        ],
    )
    top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
**CS Journal Finder** recommends Computer Science journals for an article abstract.

**Current evaluation**
- TF-IDF Top-5 accuracy: 61.85%
- Hybrid Top-5 accuracy: 53.44%
- SBERT Top-5 accuracy: 50.65%
- LDA Top-5 accuracy: 33.44%

**Dataset:** 20,944 CS articles, 410 journals, 2000-2018.
"""
    )


st.markdown('<p class="main-title">CS Journal Finder</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Enter an article abstract to find the top relevant '
    "Computer Science journals. The app includes TF-IDF, SBERT, LDA, and RRF fusion.</p>",
    unsafe_allow_html=True,
)

tab_find, tab_topics, tab_stats = st.tabs(
    ["Find Journals", "Topic Explorer", "Dataset Stats"]
)


with tab_find:
    models, load_errors = load_models()
    if load_errors:
        st.warning(
            "Some models could not be loaded. Run notebooks 01-02 first.\n\n"
            + "\n".join(load_errors)
        )

    abstract_input = st.text_area(
        "Paste your article abstract here:",
        height=200,
        placeholder="We propose a novel deep learning method for natural language processing...",
    )
    keywords_input = st.text_input(
        "Optional: author keywords (comma-separated)",
        placeholder="deep learning, NLP, transformer, text classification",
    )

    col_button, col_status = st.columns([1, 4])
    with col_button:
        find_btn = st.button("Find Journals", type="primary", use_container_width=True)
    with col_status:
        if not models:
            st.error("No models loaded. Run notebooks 01 and 02 first.")

    if find_btn and abstract_input.strip():
        from src.preprocessing import process_abstract_only

        with st.spinner("Analysing abstract..."):
            combined = abstract_input
            if keywords_input.strip():
                kws = keywords_input.strip()
                combined = f"{abstract_input} {kws} {kws} {kws}"
            query = process_abstract_only(combined)

        if not query.strip():
            st.error("The abstract is too short after preprocessing.")
        else:
            if method.startswith("TF-IDF") and "tfidf" in models:
                recs = models["tfidf"].recommend(query, top_k=top_k)
            elif method.startswith("Hybrid") and "hybrid" in models:
                recs = models["hybrid"].recommend(query, top_k=top_k)
            elif method.startswith("SBERT") and "sbert" in models:
                recs = models["sbert"].recommend(query, top_k=top_k)
            elif method.startswith("LDA") and "lda" in models:
                recs = models["lda"].recommend(query, top_k=top_k)
            else:
                recs = []
                st.error("Selected model is not available.")

            if recs:
                st.markdown(f"### Top {len(recs)} Recommended Journals")
                for result in recs:
                    rank = result["rank"]
                    score_label = (
                        f"RRF score: {result['score']:.5f}"
                        if method.startswith("Hybrid")
                        else f"Score: {result['score']:.4f}"
                    )

                    score_details = ""
                    if "score_tfidf" in result:
                        score_details = (
                            f"TF-IDF: {result.get('score_tfidf', 0):.3f} | "
                            f"SBERT: {result.get('score_sbert', 0):.3f} | "
                            f"LDA: {result.get('score_lda', 0):.3f}"
                        )

                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <span class="rec-rank">{rank}.</span>&nbsp;
                            <span class="rec-name">{result['journal_name']}</span><br>
                            <span class="rec-issn">ISSN: {result.get('journal_issn', 'N/A')}</span><br>
                            <span class="rec-score">{score_label}</span>
                            {"<br><span class='rec-score'>" + score_details + "</span>" if score_details else ""}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if "tfidf" in models:
                    top_terms = models["tfidf"].get_top_terms(query, top_n=12)
                    if top_terms:
                        st.markdown("**Key terms extracted from your abstract:**")
                        st.markdown("  ".join(f"`{term}`" for term in top_terms))

    elif find_btn:
        st.warning("Please enter an abstract first.")

    with st.expander("Try a sample abstract"):
        samples = {
            "Deep Learning / NLP": (
                "We present a novel attention-based neural architecture for machine "
                "translation. Our transformer model achieves state-of-the-art results "
                "using self-attention mechanisms."
            ),
            "Network Security": (
                "This paper proposes an intrusion detection system based on anomaly "
                "detection using statistical analysis of network traffic flows."
            ),
            "Distributed Systems": (
                "We describe a fault-tolerant distributed consensus protocol for "
                "large-scale cloud computing environments."
            ),
            "Computer Vision": (
                "We propose a convolutional neural network architecture for real-time "
                "object detection and segmentation."
            ),
            "Software Engineering": (
                "We present an automated program repair technique that uses genetic "
                "programming to generate patches for buggy programs."
            ),
        }
        chosen = st.selectbox("Select a sample:", list(samples))
        st.code(samples[chosen])


with tab_topics:
    st.markdown("### CS Topic Clusters by Subject Area")
    st.markdown(
        "The clusters were discovered with UMAP + HDBSCAN over SBERT embeddings. "
        "Labels are automatic TF-IDF summaries, so they should be read as exploratory."
    )

    subject_topics = load_subject_topics()
    if not subject_topics:
        st.info("Topic data not generated yet. Run notebook 04 first.")
    else:
        selected_subject = st.selectbox(
            "Select a CS subject area:", sorted(subject_topics.keys())
        )
        clusters = subject_topics.get(selected_subject, [])
        st.markdown(f"**{len(clusters)} sub-topic clusters found.**")

        for cluster in clusters:
            title = (
                f"Cluster {cluster['cluster_id']}: {cluster['label']} "
                f"({cluster['n_articles']} articles)"
            )
            with st.expander(title):
                st.markdown("**Example titles:**")
                for example_title in cluster.get("example_titles", []):
                    st.markdown(f"- {example_title}")


with tab_stats:
    st.markdown("### Dataset Overview")
    df_master = load_master_df()

    if df_master is None:
        st.info("Master DataFrame not found. Run notebook 01 first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Articles", f"{len(df_master):,}")
        col2.metric("Journals", f"{df_master['journal_id'].nunique():,}")
        col3.metric(
            "Year Range",
            f"{int(df_master['pub_year'].min())}-{int(df_master['pub_year'].max())}",
        )
        col4.metric("With Abstracts", f"{df_master['abstract'].notna().sum():,}")

        st.markdown("#### Articles per Journal (Top 20)")
        top20 = df_master.groupby("journal_name").size().sort_values(ascending=False).head(20)
        st.bar_chart(top20)

        st.markdown("#### Articles by Year")
        yearly = df_master["pub_year"].value_counts().sort_index()
        st.bar_chart(yearly)
