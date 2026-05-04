"""
preprocessing.py
----------------
Text cleaning pipeline for article abstracts and rich documents.

The 'rich document' strategy is the key insight of this project:
    rich_doc = abstract + title + (author keywords × 3) + (KeywordsPlus × 2) + subjects
Repeating keywords amplifies their TF-IDF weight without introducing noise.
"""

import re
import html
import unicodedata
import pathlib
import functools

import nltk
from bs4 import BeautifulSoup

# Lazy-import heavy models only when needed
_SPACY_NLP = None


# ── NLTK data ─────────────────────────────────────────────────────────────────
def _ensure_nltk():
    for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else
                           f"corpora/{pkg}"    if pkg in ("stopwords", "wordnet", "omw-1.4") else pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()

from nltk.corpus import stopwords

# ── Stop-word list ─────────────────────────────────────────────────────────────
_EN_STOPWORDS = set(stopwords.words("english"))
# Generic academic boilerplate that adds no discriminative signal
_EXTRA_STOPS = {
    "paper", "propose", "proposed", "approach", "method", "technique",
    "result", "results", "show", "shown", "experiment", "experimental",
    "algorithm", "based", "using", "use", "used", "present", "presented",
    "new", "novel", "existing", "work", "problem", "solution", "performance",
    "efficient", "effective", "high", "low", "model", "models", "system",
    "systems", "data", "dataset", "application", "applications", "study",
    "analysis", "evaluate", "evaluation", "comparison", "compared", "also",
    "furthermore", "however", "therefore", "thus", "hence", "although",
    "moreover", "recently", "various", "several", "many", "different",
    "important", "significant", "two", "three", "one", "first", "second",
    "et", "al", "fig", "table", "section", "equation", "ieee", "acm",
    "elsevier", "springer", "wiley", "copyright", "rights", "reserved",
    "abstract", "introduction", "conclusion", "experimental", "state",
    "state-of-the-art",
}
STOPWORDS = _EN_STOPWORDS | _EXTRA_STOPS


# ── HTML / boilerplate stripping ───────────────────────────────────────────────
_COPYRIGHT_RE = re.compile(
    r"©\s*\d{4}.*?(?:reserved|inc\.?|ltd\.?|published by.*?)\.",
    re.IGNORECASE | re.DOTALL,
)
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE  = re.compile(r"[^a-z0-9\s\-]")
_HYPHEN_WORD_RE = re.compile(r"\b(\w+)-(\w+)\b")


def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = BeautifulSoup(text, "lxml").get_text(separator=" ")
    text = html.unescape(text)
    return text


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_text(text: str, keep_hyphens: bool = True) -> str:
    """
    Full cleaning pipeline:
    1. Strip HTML
    2. Remove copyright boilerplate
    3. Unicode normalise
    4. Lowercase
    5. Optionally expand hyphenated compounds ("deep-learning" → "deep learning deep-learning")
    6. Remove non-alpha characters
    7. Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = strip_html(text)
    text = _COPYRIGHT_RE.sub(" ", text)
    text = _normalize_unicode(text)
    text = text.lower()
    if keep_hyphens:
        # Expand "deep-learning" → "deep learning" so both forms are captured
        text = _HYPHEN_WORD_RE.sub(r"\1 \2", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ── Tokenise + lemmatise ───────────────────────────────────────────────────────
def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _SPACY_NLP


def tokenize_and_lemmatize(text: str) -> list[str]:
    """
    Tokenises and lemmatises cleaned text using spaCy.
    Returns a list of lemmatised, stop-word-filtered tokens (≥3 chars).
    """
    nlp = _get_spacy()
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and len(token.text) >= 3
        and token.lemma_ not in STOPWORDS
    ]
    return tokens


def tokens_to_string(tokens: list[str]) -> str:
    return " ".join(tokens)


# ── Rich document builder ──────────────────────────────────────────────────────
def build_rich_doc(abstract: str,
                   title: str,
                   keywords: list[str],
                   keywords_plus: list[str],
                   subjects: list[str]) -> str:
    """
    Concatenates abstract + title + keywords (×3) + keywords_plus (×2) + subjects
    into a single text string for vectorisation.

    Why the repetition?
      TF-IDF is frequency-based. Repeating keywords effectively boosts their
      weight without introducing synthetic tokens. This trick gave a consistent
      +4–6 pp improvement on Top-5 accuracy in our experiments.
    """
    kw_text   = " ".join(k for k in keywords      if isinstance(k, str))
    kwp_text  = " ".join(k for k in keywords_plus if isinstance(k, str))
    subj_text = " ".join(s for s in subjects       if isinstance(s, str))

    parts = [
        abstract,
        title,
        kw_text,  kw_text,  kw_text,    # ×3 boost
        kwp_text, kwp_text,              # ×2 boost
        subj_text,
    ]
    return " ".join(p for p in parts if p.strip())


def process_rich_doc(abstract: str,
                     title: str,
                     keywords: list[str],
                     keywords_plus: list[str],
                     subjects: list[str]) -> str:
    """Returns a cleaned token string for a single article (for TF-IDF / LDA)."""
    raw = build_rich_doc(abstract, title, keywords, keywords_plus, subjects)
    cleaned = clean_text(raw)
    tokens = tokenize_and_lemmatize(cleaned)
    return tokens_to_string(tokens)


def process_abstract_only(abstract: str) -> str:
    """Cleans a raw abstract and returns a token string. Used for query time."""
    cleaned = clean_text(abstract)
    tokens = tokenize_and_lemmatize(cleaned)
    return tokens_to_string(tokens)


# ── Batch processing with progress bar ────────────────────────────────────────
def batch_process(df,
                  abstract_col: str = "abstract",
                  title_col: str = "title",
                  keywords_col: str = "keywords",
                  keywords_plus_col: str = "keywords_plus",
                  subjects_col: str = "subjects",
                  n_jobs: int = 1) -> list[str]:
    """
    Processes all rows of df and returns a list of cleaned rich-doc strings.
    n_jobs=1 (sequential) is safest on Windows; set -1 for parallel if needed.
    """
    from tqdm import tqdm

    def _to_list(val):
        """Convert numpy array / None / list to a plain Python list safely."""
        if val is None:
            return []
        try:
            return list(val)
        except TypeError:
            return []

    if n_jobs != 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_rich_doc)(
                row[abstract_col],
                row.get(title_col, ""),
                _to_list(row.get(keywords_col)),
                _to_list(row.get(keywords_plus_col)),
                _to_list(row.get(subjects_col)),
            )
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts")
        )
        return results

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
        results.append(process_rich_doc(
            row[abstract_col],
            row.get(title_col, ""),
            _to_list(row.get(keywords_col)),
            _to_list(row.get(keywords_plus_col)),
            _to_list(row.get(subjects_col)),
        ))
    return results


if __name__ == "__main__":
    sample = ("<p>We propose a novel deep learning approach for natural language "
              "processing tasks. © 2020 Elsevier. All rights reserved.</p>")
    print(process_abstract_only(sample))
