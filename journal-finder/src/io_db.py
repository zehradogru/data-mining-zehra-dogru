"""
io_db.py
--------
Loads data from CompSciencePub.sqlite and exports to Parquet files.
All subsequent notebooks use the Parquet files for reproducibility —
no SQLite connection needed at inference time.
"""

import sqlite3
import os
import pathlib
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR = pathlib.Path(__file__).parent
DB_PATH = THIS_DIR.parent.parent / "CompSciencePub.sqlite"   # ../CompSciencePub.sqlite
RAW_DIR = THIS_DIR.parent / "data" / "raw"

CS_SUBJECT_IDS = (454, 455, 457, 461, 464, 467, 468, 470)   # all CS-related subjects


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return sqlite3.connect(str(DB_PATH))


def load_master_df(save_parquet: bool = True) -> pd.DataFrame:
    """
    Joins all relevant tables and returns a single master DataFrame with columns:
        record_id, journal_id, journal_name, journal_issn,
        title, abstract, pub_year,
        keywords (list), keywords_plus (list), subjects (list),
        rich_doc (concatenated text for modelling)

    Only articles that have at least one CS subject are included.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / "master.parquet"
    if cached.exists():
        print(f"[io_db] Loading cached master DataFrame from {cached}")
        return pd.read_parquet(cached)

    print("[io_db] Building master DataFrame from SQLite …")
    con = _connect()

    # ── Core articles + journal ────────────────────────────────────────────────
    sql_records = """
        SELECT
            ar.AcademicRecordID        AS record_id,
            ar.PublicationId           AS journal_id,
            ar.Title                   AS title,
            ar.PubYear                 AS pub_year,
            ar.CiteCount               AS cite_count,
            ar.ImpactFactor            AS impact_factor,
            p.Name                     AS journal_name,
            p.ISSN                     AS journal_issn,
            p.Abbreviation             AS journal_abbr
        FROM AcademicRecord ar
        JOIN Publication p ON ar.PublicationId = p.PublicationID
    """
    df_records = pd.read_sql(sql_records, con)

    # ── Abstracts ──────────────────────────────────────────────────────────────
    sql_abstracts = """
        SELECT AcademicRecordId AS record_id, AbstractText AS abstract
        FROM AcademicRecordAbstract
        WHERE AbstractText IS NOT NULL AND AbstractText != ''
    """
    df_abstracts = pd.read_sql(sql_abstracts, con)

    # ── Author keywords ────────────────────────────────────────────────────────
    sql_keywords = """
        SELECT ark.AcademicRecordId AS record_id,
               GROUP_CONCAT(ak.Name, ' | ') AS keywords_raw
        FROM AcademicRecordKeyword ark
        JOIN AcademicKeyword ak ON ark.AcademicKeywordId = ak.AcademicKeywordID
        GROUP BY ark.AcademicRecordId
    """
    df_keywords = pd.read_sql(sql_keywords, con)

    # ── WoS KeywordsPlus ───────────────────────────────────────────────────────
    sql_kwplus = """
        SELECT arkp.AcademicRecordId AS record_id,
               GROUP_CONCAT(akp.Name, ' | ') AS keywords_plus_raw
        FROM AcademicRecordKeywordPlus arkp
        JOIN AcademicKeywordPlus akp ON arkp.AcademicKeywordPlusId = akp.AcademicKeywordPlusID
        GROUP BY arkp.AcademicRecordId
    """
    df_kwplus = pd.read_sql(sql_kwplus, con)

    # ── WoS Subjects ──────────────────────────────────────────────────────────
    sql_subjects = """
        SELECT ars.AcademicRecordId AS record_id,
               GROUP_CONCAT(asub.NameEn, ' | ') AS subjects_raw
        FROM AcademicRecordSubject ars
        JOIN AcademicSubject asub ON ars.AcademicSubjectId = asub.AcademicSubjectID
        GROUP BY ars.AcademicRecordId
    """
    df_subjects = pd.read_sql(sql_subjects, con)

    # ── CS filter: records that have at least one CS subject ──────────────────
    cs_ids_str = ",".join(str(i) for i in CS_SUBJECT_IDS)
    sql_cs_ids = f"""
        SELECT DISTINCT AcademicRecordId AS record_id
        FROM AcademicRecordSubject
        WHERE AcademicSubjectId IN ({cs_ids_str})
    """
    cs_record_ids = pd.read_sql(sql_cs_ids, con)["record_id"]

    con.close()

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = (
        df_records
        .merge(df_abstracts, on="record_id", how="inner")   # must have abstract
        .merge(df_keywords,  on="record_id", how="left")
        .merge(df_kwplus,    on="record_id", how="left")
        .merge(df_subjects,  on="record_id", how="left")
    )

    # Filter to CS articles only
    df = df[df["record_id"].isin(cs_record_ids)].copy()
    df.reset_index(drop=True, inplace=True)

    # Parse list columns  (regex=False: the pipe in " | " is literal, not regex OR)
    df["keywords"]      = df["keywords_raw"].str.split(" | ", regex=False)
    df["keywords_plus"] = df["keywords_plus_raw"].str.split(" | ", regex=False)
    df["subjects"]      = df["subjects_raw"].str.split(" | ", regex=False)
    df.drop(columns=["keywords_raw", "keywords_plus_raw", "subjects_raw"], inplace=True)

    # Fill NaN lists
    for col in ["keywords", "keywords_plus", "subjects"]:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    print(f"[io_db] Loaded {len(df):,} articles from {df['journal_id'].nunique():,} journals.")

    if save_parquet:
        df.to_parquet(cached, index=False)
        print(f"[io_db] Saved to {cached}")

    return df


def load_journals() -> pd.DataFrame:
    """Returns a DataFrame of all publications/journals in the database."""
    con = _connect()
    df = pd.read_sql("SELECT PublicationID AS journal_id, Name AS journal_name, ISSN AS journal_issn, Abbreviation AS journal_abbr FROM Publication", con)
    con.close()
    return df


if __name__ == "__main__":
    df = load_master_df(save_parquet=True)
    print(df.head(3).to_string())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Articles per journal (top 10):\n{df.groupby('journal_name').size().sort_values(ascending=False).head(10)}")
