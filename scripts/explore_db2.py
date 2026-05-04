import sqlite3, pandas as pd

con = sqlite3.connect('../CompSciencePub.sqlite')

# Check actual AcademicRecord IDs
print("=== ACADEMIC RECORD ID RANGE ===")
df = pd.read_sql("SELECT MIN(AcademicRecordID), MAX(AcademicRecordID), COUNT(*) FROM AcademicRecord", con)
print(df.to_string())

# Sample records
print("\n=== SAMPLE RECORDS ===")
df = pd.read_sql("""
    SELECT ar.AcademicRecordID, ar.Title, p.Name as JournalName, ar.PubYear, ar.DocumentTypeId
    FROM AcademicRecord ar
    JOIN Publication p ON ar.PublicationId = p.PublicationID
    LIMIT 5
""", con)
print(df.to_string())

# CS journals: filter by subjects
cs_subject_ids = '(454,455,457,461,464,467,468,470)'
print("\n=== CS ARTICLES (filtered by CS subjects) ===")
df = pd.read_sql(f"""
    SELECT COUNT(DISTINCT ar.AcademicRecordID) as cs_articles,
           COUNT(DISTINCT ar.PublicationId) as cs_journals
    FROM AcademicRecord ar
    JOIN AcademicRecordSubject ars ON ar.AcademicRecordID = ars.AcademicRecordId
    WHERE ars.AcademicSubjectId IN {cs_subject_ids}
""", con)
print(df.to_string())

# Articles-only filter (DocumentTypeId=1 for Article)
print("\n=== CS ARTICLES (type=Article only) ===")
df = pd.read_sql(f"""
    SELECT COUNT(DISTINCT ar.AcademicRecordID) as cs_articles,
           COUNT(DISTINCT ar.PublicationId) as cs_journals
    FROM AcademicRecord ar
    JOIN AcademicRecordSubject ars ON ar.AcademicRecordID = ars.AcademicRecordId
    WHERE ars.AcademicSubjectId IN {cs_subject_ids}
    AND ar.DocumentTypeId = 1
""", con)
print(df.to_string())

# Check if traditional subjects give 175 journals
print("\n=== TRADITIONAL CS SUBJECTS ONLY (Ascatype=traditional) ===")
df = pd.read_sql("""
    SELECT COUNT(DISTINCT ar.AcademicRecordID) as cs_articles,
           COUNT(DISTINCT ar.PublicationId) as cs_journals
    FROM AcademicRecord ar
    JOIN AcademicRecordSubject ars ON ar.AcademicRecordID = ars.AcademicRecordId
    JOIN AcademicSubject asub ON ars.AcademicSubjectId = asub.AcademicSubjectID
    WHERE asub.Ascatype = 'traditional' AND asub.Code IS NOT NULL
    AND asub.NameEn LIKE '%Computer Science%'
""", con)
print(df.to_string())

# All journals with article counts
print("\n=== ALL JOURNALS COUNT ===")
df = pd.read_sql("SELECT COUNT(DISTINCT PublicationId) as n_journals FROM AcademicRecord", con)
print(df.to_string())

# Sample abstract
print("\n=== SAMPLE ABSTRACT ===")
df = pd.read_sql("""
    SELECT ara.AbstractText FROM AcademicRecordAbstract ara LIMIT 1
""", con)
print(df['AbstractText'].iloc[0][:500])

con.close()
