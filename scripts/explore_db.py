import sqlite3, pandas as pd

con = sqlite3.connect('../CompSciencePub.sqlite')

# Sample some publications
print("=== SAMPLE PUBLICATIONS ===")
df = pd.read_sql("SELECT * FROM Publication LIMIT 10", con)
print(df.to_string())

# Check subjects
print("\n=== ACADEMIC SUBJECTS (first 30) ===")
df = pd.read_sql("SELECT * FROM AcademicSubject ORDER BY NameEn LIMIT 30", con)
print(df.to_string())

# Count CS subjects
print("\n=== CS-RELATED SUBJECTS ===")
df = pd.read_sql("SELECT * FROM AcademicSubject WHERE NameEn LIKE '%COMPUTER%' OR NameEn LIKE '%SOFTWARE%' OR NameEn LIKE '%ARTIFICIAL%' OR NameEn LIKE '%MACHINE%'", con)
print(df.to_string())

# Check database editions
print("\n=== DATABASE EDITIONS ===")
df = pd.read_sql("SELECT * FROM DatabaseEdition", con)
print(df.to_string())

# Check document types
print("\n=== DOCUMENT TYPES ===")
df = pd.read_sql("SELECT * FROM DocumentType", con)
print(df.to_string())

# Check AcademicRecord sample with extra columns
print("\n=== SAMPLE ACADEMIC RECORDS (key columns) ===")
df = pd.read_sql("""
    SELECT ar.AcademicRecordID, ar.Title, p.Name as JournalName, 
           ar.PubYear, ar.ImpactFactor, ar.Percentile, ar.CiteCount, ar.QValue,
           dt.NameEn as DocType
    FROM AcademicRecord ar
    JOIN Publication p ON ar.PublicationId = p.PublicationID
    JOIN DocumentType dt ON ar.DocumentTypeId = dt.DocumentTypeID
    WHERE ar.AcademicRecordID <= 10
""", con)
print(df.to_string())

# Articles per journal - find top CS journals
print("\n=== ARTICLES PER JOURNAL (top 20) ===")
df = pd.read_sql("""
    SELECT p.Name, COUNT(*) as ArticleCount
    FROM AcademicRecord ar
    JOIN Publication p ON ar.PublicationId = p.PublicationID
    GROUP BY p.PublicationID, p.Name
    ORDER BY ArticleCount DESC
    LIMIT 20
""", con)
print(df.to_string())

# How many articles have abstracts?
print("\n=== ABSTRACT COVERAGE ===")
df = pd.read_sql("""
    SELECT COUNT(*) as with_abstract
    FROM AcademicRecord ar
    JOIN AcademicRecordAbstract ara ON ar.AcademicRecordID = ara.AcademicRecordId
    WHERE ara.AbstractText IS NOT NULL AND ara.AbstractText != ''
""", con)
print(df.to_string())

# PubYear distribution
print("\n=== YEAR DISTRIBUTION ===")
df = pd.read_sql("SELECT PubYear, COUNT(*) as cnt FROM AcademicRecord GROUP BY PubYear ORDER BY PubYear DESC LIMIT 15", con)
print(df.to_string())

con.close()
