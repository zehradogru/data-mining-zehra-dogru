import sqlite3
con = sqlite3.connect('C:/Users/zehra/Masaüstü/dm/CompSciencePub.sqlite')
cur = con.cursor()
rows = cur.execute('SELECT AcademicSubjectID, NameEn FROM AcademicSubject LIMIT 30').fetchall()
print("=== AcademicSubject.NameEn ===")
for r in rows:
    print(r)
sample = cur.execute("""
    SELECT ars.AcademicRecordId, GROUP_CONCAT(asub.NameEn, ' | ') as subjects_raw
    FROM AcademicRecordSubject ars
    JOIN AcademicSubject asub ON ars.AcademicSubjectId = asub.AcademicSubjectID
    GROUP BY ars.AcademicRecordId LIMIT 5
""").fetchall()
print("\n=== Sample subjects_raw ===")
for r in sample:
    print(r)
con.close()
