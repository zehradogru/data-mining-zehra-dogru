import sys; sys.path.insert(0, 'C:/Users/zehra/Masaüstü/dm/journal-finder')
import sqlite3, pandas as pd
con = sqlite3.connect('C:/Users/zehra/Masaüstü/dm/CompSciencePub.sqlite')

# Check what io_db.py actually produces as subjects_raw
sql = """SELECT ars.AcademicRecordId AS record_id,
               GROUP_CONCAT(asub.NameEn, ' | ') AS subjects_raw
        FROM AcademicRecordSubject ars
        JOIN AcademicSubject asub ON ars.AcademicSubjectId = asub.AcademicSubjectID
        GROUP BY ars.AcademicRecordId LIMIT 5"""
df_s = pd.read_sql(sql, con)
print("subjects_raw repr:")
for _, row in df_s.iterrows():
    print(repr(row['subjects_raw']))

print()
# Now test the split
df_s['subjects'] = df_s['subjects_raw'].str.split(' | ')
print("After split:")
for _, row in df_s.iterrows():
    print(row['subjects'])

con.close()
