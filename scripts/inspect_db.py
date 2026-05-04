import sqlite3, sys
sys.path.insert(0, '..')

con = sqlite3.connect('../CompSciencePub.sqlite')
cur = con.cursor()

# List tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print("=== TABLES ===")
for t in tables:
    print(t)

print("\n=== ROW COUNTS ===")
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM [{t}]")
    print(f"{t}: {cur.fetchone()[0]}")

print("\n=== COLUMNS ===")
for t in tables:
    cur.execute(f"PRAGMA table_info([{t}])")
    cols = [r[1] for r in cur.fetchall()]
    print(f"{t}: {cols}")

con.close()
