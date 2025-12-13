import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from db.postgres import get_db_connection

def check():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = [row[0] for row in cur.fetchall()]
        print("Tables:", tables)
        if 'user' in tables and 'session' in tables:
            print("SUCCESS: User and Session tables found.")
        else:
            print("FAILURE: User/Session tables missing.")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check()
