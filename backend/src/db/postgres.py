import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Establishes a connection to the Neon Postgres database."""
    try:
        conn = psycopg2.connect(os.getenv("NEON_DATABASE_URL"))
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise e
