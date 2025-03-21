import sqlite3
import logging
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "drowsiness_detection.db"

def migrate():
    """Add takeup_user column to evidence_results table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if column exists first to avoid errors
            cursor.execute("PRAGMA table_info(evidence_results)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add column if it doesn't exist
            if 'takeup_user' not in columns:
                cursor.execute('''
                    ALTER TABLE evidence_results
                    ADD COLUMN takeup_user TEXT DEFAULT NULL
                ''')
            
            conn.commit()
            logging.info("Successfully added takeup_user field to evidence_results table")
            
    except sqlite3.Error as e:
        logging.error(f"Migration error: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()
