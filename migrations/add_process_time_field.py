import sqlite3
import logging
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "drowsiness_detection.db"

def migrate():
    """Add process_time column to evidence_results table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if column exists first to avoid errors
            cursor.execute("PRAGMA table_info(evidence_results)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add column if it doesn't exist
            if 'process_time' not in columns:
                cursor.execute('''
                    ALTER TABLE evidence_results
                    ADD COLUMN process_time REAL DEFAULT NULL
                ''')
            
            conn.commit()
            logging.info("Successfully added process_time field to evidence_results table")
            
    except sqlite3.Error as e:
        logging.error(f"Migration error: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()