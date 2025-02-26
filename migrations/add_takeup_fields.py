import sqlite3
import logging
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "drowsiness_detection.db"

def migrate():
    """Add takeup_memo, takeup_time, and alarm_time columns to evidence_results table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if columns exist first to avoid errors
            cursor.execute("PRAGMA table_info(evidence_results)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add columns if they don't exist
            if 'takeup_memo' not in columns:
                cursor.execute('''
                    ALTER TABLE evidence_results
                    ADD COLUMN takeup_memo TEXT
                ''')
            
            if 'takeup_time' not in columns:
                cursor.execute('''
                    ALTER TABLE evidence_results
                    ADD COLUMN takeup_time TIMESTAMP
                ''')
            
            if 'alarm_time' not in columns:
                cursor.execute('''
                    ALTER TABLE evidence_results
                    ADD COLUMN alarm_time TIMESTAMP
                ''')
            
            conn.commit()
            logging.info("Successfully added takeup fields to evidence_results table")
            
    except sqlite3.Error as e:
        logging.error(f"Migration error: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()