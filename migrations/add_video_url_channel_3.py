import sqlite3
import logging
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "drowsiness_detection.db"

def migrate():
    """Add video_url_channel_3 column to evidence_results table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Check if the column already exists
            cursor.execute("PRAGMA table_info(evidence_results)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'video_url_channel_3' not in columns:
                # Add the new column
                cursor.execute('''
                    ALTER TABLE evidence_results
                    ADD COLUMN video_url_channel_3 TEXT
                ''')

                conn.commit()
                logging.info("Successfully added video_url_channel_3 field to evidence_results table")
            else:
                logging.info("video_url_channel_3 column already exists, skipping")

    except sqlite3.Error as e:
        logging.error(f"Migration error: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()
