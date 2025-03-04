import sqlite3
import logging

def migrate_database(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if column exists
            cursor.execute("PRAGMA table_info(evidence_results)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'normal_state_frames' not in columns:
                cursor.execute('''
                    ALTER TABLE evidence_results
                    ADD COLUMN normal_state_frames INTEGER
                ''')
                conn.commit()
                logging.info("Successfully added normal_state_frames column")
            else:
                logging.info("normal_state_frames column already exists")
                
    except sqlite3.Error as e:
        logging.error(f"Migration error: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate_database("drowsiness_detection.db")