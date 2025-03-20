import sqlite3
import logging
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('migration.log')  # Save to file
    ]
)

# Load environment variables to get the model name
load_dotenv()
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
MODEL_NAME = os.path.basename(YOLO_MODEL_PATH) if YOLO_MODEL_PATH else "unknown"

def migrate_database(db_path="drowsiness_detection.db"):
    """Add model_name column to the evidence_results table."""
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(evidence_results)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "model_name" not in columns:
            logging.info("Adding model_name column to evidence_results table...")
            
            # Add the column
            cursor.execute("ALTER TABLE evidence_results ADD COLUMN model_name TEXT")
            
            # Update existing records with the current model name
            cursor.execute("UPDATE evidence_results SET model_name = ? WHERE processing_status = 'processed'", (MODEL_NAME,))
            
            # Commit the changes
            conn.commit()
            
            logging.info(f"Migration completed successfully. Added model_name column and set value to '{MODEL_NAME}' for processed records.")
        else:
            logging.info("model_name column already exists. No migration needed.")
        
        # Close the connection
        conn.close()
        
        return True
    except Exception as e:
        logging.error(f"Error during migration: {e}")
        return False

if __name__ == "__main__":
    migrate_database()
