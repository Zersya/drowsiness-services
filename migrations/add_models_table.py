import sqlite3
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('migration.log')  # Save to file
    ]
)

DB_PATH = Path(__file__).parent.parent / "drowsiness_detection.db"

def migrate():
    """Create models table and add initial model from current configuration."""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if the table already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='models'")
        if not cursor.fetchone():
            logging.info("Creating models table...")
            
            # Create the models table
            cursor.execute('''
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 0
                )
            ''')
            
            # Get the current model path from environment or use default
            from dotenv import load_dotenv
            load_dotenv()
            current_model_path = os.getenv("YOLO_MODEL_PATH", "models/model_- 21 march 2025 0_28.pt")
            model_name = os.path.basename(current_model_path)
            
            # Insert the current model as the active one
            cursor.execute('''
                INSERT INTO models (name, file_path, is_active)
                VALUES (?, ?, 1)
            ''', (model_name, current_model_path))
            
            # Commit the changes
            conn.commit()
            
            logging.info(f"Migration completed successfully. Created models table and added current model '{model_name}'.")
        else:
            logging.info("models table already exists. No migration needed.")
        
        # Close the connection
        conn.close()
        
        return True
    except Exception as e:
        logging.error(f"Error during migration: {e}")
        return False

if __name__ == "__main__":
    migrate()
