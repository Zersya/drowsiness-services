import sqlite3
import logging
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = os.getenv("LANDMARK_DB_PATH", "landmark_detection.db")

class LandmarkDatabaseManager:
    """Manages SQLite database operations for the landmark-based drowsiness detection system."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database and create necessary tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create evidence_results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS evidence_results (
                        id INTEGER PRIMARY KEY,
                        video_url TEXT,
                        process_time REAL,
                        yawn_frames INTEGER,
                        eye_closed_frames INTEGER,
                        max_consecutive_eye_closed INTEGER,
                        normal_state_frames INTEGER,
                        total_frames INTEGER,
                        is_drowsy BOOLEAN,
                        confidence REAL,
                        is_head_turned BOOLEAN,
                        is_head_down BOOLEAN,
                        processing_status TEXT,
                        details TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create processing_queue table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processing_queue (
                        id INTEGER PRIMARY KEY,
                        video_url TEXT,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create webhooks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS webhooks (
                        id INTEGER PRIMARY KEY,
                        url TEXT NOT NULL,
                        is_enabled BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
                logging.info("Landmark database initialized successfully")
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")

    def add_to_queue(self, video_url):
        """Add a video URL to the processing queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if this URL is already in the queue with status 'pending' or 'processing'
                cursor = conn.execute(
                    'SELECT id, status FROM processing_queue WHERE video_url = ? AND status IN ("pending", "processing")',
                    (video_url,)
                )
                existing = cursor.fetchone()

                if existing:
                    queue_id, status = existing
                    logging.info(f"Video already in queue: {video_url}, ID: {queue_id}, Status: {status}")
                    return queue_id

                # Add new entry to queue
                cursor = conn.execute(
                    'INSERT INTO processing_queue (video_url, status) VALUES (?, ?)',
                    (video_url, 'pending')
                )
                queue_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Added video to queue: {video_url}, ID: {queue_id}")
                return queue_id
        except sqlite3.Error as e:
            logging.error(f"Error adding to queue: {e}")
            return None

    def update_queue_status(self, queue_id, status):
        """Update the status of a queued item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE processing_queue SET status = ? WHERE id = ?',
                    (status, queue_id)
                )
                conn.commit()
                logging.info(f"Updated queue item {queue_id} status to {status}")
                return True
        except sqlite3.Error as e:
            logging.error(f"Error updating queue status: {e}")
            return False

    def get_queue_status(self, queue_id):
        """Get the status of a queued item."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id, video_url, status, created_at FROM processing_queue WHERE id = ?',
                    (queue_id,)
                )
                result = cursor.fetchone()
                if result:
                    return dict(result)
                return None
        except sqlite3.Error as e:
            logging.error(f"Error getting queue status: {e}")
            return None

    def get_next_pending_video(self):
        """Get the next pending video from the queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id, video_url FROM processing_queue WHERE status = "pending" ORDER BY created_at ASC LIMIT 1'
                )
                result = cursor.fetchone()
                if result:
                    # Update status to 'processing'
                    conn.execute(
                        'UPDATE processing_queue SET status = "processing" WHERE id = ?',
                        (result['id'],)
                    )
                    conn.commit()
                    return dict(result)
                return None
        except sqlite3.Error as e:
            logging.error(f"Error getting next pending video: {e}")
            return None

    def get_queue_stats(self):
        """Get statistics about the processing queue."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT status, COUNT(*) as count FROM processing_queue GROUP BY status'
                )
                results = cursor.fetchall()
                stats = {}
                for status, count in results:
                    stats[status] = count
                return stats
        except sqlite3.Error as e:
            logging.error(f"Error getting queue stats: {e}")
            return {}

    def get_processing_time_stats(self):
        """Calculate average processing time and queue wait time."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Calculate average processing time from completed evidence results
                cursor = conn.execute('''
                    SELECT AVG(process_time) as avg_processing_time
                    FROM evidence_results
                    WHERE processing_status = 'processed'
                    AND process_time IS NOT NULL
                ''')
                result = cursor.fetchone()
                avg_processing_time = result['avg_processing_time'] if result and result['avg_processing_time'] is not None else 0

                # Calculate average queue wait time
                cursor = conn.execute('''
                    SELECT
                        AVG(
                            (julianday(er.created_at) - julianday(pq.created_at)) * 24 * 60 * 60
                        ) as avg_queue_wait_time
                    FROM
                        evidence_results er
                    JOIN
                        processing_queue pq ON er.video_url = pq.video_url
                    WHERE
                        pq.status = 'completed'
                ''')
                result = cursor.fetchone()
                avg_queue_wait_time = result['avg_queue_wait_time'] if result and result['avg_queue_wait_time'] is not None else 0

                return {
                    'avg_processing_time': avg_processing_time,
                    'avg_queue_wait_time': avg_queue_wait_time
                }
        except sqlite3.Error as e:
            logging.error(f"Error calculating processing time stats: {e}")
            return {
                'avg_processing_time': 0,
                'avg_queue_wait_time': 0
            }

    def store_evidence_result(self, video_url, detection_results, analysis_result, process_time, head_pose=None):
        """Store evidence result in the database."""
        try:
            is_drowsy = analysis_result.get('is_drowsy')
            confidence = analysis_result.get('confidence', 0.0)

            # Extract head pose information (placeholder for landmark system)
            is_head_turned = False
            is_head_down = False
            if head_pose:
                is_head_turned = head_pose.get('head_turned', False)
                is_head_down = head_pose.get('head_down', False)

            # Combine all details for storage
            details = {
                'detection_results': detection_results,
                'analysis_result': analysis_result,
                'head_pose': head_pose
            }

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    INSERT INTO evidence_results (
                        video_url, process_time, yawn_frames, eye_closed_frames,
                        max_consecutive_eye_closed,
                        normal_state_frames, total_frames, is_drowsy, confidence,
                        is_head_turned, is_head_down, processing_status, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video_url,
                    process_time,
                    detection_results.get('yawn_frames', 0),
                    detection_results.get('eye_closed_frames', 0),
                    detection_results.get('max_consecutive_eye_closed', 0),
                    detection_results.get('normal_state_frames', 0),
                    detection_results.get('total_frames', 0),
                    is_drowsy,
                    confidence,
                    is_head_turned,
                    is_head_down,
                    'processed',
                    json.dumps(details)
                ))
                evidence_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Stored evidence result for {video_url}, ID: {evidence_id}")
                return evidence_id
        except sqlite3.Error as e:
            logging.error(f"Error storing evidence result: {e}")
            return None

    def get_evidence_result(self, evidence_id):
        """Get evidence result by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('SELECT * FROM evidence_results WHERE id = ?', (evidence_id,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except sqlite3.Error as e:
            logging.error(f"Error getting evidence result: {e}")
            return None

    def get_all_evidence_results(self, limit=100, offset=0):
        """Get all evidence results with pagination."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM evidence_results ORDER BY created_at DESC LIMIT ? OFFSET ?',
                    (limit, offset)
                )
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting all evidence results: {e}")
            return []

    def get_evidence_results_count(self):
        """Get total count of evidence results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) as count FROM evidence_results')
                result = cursor.fetchone()
                return result[0] if result else 0
        except sqlite3.Error as e:
            logging.error(f"Error getting evidence results count: {e}")
            return 0

    def get_all_queue_items(self, limit=100, offset=0):
        """Get all queue items with pagination."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM processing_queue ORDER BY created_at DESC LIMIT ? OFFSET ?',
                    (limit, offset)
                )
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting all queue items: {e}")
            return []

    def get_queue_items_count(self):
        """Get total count of queue items."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) as count FROM processing_queue')
                result = cursor.fetchone()
                return result[0] if result else 0
        except sqlite3.Error as e:
            logging.error(f"Error getting queue items count: {e}")
            return 0

    def add_webhook(self, url):
        """Add a new webhook URL."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if this URL already exists
                cursor = conn.execute(
                    'SELECT id FROM webhooks WHERE url = ?',
                    (url,)
                )
                existing = cursor.fetchone()

                if existing:
                    # If it exists but is disabled, enable it
                    conn.execute(
                        'UPDATE webhooks SET is_enabled = 1 WHERE id = ?',
                        (existing[0],)
                    )
                    conn.commit()
                    logging.info(f"Re-enabled existing webhook: {url}, ID: {existing[0]}")
                    return existing[0]

                # Add new webhook
                cursor = conn.execute(
                    'INSERT INTO webhooks (url, is_enabled) VALUES (?, ?)',
                    (url, 1)
                )
                webhook_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Added new webhook: {url}, ID: {webhook_id}")
                return webhook_id
        except sqlite3.Error as e:
            logging.error(f"Error adding webhook: {e}")
            return None

    def delete_webhook(self, webhook_id):
        """Delete a webhook by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'DELETE FROM webhooks WHERE id = ?',
                    (webhook_id,)
                )
                conn.commit()
                logging.info(f"Deleted webhook with ID: {webhook_id}")
                return True
        except sqlite3.Error as e:
            logging.error(f"Error deleting webhook: {e}")
            return False

    def get_all_webhooks(self):
        """Get all registered webhooks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM webhooks ORDER BY created_at DESC'
                )
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting all webhooks: {e}")
            return []

    def get_active_webhooks(self):
        """Get all active webhooks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT * FROM webhooks WHERE is_enabled = 1'
                )
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting active webhooks: {e}")
            return []
