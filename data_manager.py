import sqlite3
import logging
import datetime

class DataManager:
    """Manages data storage and retrieval for the drowsiness detection system."""

    def __init__(self, db_path="drowsiness_detection.db"):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database and create necessary tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Existing fetch_state table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fetch_state (
                        id INTEGER PRIMARY KEY,
                        last_fetch_time TIMESTAMP NOT NULL
                    )
                ''')

                # Updated evidence_results table with review_type, normal_state_frames, and takeup_user
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS evidence_results (
                        id INTEGER PRIMARY KEY,
                        device_id TEXT NOT NULL,
                        device_name TEXT,
                        alarm_type TEXT,
                        alarm_type_value TEXT,
                        alarm_time TIMESTAMP,
                        location TEXT,
                        speed REAL,
                        video_url TEXT,
                        video_url_channel_3 TEXT,
                        image_url TEXT,
                        is_drowsy BOOLEAN,
                        yawn_count INTEGER,
                        eye_closed_frames INTEGER,
                        normal_state_frames INTEGER,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        fleet_name TEXT,
                        alarm_guid TEXT UNIQUE,
                        processing_status TEXT DEFAULT 'pending',
                        takeup_memo TEXT,
                        takeup_time TIMESTAMP,
                        takeup_user TEXT,
                        takeType INTEGER,
                        review_type INTEGER,
                        process_time REAL,
                        model_name TEXT,
                        details TEXT
                    )
                ''')

                # New evidence_files table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS evidence_files (
                        id INTEGER PRIMARY KEY,
                        evidence_id INTEGER,
                        file_type TEXT,
                        file_url TEXT,
                        start_time TIMESTAMP,
                        stop_time TIMESTAMP,
                        channel INTEGER,
                        FOREIGN KEY (evidence_id) REFERENCES evidence_results (id)
                    )
                ''')

                # Models table for model management
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active INTEGER DEFAULT 0
                    )
                ''')

                conn.commit()
                logging.info("Database initialized successfully")
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise

    def get_last_fetch_time(self):
        """TODO: REMOVE THIS FOR PRODUCTION"""
        # return datetime.datetime.now() - datetime.timedelta(hours=3)

        """Retrieve the last fetch time from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT last_fetch_time FROM fetch_state ORDER BY id DESC LIMIT 1')
                result = cursor.fetchone()

                if result:
                    return datetime.datetime.fromisoformat(result[0])
                else:
                    # If no record exists, insert default time (30 minutes ago) and return it
                    default_time = datetime.datetime.now() - datetime.timedelta(minutes=30)
                    cursor.execute('INSERT INTO fetch_state (last_fetch_time) VALUES (?)',
                                 (default_time.isoformat(),))
                    conn.commit()
                    return default_time

        except sqlite3.Error as e:
            logging.error(f"Error retrieving last fetch time: {e}")
            # Fallback to 30 minutes ago if database access fails
            return datetime.datetime.now() - datetime.timedelta(minutes=30)

    def update_last_fetch_time(self, fetch_time):
        """Update the last fetch time in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO fetch_state (last_fetch_time) VALUES (?)',
                             (fetch_time.isoformat(),))
                conn.commit()
                logging.info(f"Updated last fetch time to {fetch_time}")
        except sqlite3.Error as e:
            logging.error(f"Error updating last fetch time: {e}")

    def store_evidence_result(self, evidence_data, detection_results=None):
        """Store evidence and its processing results in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Extract video URLs for both channels
                video_url = None
                video_url_channel_3 = None
                # Look for video URLs in the files array
                if evidence_data.get('alarmFile'):
                    for file_data in evidence_data['alarmFile']:
                        # fileType "2" indicates a video file
                        if file_data.get('fileType') == "2":
                            if file_data.get('channel') == 2:
                                video_url = file_data.get('downUrl')
                            elif file_data.get('channel') == 3:
                                video_url_channel_3 = file_data.get('downUrl')

                            # Break if we found both videos
                            if video_url and video_url_channel_3:
                                break

                # If no video found in alarmFile, try the legacy videoUrl field
                if not video_url:
                    video_url = evidence_data.get('videoUrl')

                # Determine processing status
                # If no video URL is found, mark as 'skipped' instead of 'pending'
                if not video_url:
                    processing_status = 'skipped'
                elif detection_results is not None:
                    processing_status = 'processed'
                else:
                    processing_status = 'pending'

                logging.debug(f"Extracted video URL: {video_url}")
                logging.info(f"Setting processing status to: {processing_status} {evidence_data.get('reviewType')}")

                # Convert analysis_details to JSON string if it exists in detection_results
                details_json = None
                if detection_results and 'analysis_details' in detection_results:
                    import json
                    details_json = json.dumps(detection_results.get('analysis_details'))

                # Insert main evidence record with normal_state_frames, process_time, model_name, takeup_user, and details
                cursor.execute('''
                    INSERT OR REPLACE INTO evidence_results (
                        device_id, device_name, alarm_type, alarm_type_value,
                        alarm_time, location, speed, video_url, video_url_channel_3, image_url,
                        is_drowsy, yawn_count, eye_closed_frames, normal_state_frames,
                        fleet_name, alarm_guid, processing_status,
                        takeup_memo, takeup_time, takeup_user, takeType, review_type, process_time, model_name, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    evidence_data.get('deviceID'),
                    evidence_data.get('deviceName'),
                    evidence_data.get('alarmType'),
                    evidence_data.get('alarmTypeValue'),
                    evidence_data.get('alarmTime'),
                    evidence_data.get('location'),
                    evidence_data.get('speed'),
                    video_url,
                    video_url_channel_3,
                    evidence_data.get('imageUrl'),
                    detection_results.get('is_drowsy') if detection_results else None,
                    detection_results.get('yawn_count') if detection_results else None,
                    detection_results.get('eye_closed_frames') if detection_results else None,
                    detection_results.get('normal_state_frames') if detection_results else None,
                    evidence_data.get('fleetName'),
                    evidence_data.get('alarmGuid'),
                    processing_status,
                    evidence_data.get('takeupMemo'),
                    evidence_data.get('takeupTime'),
                    evidence_data.get('takeupUser'),
                    evidence_data.get('takeType'),
                    evidence_data.get('reviewType'),
                    detection_results.get('process_time') if detection_results else None,
                    detection_results.get('model_name') if detection_results else None,
                    details_json
                ))

                evidence_id = cursor.lastrowid

                # Insert associated files
                if evidence_data.get('files'):
                    for file_data in evidence_data['files']:
                        cursor.execute('''
                            INSERT INTO evidence_files (
                                evidence_id, file_type, file_url,
                                start_time, stop_time, channel
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            evidence_id,
                            file_data.get('fileType'),
                            file_data.get('downUrl'),  # Changed from 'url' to 'downUrl'
                            file_data.get('fileStartTime'),  # Changed from 'startTime'
                            file_data.get('fileStopTime'),  # Changed from 'stopTime'
                            file_data.get('channel')
                        ))

                conn.commit()
                logging.info(f"Stored evidence result for device {evidence_data.get('deviceName')} with status {processing_status}")
                return evidence_id

        except sqlite3.Error as e:
            logging.error(f"Error storing evidence result: {e}")
            return None

    def update_evidence_result(self, evidence_id, detection_results):
        """Updates an existing evidence record with detection results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Convert analysis_details to JSON string if it exists
                details_json = None
                if 'analysis_details' in detection_results:
                    import json
                    details_json = json.dumps(detection_results.get('analysis_details'))

                cursor.execute('''
                    UPDATE evidence_results
                    SET is_drowsy = ?,
                        yawn_count = ?,
                        eye_closed_frames = ?,
                        normal_state_frames = ?,
                        processing_status = ?,
                        process_time = ?,
                        model_name = ?,
                        details = ?
                    WHERE id = ?
                ''', (
                    detection_results.get('is_drowsy'),
                    detection_results.get('yawn_count'),
                    detection_results.get('eye_closed_frames'),
                    detection_results.get('normal_state_frames'),
                    'processed',
                    detection_results.get('process_time'),
                    detection_results.get('model_name'),
                    details_json,
                    evidence_id
                ))
                conn.commit()
                logging.info(f"Updated evidence result for ID {evidence_id}")
        except sqlite3.Error as e:
            logging.error(f"Error updating evidence result: {e}")

    def update_evidence_status(self, evidence_id, status):
        """Updates the processing status of an evidence record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE evidence_results
                    SET processing_status = ?
                    WHERE id = ?
                ''', (status, evidence_id))
                conn.commit()
                logging.info(f"Updated evidence ID {evidence_id} status to {status}")
                return True
        except Exception as e:
            logging.error(f"Error updating evidence status: {e}")
            return False

    def get_pending_evidence_count(self):
        """Get a count of pending evidence items for diagnostic purposes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*)
                    FROM evidence_results
                    WHERE processing_status = 'pending'
                    AND video_url IS NOT NULL
                ''')
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logging.error(f"Error counting pending evidence: {e}")
            return 0

    def get_pending_evidence(self):
        """Retrieve pending evidence that needs processing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")

                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, video_url, device_name
                    FROM evidence_results
                    WHERE processing_status = 'pending'
                    AND video_url IS NOT NULL
                    ORDER BY alarm_time ASC
                ''')

                # Log the raw SQL query for debugging
                logging.debug("Executing SQL query for pending evidence")

                results = cursor.fetchall()

                # Log the raw SQL results for debugging
                logging.debug(f"Raw SQL results for pending evidence: {results}")

                return results
        except sqlite3.Error as e:
            logging.error(f"Error retrieving pending evidence: {e}")
            return []

    def get_evidence_by_id(self, evidence_id):
        """Retrieve a specific evidence record by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM evidence_results WHERE id = ?
                ''', (evidence_id,))
                evidence = cursor.fetchone()

                if evidence:
                    # Get associated files
                    cursor.execute('''
                        SELECT * FROM evidence_files WHERE evidence_id = ?
                    ''', (evidence_id,))
                    files = cursor.fetchall()

                    # Convert to dict for easier handling
                    evidence_dict = dict(evidence)
                    evidence_dict['files'] = [dict(file) for file in files]

                    return evidence_dict
                return None
        except sqlite3.Error as e:
            logging.error(f"Error retrieving evidence by ID: {e}")
            return None

    def get_evidence_by_alarm_guid(self, alarm_guid):
        """Retrieve evidence by its unique alarm GUID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM evidence_results WHERE alarm_guid = ?
                ''', (alarm_guid,))
                return dict(cursor.fetchone()) if cursor.fetchone() else None
        except sqlite3.Error as e:
            logging.error(f"Error retrieving evidence by alarm GUID: {e}")
            return None

    def get_evidence_count_by_status(self):
        """Get count of evidence records grouped by processing status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT processing_status, COUNT(*) as count
                    FROM evidence_results
                    GROUP BY processing_status
                ''')
                return dict(cursor.fetchall())
        except sqlite3.Error as e:
            logging.error(f"Error retrieving evidence count by status: {e}")
            return {}

    def get_drowsiness_statistics(self, start_date=None, end_date=None):
        """Get statistics about drowsiness detections within a date range."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = '''
                    SELECT
                        COUNT(*) as total_processed,
                        SUM(CASE WHEN is_drowsy = 1 THEN 1 ELSE 0 END) as drowsy_count,
                        AVG(CASE WHEN is_drowsy = 1 THEN yawn_count ELSE NULL END) as avg_yawn_count,
                        AVG(CASE WHEN is_drowsy = 1 THEN eye_closed_frames ELSE NULL END) as avg_eye_closed_frames,
                        COUNT(DISTINCT device_id) as unique_devices,
                        COUNT(DISTINCT fleet_name) as unique_fleets
                    FROM evidence_results
                    WHERE processing_status = 'processed'
                '''

                params = []
                if start_date and end_date:
                    query += ' AND alarm_time BETWEEN ? AND ?'
                    params.extend([start_date.isoformat(), end_date.isoformat()])

                cursor.execute(query, params)
                return dict(cursor.fetchone())
        except sqlite3.Error as e:
            logging.error(f"Error retrieving drowsiness statistics: {e}")
            return {}

    def delete_old_records(self, days_to_keep=30):
        """Delete records older than the specified number of days."""
        try:
            cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days_to_keep)).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # First, get IDs of evidence to delete
                cursor.execute('''
                    SELECT id FROM evidence_results
                    WHERE alarm_time < ?
                ''', (cutoff_date,))

                evidence_ids = [row[0] for row in cursor.fetchall()]

                # Delete associated files first (foreign key constraint)
                if evidence_ids:
                    placeholders = ','.join(['?'] * len(evidence_ids))
                    cursor.execute(f'''
                        DELETE FROM evidence_files
                        WHERE evidence_id IN ({placeholders})
                    ''', evidence_ids)

                    # Then delete the evidence records
                    cursor.execute(f'''
                        DELETE FROM evidence_results
                        WHERE id IN ({placeholders})
                    ''', evidence_ids)

                # Also clean up old fetch state records, keeping only the most recent 100
                cursor.execute('''
                    DELETE FROM fetch_state
                    WHERE id NOT IN (
                        SELECT id FROM fetch_state
                        ORDER BY id DESC
                        LIMIT 100
                    )
                ''')

                conn.commit()
                logging.info(f"Deleted {len(evidence_ids)} old evidence records older than {days_to_keep} days")

                return len(evidence_ids)
        except sqlite3.Error as e:
            logging.error(f"Error deleting old records: {e}")
            return 0

    def get_models(self):
        """Get all models from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT id, name, file_path, upload_date, is_active
                    FROM models
                    ORDER BY upload_date DESC
                ''')
                return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error retrieving models: {e}")
            return []

    def get_active_model(self):
        """Get the currently active model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT id, name, file_path, upload_date
                    FROM models
                    WHERE is_active = 1
                    LIMIT 1
                ''')
                return cursor.fetchone()
        except sqlite3.Error as e:
            logging.error(f"Error retrieving active model: {e}")
            return None

    def set_active_model(self, model_id):
        """Set a model as active and all others as inactive."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, reset all models to inactive
                conn.execute('UPDATE models SET is_active = 0')

                # Then set the selected model as active
                conn.execute('UPDATE models SET is_active = 1 WHERE id = ?', (model_id,))
                conn.commit()

                logging.info(f"Set model ID {model_id} as active")
                return True
        except sqlite3.Error as e:
            logging.error(f"Error setting active model: {e}")
            return False

    def add_model(self, name, file_path):
        """Add a new model to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    INSERT INTO models (name, file_path, upload_date, is_active)
                    VALUES (?, ?, ?, 0)
                ''', (name, file_path, datetime.datetime.now().isoformat()))
                model_id = cursor.lastrowid
                conn.commit()

                logging.info(f"Added new model: {name} at {file_path}")
                return model_id
        except sqlite3.Error as e:
            logging.error(f"Error adding model: {e}")
            return None

    def delete_model(self, model_id):
        """Delete a model from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First check if the model is active
                cursor = conn.execute('SELECT is_active, file_path FROM models WHERE id = ?', (model_id,))
                model = cursor.fetchone()

                if not model:
                    logging.warning(f"Attempted to delete non-existent model ID: {model_id}")
                    return False, "Model not found"

                if model[0] == 1:  # is_active
                    logging.warning(f"Attempted to delete active model ID: {model_id}")
                    return False, "Cannot delete active model"

                # Delete the model
                conn.execute('DELETE FROM models WHERE id = ?', (model_id,))
                conn.commit()

                logging.info(f"Deleted model ID: {model_id}")
                return True, model[1]  # Return success and file_path
        except sqlite3.Error as e:
            logging.error(f"Error deleting model: {e}")
            return False, str(e)

    def mark_evidence_as_reviewed(self, evidence_id, memo=None, take_type=None, user=None):
        """Mark an evidence record as reviewed with an optional memo, take type, and user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                update_query = '''
                    UPDATE evidence_results
                    SET takeup_time = ?,
                        takeup_memo = ?,
                        takeup_user = ?
                '''

                params = [
                    datetime.datetime.now().isoformat(),
                    memo,
                    user
                ]

                # Add takeType to the update if provided
                if take_type is not None:
                    update_query += ", takeType = ?"
                    params.append(take_type)

                update_query += " WHERE id = ?"
                params.append(evidence_id)

                cursor.execute(update_query, params)
                conn.commit()

                logging.info(f"Marked evidence ID {evidence_id} as reviewed by user {user}")
                return True

        except sqlite3.Error as e:
            logging.error(f"Error marking evidence as reviewed: {e}")
            return False

    def get_device_statistics(self):
        """Get drowsiness statistics grouped by device."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        device_id,
                        device_name,
                        fleet_name,
                        COUNT(*) as total_events,
                        SUM(CASE WHEN is_drowsy = 1 THEN 1 ELSE 0 END) as drowsy_events,
                        MAX(alarm_time) as last_event_time
                    FROM evidence_results
                    WHERE processing_status = 'processed'
                    GROUP BY device_id
                    ORDER BY drowsy_events DESC
                ''')

                return [dict(zip([column[0] for column in cursor.description], row))
                        for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Error retrieving device statistics: {e}")
            return []

    def get_fleet_statistics(self):
        """Get drowsiness statistics grouped by fleet."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT
                        fleet_name,
                        COUNT(*) as total_events,
                        SUM(CASE WHEN is_drowsy = 1 THEN 1 ELSE 0 END) as drowsy_events,
                        COUNT(DISTINCT device_id) as device_count,
                        MAX(alarm_time) as last_event_time
                    FROM evidence_results
                    WHERE processing_status = 'processed'
                    GROUP BY fleet_name
                    ORDER BY drowsy_events DESC
                ''')

                return [dict(zip([column[0] for column in cursor.description], row))
                        for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Error retrieving fleet statistics: {e}")
            return []
