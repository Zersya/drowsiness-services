from flask import Flask, render_template, jsonify
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

DB_PATH = "fetch_state.db"  # Same as in drowsiness_detector.py

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

@app.route('/')
def index():
    """Render the main dashboard page."""
    try:
        conn = get_db_connection()
        
        # Get latest fetch time
        cursor = conn.execute('''
            SELECT last_fetch_time 
            FROM fetch_state 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        last_fetch = cursor.fetchone()
        last_fetch_time = datetime.fromisoformat(last_fetch['last_fetch_time']) if last_fetch else None

        # Get latest evidence results
        cursor = conn.execute('''
            SELECT 
                er.device_name,
                er.alarm_type,
                er.alarm_type_value,
                er.alarm_time,
                er.location,
                er.speed,
                er.is_drowsy,
                er.yawn_count,
                er.eye_closed_frames,
                er.processing_status,
                er.fleet_name
            FROM evidence_results er
            ORDER BY er.alarm_time DESC
            LIMIT 50
        ''')
        evidence_results = cursor.fetchall()
        
        # Get some basic statistics
        cursor = conn.execute('''
            SELECT 
                COUNT(*) as total_events,
                SUM(CASE WHEN is_drowsy = 1 THEN 1 ELSE 0 END) as drowsy_events,
                COUNT(DISTINCT device_id) as unique_devices,
                COUNT(DISTINCT fleet_name) as unique_fleets,
                SUM(CASE WHEN processing_status = 'processed' THEN 1 ELSE 0 END) as processed_events,
                SUM(CASE WHEN processing_status = 'pending' THEN 1 ELSE 0 END) as pending_events,
                SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) as failed_events
            FROM evidence_results
        ''')
        stats = cursor.fetchone()
        
        conn.close()
        
        return render_template('dashboard.html', 
                             evidence_results=evidence_results,
                             stats=stats,
                             last_fetch_time=last_fetch_time)
                             
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/events')
def get_events():
    """API endpoint to get events data."""
    try:
        conn = get_db_connection()
        cursor = conn.execute('''
            SELECT 
                er.*,
                ef.file_type,
                ef.file_url
            FROM evidence_results er
            LEFT JOIN evidence_files ef ON er.id = ef.evidence_id
            ORDER BY er.alarm_time DESC
            LIMIT 100
        ''')
        events = cursor.fetchall()
        conn.close()
        
        # Convert rows to dictionaries
        events_list = []
        for event in events:
            event_dict = dict(event)
            # Convert datetime to string for JSON serialization
            if event_dict['alarm_time']:
                event_dict['alarm_time'] = datetime.fromisoformat(event_dict['alarm_time']).strftime('%Y-%m-%d %H:%M:%S')
            events_list.append(event_dict)
            
        return jsonify(events_list)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4848, debug=True)
