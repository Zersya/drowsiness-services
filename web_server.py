from flask import Flask, render_template, jsonify, request
import sqlite3
from datetime import datetime
import os
import math

app = Flask(__name__)

# Add max and min functions to template context
app.jinja_env.globals.update(max=max, min=min)

DB_PATH = "drowsiness_detection.db"  # Same as in drowsiness_detector.py

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
        
        # Pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = 10  # Number of items per page
        offset = (page - 1) * per_page
        
        # Get latest fetch time
        cursor = conn.execute('''
            SELECT last_fetch_time 
            FROM fetch_state 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        last_fetch = cursor.fetchone()
        last_fetch_time = datetime.fromisoformat(last_fetch['last_fetch_time']) if last_fetch else None

        # Get total count of evidence results
        cursor = conn.execute('SELECT COUNT(*) as total FROM evidence_results')
        total_records = cursor.fetchone()['total']
        total_pages = math.ceil(total_records / per_page)

        # Get paginated evidence results
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
                er.fleet_name,
                er.takeup_memo,
                er.takeup_time
            FROM evidence_results er
            ORDER BY er.alarm_time DESC
            LIMIT ? OFFSET ?
        ''', (per_page, offset))
        evidence_results = cursor.fetchall()
        
        # Get statistics
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
                             last_fetch_time=last_fetch_time,
                             pagination={
                                 'page': page,
                                 'per_page': per_page,
                                 'total_pages': total_pages,
                                 'total_records': total_records
                             })
                             
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
