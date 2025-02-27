from flask import Flask, render_template, jsonify, request
import sqlite3
from datetime import datetime, timedelta
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
        
        # Filter parameters - get multiple event types
        event_types = request.args.getlist('event_type')

        # Set default event types if none are provided
        if not event_types:
            event_types = ['yawning', 'eye_closed']
        
        # Get latest fetch time
        cursor = conn.execute('''
            SELECT last_fetch_time 
            FROM fetch_state 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        last_fetch = cursor.fetchone()
        last_fetch_time = datetime.fromisoformat(last_fetch['last_fetch_time']) if last_fetch else None

        # Get date range filters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Initialize params list for SQL queries
        params = []
        
        # Base query conditions
        conditions = []
        
        # Add date range conditions
        if start_date:
            conditions.append("DATE(er.alarm_time) >= DATE(?)")
            params.append(start_date)
        if end_date:
            conditions.append("DATE(er.alarm_time) <= DATE(?)")
            params.append(end_date)
        
        # Add event type conditions
        if event_types and 'all' not in event_types:
            event_conditions = []
            for event_type in event_types:
                if event_type == 'yawning':
                    event_conditions.append("er.alarm_type_value LIKE '%Yawn%'")
                elif event_type == 'eye_closed':
                    event_conditions.append("er.alarm_type_value LIKE '%Eye closed%'")
            
            if event_conditions:
                conditions.append("(" + " OR ".join(event_conditions) + ")")
        
        # Construct the WHERE clause
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Get total count of evidence results with filters applied
        count_query = f'''
            SELECT COUNT(*) as total 
            FROM evidence_results er
            {where_clause}
        '''
        cursor = conn.execute(count_query, params)
        total_records = cursor.fetchone()['total']
        total_pages = math.ceil(total_records / per_page)

        # Get paginated evidence results with filters applied
        results_query = f'''
            SELECT 
                er.id,
                er.device_id,
                er.device_name,
                er.alarm_type,
                er.alarm_type_value,
                er.alarm_time,
                er.location,
                er.speed,
                er.video_url,
                er.is_drowsy,
                er.yawn_count,
                er.eye_closed_frames,
                er.processing_status,
                er.fleet_name,
                er.takeup_memo,
                er.takeup_time,
                er.takeType
            FROM evidence_results er
            {where_clause}
            ORDER BY er.alarm_time DESC
            LIMIT ? OFFSET ?
        '''
        # Create a new params list for the results query
        results_params = params + [per_page, offset]
        cursor = conn.execute(results_query, results_params)
        evidence_results = cursor.fetchall()
        
        # Get statistics with filters applied
        stats_query = f'''
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT device_name) as unique_devices,
                COUNT(DISTINCT fleet_name) as unique_fleets,
                COALESCE(SUM(CASE WHEN is_drowsy = 1 THEN 1 ELSE 0 END), 0) as drowsy_events,
                COALESCE(SUM(CASE WHEN processing_status = 'processed' THEN 1 ELSE 0 END), 0) as processed_events,
                COALESCE(SUM(CASE WHEN processing_status = 'pending' THEN 1 ELSE 0 END), 0) as pending_events,
                COALESCE(SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END), 0) as failed_events,
                COALESCE(SUM(CASE WHEN is_drowsy = 1 AND takeType = 0 THEN 1 ELSE 0 END), 0) as true_positives,
                COALESCE(SUM(CASE WHEN is_drowsy = 1 AND takeType = 1 THEN 1 ELSE 0 END), 0) as false_positives,
                COALESCE(SUM(CASE WHEN is_drowsy = 0 AND takeType = 1 THEN 1 ELSE 0 END), 0) as true_negatives,
                COALESCE(SUM(CASE WHEN is_drowsy = 0 AND takeType = 0 THEN 1 ELSE 0 END), 0) as false_negatives
            FROM evidence_results er
            {" WHERE " + " AND ".join(["processing_status = 'processed'"] + conditions) if conditions else " WHERE processing_status = 'processed'"}
        '''
        cursor = conn.execute(stats_query, params)
        stats = dict(cursor.fetchone())
        
        # Calculate accuracy and sensitivity
        total_predictions = (stats['true_positives'] + stats['true_negatives'] + 
                           stats['false_positives'] + stats['false_negatives'])
        
        if total_predictions > 0:
            stats['accuracy'] = ((stats['true_positives'] + stats['true_negatives']) / 
                               total_predictions) * 100
        else:
            stats['accuracy'] = 0.0
            
        if (stats['true_positives'] + stats['false_negatives']) > 0:
            stats['sensitivity'] = (stats['true_positives'] / 
                                  (stats['true_positives'] + stats['false_negatives'])) * 100
        else:
            stats['sensitivity'] = 0.0
        
        # Get available event types for filter dropdown
        cursor = conn.execute('''
            SELECT DISTINCT alarm_type_value
            FROM evidence_results
            ORDER BY alarm_type_value
        ''')
        available_event_types = [row['alarm_type_value'] for row in cursor.fetchall()]
        
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
                             },
                             filters={
                                 'event_types': event_types,
                                 'available_event_types': available_event_types,
                                 'start_date': start_date,
                                 'end_date': end_date
                             })
                             
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
