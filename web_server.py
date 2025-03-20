from flask import Flask, render_template, jsonify, request, session, redirect, url_for, Response
from flask_cors import CORS
import sqlite3
from datetime import datetime, timedelta
import os
import math
from dotenv import load_dotenv
from functools import wraps
from services.auth_service import KeycloakAuth
import csv
from io import StringIO

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, supports_credentials=True)

# Set secure cookie
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.secret_key = os.urandom(24)

# Load environment variables
load_dotenv()

# Authentication configuration
AUTH_TYPE = os.getenv("AUTH_TYPE", "PIN").upper()  # Default to PIN auth
WEB_ACCESS_PIN = os.getenv("WEB_ACCESS_PIN", "123456")

# Add max and min functions to template context
app.jinja_env.globals.update(max=max, min=min)


DB_PATH = "drowsiness_detection.db"  # Same as in drowsiness_detector.py


# Get port from environment variable or use 8000 as default
PORT = int(os.getenv('FLASK_PORT', 8000))

# Initialize Keycloak auth if needed
auth_service = KeycloakAuth() if AUTH_TYPE == "KEYCLOAK" else None

def is_ajax_request():
    """Check if the request is an AJAX request."""
    return request.headers.get('X-Requested-With') == 'XMLHttpRequest'

def verify_auth():
    """Verify authentication based on AUTH_TYPE."""
    if AUTH_TYPE == "KEYCLOAK":
        if 'token' not in session:
            return False
        try:
            return auth_service.verify_token(session['token']['access_token'])
        except:
            return False
    else:  # PIN-based auth
        return session.get('authenticated', False)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not verify_auth():
            if is_ajax_request():
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Add before_request handler
@app.before_request
def before_request():
    if request.endpoint in ['login', 'static']:
        return None

    if not verify_auth():
        if is_ajax_request():
            return jsonify({'error': 'Unauthorized'}), 401
        return redirect(url_for('login'))

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Check if already authenticated
    if verify_auth():
        return redirect(url_for('index'))

    if request.method == 'POST':
        if AUTH_TYPE == "KEYCLOAK":
            username = request.form.get('username')
            password = request.form.get('password')

            result = auth_service.authenticate(username, password)

            if result['success']:
                session['token'] = result['token']
                session['user_info'] = result['user_info']
                return redirect(url_for('index'))

            return render_template('login.html', error='Invalid credentials', auth_type=AUTH_TYPE)
        else:
            # PIN-based authentication
            pin = request.form.get('pin')
            if pin == WEB_ACCESS_PIN:
                session['authenticated'] = True
                return redirect(url_for('index'))

            return render_template('login.html', error='Invalid PIN', auth_type=AUTH_TYPE)

    return render_template('login.html', auth_type=AUTH_TYPE)

@app.route('/logout')
def logout():
    if AUTH_TYPE == "KEYCLOAK" and 'token' in session:
        try:
            auth_service.logout(session['token']['refresh_token'])
        except:
            pass
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
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

        # Get status filter parameters
        status_types = request.args.getlist('status')

        # Set default status types if none are provided
        if not status_types:
            status_types = ['all']

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
                    event_conditions.append("er.alarm_type_value LIKE '%Yawning%'")
                elif event_type == 'eye_closed':
                    event_conditions.append("er.alarm_type_value LIKE '%Eye closed%'")

            if event_conditions:
                conditions.append("(" + " OR ".join(event_conditions) + ")")

        # Add status conditions
        if status_types and 'all' not in status_types:
            status_conditions = []
            for status in status_types:
                status_conditions.append(f"er.processing_status = '{status}'")

            if status_conditions:
                conditions.append("(" + " OR ".join(status_conditions) + ")")

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
                er.normal_state_frames,
                er.processing_status,
                er.fleet_name,
                er.takeup_memo,
                er.takeup_time,
                er.takeType,
                er.review_type,
                er.process_time,
                er.model_name
            FROM evidence_results er
            {where_clause}
            ORDER BY er.alarm_time DESC
            LIMIT ? OFFSET ?
        '''
        # Create a new params list for the results query
        results_params = params + [per_page, offset]
        cursor = conn.execute(results_query, results_params)
        evidence_results = cursor.fetchall()

        # First, let's create a base condition for pending events that will be used consistently
        pending_condition = "processing_status = 'pending' AND video_url IS NOT NULL"

        # Stats query with corrected pending events counting
        stats_query = f'''
            WITH pending_count AS (
                SELECT COUNT(*) as count
                FROM evidence_results er
                WHERE {pending_condition}
                {" AND " + " AND ".join(conditions) if conditions else ""}
            )
            SELECT
                COUNT(*) as total_events,
                COUNT(DISTINCT device_name) as unique_devices,
                COUNT(DISTINCT fleet_name) as unique_fleets,
                COALESCE(SUM(CASE WHEN is_drowsy = 1 THEN 1 ELSE 0 END), 0) as drowsy_events,
                COALESCE(SUM(CASE WHEN processing_status = 'processed' THEN 1 ELSE 0 END), 0) as processed_events,
                (SELECT count FROM pending_count) as pending_events,
                COALESCE(SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END), 0) as failed_events,
                -- Take Type metrics (only processed records)
                COALESCE(SUM(CASE WHEN is_drowsy = 1 AND takeType = 0 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as take_true_positives,
                COALESCE(SUM(CASE WHEN is_drowsy = 1 AND takeType = 1 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as take_false_positives,
                COALESCE(SUM(CASE WHEN is_drowsy = 0 AND takeType = 1 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as take_true_negatives,
                COALESCE(SUM(CASE WHEN is_drowsy = 0 AND takeType = 0 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as take_false_negatives,
                -- Review Type vs Take Type metrics (only processed records)
                COALESCE(SUM(CASE WHEN review_type = 0 AND takeType = 0 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as review_true_positives,
                COALESCE(SUM(CASE WHEN review_type = 1 AND takeType = 0 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as review_false_positives,
                COALESCE(SUM(CASE WHEN review_type = 1 AND takeType = 1 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as review_true_negatives,
                COALESCE(SUM(CASE WHEN review_type = 0 AND takeType = 1 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as review_false_negatives,
                -- Our Type vs Take Type metrics (based on alarm_type_value and detection values)
                -- Only include records where takeType is not NULL and processing_status = 'processed'
                COALESCE(SUM(CASE
                    -- For Yawning alarms, check if yawn_count is highest
                    WHEN alarm_type_value LIKE '%Yawning%'
                        AND takeType = 0
                        AND processing_status = 'processed'
                        AND yawn_count > eye_closed_frames
                        AND (normal_state_frames IS NULL OR yawn_count >= normal_state_frames) THEN 1
                    -- For Eye closed alarms, check if eye_closed_frames is highest
                    WHEN alarm_type_value LIKE '%Eye closed%'
                        AND takeType = 0
                        AND processing_status = 'processed'
                        AND eye_closed_frames > yawn_count
                        AND (normal_state_frames IS NULL OR eye_closed_frames >= normal_state_frames) THEN 1
                    -- For normal state, check if normal_state_frames is highest
                    WHEN normal_state_frames IS NOT NULL
                        AND normal_state_frames > yawn_count
                        AND normal_state_frames > eye_closed_frames
                        AND takeType = 1
                        AND processing_status = 'processed' THEN 1
                    ELSE 0
                END), 0) as our_true_positives,
                COALESCE(SUM(CASE
                    -- For Yawning alarms, check if yawn_count is NOT highest or takeType is wrong
                    WHEN alarm_type_value LIKE '%Yawning%'
                        AND processing_status = 'processed'
                        AND (
                            takeType = 1 OR
                            yawn_count <= eye_closed_frames OR
                            (normal_state_frames IS NOT NULL AND yawn_count < normal_state_frames)
                        ) THEN 1
                    -- For Eye closed alarms, check if eye_closed_frames is NOT highest or takeType is wrong
                    WHEN alarm_type_value LIKE '%Eye closed%'
                        AND processing_status = 'processed'
                        AND (
                            takeType = 1 OR
                            eye_closed_frames <= yawn_count OR
                            (normal_state_frames IS NOT NULL AND eye_closed_frames < normal_state_frames)
                        ) THEN 1
                    -- For normal state, check if normal_state_frames is NOT highest or takeType is wrong
                    WHEN normal_state_frames IS NOT NULL
                        AND processing_status = 'processed'
                        AND (
                            (normal_state_frames <= yawn_count OR normal_state_frames <= eye_closed_frames) OR
                            takeType = 0
                        ) THEN 1
                    ELSE 0
                END), 0) as our_false_positives,
                COALESCE(SUM(CASE
                    -- True negatives: Correctly identified as not matching the alarm type
                    WHEN processing_status = 'processed'
                        AND takeType = 1
                        AND (
                            (alarm_type_value LIKE '%Yawning%' AND (yawn_count <= eye_closed_frames OR (normal_state_frames IS NOT NULL AND yawn_count < normal_state_frames))) OR
                            (alarm_type_value LIKE '%Eye closed%' AND (eye_closed_frames <= yawn_count OR (normal_state_frames IS NOT NULL AND eye_closed_frames < normal_state_frames)))
                        ) THEN 1
                    ELSE 0
                END), 0) as our_true_negatives,
                COALESCE(SUM(CASE
                    -- False negatives: Incorrectly identified as not matching the alarm type
                    WHEN processing_status = 'processed'
                        AND takeType = 0
                        AND (
                            (alarm_type_value LIKE '%Yawning%' AND (yawn_count <= eye_closed_frames OR (normal_state_frames IS NOT NULL AND yawn_count < normal_state_frames))) OR
                            (alarm_type_value LIKE '%Eye closed%' AND (eye_closed_frames <= yawn_count OR (normal_state_frames IS NOT NULL AND eye_closed_frames < normal_state_frames)))
                        ) THEN 1
                    ELSE 0
                END), 0) as our_false_negatives
            FROM evidence_results er
            {" WHERE " + " AND ".join(conditions) if conditions else ""}
        '''
        # Duplicate the params for both the CTE and main query parts when conditions exist
        stats_params = params * 2 if conditions else []
        cursor = conn.execute(stats_query, stats_params)
        stats = dict(cursor.fetchone())

        # Calculate Take Type metrics
        take_total_predictions = (stats['take_true_positives'] + stats['take_true_negatives'] +
                                 stats['take_false_positives'] + stats['take_false_negatives'])

        if take_total_predictions > 0:
            stats['take_accuracy'] = ((stats['take_true_positives'] + stats['take_true_negatives']) /
                                     take_total_predictions) * 100
            stats['take_sensitivity'] = (stats['take_true_positives'] /
                                       (stats['take_true_positives'] + stats['take_false_negatives'])) * 100 if (stats['take_true_positives'] + stats['take_false_negatives']) > 0 else 0.0
        else:
            stats['take_accuracy'] = 0.0
            stats['take_sensitivity'] = 0.0

        # Calculate Review Type metrics
        review_total_predictions = (stats['review_true_positives'] + stats['review_true_negatives'] +
                                  stats['review_false_positives'] + stats['review_false_negatives'])

        if review_total_predictions > 0:
            stats['review_accuracy'] = ((stats['review_true_positives'] + stats['review_true_negatives']) /
                                       review_total_predictions) * 100
            stats['review_sensitivity'] = (stats['review_true_positives'] /
                                         (stats['review_true_positives'] + stats['review_false_negatives'])) * 100 if (stats['review_true_positives'] + stats['review_false_negatives']) > 0 else 0.0
        else:
            stats['review_accuracy'] = 0.0
            stats['review_sensitivity'] = 0.0

        # Calculate Our Type vs Take Type metrics
        our_total_predictions = (stats['our_true_positives'] + stats['our_true_negatives'] +
                               stats['our_false_positives'] + stats['our_false_negatives'])

        if our_total_predictions > 0:
            stats['our_accuracy'] = ((stats['our_true_positives'] + stats['our_true_negatives']) /
                                    our_total_predictions) * 100
            stats['our_sensitivity'] = (stats['our_true_positives'] /
                                      (stats['our_true_positives'] + stats['our_false_negatives'])) * 100 if (stats['our_true_positives'] + stats['our_false_negatives']) > 0 else 0.0
        else:
            stats['our_accuracy'] = 0.0
            stats['our_sensitivity'] = 0.0

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
                                 'status_types': status_types,
                                 'start_date': start_date,
                                 'end_date': end_date
                             })

    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/export')
@login_required
def export_data():
    try:
        conn = get_db_connection()

        # Get filter parameters
        event_types = request.args.getlist('event_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        status_types = request.args.getlist('status')

        # Set default event types if none are provided
        if not event_types:
            event_types = ['yawning', 'eye_closed']

        # Set default status types if none are provided
        if not status_types:
            status_types = ['all']

        # Initialize params list for SQL queries
        params = []
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
                    event_conditions.append("er.alarm_type_value LIKE '%Yawning%'")
                elif event_type == 'eye_closed':
                    event_conditions.append("er.alarm_type_value LIKE '%Eye closed%'")

            if event_conditions:
                conditions.append("(" + " OR ".join(event_conditions) + ")")

        # Add status conditions
        if status_types and 'all' not in status_types:
            status_conditions = []
            for status in status_types:
                status_conditions.append(f"er.processing_status = '{status}'")

            if status_conditions:
                conditions.append("(" + " OR ".join(status_conditions) + ")")

        # Construct the WHERE clause
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        # Get all filtered evidence results
        query = f'''
            SELECT
                er.device_name,
                er.alarm_time,
                er.alarm_type_value as event_type,
                er.speed,
                er.is_drowsy,
                er.yawn_count,
                er.eye_closed_frames,
                er.process_time,
                er.model_name,
                er.processing_status,
                CASE
                    WHEN er.takeType = 0 THEN 'True Alarm'
                    WHEN er.takeType = 1 THEN 'False Alarm'
                    ELSE '-'
                END as take_type,
                er.takeup_memo as memo,
                er.takeup_time as memo_time,
                CASE
                    WHEN er.review_type = 0 THEN 'True Alarm'
                    WHEN er.review_type = 1 THEN 'False Alarm'
                    ELSE '-'
                END as review_type,
                er.video_url
            FROM evidence_results er
            {where_clause}
            ORDER BY er.alarm_time DESC
        '''

        cursor = conn.execute(query, params)
        results = cursor.fetchall()

        # Create CSV in memory
        si = StringIO()
        writer = csv.writer(si)

        # Write headers
        writer.writerow([
            'Device', 'Time', 'Event Type', 'Speed (km/h)', 'Drowsy',
            'Yawn Count', 'Eyes Closed Frames', 'Process Time (sec)', 'Model Name', 'Status', 'Take Type',
            'Memo', 'Memo Time', 'Review Type', 'Video URL'
        ])

        # Write data
        for row in results:
            writer.writerow([
                row['device_name'],
                row['alarm_time'],
                row['event_type'],
                row['speed'],
                'Yes' if row['is_drowsy'] else 'No',
                row['yawn_count'] or 0,
                row['eye_closed_frames'] or 0,
                f"{row['process_time']:.2f}" if row['process_time'] is not None else '-',
                row['model_name'] or '-',
                row['processing_status'],
                row['take_type'],
                row['memo'] or '-',
                row['memo_time'] or '-',
                row['review_type'],
                row['video_url'] or '-'
            ])

        # Create response
        output = si.getvalue()
        si.close()

        return Response(
            output,
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename=dashboard_export.csv',
                'Content-Type': 'text/csv'
            }
        )

    except Exception as e:
        return f"Error: {str(e)}", 500
    finally:
        conn.close()

# Error handlers
@app.errorhandler(403)
def forbidden_error(error):
    if is_ajax_request():  # For AJAX requests
        return jsonify({'error': 'Forbidden'}), 403
    return redirect(url_for('login'))

@app.errorhandler(401)
def unauthorized_error(error):
    if is_ajax_request():  # For AJAX requests
        return jsonify({'error': 'Unauthorized'}), 401
    return redirect(url_for('login'))

@app.errorhandler(404)
def not_found_error(error):
    if is_ajax_request():  # For AJAX requests
        return jsonify({'error': 'Not Found'}), 404
    if 'token' not in session:
        return redirect(url_for('login'))
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    if is_ajax_request():  # For AJAX requests
        return jsonify({'error': 'Internal Server Error'}), 500
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)
