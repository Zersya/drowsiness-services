from flask import Flask, render_template, jsonify, request, session, redirect, url_for, Response
from flask_cors import CORS
import sqlite3
from datetime import datetime, timedelta
import os
import math
import logging
import subprocess
import shutil
import werkzeug
import time
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
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=12)  # Set session lifetime to 12 hours
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
            # First, try to verify the current access token
            if auth_service.verify_token(session['token']['access_token']):
                return True

            # If token verification fails, try to refresh the token
            if 'refresh_token' in session['token']:
                refresh_result = auth_service.refresh_token(session['token']['refresh_token'])
                if refresh_result['success']:
                    # Update the session with the new token
                    session['token'] = refresh_result['token']
                    # Get updated user info with the new token
                    try:
                        session['user_info'] = auth_service.keycloak_openid.userinfo(session['token']['access_token'])
                    except Exception as e:
                        logging.warning(f"Failed to update user info after token refresh: {str(e)}")
                    return True

            # If we get here, both verification and refresh failed
            return False
        except Exception as e:
            logging.error(f"Auth verification error: {str(e)}")
            return False
    else:  # PIN-based auth
        return session.get('authenticated', False)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not verify_auth():
            if is_ajax_request():
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login', next=request.url))
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
        return redirect(url_for('login', next=request.url))

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Check if already authenticated
    if verify_auth():
        # If there's a next parameter, redirect to that URL
        next_url = request.args.get('next')
        if next_url:
            return redirect(next_url)
        return redirect(url_for('index'))

    if request.method == 'POST':
        if AUTH_TYPE == "KEYCLOAK":
            username = request.form.get('username')
            password = request.form.get('password')

            result = auth_service.authenticate(username, password)

            if result['success']:
                # Set session as permanent to use the configured lifetime
                session.permanent = True
                session['token'] = result['token']
                session['user_info'] = result['user_info']

                # If there's a next parameter in form data or URL, redirect to that URL
                next_url = request.form.get('next') or request.args.get('next')
                if next_url:
                    return redirect(next_url)
                return redirect(url_for('index'))

            next_url = request.form.get('next') or request.args.get('next')
            return render_template('login.html', error='Invalid credentials', auth_type=AUTH_TYPE, next=next_url)
        else:
            # PIN-based authentication
            pin = request.form.get('pin')
            if pin == WEB_ACCESS_PIN:
                # Set session as permanent to use the configured lifetime
                session.permanent = True
                session['authenticated'] = True

                # If there's a next parameter in form data or URL, redirect to that URL
                next_url = request.form.get('next') or request.args.get('next')
                if next_url:
                    return redirect(next_url)
                return redirect(url_for('index'))

            next_url = request.form.get('next') or request.args.get('next')
            return render_template('login.html', error='Invalid PIN', auth_type=AUTH_TYPE, next=next_url)

    return render_template('login.html', auth_type=AUTH_TYPE, next=request.args.get('next'))

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

        # Get model_name filter parameters
        model_names = request.args.getlist('model_name')

        # Set default model_names if none are provided
        if not model_names:
            model_names = ['all']

        # Get takeType filter parameters
        take_types = request.args.getlist('take_type')

        # Set default take_types if none are provided
        if not take_types:
            take_types = ['all']

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

        # Add model_name conditions
        if model_names and 'all' not in model_names:
            model_conditions = []
            for model_name in model_names:
                model_conditions.append(f"er.model_name = '{model_name}'")

            if model_conditions:
                conditions.append("(" + " OR ".join(model_conditions) + ")")

        # Add takeType conditions
        if take_types and 'all' not in take_types:
            take_type_conditions = []
            for take_type in take_types:
                if take_type == 'not_empty':
                    take_type_conditions.append("er.takeType IS NOT NULL")
                elif take_type in ['0', '1']:
                    take_type_conditions.append(f"er.takeType = {take_type}")

            if take_type_conditions:
                conditions.append("(" + " OR ".join(take_type_conditions) + ")")

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
                er.video_url_channel_3,
                er.is_drowsy,
                er.yawn_count,
                er.eye_closed_frames,
                er.normal_state_frames,
                er.processing_status,
                er.fleet_name,
                er.takeup_memo,
                er.takeup_time,
                er.takeup_user,
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
                -- Review Type vs Take Type metrics (only processed records) - CORRECTED
                COALESCE(SUM(CASE WHEN review_type = 0 AND takeType = 0 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as review_true_positives,
                COALESCE(SUM(CASE WHEN review_type = 0 AND takeType = 1 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as review_false_positives,
                COALESCE(SUM(CASE WHEN review_type = 1 AND takeType = 1 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as review_true_negatives,
                COALESCE(SUM(CASE WHEN review_type = 1 AND takeType = 0 AND processing_status = 'processed' THEN 1 ELSE 0 END), 0) as review_false_negatives, 
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
            stats['take_precision'] = (stats['take_true_positives'] /
                                     (stats['take_true_positives'] + stats['take_false_positives'])) * 100 if (stats['take_true_positives'] + stats['take_false_positives']) > 0 else 0.0
        else:
            stats['take_accuracy'] = 0.0
            stats['take_sensitivity'] = 0.0
            stats['take_precision'] = 0.0

        # Calculate Review Type metrics
        review_total_predictions = (stats['review_true_positives'] + stats['review_true_negatives'] +
                                  stats['review_false_positives'] + stats['review_false_negatives'])

        if review_total_predictions > 0:
            stats['review_accuracy'] = ((stats['review_true_positives'] + stats['review_true_negatives']) /
                                       review_total_predictions) * 100
            stats['review_sensitivity'] = (stats['review_true_positives'] /
                                         (stats['review_true_positives'] + stats['review_false_negatives'])) * 100 if (stats['review_true_positives'] + stats['review_false_negatives']) > 0 else 0.0
            stats['review_precision'] = (stats['review_true_positives'] /
                                       (stats['review_true_positives'] + stats['review_false_positives'])) * 100 if (stats['review_true_positives'] + stats['review_false_positives']) > 0 else 0.0
        else:
            stats['review_accuracy'] = 0.0
            stats['review_sensitivity'] = 0.0
            stats['review_precision'] = 0.0

        # Calculate Our Type vs Take Type metrics
        our_total_predictions = (stats['our_true_positives'] + stats['our_true_negatives'] +
                               stats['our_false_positives'] + stats['our_false_negatives'])

        if our_total_predictions > 0:
            stats['our_accuracy'] = ((stats['our_true_positives'] + stats['our_true_negatives']) /
                                    our_total_predictions) * 100
            stats['our_sensitivity'] = (stats['our_true_positives'] /
                                      (stats['our_true_positives'] + stats['our_false_negatives'])) * 100 if (stats['our_true_positives'] + stats['our_false_negatives']) > 0 else 0.0
            stats['our_precision'] = (stats['our_true_positives'] /
                                    (stats['our_true_positives'] + stats['our_false_positives'])) * 100 if (stats['our_true_positives'] + stats['our_false_positives']) > 0 else 0.0
        else:
            stats['our_accuracy'] = 0.0
            stats['our_sensitivity'] = 0.0
            stats['our_precision'] = 0.0

        # Get available event types for filter dropdown
        cursor = conn.execute('''
            SELECT DISTINCT alarm_type_value
            FROM evidence_results
            ORDER BY alarm_type_value
        ''')
        available_event_types = [row['alarm_type_value'] for row in cursor.fetchall()]

        # Get available model names for filter dropdown
        cursor = conn.execute('''
            SELECT DISTINCT model_name
            FROM evidence_results
            WHERE model_name IS NOT NULL AND model_name != ''
            ORDER BY model_name
        ''')
        available_model_names = [row['model_name'] for row in cursor.fetchall()]

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
                                 'model_names': model_names,
                                 'available_model_names': available_model_names,
                                 'take_types': take_types,
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
        model_names = request.args.getlist('model_name')
        take_types = request.args.getlist('take_type')

        # Set default event types if none are provided
        if not event_types:
            event_types = ['yawning', 'eye_closed']

        # Set default status types if none are provided
        if not status_types:
            status_types = ['all']

        # Set default model_names if none are provided
        if not model_names:
            model_names = ['all']

        # Set default take_types if none are provided
        if not take_types:
            take_types = ['all']

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

        # Add model_name conditions
        if model_names and 'all' not in model_names:
            model_conditions = []
            for model_name in model_names:
                model_conditions.append(f"er.model_name = '{model_name}'")

            if model_conditions:
                conditions.append("(" + " OR ".join(model_conditions) + ")")

        # Add takeType conditions
        if take_types and 'all' not in take_types:
            take_type_conditions = []
            for take_type in take_types:
                if take_type == 'not_empty':
                    take_type_conditions.append("er.takeType IS NOT NULL")
                elif take_type in ['0', '1']:
                    take_type_conditions.append(f"er.takeType = {take_type}")

            if take_type_conditions:
                conditions.append("(" + " OR ".join(take_type_conditions) + ")")

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
                er.takeup_user as memo_user,
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
            'Memo', 'Memo User', 'Memo Time', 'Review Type', 'Video URL'
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
                row['memo_user'] or '-',
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

@app.route('/export_incorrect_predictions')
@login_required
def export_incorrect_predictions():
    try:
        conn = get_db_connection()

        # Get filter parameters
        event_types = request.args.getlist('event_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        status_types = request.args.getlist('status')
        model_names = request.args.getlist('model_name')

        # Set default event types if none are provided
        if not event_types:
            event_types = ['yawning', 'eye_closed']

        # Set default status types if none are provided
        if not status_types:
            status_types = ['all']

        # Set default model_names if none are provided
        if not model_names:
            model_names = ['all']

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

        # Add model_name conditions
        if model_names and 'all' not in model_names:
            model_conditions = []
            for model_name in model_names:
                model_conditions.append(f"er.model_name = '{model_name}'")

            if model_conditions:
                conditions.append("(" + " OR ".join(model_conditions) + ")")

        # Add condition for incorrect predictions
        # This is the key part: we want records where takeType and is_drowsy don't match
        # True Alarm (takeType=0) but Not Drowsy (is_drowsy=0) OR False Alarm (takeType=1) but Drowsy (is_drowsy=1)
        conditions.append("((er.takeType = 0 AND er.is_drowsy = 0) OR (er.takeType = 1 AND er.is_drowsy = 1))")

        # Ensure we only get processed records with takeType values
        conditions.append("er.processing_status = 'processed'")
        conditions.append("er.takeType IS NOT NULL")

        # Construct the WHERE clause
        where_clause = " WHERE " + " AND ".join(conditions)

        # Get all filtered evidence results
        query = f'''
            SELECT
                er.alarm_type_value as event_type,
                er.is_drowsy,
                er.yawn_count,
                er.eye_closed_frames,
                er.processing_status,
                CASE
                    WHEN er.takeType = 0 THEN 'True Alarm'
                    WHEN er.takeType = 1 THEN 'False Alarm'
                    ELSE '-'
                END as take_type,
                er.takeup_memo as memo,
                er.takeup_time as memo_time,
                er.takeup_user as memo_user,
                CASE
                    WHEN er.review_type = 0 THEN 'True Alarm'
                    WHEN er.review_type = 1 THEN 'False Alarm'
                    ELSE '-'
                END as review_type,
                er.details
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
            'Event Type', 'Drowsy',
            'Yawn Count', 'Eyes Closed Frames', 'Status', 'Take Type',
            'Memo', 'Memo User', 'Memo Time', 'Review Type', 'Details'
        ])

        # Write data
        for row in results:
            # Parse details JSON if it exists
            details_str = '-'
            if row['details']:
                try:
                    import json
                    details_dict = json.loads(row['details'])
                    # Format the details for better display
                    details_str = json.dumps(details_dict, indent=2)
                except Exception as e:
                    details_str = row['details']

            writer.writerow([
                row['event_type'],
                'Yes' if row['is_drowsy'] else 'No',
                row['yawn_count'] or 0,
                row['eye_closed_frames'] or 0,
                row['processing_status'],
                row['take_type'],
                row['memo'] or '-',
                row['memo_user'] or '-',
                row['memo_time'] or '-',
                row['review_type'],
                details_str
            ])

        # Create response
        output = si.getvalue()
        si.close()

        return Response(
            output,
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename=incorrect_drowsiness_predictions.csv',
                'Content-Type': 'text/csv'
            }
        )

    except Exception as e:
        return f"Error: {str(e)}", 500
    finally:
        conn.close()

@app.route('/export_all_predictions')
@login_required
def export_all_predictions():
    try:
        conn = get_db_connection()

        # Get filter parameters
        event_types = request.args.getlist('event_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        status_types = request.args.getlist('status')
        model_names = request.args.getlist('model_name')

        # Set default event types if none are provided
        if not event_types:
            event_types = ['yawning', 'eye_closed']

        # Set default status types if none are provided
        if not status_types:
            status_types = ['all']

        # Set default model_names if none are provided
        if not model_names:
            model_names = ['all']

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

        # Add model_name conditions
        if model_names and 'all' not in model_names:
            model_conditions = []
            for model_name in model_names:
                model_conditions.append(f"er.model_name = '{model_name}'")

            if model_conditions:
                conditions.append("(" + " OR ".join(model_conditions) + ")")

        # Ensure we only get processed records with takeType values
        conditions.append("er.processing_status = 'processed'")
        conditions.append("er.takeType IS NOT NULL")

        # Construct the WHERE clause
        where_clause = " WHERE " + " AND ".join(conditions)

        # Get all filtered evidence results
        query = f'''
            SELECT
                er.alarm_type_value as event_type,
                er.is_drowsy,
                er.yawn_count,
                er.eye_closed_frames,
                er.processing_status,
                CASE
                    WHEN er.takeType = 0 THEN 'True Alarm'
                    WHEN er.takeType = 1 THEN 'False Alarm'
                    ELSE '-'
                END as take_type,
                er.takeup_memo as memo,
                er.takeup_time as memo_time,
                er.takeup_user as memo_user,
                CASE
                    WHEN er.review_type = 0 THEN 'True Alarm'
                    WHEN er.review_type = 1 THEN 'False Alarm'
                    ELSE '-'
                END as review_type,
                er.details
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
            'Event Type', 'Drowsy',
            'Yawn Count', 'Eyes Closed Frames', 'Status', 'Take Type',
            'Memo', 'Memo User', 'Memo Time', 'Review Type', 'Details'
        ])

        # Write data
        for row in results:
            # Parse details JSON if it exists
            details_str = '-'
            if row['details']:
                try:
                    import json
                    details_dict = json.loads(row['details'])
                    # Format the details for better display
                    details_str = json.dumps(details_dict, indent=2)
                except Exception as e:
                    details_str = row['details']

            writer.writerow([
                row['event_type'],
                'Yes' if row['is_drowsy'] else 'No',
                row['yawn_count'] or 0,
                row['eye_closed_frames'] or 0,
                row['processing_status'],
                row['take_type'],
                row['memo'] or '-',
                row['memo_user'] or '-',
                row['memo_time'] or '-',
                row['review_type'],
                details_str
            ])

        # Create response
        output = si.getvalue()
        si.close()

        return Response(
            output,
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename=all_drowsiness_predictions.csv',
                'Content-Type': 'text/csv'
            }
        )

    except Exception as e:
        return f"Error: {str(e)}", 500
    finally:
        conn.close()

# Detail view route
@app.route('/detail/<int:row_id>')
def detail_view(row_id):
    """Render the detail view for a specific row."""
    try:
        # Check if user is authenticated
        if not verify_auth():
            # Store the current URL in the session for redirect after login
            return redirect(url_for('login', next=request.url))

        conn = get_db_connection()

        # Get the specific row data
        cursor = conn.execute('''
            SELECT *
            FROM evidence_results
            WHERE id = ?
        ''', (row_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            # Return a custom 404 message for non-existent rows
            return render_template('404.html', message="The requested event could not be found."), 404

        # Parse the details JSON if it exists
        formatted_details = None
        if row['details']:
            try:
                import json
                details_dict = json.loads(row['details'])
                # Format the details for better display
                formatted_details = json.dumps(details_dict, indent=2)
            except Exception as e:
                logging.error(f"Error parsing details JSON: {e}")
                formatted_details = row['details']

        conn.close()

        return render_template('detail.html', row=row, formatted_details=formatted_details)

    except Exception as e:
        return f"Error: {str(e)}", 500

# Error handlers
@app.errorhandler(403)
def forbidden_error(error):
    if is_ajax_request():  # For AJAX requests
        return jsonify({'error': 'Forbidden'}), 403
    return redirect(url_for('login', next=request.url))

@app.errorhandler(401)
def unauthorized_error(error):
    if is_ajax_request():  # For AJAX requests
        return jsonify({'error': 'Unauthorized'}), 401
    return redirect(url_for('login', next=request.url))

@app.errorhandler(404)
def not_found_error(error):
    if is_ajax_request():  # For AJAX requests
        return jsonify({'error': 'Not Found'}), 404
    if not verify_auth():
        return redirect(url_for('login', next=request.url))
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    if is_ajax_request():  # For AJAX requests
        return jsonify({'error': 'Internal Server Error'}), 500
    return render_template('500.html'), 500

# Model Management Routes
@app.route('/models')
@login_required
def models():
    """Render the model management page."""
    try:
        conn = get_db_connection()
        cursor = conn.execute('''
            SELECT id, name, file_path, upload_date, is_active
            FROM models
            ORDER BY upload_date DESC
        ''')
        models = cursor.fetchall()
        conn.close()

        return render_template('model_management.html', models=models)
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/upload_model', methods=['POST'])
@login_required
def upload_model():
    """Handle model file upload."""
    try:
        # Check if the post request has the file part
        if 'model_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400

        file = request.files['model_file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        # Check file extension
        if not file.filename.endswith('.pt'):
            return jsonify({'success': False, 'error': 'Invalid file type. Only .pt files are allowed'}), 400

        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Get model name (use provided name or filename)
        model_name = request.form.get('model_name', '').strip()
        if not model_name:
            model_name = os.path.basename(file.filename)

        # Generate a unique filename to avoid overwriting
        filename = werkzeug.utils.secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{model_name}")
        if not filename.endswith('.pt'):
            filename += '.pt'

        # Save the file
        file_path = os.path.join(models_dir, filename)
        file.save(file_path)

        # Store model info in database
        conn = get_db_connection()
        cursor = conn.execute('''
            INSERT INTO models (name, file_path, upload_date, is_active)
            VALUES (?, ?, ?, 0)
        ''', (model_name, os.path.join('models', filename), datetime.now().isoformat()))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Model uploaded successfully'})
    except Exception as e:
        logging.error(f"Error uploading model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/set_active_model', methods=['POST'])
@login_required
def set_active_model():
    """Set a model as active."""
    try:
        data = request.json
        model_id = data.get('model_id')

        if not model_id:
            return jsonify({'success': False, 'error': 'Model ID is required'}), 400

        conn = get_db_connection()

        # First, get the model path
        cursor = conn.execute('SELECT file_path FROM models WHERE id = ?', (model_id,))
        model = cursor.fetchone()

        if not model:
            conn.close()
            return jsonify({'success': False, 'error': 'Model not found'}), 404

        # Reset all models to inactive
        conn.execute('UPDATE models SET is_active = 0')

        # Set the selected model as active
        conn.execute('UPDATE models SET is_active = 1 WHERE id = ?', (model_id,))

        # Update the YOLO_MODEL_PATH in the .env file
        model_path = model['file_path']
        update_env_file('YOLO_MODEL_PATH', model_path)

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Model set as active successfully'})
    except Exception as e:
        logging.error(f"Error setting active model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_model', methods=['POST'])
@login_required
def delete_model():
    """Delete a model from the database and file system."""
    try:
        data = request.json
        model_id = data.get('model_id')

        if not model_id:
            return jsonify({'success': False, 'error': 'Model ID is required'}), 400

        conn = get_db_connection()

        # First, check if the model exists and is not active
        cursor = conn.execute('SELECT file_path, is_active FROM models WHERE id = ?', (model_id,))
        model = cursor.fetchone()

        if not model:
            conn.close()
            return jsonify({'success': False, 'error': 'Model not found'}), 404

        if model['is_active'] == 1:
            conn.close()
            return jsonify({'success': False, 'error': 'Cannot delete the active model. Please set another model as active first.'}), 400

        # Get the file path to delete the file
        file_path = model['file_path']
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)

        # Delete the model from the database
        conn.execute('DELETE FROM models WHERE id = ?', (model_id,))
        conn.commit()
        conn.close()

        # Delete the file if it exists
        if os.path.exists(full_path):
            os.remove(full_path)
            logging.info(f"Deleted model file: {full_path}")

        return jsonify({'success': True, 'message': 'Model deleted successfully'})
    except Exception as e:
        logging.error(f"Error deleting model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/restart_detector', methods=['POST'])
@login_required
def restart_detector():
    """Restart the drowsiness detector process."""
    try:
        # Get the active model path
        conn = get_db_connection()
        cursor = conn.execute('SELECT file_path FROM models WHERE is_active = 1')
        active_model = cursor.fetchone()
        conn.close()

        if not active_model:
            return jsonify({'success': False, 'error': 'No active model found'}), 400

        # Try to find and kill the existing drowsiness_detector process
        try:
            # First, check if PID file exists
            pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drowsiness_detector.pid')
            if os.path.exists(pid_file):
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())

                    logging.info(f"Found PID file with process ID: {pid}")

                    # Try to terminate the process
                    if os.name == 'nt':  # Windows
                        subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:  # Linux/Mac
                        subprocess.run(['kill', '-9', str(pid)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Remove the PID file
                    os.remove(pid_file)
                    logging.info(f"Removed PID file after killing process")
                except Exception as e:
                    logging.warning(f"Error terminating process from PID file: {e}")

            # Fallback methods if PID file doesn't exist or process termination failed
            if os.name == 'nt':  # Windows
                # Find all Python processes running drowsiness_detector.py
                result = subprocess.run(['wmic', 'process', 'where',
                                      "commandline like '%drowsiness_detector.py%' and name like '%python%'",
                                      'get', 'processid'],
                                     capture_output=True, text=True, check=False)

                # Extract process IDs from the output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # First line is header (ProcessId)
                    for line in lines[1:]:  # Skip header
                        if line.strip():  # Skip empty lines
                            try:
                                pid = int(line.strip())
                                logging.info(f"Killing drowsiness_detector.py process with PID: {pid}")
                                subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            except ValueError:
                                continue

                # Additional fallback method
                subprocess.run(['taskkill', '/F', '/FI', "WINDOWTITLE eq *drowsiness_detector*"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:  # Linux/Mac
                subprocess.run(['pkill', '-f', 'drowsiness_detector.py'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Give processes time to terminate
            time.sleep(2)
        except Exception as e:
            logging.warning(f"Could not kill existing process: {e}")

        # Check if we're in a TMUX environment
        is_tmux = False
        try:
            # Check if TMUX is installed
            result = subprocess.run(['which', 'tmux'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                # Check if there's a TMUX session named "service1" (from the script)
                result = subprocess.run(['tmux', 'has-session', '-t', 'service1'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                is_tmux = result.returncode == 0
                logging.info(f"TMUX environment detected: {is_tmux}")
        except Exception as e:
            logging.warning(f"Error checking for TMUX environment: {e}")

        # Start the drowsiness_detector.py in a new process
        if os.name == 'nt':  # Windows
            subprocess.Popen(['python', 'drowsiness_detector.py'],
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux/Mac
            if is_tmux:
                # Restart in TMUX session
                logging.info("Restarting drowsiness_detector.py in TMUX session 'service1'")
                work_dir = os.path.dirname(os.path.abspath(__file__))

                # Check if the session exists
                result = subprocess.run(['tmux', 'has-session', '-t', 'service1'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if result.returncode == 0:
                    # Session exists, send commands to it
                    subprocess.run(['tmux', 'send-keys', '-t', 'service1', 'C-c', 'C-m'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(1)  # Give time for any cleanup

                    # Send commands to start the service
                    subprocess.run(['tmux', 'send-keys', '-t', 'service1', f'cd {work_dir}', 'C-m'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Check if virtual environment exists and activate it
                    venv_path = os.path.join(work_dir, '.venv/bin/activate')
                    if os.path.exists(venv_path):
                        subprocess.run(['tmux', 'send-keys', '-t', 'service1', f'source {venv_path}', 'C-m'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Start the detector
                    subprocess.run(['tmux', 'send-keys', '-t', 'service1', 'python3 drowsiness_detector.py', 'C-m'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    # Session doesn't exist, create it
                    logging.info("TMUX session 'service1' not found, creating new session")
                    subprocess.run(['tmux', 'new-session', '-d', '-s', 'service1'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Send commands to start the service
                    subprocess.run(['tmux', 'send-keys', '-t', 'service1', f'cd {work_dir}', 'C-m'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Check if virtual environment exists and activate it
                    venv_path = os.path.join(work_dir, '.venv/bin/activate')
                    if os.path.exists(venv_path):
                        subprocess.run(['tmux', 'send-keys', '-t', 'service1', f'source {venv_path}', 'C-m'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Start the detector
                    subprocess.run(['tmux', 'send-keys', '-t', 'service1', 'python3 drowsiness_detector.py', 'C-m'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                # Regular start without TMUX
                logging.info("Starting drowsiness_detector.py without TMUX")
                subprocess.Popen(['python3', 'drowsiness_detector.py'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return jsonify({'success': True, 'message': 'Drowsiness detector restarted successfully'})
    except Exception as e:
        logging.error(f"Error restarting detector: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Helper function to update .env file
def update_env_file(key, value):
    """Update a key in the .env file."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

    # Read the current .env file
    if os.path.exists(env_path):
        with open(env_path, 'r') as file:
            lines = file.readlines()
    else:
        lines = []

    # Find and replace the key, or add it if not found
    key_found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
            break

    if not key_found:
        lines.append(f"{key}={value}\n")

    # Write back to the .env file
    with open(env_path, 'w') as file:
        file.writelines(lines)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=PORT)
