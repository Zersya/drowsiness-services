#!/usr/bin/env python3
"""
Landmark-based Drowsiness Detection API
======================================

A modular drowsiness detection system based on facial landmarks and PERCLOS analysis.
This system maintains the same API interface as simplify.py while using landmark-based
detection instead of YOLO-based detection.

Features:
- Facial landmark detection using dlib
- PERCLOS (Percentage of Eye Closure) analysis
- Eye Aspect Ratio (EAR) calculation
- Blink frequency analysis
- Queue-based video processing
- Webhook notifications
- SQLite database storage
- RESTful API interface

Author: Augment Agent
Date: December 2024
"""

import os
import logging
import signal
import sqlite3
import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# Import landmark system modules
from landmark_database import LandmarkDatabaseManager
from landmark_queue import LandmarkQueueManager
from landmark_worker import create_landmark_worker
from landmark_webhook import LandmarkWebhookManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler('landmark_detection.log')  # Save to file
    ]
)

# Load environment variables
load_dotenv()

# Configuration
PORT = int(os.getenv('LANDMARK_PORT', 8003))
MAX_WORKERS = int(os.getenv('LANDMARK_MAX_WORKERS', 1))
QUEUE_CHECK_INTERVAL = int(os.getenv('LANDMARK_QUEUE_CHECK_INTERVAL', 5))
DB_PATH = os.getenv("LANDMARK_DB_PATH", "landmark_detection.db")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
db_manager = LandmarkDatabaseManager()
queue_manager = LandmarkQueueManager(db_manager, MAX_WORKERS, QUEUE_CHECK_INTERVAL)
worker = create_landmark_worker(db_manager)
webhook_manager = LandmarkWebhookManager(db_manager)

# Start the queue worker
queue_manager.start_queue_worker(worker.process_video_task)

# API Endpoints
@app.route('/')
def index():
    """API root endpoint."""
    return jsonify({
        'message': 'Landmark-based Drowsiness Detection API',
        'version': '1.0.0',
        'detection_method': 'facial_landmarks',
        'features': [
            'PERCLOS analysis',
            'Eye Aspect Ratio (EAR) calculation',
            'Blink frequency analysis',
            'Queue-based processing',
            'Webhook notifications'
        ],
        'endpoints': [
            '/api/process',          # Add a video to the processing queue
            '/api/queue/<id>',       # Check the status of a queued video
            '/api/queue',            # Get queue statistics
            '/api/results',          # Get all processed videos
            '/api/result/<id>',      # Get details for a specific processed video
            '/api/webhook',          # Manage webhooks (GET, POST, DELETE)
            '/api/download/db',      # Download the SQLite database file
            '/api/precision',        # Calculate precision metrics
        ]
    })

@app.route('/api/process', methods=['POST'])
def process_video():
    """Add a video to the processing queue."""
    try:
        # Get video URL from request
        data = request.get_json()
        if not data or 'video_url' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing video_url in request'
            }), 400

        video_url = data['video_url']

        # Add to processing queue
        queue_id = queue_manager.add_to_queue(video_url)
        if not queue_id:
            return jsonify({
                'success': False,
                'error': 'Failed to add video to processing queue'
            }), 500

        # Return queue ID immediately
        return jsonify({
            'success': True,
            'message': 'Video added to landmark processing queue',
            'data': {
                'queue_id': queue_id,
                'status': 'pending',
                'status_url': f'/api/queue/{queue_id}',
                'detection_method': 'facial_landmarks'
            }
        })

    except Exception as e:
        logging.error(f"Error adding video to landmark queue: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/queue/<int:queue_id>', methods=['GET'])
def get_queue_status(queue_id):
    """Get the status of a queued video."""
    try:
        # Get queue status
        queue_status = queue_manager.get_queue_status(queue_id)
        if not queue_status:
            return jsonify({
                'success': False,
                'error': f'Queue item with ID {queue_id} not found'
            }), 404

        # Check if processing is complete
        if queue_status['status'] == 'completed':
            # Find the evidence result
            with sqlite3.connect(db_manager.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    'SELECT id FROM evidence_results WHERE video_url = ? ORDER BY created_at DESC LIMIT 1',
                    (queue_status['video_url'],)
                )
                evidence = cursor.fetchone()
                evidence_id = evidence['id'] if evidence else None

            return jsonify({
                'success': True,
                'data': {
                    'queue_id': queue_id,
                    'status': queue_status['status'],
                    'video_url': queue_status['video_url'],
                    'created_at': queue_status['created_at'],
                    'evidence_id': evidence_id,
                    'result_url': f'/api/result/{evidence_id}' if evidence_id else None,
                    'detection_method': 'facial_landmarks'
                }
            })
        elif queue_status['status'] == 'failed':
            return jsonify({
                'success': True,
                'data': {
                    'queue_id': queue_id,
                    'status': queue_status['status'],
                    'video_url': queue_status['video_url'],
                    'created_at': queue_status['created_at'],
                    'error': 'Video processing failed',
                    'detection_method': 'facial_landmarks'
                }
            })
        else:  # pending or processing
            return jsonify({
                'success': True,
                'data': {
                    'queue_id': queue_id,
                    'status': queue_status['status'],
                    'video_url': queue_status['video_url'],
                    'created_at': queue_status['created_at'],
                    'detection_method': 'facial_landmarks'
                }
            })

    except Exception as e:
        logging.error(f"Error getting landmark queue status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/queue', methods=['GET'])
def get_queue_stats():
    """Get statistics about the processing queue."""
    try:
        # Check if this is a request for paginated queue items
        page = request.args.get('page', type=int)

        if page is not None:
            # Return paginated queue items
            return get_queue_items_paginated()

        # Get queue stats (original functionality)
        stats_data = queue_manager.get_queue_stats()

        return jsonify({
            'success': True,
            'data': {
                **stats_data,
                'detection_method': 'facial_landmarks',
                'system': 'landmark_drowsiness_detection'
            }
        })

    except Exception as e:
        logging.error(f"Error getting landmark queue stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_queue_items_paginated():
    """Get paginated queue items."""
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        # Validate pagination parameters
        page = max(1, page)
        per_page = max(1, min(100, per_page))  # Limit per_page to 100

        # Calculate offset
        offset = (page - 1) * per_page

        # Get total count and results from database
        total_count = queue_manager.get_queue_items_count()
        results = queue_manager.get_all_queue_items(per_page, offset)

        # Calculate pagination metadata
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
        has_next = page < total_pages
        has_previous = page > 1

        return jsonify({
            'success': True,
            'data': results,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_items': total_count,
                'has_next': has_next,
                'has_previous': has_previous,
                'next_page': page + 1 if has_next else None,
                'previous_page': page - 1 if has_previous else None
            },
            'detection_method': 'facial_landmarks'
        })

    except Exception as e:
        logging.error(f"Error getting paginated landmark queue items: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/results')
def get_results():
    """Get all evidence results with pagination."""
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        # Validate pagination parameters
        page = max(1, page)
        per_page = max(1, min(100, per_page))  # Limit per_page to 100

        # Calculate offset
        offset = (page - 1) * per_page

        # Get total count and results from database
        total_count = db_manager.get_evidence_results_count()
        results = db_manager.get_all_evidence_results(per_page, offset)

        # Calculate pagination metadata
        total_pages = (total_count + per_page - 1) // per_page if total_count > 0 else 1
        has_next = page < total_pages
        has_previous = page > 1

        return jsonify({
            'success': True,
            'data': results,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_items': total_count,
                'has_next': has_next,
                'has_previous': has_previous,
                'next_page': page + 1 if has_next else None,
                'previous_page': page - 1 if has_previous else None
            },
            'detection_method': 'facial_landmarks'
        })

    except Exception as e:
        logging.error(f"Error getting landmark results: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/result/<int:evidence_id>')
def get_result(evidence_id):
    """Get a specific evidence result by ID."""
    try:
        # Get result from database
        result = db_manager.get_evidence_result(evidence_id)

        if not result:
            return jsonify({
                'success': False,
                'error': f'Evidence result with ID {evidence_id} not found'
            }), 404

        # Add detection method info
        result['detection_method'] = 'facial_landmarks'

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logging.error(f"Error getting landmark result: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/webhook', methods=['GET', 'POST', 'DELETE'])
def manage_webhooks():
    """Manage webhooks for video processing notifications."""
    try:
        # GET: List all webhooks
        if request.method == 'GET':
            webhooks = webhook_manager.get_all_webhooks()
            return jsonify({
                'success': True,
                'data': webhooks,
                'detection_method': 'facial_landmarks'
            })

        # POST: Add a new webhook
        elif request.method == 'POST':
            data = request.get_json()
            if not data or 'url' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing url in request'
                }), 400

            webhook_url = data['url']

            # Validate URL format
            try:
                parsed_url = requests.utils.urlparse(webhook_url)
                if not all([parsed_url.scheme, parsed_url.netloc]):
                    return jsonify({
                        'success': False,
                        'error': 'Invalid URL format'
                    }), 400
            except Exception:
                return jsonify({
                    'success': False,
                    'error': 'Invalid URL format'
                }), 400

            # Add webhook to database
            webhook_id = webhook_manager.add_webhook(webhook_url)
            if not webhook_id:
                return jsonify({
                    'success': False,
                    'error': 'Failed to add webhook'
                }), 500

            return jsonify({
                'success': True,
                'message': 'Webhook added successfully',
                'data': {
                    'webhook_id': webhook_id,
                    'url': webhook_url,
                    'detection_method': 'facial_landmarks'
                }
            })

        # DELETE: Remove a webhook
        elif request.method == 'DELETE':
            data = request.get_json()
            if not data or 'webhook_id' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing webhook_id in request'
                }), 400

            webhook_id = data['webhook_id']

            # Delete webhook from database
            success = webhook_manager.delete_webhook(webhook_id)
            if not success:
                return jsonify({
                    'success': False,
                    'error': f'Failed to delete webhook with ID {webhook_id}'
                }), 500

            return jsonify({
                'success': True,
                'message': f'Webhook with ID {webhook_id} deleted successfully'
            })

    except Exception as e:
        logging.error(f"Error managing landmark webhooks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/download/db', methods=['GET'])
def download_database():
    """Download the SQLite database file."""
    try:
        # Check if the database file exists
        if not os.path.exists(DB_PATH):
            return jsonify({
                'success': False,
                'error': f'Database file not found at {DB_PATH}'
            }), 404

        # Get the file size for logging
        file_size = os.path.getsize(DB_PATH)
        logging.info(f"Serving landmark database file: {DB_PATH} (size: {file_size / 1024:.2f} KB)")

        # Generate a filename for the download
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"landmark_detection_{timestamp}.db"

        # Return the file as an attachment
        return send_file(
            DB_PATH,
            as_attachment=True,
            download_name=download_name,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        logging.error(f"Error downloading landmark database file: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/precision', methods=['GET'])
def calculate_precision():
    """Calculate precision metrics for the landmark drowsiness detection system."""
    try:
        # Get all processed evidence results
        with sqlite3.connect(db_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT
                    id, is_drowsy, confidence, details, video_url
                FROM evidence_results
                WHERE processing_status = 'processed'
                ORDER BY created_at DESC
            ''')
            results = cursor.fetchall()

        if not results:
            return jsonify({
                'success': False,
                'error': 'No processed results found for precision calculation'
            }), 404

        # Calculate basic metrics
        total_predictions = len(results)
        drowsy_predictions = sum(1 for r in results if r['is_drowsy'])
        non_drowsy_predictions = total_predictions - drowsy_predictions

        # Calculate confidence statistics
        confidences = [r['confidence'] for r in results if r['confidence'] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # High confidence predictions (>0.7)
        high_confidence_predictions = sum(1 for c in confidences if c > 0.7)

        # Calculate precision-related metrics
        precision_metrics = {
            'total_predictions': total_predictions,
            'drowsy_predictions': drowsy_predictions,
            'non_drowsy_predictions': non_drowsy_predictions,
            'drowsy_percentage': (drowsy_predictions / total_predictions * 100) if total_predictions > 0 else 0,
            'average_confidence': round(avg_confidence, 3),
            'high_confidence_predictions': high_confidence_predictions,
            'high_confidence_percentage': (high_confidence_predictions / total_predictions * 100) if total_predictions > 0 else 0,
            'detection_method': 'facial_landmarks',
            'model_version': 'Landmark-Based v1.0',
            'features': [
                'PERCLOS (Percentage of Eye Closure) analysis',
                'Eye Aspect Ratio (EAR) calculation',
                'Blink frequency analysis',
                'Facial landmark detection using dlib',
                'Real-time video processing'
            ]
        }

        return jsonify({
            'success': True,
            'data': precision_metrics,
            'note': 'Precision calculation requires ground truth labels for accurate TP/FP/TN/FN metrics'
        })

    except Exception as e:
        logging.error(f"Error calculating landmark precision: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Signal handler for graceful shutdown
def signal_handler(sig, _):
    logging.info(f"Received signal {sig}, shutting down landmark system...")
    # Stop the queue worker
    queue_manager.stop_queue_worker()
    logging.info("Landmark system shutdown complete")
    import sys
    sys.exit(0)


# Main function
if __name__ == '__main__':
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info(f"Starting Landmark-based Drowsiness Detection API on port {PORT}")
    try:
        app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        # This should be handled by the signal handler, but just in case
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        signal_handler(signal.SIGTERM, None)
