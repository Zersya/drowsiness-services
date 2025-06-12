import logging
import time
import json
import datetime
import sqlite3
from landmark_database import LandmarkDatabaseManager
from landmark_processor import LandmarkDrowsinessProcessor
from landmark_analyzer import create_landmark_analyzer
from landmark_webhook import LandmarkWebhookManager

class LandmarkVideoWorker:
    """Worker class for processing videos using landmark-based drowsiness detection."""
    
    def __init__(self, db_manager: LandmarkDatabaseManager):
        self.db_manager = db_manager
        self.processor = LandmarkDrowsinessProcessor()
        self.analyzer = create_landmark_analyzer(analyzer_type="landmark")
        self.webhook_manager = LandmarkWebhookManager(db_manager)
    
    def process_video_task(self, queue_item):
        """
        Process a video from the queue using landmark-based detection.
        
        Args:
            queue_item: Dictionary containing 'id' and 'video_url'
        """
        queue_id = queue_item['id']
        video_url = queue_item['video_url']

        logging.info(f"Processing landmark video from queue: {queue_id}, URL: {video_url}")

        try:
            # Send webhook notification that processing has started
            self.webhook_manager.send_processing_started(queue_id, video_url)
            
            # Process the video
            start_time = time.time()
            processing_success, detection_results = self.processor.process_video(video_url)
            process_time = time.time() - start_time

            if not processing_success:
                # Update queue status to failed
                self.db_manager.update_queue_status(queue_id, 'failed')

                # Store the failed evidence with status 'failed'
                try:
                    # Create a minimal evidence record for the failed video
                    failed_evidence = {
                        'video_url': video_url,
                        'processing_status': 'failed',
                        'process_time': process_time,
                        'details': json.dumps({
                            'error': 'Failed to process video with landmark detection',
                            'detection_results': detection_results,
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                    }

                    # Insert into database with a direct SQL query to handle the failed case
                    with sqlite3.connect(self.db_manager.db_path) as conn:
                        conn.execute('''
                            INSERT INTO evidence_results (
                                video_url, process_time, processing_status, details
                            ) VALUES (?, ?, ?, ?)
                        ''', (
                            video_url,
                            process_time,
                            'failed',
                            failed_evidence['details']
                        ))
                        conn.commit()
                        logging.info(f"Stored failed landmark evidence record for {video_url}")

                    # Send webhook notification for failed processing
                    error_message = detection_results.get('error', 'Failed to process video')
                    self.webhook_manager.send_processing_failed(queue_id, video_url, error_message)
                    
                except Exception as e:
                    logging.error(f"Error storing failed landmark evidence record: {e}")

                return

            # Analyze drowsiness using landmark-based analyzer
            analysis_result = self.analyzer.analyze(detection_results)

            # Store results in database
            evidence_id = self.db_manager.store_evidence_result(
                video_url,
                detection_results,
                analysis_result,
                process_time,
                detection_results.get('head_pose')
            )

            # Update queue status
            self.db_manager.update_queue_status(queue_id, 'completed')

            logging.info(f"Completed processing landmark video from queue: {queue_id}, Evidence ID: {evidence_id}")

            # Send webhook notification for successful processing
            # Get the full evidence result to include in the webhook
            evidence_result = self.db_manager.get_evidence_result(evidence_id)
            self.webhook_manager.send_processing_completed(queue_id, evidence_id, video_url, evidence_result)

        except Exception as e:
            logging.error(f"Error processing landmark video from queue: {e}")
            self.db_manager.update_queue_status(queue_id, 'failed')

            # Send webhook notification for error
            self.webhook_manager.send_processing_failed(queue_id, video_url, str(e))

def create_landmark_worker(db_manager: LandmarkDatabaseManager):
    """Create and return a landmark video worker instance."""
    return LandmarkVideoWorker(db_manager)
