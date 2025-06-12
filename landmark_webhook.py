import logging
import datetime
import requests
from landmark_database import LandmarkDatabaseManager

def send_webhook_notification(db_manager: LandmarkDatabaseManager, queue_id, evidence_id, status, video_url, results=None):
    """
    Send webhook notifications to all registered webhook URLs.
    
    Args:
        db_manager: Database manager instance
        queue_id: Queue item ID
        evidence_id: Evidence result ID (None if failed)
        status: Processing status ('completed', 'failed', 'processing')
        video_url: URL of the processed video
        results: Processing results (optional)
    """
    active_webhooks = db_manager.get_active_webhooks()
    if not active_webhooks:
        logging.info("No active webhooks found, skipping notifications")
        return

    # Prepare webhook payload
    payload = {
        'queue_id': queue_id,
        'evidence_id': evidence_id,
        'status': status,
        'video_url': video_url,
        'timestamp': datetime.datetime.now().isoformat(),
        'system': 'landmark_drowsiness_detection'  # Identifier for landmark system
    }

    # Add results if available
    if results:
        payload['results'] = results

    # Send to all active webhooks
    for webhook in active_webhooks:
        try:
            webhook_url = webhook['url']
            logging.info(f"Sending landmark webhook notification to {webhook_url}")

            response = requests.post(
                webhook_url,
                json=payload,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'LandmarkDrowsinessDetection/1.0'
                },
                timeout=10  # 10 second timeout
            )

            if response.status_code >= 200 and response.status_code < 300:
                logging.info(f"Webhook notification sent successfully to {webhook_url}")
            else:
                logging.warning(f"Webhook notification failed: {webhook_url}, Status: {response.status_code}, Response: {response.text}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Error sending webhook notification to {webhook['url']}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error sending webhook notification: {e}")

class LandmarkWebhookManager:
    """Manager class for webhook operations in the landmark system."""
    
    def __init__(self, db_manager: LandmarkDatabaseManager):
        self.db_manager = db_manager
    
    def send_processing_started(self, queue_id, video_url):
        """Send notification when processing starts."""
        send_webhook_notification(
            self.db_manager,
            queue_id=queue_id,
            evidence_id=None,
            status='processing',
            video_url=video_url,
            results=None
        )
    
    def send_processing_completed(self, queue_id, evidence_id, video_url, evidence_result):
        """Send notification when processing completes successfully."""
        # Format results to match expected webhook structure
        formatted_results = {
            'id': evidence_result.get('id'),
            'video_url': evidence_result.get('video_url'),
            'process_time': evidence_result.get('process_time'),
            'total_frames': evidence_result.get('total_frames'),
            'is_drowsy': 1 if evidence_result.get('is_drowsy') else 0,  # Convert boolean to 0/1
            'confidence': evidence_result.get('confidence'),
            'processing_status': evidence_result.get('processing_status'),
            'created_at': evidence_result.get('created_at')
        }

        send_webhook_notification(
            self.db_manager,
            queue_id=queue_id,
            evidence_id=evidence_id,
            status='completed',
            video_url=video_url,
            results=formatted_results
        )
    
    def send_processing_failed(self, queue_id, video_url, error_message):
        """Send notification when processing fails."""
        send_webhook_notification(
            self.db_manager,
            queue_id=queue_id,
            evidence_id=None,
            status='failed',
            video_url=video_url,
            results={'error': error_message}
        )
    
    def add_webhook(self, url):
        """Add a new webhook URL."""
        return self.db_manager.add_webhook(url)
    
    def delete_webhook(self, webhook_id):
        """Delete a webhook by ID."""
        return self.db_manager.delete_webhook(webhook_id)
    
    def get_all_webhooks(self):
        """Get all registered webhooks."""
        return self.db_manager.get_all_webhooks()
    
    def get_active_webhooks(self):
        """Get all active webhooks."""
        return self.db_manager.get_active_webhooks()
