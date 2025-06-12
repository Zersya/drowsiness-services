import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from landmark_database import LandmarkDatabaseManager

class LandmarkQueueManager:
    """Queue manager for landmark-based drowsiness detection system."""
    
    def __init__(self, db_manager: LandmarkDatabaseManager, max_workers=1, queue_check_interval=5):
        self.db_manager = db_manager
        self.max_workers = max_workers
        self.queue_check_interval = queue_check_interval
        
        # Create thread pool for processing videos
        self.processing_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="LandmarkVideoProcessor"
        )
        logging.info(f"Created ThreadPoolExecutor with {max_workers} workers for landmark processing")
        
        # Flag to control the background worker
        self.shutdown_flag = threading.Event()
        
        # Background worker thread
        self.queue_worker_thread = None
        
    def start_queue_worker(self, process_video_task_func):
        """Start the background queue worker thread."""
        if self.queue_worker_thread is not None:
            logging.warning("Queue worker already started")
            return
            
        self.process_video_task_func = process_video_task_func
        self.queue_worker_thread = threading.Thread(target=self._queue_worker, daemon=False)
        self.queue_worker_thread.start()
        logging.info(f"Landmark queue worker thread started with ID: {self.queue_worker_thread.ident}")
    
    def stop_queue_worker(self):
        """Stop the background queue worker thread."""
        if self.queue_worker_thread is None:
            return
            
        logging.info("Stopping landmark queue worker...")
        self.shutdown_flag.set()
        
        # Wait for the queue worker to finish
        logging.info("Waiting for landmark queue worker to finish...")
        self.queue_worker_thread.join(timeout=10)
        
        # Shutdown the thread pool
        logging.info("Shutting down landmark thread pool...")
        self.processing_pool.shutdown(wait=True, cancel_futures=False)
        logging.info("Landmark queue worker shutdown complete")
    
    def _queue_worker(self):
        """Background worker to process videos from the queue."""
        logging.info("Starting landmark queue worker thread")

        while not self.shutdown_flag.is_set():
            try:
                # Get current processing status
                stats = self.db_manager.get_queue_stats()
                processing_count = stats.get('processing', 0)
                pending_count = stats.get('pending', 0)

                # Calculate how many more tasks we can submit
                available_workers = self.max_workers - processing_count

                logging.info(f"Landmark queue worker check - Processing: {processing_count}, Pending: {pending_count}, Available workers: {available_workers}, Max workers: {self.max_workers}")

                # If we have available workers and pending tasks, submit them
                if available_workers > 0 and pending_count > 0:
                    # Submit up to available_workers tasks
                    for _ in range(min(available_workers, pending_count)):
                        queue_item = self.db_manager.get_next_pending_video()
                        if queue_item:
                            # Submit the video for processing
                            self.processing_pool.submit(self.process_video_task_func, queue_item)
                            logging.info(f"Submitted landmark video for processing: {queue_item['id']}")
                        else:
                            # This shouldn't happen, but just in case
                            logging.warning("Expected pending video but none found")
                            break
                elif pending_count == 0:
                    logging.info("No pending videos in landmark queue")
                else:
                    logging.info(f"All landmark workers busy ({processing_count}/{self.max_workers})")

                # Wait before checking the queue again
                time.sleep(self.queue_check_interval)

            except Exception as e:
                logging.error(f"Error in landmark queue worker: {e}")
                time.sleep(self.queue_check_interval)

        logging.info("Landmark queue worker thread shutting down")
    
    def add_to_queue(self, video_url):
        """Add a video URL to the processing queue."""
        return self.db_manager.add_to_queue(video_url)
    
    def get_queue_status(self, queue_id):
        """Get the status of a queued item."""
        return self.db_manager.get_queue_status(queue_id)
    
    def get_queue_stats(self):
        """Get statistics about the processing queue."""
        stats = self.db_manager.get_queue_stats()
        
        # Get thread pool information
        active_workers = min(self.max_workers, stats.get('processing', 0))
        pending_tasks = stats.get('pending', 0)
        completed_tasks = stats.get('completed', 0)
        failed_tasks = stats.get('failed', 0)

        # Calculate utilization percentage
        worker_utilization = (active_workers / self.max_workers) * 100 if self.max_workers > 0 else 0

        # Get processing time statistics
        time_stats = self.db_manager.get_processing_time_stats()

        return {
            'stats': stats,
            'worker_threads': {
                'max': self.max_workers,
                'active': active_workers,
                'utilization_percent': worker_utilization
            },
            'tasks': {
                'pending': pending_tasks,
                'processing': stats.get('processing', 0),
                'completed': completed_tasks,
                'failed': failed_tasks,
                'total': pending_tasks + stats.get('processing', 0) + completed_tasks + failed_tasks
            },
            'time_metrics': {
                'average_processing_task_time': round(time_stats['avg_processing_time'], 2),
                'average_queue_wait_time': round(time_stats['avg_queue_wait_time'], 2)
            },
            'queue_check_interval': self.queue_check_interval
        }
    
    def get_all_queue_items(self, limit=100, offset=0):
        """Get all queue items with pagination."""
        return self.db_manager.get_all_queue_items(limit, offset)
    
    def get_queue_items_count(self):
        """Get total count of queue items."""
        return self.db_manager.get_queue_items_count()
