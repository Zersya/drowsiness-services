#!/usr/bin/env python3
"""
Startup script for the Landmark-based Drowsiness Detection System
================================================================

This script provides a convenient way to start the landmark system with
proper initialization and error handling.

Usage:
    python start_landmark_system.py [options]

Options:
    --port PORT         Port to run the server on (default: 8003)
    --workers N         Number of worker threads (default: 1)
    --debug             Enable debug mode
    --config FILE       Path to configuration file (default: .env)
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('landmark_detection.log')
        ]
    )

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask', 'flask_cors', 'cv2', 'numpy', 'dlib', 
        'scipy', 'requests', 'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'flask_cors':
                import flask_cors
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r landmark_requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_model_files():
    """Check if required model files are available."""
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(predictor_path):
        print(f"⚠️  Facial landmark predictor not found at {predictor_path}")
        print("The system will attempt to download it automatically on first run.")
        print("This may take a few minutes depending on your internet connection.")
    else:
        print("✅ Facial landmark predictor found")
    
    return True

def main():
    """Main function to start the landmark system."""
    parser = argparse.ArgumentParser(
        description='Start the Landmark-based Drowsiness Detection System'
    )
    parser.add_argument('--port', type=int, default=8003,
                       help='Port to run the server on (default: 8003)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker threads (default: 1)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--config', default='.env',
                       help='Path to configuration file (default: .env)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    
    print("🚀 Starting Landmark-based Drowsiness Detection System")
    print("=" * 60)
    
    # Load configuration
    if os.path.exists(args.config):
        load_dotenv(args.config)
        print(f"✅ Configuration loaded from {args.config}")
    else:
        print(f"⚠️  Configuration file {args.config} not found, using defaults")
    
    # Override with command line arguments
    os.environ['LANDMARK_PORT'] = str(args.port)
    os.environ['LANDMARK_MAX_WORKERS'] = str(args.workers)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        sys.exit(1)
    
    print(f"🌐 Server will start on port {args.port}")
    print(f"👥 Using {args.workers} worker thread(s)")
    print(f"🔧 Debug mode: {'enabled' if args.debug else 'disabled'}")
    print("=" * 60)
    
    try:
        # Import and start the landmark API
        from landmark_api import app
        
        print("✅ Landmark system modules loaded successfully")
        print("🎯 Starting Flask server...")
        
        app.run(
            host='0.0.0.0',
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import landmark system modules: {e}")
        print("Please ensure all files are in the correct location:")
        required_files = [
            'landmark_api.py',
            'landmark_database.py',
            'landmark_processor.py',
            'landmark_analyzer.py',
            'landmark_queue.py',
            'landmark_webhook.py',
            'landmark_worker.py'
        ]
        for file in required_files:
            status = "✅" if os.path.exists(file) else "❌"
            print(f"  {status} {file}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal, shutting down...")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.exception("Unexpected error during startup")
        sys.exit(1)

if __name__ == '__main__':
    main()
