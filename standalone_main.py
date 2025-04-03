#!/usr/bin/env python3
"""
EternaLattice Standalone Node Main Entry Point

This script provides a convenient way to run EternaLattice in standalone mode,
which includes both the node and the web interface.
"""
import Blockchain
import os
import sys
import logging
import argparse
import time
import webbrowser
from threading import Thread

from eternalattice_node import start_node, initialize_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eternalattice_standalone.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('eternalattice.standalone')

def main():
    """Main entry point for the standalone node."""
    parser = argparse.ArgumentParser(description="EternaLattice Standalone Node")
    parser.add_argument('--port', type=int, default=5000, help='Port for web interface (default: 5000)')
    parser.add_argument('--data-dir', help='Data directory path (default: ~/.eternalattice)')
    parser.add_argument('--peers', help='Comma-separated list of peer addresses to connect to')
    parser.add_argument('--no-browser', action='store_true', help='Do not automatically open browser')
    args = parser.parse_args()
    
    try:
        logger.info("Starting EternaLattice standalone node")
        
        # Initialize database
        db_path = None
        if args.data_dir:
            db_path = os.path.join(args.data_dir, "eternalattice.db")
        initialize_database(db_path)
        
        # Convert node start parameters to match eternalattice_node.py format
        node_args = argparse.Namespace()
        node_args.peers = args.peers
        node_args.port = args.port
        node_args.web = True
        
        # Start the node (this will also start the web interface)
        start_result = start_node(node_args)
        
        if start_result:
            logger.info("Standalone node started successfully")
            
            # If --no-browser was not specified, open the browser
            if not args.no_browser:
                # Small delay to ensure the server is ready
                time.sleep(2)
                webbrowser.open(f"http://localhost:{args.port}")
                logger.info(f"Opened browser to http://localhost:{args.port}")
            
            try:
                # Keep the main thread running
                print("\nEternaLattice node is running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping node...")
                # Cleanup would go here
                logger.info("Node stopped by user")
                sys.exit(0)
        else:
            logger.error("Failed to start node")
            sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running standalone node: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()