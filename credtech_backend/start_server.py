#!/usr/bin/env python3
"""
Start script for the CredTech Backend server.
"""
import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the application
from app.main import app
import uvicorn

if __name__ == "__main__":
    print("Starting CredTech Backend Server...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
