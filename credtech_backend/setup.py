#!/usr/bin/env python3
"""
Setup script for CredTech Backend system.
This script helps with initial configuration and database setup.
"""
import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print the setup banner."""
    print("=" * 60)
    print("  CredTech Backend - Dynamic Hybrid Expert Model")
    print("  Setup and Configuration Script")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Error: Python 3.9 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required system dependencies are available."""
    dependencies = ['pip', 'git']
    missing = []
    
    for dep in dependencies:
        try:
            subprocess.run([dep, '--version'], capture_output=True, check=True)
            print(f"✅ {dep} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {dep} is not available")
            missing.append(dep)
    
    return len(missing) == 0


def install_packages():
    """Install Python packages from requirements.txt."""
    print("\n📦 Installing Python packages...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False


def setup_environment():
    """Set up environment variables."""
    print("\n🔧 Setting up environment configuration...")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists():
        if env_example.exists():
            # Copy example to .env
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ Created .env file from .env.example")
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
    
    print("\n⚠️  Please edit the .env file with your configuration:")
    print("   - Database credentials (PostgreSQL)")
    print("   - News API key (optional)")
    print("   - Elasticsearch URL (optional)")
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        'models',
        'logs',
        'data'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def display_next_steps():
    """Display next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Setup completed! Next steps:")
    print("=" * 60)
    print()
    print("1. 🗄️  Set up PostgreSQL database:")
    print("   - Install PostgreSQL if not already installed")
    print("   - Create database: credtech_db")
    print("   - Create user with appropriate permissions")
    print("   - Update DATABASE_URL in .env file")
    print()
    print("2. 🔧 Configure environment:")
    print("   - Edit .env file with your settings")
    print("   - Add News API key for real news data (optional)")
    print("   - Configure Elasticsearch if available (optional)")
    print()
    print("3. 🚀 Start the application:")
    print("   uvicorn app.main:app --reload")
    print()
    print("4. 📖 Access the API documentation:")
    print("   http://127.0.0.1:8000/docs")
    print()
    print("5. 🧪 Test the system:")
    print("   - Add companies via API")
    print("   - Train the model")
    print("   - Generate explanations")
    print()
    print("For detailed instructions, see README.md")
    print("=" * 60)


def main():
    """Main setup function."""
    print_banner()
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        print("\nPlease install missing dependencies and run setup again.")
        sys.exit(1)
    
    # Install packages
    if not install_packages():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Show next steps
    display_next_steps()


if __name__ == "__main__":
    main()
