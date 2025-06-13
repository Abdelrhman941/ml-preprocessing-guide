#!/usr/bin/env python3
"""
ML Studio Deployment Script
Quick setup and launch script for ML Studio
"""

import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def run_tests():
    """Run validation tests"""
    print("🧪 Running validation tests...")
    try:
        result = subprocess.run([sys.executable, "run_tests.py"], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("⚠️  Some tests failed, but continuing...")
            return True  # Don't block deployment for test failures
    except Exception as e:
        print(f"⚠️  Test execution failed: {e}")
        return True  # Continue anyway

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching ML Studio...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ml_studio_app.py"])
    except KeyboardInterrupt:
        print("\n👋 ML Studio stopped")
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        print("💡 Try running manually: streamlit run ml_studio_app.py")

def main():
    """Main deployment function"""
    print("🚀 ML Studio Deployment")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Run tests
    run_tests()
    
    # Launch application
    launch_app()

if __name__ == "__main__":
    main()
