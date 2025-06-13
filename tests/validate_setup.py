#!/usr/bin/env python3
"""
Quick validation script for ML Studio after restructuring
"""

import sys
import traceback
import os

# Add parent directory to path so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_critical_imports():
    """Test the most critical imports that caused the error"""
    
    print("🧪 Testing Critical Imports")
    print("=" * 50)
    
    tests = [
        ("preprocessing.preprocessor", "MLPreprocessor"),
        ("utils.helpers", "load_sample_dataset"),
        ("config.settings", "APP_CONFIG"),
        ("pages.main_pages", "render_home_page"),
        ("utils.navigation", "initialize_session_state"),
        ("training.trainer", "create_model")
    ]
    
    passed = 0
    failed = 0
    
    for module, item in tests:
        try:
            exec(f"from {module} import {item}")
            print(f"✅ {module}.{item}")
            passed += 1
        except ImportError as e:
            print(f"❌ {module}.{item} - {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {module}.{item} - {e}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0

def test_main_app():
    """Test if the main app can be imported"""
    print("\n🧪 Testing Main Application")
    print("=" * 50)
    
    try:
        import ml_studio_app
        print("✅ Main application imports successfully")
        return True
    except Exception as e:
        print(f"❌ Main application failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_compatibility():
    """Test if Streamlit components work"""
    print("\n🧪 Testing Streamlit Compatibility")
    print("=" * 50)
    
    try:
        import streamlit as st
        from utils.navigation import initialize_session_state
        from pages.main_pages import render_home_page
        
        print("✅ Streamlit imports work")
        print("✅ Navigation functions available")
        print("✅ Page rendering functions available")
        return True
    except Exception as e:
        print(f"❌ Streamlit compatibility issue: {e}")
        return False

def main():
    print("🚀 ML Studio - Post-Restructuring Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Test critical imports
    if not test_critical_imports():
        all_passed = False
    
    # Test main app
    if not test_main_app():
        all_passed = False
    
    # Test Streamlit compatibility
    if not test_streamlit_compatibility():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! ML Studio is ready to use.")
        print("\nTo run the application:")
        print("  streamlit run ml_studio_app.py")
        print("\nThen open: http://localhost:8501")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
