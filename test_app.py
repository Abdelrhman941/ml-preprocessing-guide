import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing imports...")
    
    try:
        import gui.utils as utils
        print("✅ utils module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils: {e}")
        return False
    
    try:
        import config
        print("✅ config module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    try:
        import gui.preprocessor as preprocessor
        print("✅ preprocessor module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import preprocessor: {e}")
        return False
    
    try:
        import gui.navigation as navigation
        print("✅ navigation module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import navigation: {e}")
        return False
    
    try:
        import gui.pages as pages
        print("✅ pages module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pages: {e}")
        return False
    
    return True

def test_utility_functions():
    """Test utility functions."""
    print("\n🧪 Testing utility functions...")
    
    try:
        from gui.utils import load_sample_dataset, get_model_params, create_model
        
        # Test loading sample dataset
        df, task_type = load_sample_dataset("Iris")
        if df is not None and task_type == 'classification':
            print("✅ Sample dataset loading works")
        else:
            print("❌ Sample dataset loading failed")
            return False
        
        # Test model parameters
        params = get_model_params("Random Forest")
        if isinstance(params, dict) and len(params) > 0:
            print("✅ Model parameter retrieval works")
        else:
            print("❌ Model parameter retrieval failed")
            return False
        
        # Test model creation
        model = create_model("Random Forest", {'n_estimators': 10}, 'classification')
        if model is not None:
            print("✅ Model creation works")
        else:
            print("❌ Model creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        traceback.print_exc()
        return False

def test_preprocessor():
    """Test MLPreprocessor class."""
    print("\n🧪 Testing MLPreprocessor...")
    
    try:
        from gui.preprocessor import MLPreprocessor
        from gui.utils import load_sample_dataset
        
        # Load test data
        df, _ = load_sample_dataset("Iris")
        
        # Initialize preprocessor
        preprocessor = MLPreprocessor()
        
        # Test data overview
        overview = preprocessor.data_overview(df, sample_size=3)
        if isinstance(overview, dict) and 'shape' in overview:
            print("✅ Data overview works")
        else:
            print("❌ Data overview failed")
            return False
        
        # Test missing data handling
        df_processed = preprocessor.handle_missing_data(df)
        if df_processed is not None:
            print("✅ Missing data handling works")
        else:
            print("❌ Missing data handling failed")
            return False
        
        # Test duplicate detection
        duplicate_info = preprocessor.detect_duplicates(df)
        if isinstance(duplicate_info, dict):
            print("✅ Duplicate detection works")
        else:
            print("❌ Duplicate detection failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessor test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration module."""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import (
            APP_CONFIG, MODEL_CONFIG, PREPROCESSING_CONFIG, 
            HYPERPARAMETER_GRIDS, COLOR_SCHEMES
        )
        
        # Check required configurations exist
        required_configs = [
            APP_CONFIG, MODEL_CONFIG, PREPROCESSING_CONFIG,
            HYPERPARAMETER_GRIDS, COLOR_SCHEMES
        ]
        
        for config in required_configs:
            if not isinstance(config, dict) or len(config) == 0:
                print("❌ Configuration validation failed")
                return False
        
        print("✅ Configuration module works")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting ML Studio Test Suite\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Utility Functions", test_utility_functions),
        ("Preprocessor Class", test_preprocessor),
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print('='*60)
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'), 
        ('sklearn', 'scikit-learn'), 
        ('plotly', 'plotly'),
        ('streamlit', 'streamlit'), 
        ('xgboost', 'xgboost'), 
        ('lightgbm', 'lightgbm'), 
        ('imblearn', 'imbalanced-learn')
    ]    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (missing)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 All dependencies are installed!")
        return True

if __name__ == "__main__":
    print("🚀 ML Studio Application Validator")
    print("="*50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Run tests
    if run_all_tests():
        print("\n🚀 Application validation successful!")
        print("You can now run the application using:")
        print("streamlit run ml_studio_app.py")
        sys.exit(0)
    else:
        print("\n❌ Application validation failed!")
        sys.exit(1)
