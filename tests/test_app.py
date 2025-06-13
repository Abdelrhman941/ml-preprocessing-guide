import sys
import time
import traceback
import os

# Add parent directory to path so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------ Test that all modules can be imported successfully ------
def test_imports():
    print("🧪 Testing imports...")
    
    try:
        import utils as utils
        print("✅ utils module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils: {e}")
        return False
    
    try:
        import config as config
        print("✅ config module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False
    try:
        import preprocessing.preprocessor as preprocessor
        print("✅ preprocessor module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import preprocessor: {e}")
        return False
    
    try:
        import utils.navigation as navigation
        print("✅ navigation module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import navigation: {e}")
        return False
    
    try:
        import pages as pages
        print("✅ pages module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pages: {e}")
        return False
    
    return True

def test_utility_functions():
    print("\n🧪 Testing utility functions...")
    
    try:
        from utils.helpers import load_sample_dataset, get_model_params, create_model
        
        # Test loading sample dataset
        df, task_type = load_sample_dataset("Iris (Classification)")    # Replace with your actual dataset name, if sample dataset : write (...)
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
    print("\n🧪 Testing MLPreprocessor...")    
    try:
        from preprocessing.preprocessor import MLPreprocessor
        from utils.helpers import load_sample_dataset
        
        # Load test data
        df, _ = load_sample_dataset("Iris (Classification)")  # Replace with your actual dataset name, if sample dataset : write (...)
        
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
    print("\n🧪 Testing configuration...")
    
    try:
        from config.settings import (
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
    print("🚀 Starting ML Studio Test Suite\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Utility Functions", test_utility_functions),
        ("Preprocessor Class", test_preprocessor),
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total  = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50} Running: {test_name} {'='*50}")
        
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} CRASHED: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*60} TEST RESULTS: {passed}/{total} tests passed {'='*60}")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to run.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

def check_dependencies():
    print("📦 Checking dependencies...")
    
    required_packages = [
        ('numpy', 'numpy'), 
        ('pandas', 'pandas'), 
        ('plotly', 'plotly'),
        ('sklearn', 'scikit-learn'), 
        ('xgboost', 'xgboost'), 
        ('lightgbm', 'lightgbm'), 
        ('imblearn', 'imbalanced-learn'),
        ('streamlit', 'streamlit')
    ]    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (missing). Try: pip install {package_name}")
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
    
    start_time = time.time()
    
    # Run tests
    tests_passed = run_all_tests()
    
    duration = time.time() - start_time
    print(f"⏱️  Duration: {duration:.2f}s")
    
    if tests_passed:
        print("\n🚀 Application validation successful!")
        print("You can now run the application using:")
        print("streamlit run ml_studio_app.py")
        sys.exit(0)
    else:
        print("\n❌ Application validation failed!")
        sys.exit(1)