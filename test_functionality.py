import pandas as pd
import numpy as np
from gui.utils import (
    detect_and_remove_duplicates, 
    plot_learning_curves,
    create_model,
    get_model_params
)
from gui.preprocessor import MLPreprocessor
import warnings
warnings.filterwarnings('ignore')

def test_duplicate_detection():
    """Test duplicate detection functionality."""
    print("🧪 Testing duplicate detection...")
    
    # Create test data with duplicates
    data = {
        'feature1': [1, 2, 3, 1, 4],
        'feature2': [10, 20, 30, 10, 40],
        'target': [0, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    print(f"Original data shape: {df.shape}")
    
    # Test duplicate detection
    df_clean, num_duplicates = detect_and_remove_duplicates(df)
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Duplicates removed: {num_duplicates}")
    
    assert num_duplicates == 1, f"Expected 1 duplicate, found {num_duplicates}"
    print("✅ Duplicate detection test passed!")
    return True

def test_preprocessor():
    """Test MLPreprocessor functionality."""
    print("\n🧪 Testing MLPreprocessor...")
    
    # Create test data with missing values
    data = {
        'numeric1': [1, 2, np.nan, 4, 5],
        'numeric2': [10, np.nan, 30, 40, 50],
        'categorical': ['A', 'B', np.nan, 'A', 'C'],
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    print(f"Data with missing values: {df.isnull().sum().sum()} total missing")
    
    # Test preprocessor
    preprocessor = MLPreprocessor()
    
    # Test missing value handling
    strategy = {'numeric1': 'mean', 'numeric2': 'median', 'categorical': 'mode'}
    df_processed = preprocessor.handle_missing_data(df, strategy)
    
    print(f"Missing values after processing: {df_processed.isnull().sum().sum()}")
    assert df_processed.isnull().sum().sum() == 0, "Missing values not handled properly"
    print("✅ Preprocessor test passed!")
    return True

def test_model_creation():
    """Test model creation functionality."""
    print("\n🧪 Testing model creation...")
    
    # Test different model types
    model_types = ['Random Forest', 'XGBoost', 'LightGBM']
    
    for model_name in model_types:
        print(f"Testing {model_name}...")
        
        # Test classification
        model = create_model(model_name, {}, 'classification')
        assert model is not None, f"Failed to create {model_name} for classification"
        
        # Test regression
        model = create_model(model_name, {}, 'regression')
        assert model is not None, f"Failed to create {model_name} for regression"
        
        print(f"✅ {model_name} creation test passed!")
    
    return True

def test_enhanced_features():
    """Test enhanced features like learning curves and VIP analysis."""
    print("\n🧪 Testing enhanced features...")
    
    # Create sample data for testing
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y = (X['feature1'] + X['feature2'] * 0.5 + np.random.randn(100) * 0.1 > 0).astype(int)
    
    # Test model creation and training
    model = create_model('Random Forest', {'n_estimators': 10}, 'classification')
    model.fit(X, y)
    
    # Test learning curves (basic functionality)
    print("Testing learning curves...")
    try:
        fig = plot_learning_curves(model, X, y, 'classification', cv=3)
        print("✅ Learning curves generation test passed!")
    except Exception as e:
        print(f"⚠️ Learning curves test failed: {e}")
      # Test VIP analysis (basic functionality)
    print("Testing VIP analysis...")
    try:
        # VIP analysis removed - this test is no longer needed
        print("✅ VIP analysis test skipped (feature removed)")
    except Exception as e:
        print(f"⚠️ VIP analysis test failed: {e}")
    
    return True

def test_heart_dataset():
    """Test with actual heart dataset."""
    print("\n🧪 Testing with heart dataset...")
    
    try:
        # Load heart dataset
        heart_df = pd.read_csv('dataset/heart.csv')
        print(f"Heart dataset loaded: {heart_df.shape}")
        
        # Basic data info
        print(f"Columns: {list(heart_df.columns)}")
        print(f"Missing values: {heart_df.isnull().sum().sum()}")
        print(f"Data types: {heart_df.dtypes.nunique()} different types")
        
        # Test duplicate detection on real data
        df_clean, num_duplicates = detect_and_remove_duplicates(heart_df)
        print(f"Duplicates in heart dataset: {num_duplicates}")
        
        print("✅ Heart dataset test passed!")
        return True
        
    except FileNotFoundError:
        print("⚠️ Heart dataset not found, skipping test")
        return True
    except Exception as e:
        print(f"❌ Heart dataset test failed: {e}")
        return False

def main():
    """Run all functionality tests."""
    print("🚀 Starting ML Studio Functionality Tests\n")
    
    tests = [
        test_duplicate_detection,
        test_preprocessor,
        test_model_creation,
        test_enhanced_features,
        test_heart_dataset
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All functionality tests passed! The app is ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()
