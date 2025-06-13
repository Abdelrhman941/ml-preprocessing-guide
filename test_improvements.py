"""
Quick test script to validate the ML Studio improvements.
Run this to verify that the training error is fixed and metrics work correctly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import detect_task_type, create_model, get_model_metrics_summary
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes

def test_classification():
    """Test classification functionality with proper task detection."""
    print("🧪 Testing Classification...")
    
    # Load iris dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Test task type detection
    detected_task = detect_task_type(y)
    print(f"✅ Task type detected: {detected_task}")
    assert detected_task == 'classification', f"Expected 'classification', got '{detected_task}'"
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train model
    model = create_model("Random Forest", {'n_estimators': 10}, 'classification')
    model.fit(X_train, y_train)
    
    # Get metrics
    metrics = get_model_metrics_summary(model, X_test, y_test, 'classification')
    print(f"✅ Classification metrics: {metrics}")
    
    # Verify expected metrics are present
    expected_metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
    
    print("✅ Classification test passed!")
    return True

def test_regression():
    """Test regression functionality with proper task detection."""
    print("\n🧪 Testing Regression...")
    
    # Load diabetes dataset
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Test task type detection
    detected_task = detect_task_type(y)
    print(f"✅ Task type detected: {detected_task}")
    assert detected_task == 'regression', f"Expected 'regression', got '{detected_task}'"
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_model("Random Forest", {'n_estimators': 10}, 'regression')
    model.fit(X_train, y_train)
    
    # Get metrics
    metrics = get_model_metrics_summary(model, X_test, y_test, 'regression')
    print(f"✅ Regression metrics: {metrics}")
    
    # Verify expected metrics are present
    expected_metrics = ['R² Score', 'MSE', 'MAE', 'RMSE']
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
    
    print("✅ Regression test passed!")
    return True

def test_edge_cases():
    """Test edge cases that previously caused errors."""
    print("\n🧪 Testing Edge Cases...")
    
    # Test with continuous target that should be classification
    y_continuous_class = pd.Series([0.0, 1.0, 0.0, 1.0, 2.0, 2.0] * 10)  # Continuous but discrete classes
    detected = detect_task_type(y_continuous_class)
    print(f"✅ Continuous discrete target detected as: {detected}")
    assert detected == 'classification'
    
    # Test with truly continuous target
    y_continuous_reg = pd.Series(np.random.normal(0, 1, 100))
    detected = detect_task_type(y_continuous_reg)
    print(f"✅ Continuous target detected as: {detected}")
    assert detected == 'regression'
    
    # Test with string labels
    y_string = pd.Series(['cat', 'dog', 'cat', 'bird'] * 25)
    detected = detect_task_type(y_string)
    print(f"✅ String target detected as: {detected}")
    assert detected == 'classification'
    
    print("✅ Edge cases test passed!")
    return True

if __name__ == "__main__":
    print("🚀 ML Studio Validation Test")
    print("=" * 50)
    
    try:
        # Run all tests
        test_classification()
        test_regression() 
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Training error is fixed")
        print("✅ Task type detection works correctly")
        print("✅ All metrics are properly implemented")
        print("✅ ML Studio is ready for use!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Please check the implementation and try again.")
        sys.exit(1)
