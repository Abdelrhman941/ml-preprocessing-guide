"""
Test Arrow compatibility fixes for ML Studio app
"""
import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import sys

def test_arrow_compatibility():
    """Test various scenarios that could cause Arrow serialization issues"""
    print("Testing Arrow Compatibility Fixes")
    print("=" * 50)
    
    # Test 1: Target statistics DataFrame (mixed numeric types)
    print("\n1. Testing Target Statistics DataFrame...")
    try:
        # Simulate target column data
        target_data = pd.Series([1.5, 2.3, 3.1, 4.2, 5.8, 6.4, 7.1])
        
        # Create stats DataFrame with proper string formatting
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{target_data.mean():.4f}",
                f"{target_data.median():.4f}",
                f"{target_data.std():.4f}",
                f"{target_data.min():.4f}",
                f"{target_data.max():.4f}",
                f"{target_data.skew():.4f}",
                f"{target_data.kurtosis():.4f}"
            ]
        })
        
        # Test Arrow serialization
        import pyarrow as pa
        table = pa.Table.from_pandas(stats_df)
        print("✅ Target statistics DataFrame - Arrow compatible")
        
    except Exception as e:
        print(f"❌ Target statistics DataFrame failed: {e}")
        return False
    
    # Test 2: Prediction statistics DataFrame (mixed types)
    print("\n2. Testing Prediction Statistics DataFrame...")
    try:
        # Simulate prediction comparison data
        comparison_df = pd.DataFrame({
            'Error': [0.1, -0.2, 0.3, -0.1, 0.4],
            'Absolute Error': [0.1, 0.2, 0.3, 0.1, 0.4],
            'Correct': [True, False, True, True, False]
        })
        
        # Test regression stats (all formatted as strings)
        reg_stats_df = pd.DataFrame({
            'Metric': ['Mean Error', 'Mean Absolute Error', 'Max Error', 'Min Error'],
            'Value': [
                f"{comparison_df['Error'].mean():.4f}",
                f"{comparison_df['Absolute Error'].mean():.4f}",
                f"{comparison_df['Error'].max():.4f}",
                f"{comparison_df['Error'].min():.4f}"
            ]
        })
        
        # Test classification stats (all formatted as strings)
        accuracy = comparison_df['Correct'].mean()
        class_stats_df = pd.DataFrame({
            'Metric': ['Accuracy (Sample)', 'Correct Predictions', 'Total Predictions'],
            'Value': [f"{accuracy:.3f}", str(comparison_df['Correct'].sum()), str(len(comparison_df))]
        })
        
        # Test Arrow serialization for both
        table1 = pa.Table.from_pandas(reg_stats_df)
        table2 = pa.Table.from_pandas(class_stats_df)
        print("✅ Prediction statistics DataFrames - Arrow compatible")
        
    except Exception as e:
        print(f"❌ Prediction statistics DataFrames failed: {e}")
        return False
    
    # Test 3: Feature importance DataFrame
    print("\n3. Testing Feature Importance DataFrame...")
    try:
        feature_importance_df = pd.DataFrame({
            'Feature': ['feature1', 'feature2', 'feature3'],
            'Importance': [0.45, 0.32, 0.23]
        })
        
        table = pa.Table.from_pandas(feature_importance_df)
        print("✅ Feature importance DataFrame - Arrow compatible")
        
    except Exception as e:
        print(f"❌ Feature importance DataFrame failed: {e}")
        return False
    
    # Test 4: Mixed object types that could cause issues
    print("\n4. Testing Mixed Object Types...")
    try:
        # Simulate problematic DataFrame with mixed types in object column
        problematic_df = pd.DataFrame({
            'Column1': ['a', 'b', 'c'],
            'Column2': [1, 2.5, 'mixed'],  # Mixed types
            'Column3': [np.int64(1), np.int64(2), np.int64(3)]  # numpy integers
        })
        
        # Apply our compatibility fix
        from preprocessor import MLPreprocessor
        preprocessor = MLPreprocessor()
        fixed_df = preprocessor.ensure_arrow_compatibility(problematic_df)
        
        table = pa.Table.from_pandas(fixed_df)
        print("✅ Mixed types DataFrame - Arrow compatible after fix")
        
    except Exception as e:
        print(f"❌ Mixed types DataFrame failed: {e}")
        return False
    
    # Test 5: Binning results that could have interval types
    print("\n5. Testing Binning Results...")
    try:
        # Simulate binned data
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        binned = pd.cut(data, bins=3, labels=['Low', 'Medium', 'High'])
        
        binned_df = pd.DataFrame({
            'Original': data,
            'Binned': binned.astype(str)  # Convert to string to avoid interval issues
        })
        
        table = pa.Table.from_pandas(binned_df)
        print("✅ Binning results DataFrame - Arrow compatible")
        
    except Exception as e:
        print(f"❌ Binning results DataFrame failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All Arrow compatibility tests passed!")
    return True

if __name__ == "__main__":
    test_arrow_compatibility()
