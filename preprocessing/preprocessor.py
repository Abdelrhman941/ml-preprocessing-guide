import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import List, Dict, Tuple, Optional

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer

# Imbalanced learning imports
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
##############################################################
class MLPreprocessor:
    def __init__(self):
        self.scalers        = {}
        self.imputers       = {}
        self.encoders       = {}
        self.feature_names  = []
        self.steps_log      = []
        self.balancing_info = {}
    
    def log_step(self, step_name: str, details: str) -> None:
        """
        → Log preprocessing steps for tracking and reproducibility.
        Args:
            step_name : Name of the preprocessing step
            details   : Description of what was done
        """
        log_entry = f"✅ {step_name} : {details}"
        self.steps_log.append(log_entry)
        print(log_entry)
    
    # Get a summary of all preprocessing steps performed
    def get_preprocessing_summary(self) -> List[str]:
        return self.steps_log.copy()
    
    # Clear the preprocessing log
    def clear_log(self) -> None:
        self.steps_log = []
    # =================== DATA UNDERSTANDING & INSPECTION ===================
    
    def data_overview(self, df: pd.DataFrame, sample_size: int = 5) -> Dict:
        """
        → Provide comprehensive overview of the dataset.
        Args:
            df          : Input DataFrame
            sample_size : Number of random samples to display
        
        Returns → Dictionary containing overview metrics
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame")
        if sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        
        print(f"{'=' * 20} 📊 DATA OVERVIEW {'=' * 20}")
        
        try:
            overview = {
                'shape'          : df.shape,
                'duplicated_rows': df.duplicated().sum(),
                'missing_values' : df.isnull().sum().sum(),
                'data_types'     : df.dtypes.value_counts().to_dict()
            }
        except Exception as e:
            raise RuntimeError(f"Error computing overview metrics: {e}")
        
        # Display overview metrics
        print(f"Dataset Shape        : {overview['shape']}")
        print(f"Duplicated Rows      : {overview['duplicated_rows']}")
        print(f"Total Missing Values : {overview['missing_values']}")
        print(f"Data Types           : {overview['data_types']}")
        
        # Display basic info
        print("\nℹ️ STRUCTURE (.info()):")
        buffer = io.StringIO()
        df.info(buf=buffer)
        print(buffer.getvalue())
        
        if len(df) == 0:
            print("\n⚠️ Empty DataFrame: No statistics or samples available")
            self.log_step("Data Overview Complete", f"Shape: {overview['shape']} (empty)")
            return overview
        
        # Display statistics and samples
        print("\n📋 BASIC STATISTICS:")
        print(df.describe(include='all').round(2))
        
        print(f"\n🎲 RANDOM SAMPLE ({sample_size} rows):")
        print(df.sample(min(sample_size, len(df))))
        
        # Missing values analysis
        print("\n❌ MISSING VALUES BY COLUMN:")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            missing_pct = (missing_data / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing_Count': missing_data,
                'Missing_Percentage': missing_pct.round(2)
            })
            print(missing_df)
        else:
            print("No missing values found! 🎉")
        
        self.log_step("Data Overview Complete", f"Shape: {overview['shape']}")
        return overview
    
    def plot_missing_data(self, df: pd.DataFrame) -> None:
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) == 0:
            print("No missing data to visualize! 🎉")
            return
        
        plt.figure(figsize=(10, 6))
        missing_data.plot(kind='bar')
        plt.title('Missing Data by Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        self.log_step("Missing Data Visualization", f"Plotted {len(missing_data)} columns with missing data")
    # =================== MISSING DATA HANDLING ===================
    
    def _auto_missing_strategy(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        → Automatically determine the best missing value strategy for each column.
        
        Returns → Dictionary mapping column names to strategies
        """
        strategy = {}
        
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            
            missing_pct = df[col].isnull().sum() / len(df)
            
            if missing_pct > 0.7:
                strategy[col] = 'drop'
            elif df[col].dtype in ['object', 'category']:
                strategy[col] = 'mode'
            elif df[col].dtype in ['int64', 'float64']:
                if abs(df[col].skew()) > 1:  # Skewed data
                    strategy[col] = 'median'
                else:
                    strategy[col] = 'mean'
            else:
                strategy[col] = 'mode'
        
        return strategy
    
    def handle_missing_data(self, df: pd.DataFrame, strategy: Optional[Dict[str, str]] = None,advanced_imputation: bool = False) -> pd.DataFrame:
        """
        → Handle missing data using various strategies.
        
        Args:
            df                  : Input DataFrame
            strategy            : Dictionary mapping column names to strategies
            drop_threshold      : Threshold for dropping columns with too many missing values
            advanced_imputation : Whether to use advanced imputation methods
        
        Returns → DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        if strategy is None:
            strategy = self._auto_missing_strategy(df_processed)
            self.log_step("Missing Strategy", "Auto-detected missing value strategies")
        
        for col, strat in strategy.items():
            if col not in df_processed.columns:
                continue
                
            missing_count = df_processed[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strat == 'drop':
                df_processed = df_processed.drop(columns=[col])
                self.log_step("Drop Column", f"Dropped {col} ({missing_count} missing values)")
            
            elif strat == 'mean':
                mean_val = df_processed[col].mean()
                df_processed[col] = df_processed[col].fillna(mean_val)
                self.log_step("Mean Imputation", f"Filled {col} with mean ({mean_val:.2f})")
            
            elif strat == 'median':
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
                self.log_step("Median Imputation", f"Filled {col} with median ({median_val:.2f})")
            
            elif strat == 'mode':
                mode_val = df_processed[col].mode()
                if len(mode_val) > 0:
                    df_processed[col] = df_processed[col].fillna(mode_val[0])
                    self.log_step("Mode Imputation", f"Filled {col} with mode ({mode_val[0]})")
                else:
                    df_processed[col] = df_processed[col].fillna('Unknown')
                    self.log_step("Mode Imputation", f"Filled {col} with 'Unknown' (no mode found)")
        
        # Advanced imputation if requested
        if advanced_imputation:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
                self.log_step("KNN Imputation", f"Applied KNN imputation to {len(numeric_cols)} numeric columns")
        
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first', ignore_index: bool = False) -> pd.DataFrame:
        """
        → Remove duplicate rows from DataFrame.
        
        Args:
            df           : Input DataFrame
            subset       : Column labels to consider for identifying duplicates
            keep         : Which duplicates to keep ('first', 'last', False)
            ignore_index : Whether to reset index after removing duplicates
        
        Returns → DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df_processed = df.drop_duplicates(subset=subset, keep=keep, ignore_index=ignore_index)
        removed_rows = initial_rows - len(df_processed)
        
        if removed_rows > 0:
            self.log_step("Remove Duplicates", f"Removed {removed_rows} duplicate rows")
        else:
            self.log_step("Remove Duplicates", "No duplicates found")
        
        return df_processed
    
    def detect_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict:
        """
        → Detect and analyze duplicate rows.
        
        Args:
            df     : Input DataFrame
            subset : Column labels to consider for identifying duplicates
        
        Returns → Dictionary with duplicate analysis results
        """
        duplicated_mask = df.duplicated(subset=subset, keep=False)
        duplicate_count = duplicated_mask.sum()
        
        result = {
            'total_duplicates'        : duplicate_count,
            'unique_duplicate_groups' : df[duplicated_mask].duplicated(subset=subset).sum() if duplicate_count > 0 else 0,
            'duplicate_percentage'    : (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        }
        
        if duplicate_count > 0:
            print(f"Found {duplicate_count} duplicate rows ({result['duplicate_percentage']:.2f}%)")
            print("Sample duplicate rows:")
            print(df[duplicated_mask].head())
        else:
            print("No duplicate rows found! 🎉")
        
        self.log_step("Duplicate Detection", f"Found {duplicate_count} duplicate rows")
        return result
    # =================== OUTLIER DETECTION & TREATMENT ===================
    
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict:
        """
        → Detect outliers using the Interquartile Range (IQR) method.
        
        Args:
            df      : Input DataFrame
            columns : List of columns to check for outliers
            
        Returns → Dictionary with outlier information
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            outliers_info[col] = {
                'count'          : len(outliers),
                'percentage'     : (len(outliers) / len(df)) * 100,
                'lower_bound'    : lower_bound,
                'upper_bound'    : upper_bound,
                'outlier_values' : outliers[col].tolist()
            }
        
        total_outliers = sum(info['count'] for info in outliers_info.values())
        self.log_step("Outlier Detection (IQR)", f"Found {total_outliers} outliers across {len(columns)} columns")
        
        return outliers_info
    
    def remove_outliers_iqr(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        → Remove outliers using the IQR method.
        
        Args:
            df      : Input DataFrame
            columns : List of columns to remove outliers from
            
        Returns → DataFrame with outliers removed
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_processed = df.copy()
        initial_rows = len(df_processed)
        
        for col in columns:
            if col not in df_processed.columns:
                continue
                
            Q1  = df_processed[col].quantile(0.25)
            Q3  = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_processed = df_processed[(df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]
        removed_rows = initial_rows - len(df_processed)
        self.log_step("Remove Outliers (IQR)", f"Removed {removed_rows} rows with outliers")
        return df_processed
    
    # =================== FEATURE ENGINEERING ===================
    
    def create_datetime_features(self, df: pd.DataFrame, datetime_cols: List[str], features: List[str] = None) -> pd.DataFrame:
        """
        → Create datetime features from datetime columns.
        
        Args:
            df            : Input DataFrame
            datetime_cols : List of datetime column names
            features      : List of features to create. Options: 
                    ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'is_weekend', 'quarter', 'day_of_year', 'week_of_year']
                    If None, creates all features
        
        Returns → DataFrame with new datetime features
        """
        if features is None:
            features = ['year', 'month', 'day', 'weekday', 'is_weekend', 'quarter']
        
        df_engineered = df.copy()
        created_features = []
        
        for col in datetime_cols:
            if col not in df_engineered.columns:
                print(f"⚠️ Warning: Column '{col}' not found")
                continue
            
            # Ensure column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_engineered[col]):
                df_engineered[col] = pd.to_datetime(df_engineered[col], errors='coerce')
            
            # Create features
            if 'year' in features:
                df_engineered[f'{col}_year'] = df_engineered[col].dt.year
                created_features.append(f'{col}_year')
            
            if 'month' in features:
                df_engineered[f'{col}_month'] = df_engineered[col].dt.month
                created_features.append(f'{col}_month')
            
            if 'day' in features:
                df_engineered[f'{col}_day'] = df_engineered[col].dt.day
                created_features.append(f'{col}_day')
            
            if 'hour' in features:
                df_engineered[f'{col}_hour'] = df_engineered[col].dt.hour
                created_features.append(f'{col}_hour')
            
            if 'minute' in features:
                df_engineered[f'{col}_minute'] = df_engineered[col].dt.minute
                created_features.append(f'{col}_minute')
            
            if 'weekday' in features:
                df_engineered[f'{col}_weekday'] = df_engineered[col].dt.dayofweek
                created_features.append(f'{col}_weekday')
            
            if 'is_weekend' in features:
                df_engineered[f'{col}_is_weekend'] = (df_engineered[col].dt.dayofweek >= 5).astype(int)
                created_features.append(f'{col}_is_weekend')
            
            if 'quarter' in features:
                df_engineered[f'{col}_quarter'] = df_engineered[col].dt.quarter
                created_features.append(f'{col}_quarter')
            
            if 'day_of_year' in features:
                df_engineered[f'{col}_day_of_year'] = df_engineered[col].dt.dayofyear
                created_features.append(f'{col}_day_of_year')
            
            if 'week_of_year' in features:
                df_engineered[f'{col}_week_of_year'] = df_engineered[col].dt.isocalendar().week
                created_features.append(f'{col}_week_of_year')
        
        step_message = f"Datetime feature engineering: {len(created_features)} features created"
        self.steps_log.append(step_message)
        print(f"✅ {step_message}")
        print(f"📝 Created features: {', '.join(created_features)}")
        
        return df_engineered
    
    def create_mathematical_features(self, df: pd.DataFrame, feature_operations: Dict[str, Dict]) -> pd.DataFrame:
        """
        → Create mathematical features from existing numeric columns.
        
        Args:
            df                 : Input DataFrame
            feature_operations : Dictionary defining mathematical operations
                            {
                                'new_feature_name': {
                                'operation': 'ratio|difference|product|sum',
                                'columns': ['col1', 'col2']
                                }
                            }
        
        Returns → DataFrame with new mathematical features
        """
        df_engineered = df.copy()
        created_features = []
        
        for feature_name, operation_config in feature_operations.items():
            operation = operation_config.get('operation')
            columns = operation_config.get('columns', [])
            
            if len(columns) < 2:
                print(f"⚠️ Warning: Need at least 2 columns for operation. Skipping {feature_name}")
                continue
            
            # Check if columns exist
            missing_cols = [col for col in columns if col not in df_engineered.columns]
            if missing_cols:
                print(f"⚠️ Warning: Missing columns {missing_cols}. Skipping {feature_name}")
                continue
            
            try:
                if operation == 'ratio':
                    df_engineered[feature_name] = df_engineered[columns[0]] / (df_engineered[columns[1]] + 1e-8)  # Add small value to avoid division by zero
                elif operation == 'difference':
                    df_engineered[feature_name] = df_engineered[columns[0]] - df_engineered[columns[1]]
                elif operation == 'product':
                    df_engineered[feature_name] = df_engineered[columns[0]] * df_engineered[columns[1]]
                elif operation == 'sum':
                    df_engineered[feature_name] = df_engineered[columns].sum(axis=1)
                else:
                    print(f"⚠️ Warning: Unknown operation '{operation}'. Skipping {feature_name}")
                    continue
                
                created_features.append(feature_name)
                print(f"✅ Created '{feature_name}': {operation} of {columns}")
                
            except Exception as e:
                print(f"❌ Error creating feature '{feature_name}': {str(e)}")
                continue
        
        step_message = f"Mathematical feature engineering: {len(created_features)} features created"
        self.steps_log.append(step_message)
        print(f"📝 {step_message}")
        
        return df_engineered
    
    def create_binning_features(self, df: pd.DataFrame, binning_config: Dict[str, Dict]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        → Create binned/discretized features from continuous variables.
        
        Args:
            df             : Input DataFrame
            binning_config : Dictionary defining binning operations
                        {
                            'new_feature_name': {
                            'column': 'source_column',
                            'method': 'equal_width|equal_freq|custom',
                            'bins': 5,  # for equal_width/equal_freq
                            'bin_edges': [0, 25, 50, 75, 100],  # for custom
                            'labels': ['Low', 'Medium', 'High']  # optional
                            }
                        }
        
        Returns → Tuple of (DataFrame with new binned features, success_messages, error_messages)
        """
        df_engineered    = df.copy()
        created_features = []
        success_messages = []
        error_messages = []
        
        for feature_name, config in binning_config.items():
            source_col = config.get('column')
            method     = config.get('method', 'equal_width')
            bins       = config.get('bins', 5)
            bin_edges  = config.get('bin_edges')
            labels     = config.get('labels')
            
            if source_col not in df_engineered.columns:
                error_messages.append(f"⚠️ Column '{source_col}' not found. Skipping '{feature_name}'")
                continue
            # Validate labels if provided
            if labels:
                if method in ['equal_width', 'equal_freq']:
                    expected_labels = bins  # For equal_width/equal_freq, bins parameter = number of intervals = number of labels needed
                    if len(labels) != expected_labels:
                        error_messages.append(f"❌ For '{feature_name}': Labels count ({len(labels)}) must be {expected_labels} (same as bins). Provided labels: {labels}")
                        continue
                elif method == 'custom' and bin_edges:
                    expected_labels = len(bin_edges) - 1  # For custom, bin_edges-1 = number of intervals = number of labels needed
                    if len(labels) != expected_labels:
                        error_messages.append(f"❌ For '{feature_name}': Labels count ({len(labels)}) must be {expected_labels} (bin_edges-1). Provided: {len(labels)} labels for {len(bin_edges)} bin edges")
                        continue
            
            try:
                if method == 'equal_width':
                    binned_col = pd.cut(df_engineered[source_col], bins=bins, labels=labels)
                elif method == 'equal_freq':
                    binned_col = pd.qcut(df_engineered[source_col], q=bins, labels=labels)
                elif method == 'custom' and bin_edges:
                    binned_col = pd.cut(df_engineered[source_col], bins=bin_edges, labels=labels)
                else:
                    error_messages.append(f"⚠️ Invalid binning configuration for '{feature_name}'. Skipping")
                    continue
                
                # Convert categorical to string to avoid Arrow serialization issues
                df_engineered[feature_name] = binned_col.astype(str)
                
                created_features.append(feature_name)
                success_messages.append(f"✅ Created binned feature '{feature_name}' from '{source_col}' using {method}")
                
            except Exception as e:
                error_messages.append(f"❌ Error creating binned feature '{feature_name}': {str(e)}")
                continue
        
        step_message = f"Binning feature engineering: {len(created_features)} features created"
        self.steps_log.append(step_message)
        
        if created_features:
            success_messages.append(f"📝 {step_message}")
        
        return df_engineered, success_messages, error_messages
    # =================== FEATURE SCALING ===================
    
    def scale_features(self,df: pd.DataFrame,columns: Optional[List[str]] = None,method: str = 'standard') -> pd.DataFrame:
        """
        → Scale numerical features using various methods.
        
        Args:
            df     : Input DataFrame
            columns: List of columns to scale (if None, scales all numeric columns)
            method : Scaling method ('standard', 'minmax', 'robust')
        
        Returns → DataFrame with scaled features
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_processed = df.copy()
        
        # Select scaler based on method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Apply scaling
        df_processed[columns] = scaler.fit_transform(df_processed[columns])
        
        # Store scaler for potential inverse transformation
        self.scalers[method] = scaler
        
        self.log_step("Feature Scaling", f"Applied {method} scaling to {len(columns)} columns")
        
        return df_processed
    # =================== DATA BALANCING ===================
    
    def balance_data(self,X: pd.DataFrame,y: pd.Series,method: str = 'smote',random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """
        → Balance imbalanced datasets using various techniques.
        
        Args:
            X           : Feature DataFrame
            y           : Target Series
            method      : Balancing method ('smote', 'random_over', 'random_under', 'smote_tomek')
            random_state: Random state for reproducibility
        
        Returns → Tuple of balanced (X, y)
        """
        # Record original class distribution
        original_counts = y.value_counts().to_dict()
        
        # Select balancing method
        if method == 'smote':
            balancer = SMOTE(random_state=random_state)
        elif method == 'random_over':
            balancer = RandomOverSampler(random_state=random_state)
        elif method == 'random_under':
            balancer = RandomUnderSampler(random_state=random_state)
        elif method == 'smote_tomek':
            balancer = SMOTETomek(random_state=random_state)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        # Apply balancing
        X_balanced, y_balanced = balancer.fit_resample(X, y)
        
        # Record new class distribution
        new_counts = pd.Series(y_balanced).value_counts().to_dict()
        
        # Store balancing information
        self.balancing_info = {
            'method'         : method,
            'original_counts': original_counts,
            'new_counts'     : new_counts,
            'original_total' : len(y),
            'new_total'      : len(y_balanced)}
        
        self.log_step("Data Balancing", f"Applied {method}: {len(y)} → {len(y_balanced)} samples")
        
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name=y.name)
    # =================== UTILITY METHODS ===================
    
    # Get summary of data balancing operations
    def get_balancing_summary(self) -> Dict:
        return self.balancing_info.copy()
    
    # Reset all stored preprocessor state
    def reset_preprocessor(self) -> None:
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_names = []
        self.steps_log = []
        self.balancing_info = {}
        print("✅ Preprocessor state reset")    
    # Ensure DataFrame is compatible with Arrow serialization for Streamlit display.
    def ensure_arrow_compatibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        → Ensure DataFrame is compatible with Arrow serialization for Streamlit display.
        
        Args:
            df : Input DataFrame
        
        Returns → Arrow-compatible DataFrame
        """
        df_compatible = df.copy()
        
        for col in df_compatible.columns:
            # Handle categorical columns
            if df_compatible[col].dtype.name == 'category':
                df_compatible[col] = df_compatible[col].astype(str)
            
            # Handle object columns with mixed types
            elif df_compatible[col].dtype == 'object':
                # Convert to string to ensure consistency
                df_compatible[col] = df_compatible[col].astype(str)
            
            # Handle interval columns (from pd.cut)
            elif str(df_compatible[col].dtype).startswith('interval'):
                df_compatible[col] = df_compatible[col].astype(str)
            
            # Handle nullable integer types
            elif str(df_compatible[col].dtype).startswith('Int'):
                df_compatible[col] = df_compatible[col].astype('int64', errors='ignore')
            
            # Handle nullable float types
            elif str(df_compatible[col].dtype).startswith('Float'):
                df_compatible[col] = df_compatible[col].astype('float64', errors='ignore')
            
            # Handle any remaining complex types by converting to string
            elif df_compatible[col].dtype == 'object' or 'complex' in str(df_compatible[col].dtype):
                df_compatible[col] = df_compatible[col].astype(str)
        
        # Final safety check: ensure no mixed types in any column
        for col in df_compatible.columns:
            if df_compatible[col].dtype == 'object':
                # Check if all values can be converted to the same type
                sample_values = df_compatible[col].dropna().head(10)
                if len(sample_values) > 0:
                    try:
                        # Try to convert to numeric first
                        pd.to_numeric(sample_values)
                        df_compatible[col] = pd.to_numeric(df_compatible[col], errors='coerce')
                    except (ValueError, TypeError):
                        # If not numeric, ensure all are strings
                        df_compatible[col] = df_compatible[col].astype(str)
        
        return df_compatible