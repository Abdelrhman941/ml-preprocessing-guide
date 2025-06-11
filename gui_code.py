import io 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union
import json

# Scikit-learn: Core ML utilities
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer, KNNImputer

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
#####################################################################################3
class MLPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_names = []
        self.steps_log = []             # to record every preprocessing step as  writing down in a notepad
        self.balancing_info = {}        # to record info in balance(eg. number of samples before and after SMOTE or undersampling).
    
    def log_step(self, step_name: str, details: str):
        """Log preprocessing steps for tracking."""
        log_entry = f"✅ {step_name}: {details}"
        self.steps_log.append(log_entry)
        print(log_entry)
    # =================== 1. DATA UNDERSTANDING & INSPECTION ===================
    def data_overview(self, df, sample_size= 5):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame")
        if sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        
        print(f"{'=' * 20} 📊 DATA OVERVIEW {'=' * 20}")
        
        # Compute overview metrics
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
        
        print("\nℹ️ STRUCTURE (.info()):")
        buffer = io.StringIO()
        df.info(buf=buffer)      
        print(buffer.getvalue())
        
        # Handle empty DataFrame
        if len(df) == 0:
            print("\n⚠️ Empty DataFrame: No statistics or samples available")
            self.log_step("Data Overview Complete", f"Shape: {overview['shape']} (empty)")
            return overview
        
        # Display sample data and basic statistics
        print("\n📋 BASIC STATISTICS:")
        display(df.describe(include='all').round(2))
        print(f"\n🎲 RANDOM SAMPLE ({sample_size} rows):")
        display(df.sample(min(sample_size, len(df))))
        
        # Display missing values by column and percentage
        print("\n❌ MISSING VALUES BY COLUMN:")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            missing_pct = (missing_data / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing_Count'     : missing_data,
                'Missing_Percentage': missing_pct.round(2)
            })
            print(missing_df)
        else:
            print("No missing values found! 🎉")
        
        # record the overview step
        self.log_step("Data Overview Complete", f"Shape: {overview['shape']}")
        return overview
    
    def plot_missing_data(self, df):
        if df.isnull().sum().sum() == 0:
            print("🎉 No missing data to visualize.")
            self.log_step("Plot Missing Data", "No missing values found")
            return
        
        try:
            import missingno as msno
            plt.figure(figsize=(12, 6))
            msno.matrix(df)
            plt.title("Missing Data Pattern")
            plt.show()
            
            plt.figure(figsize=(10, 6))
            msno.bar(df)
            plt.title("Missing Data Count by Column")
            plt.show()
            
            self.log_step("Plotted Missing Data", "using missingno")
            
        except ImportError:
            print("⚠️ Install missingno for better visualization: pip install missingno")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                plt.figure(figsize=(10, 6))
                missing_data.plot(kind='bar')
                plt.title("Missing Values by Column")
                plt.xlabel("Columns")
                plt.ylabel("Missing Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                
                self.log_step("Plotted Missing Data", "fallback with matplotlib")
    # =================== 2. HANDLING MISSING DATA ===================
    def _auto_missing_strategy(self, df: pd.DataFrame) -> Dict[str, str]:
        strategy = {}
        
        for column in df.columns:
            if df[column].isnull().any():
                if pd.api.types.is_numeric_dtype(df[column]):
                    non_null_data = df[column].dropna()
                    if len(non_null_data) > 8:
                        try:
                            _, p_value = stats.normaltest(non_null_data)
                            strategy[column] = 'mean' if p_value > 0.05 else 'median'
                        except Exception:
                            strategy[column] = 'median'
                    else:
                        strategy[column] = 'median'
                else:
                    strategy[column] = 'most_frequent'
        self.log_step("Auto Imputation Strategy", f"{len(strategy)} columns will be imputed")
        return strategy
    
    def handle_missing_data(self, df: pd.DataFrame, strategy: Dict[str, str] = None, drop_threshold: float = 0.7, advanced_imputation: bool = False) -> pd.DataFrame:
        df_processed = df.copy()
        print(f"\n{'=' * 20} 🔧 HANDLING MISSING DATA {'=' * 20}")
        
        # Drop columns with excessive missing values (drop_threshold= 70% default)
        missing_pct  = df_processed.isnull().sum() / len(df_processed)
        cols_to_drop = missing_pct[missing_pct > drop_threshold].index.tolist()
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)
            self.log_step("Dropped columns", f"{cols_to_drop} (>{drop_threshold*100}% missing)")
        
        # Apply imputation strategies
        if strategy is None:
            strategy = self._auto_missing_strategy(df_processed)
        
        for column, method in strategy.items():
            if column not in df_processed.columns:
                continue
            if method == 'drop':
                df_processed = df_processed.dropna(subset=[column])
                self.log_step(f"Dropped rows for {column}", "NaN rows removed")
            elif method in ['mean', 'median', 'most_frequent']:
                if df_processed[column].dtype in ['int64', 'float64']:
                    imputer = SimpleImputer(strategy=method)
                    df_processed[column]  = imputer.fit_transform(df_processed[[column]]).ravel()
                    self.imputers[column] = imputer
                    self.log_step(f"Imputed {column}", f"Strategy: {method}")
            elif method in ['ffill', 'bfill']:
                df_processed[column] = df_processed[column].fillna(method=method)
                self.log_step(f"Forward/Backward fill {column}", f"Method: {method}")
        
        # Advanced imputation for numerical columns
        if advanced_imputation:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            missing_numeric = [col for col in numeric_cols if df_processed[col].isnull().any()]
            
            if missing_numeric:
                knn_imputer = KNNImputer(n_neighbors=5)
                df_processed[missing_numeric] = knn_imputer.fit_transform(df_processed[missing_numeric])
                self.imputers['knn_numeric'] = knn_imputer
                self.log_step("KNN Imputation", f"Applied to {missing_numeric}")
        
        return df_processed
    # =================== 3. HANDLING OUTLIERS ===================
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', columns: List[str] = None) -> Dict[str, List[int]]:
        outliers = {}
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if not np.issubdtype(df[column].dtype, np.number):
                continue
            series = df[column] 
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (series < lower_bound) | (series > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(series.fillna(series.mean()))) 
                outlier_mask = z_scores > 3
            elif method == 'modified_zscore':
                median = series.median()
                mad = np.median(np.abs(series - median))
                if mad == 0:
                    outliers[column] = []
                    continue
                modified_z_scores = 0.6745 * (series - median) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
            outliers[column] = series[outlier_mask].index.tolist()
        
        summary_df = pd.DataFrame({
            'Column': list(outliers.keys()),
            'Num_Outliers': [len(idxs) for idxs in outliers.values()],
            'Outlier_Indices': list(outliers.values())
        })
        display(summary_df) 
        
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'winsorize', detection_method: str = 'iqr',
                        columns: List[str] = None, percentiles: Tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
        df_processed = df.copy()
        print(f"\n{'=' * 20} 🎯 HANDLING OUTLIERS {'=' * 20}")
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in df_processed.columns:
                continue
            if df_processed[column].dropna().empty:
                continue  
            
            if method == 'remove':
                outliers = self.detect_outliers(df_processed, detection_method, [column])
                df_processed = df_processed.drop(index=outliers[column])
                self.log_step(f"Removed outliers from {column}", f"Removed {len(outliers[column])} rows")
            
            elif method == 'winsorize':
                lower_pct, upper_pct = percentiles
                lower_bound = df_processed[column].quantile(lower_pct)
                upper_bound = df_processed[column].quantile(upper_pct)
                df_processed[column] = df_processed[column].clip(lower_bound, upper_bound)
                self.log_step(f"Winsorized {column:<8}", f"Bounds : [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            elif method == 'log_transform':
                col_values = df_processed[column].dropna()
                if (col_values > 0).all():
                    df_processed[column] = np.log1p(df_processed[column])
                    self.log_step(f"Log transformed {column}", "Applied log1p transformation")
                else:
                    print(f"⚠️ Cannot log-transform '{column}' — contains non-positive or NaN values")
            
            elif method == 'cap':
                Q1 = df_processed[column].quantile(0.25)
                Q3 = df_processed[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_processed[column] = df_processed[column].clip(lower_bound, upper_bound)
                self.log_step(f"Capped {column}", "IQR bounds applied")
        
        return df_processed
    
    def plot_outliers(self, df: pd.DataFrame, columns: List[str] = None, save_path: str = None):
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle("Boxplots of Outlier Detection", fontsize=16, y=1.02)
        
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(columns):
            sns.boxplot(data=df, y=column, ax=axes[i])
            axes[i].set_title(f'Outliers in {column}')
        
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    # =================== 4. DATA TYPE CONVERSION ===================
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df_optimized = df.copy()
        print(f"\n {'=' * 20} 🔧 OPTIMIZING DATA TYPES {'=' * 20}")
        
        original_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize integers
        int_columns = df_optimized.select_dtypes(include=['int64']).columns
        for col in int_columns:
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif col_max <= 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        # Optimize floats
        float_columns = df_optimized.select_dtypes(include=['float64']).columns
        for col in float_columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Convert to category for low cardinality objects
        object_columns = df_optimized.select_dtypes(include=['object']).columns
        for col in object_columns:
            unique_ratio = df_optimized[col].nunique() / len(df_optimized)
            if unique_ratio < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = (1 - optimized_memory / original_memory) * 100
        
        self.log_step("Data types optimized", f"Memory reduction: {memory_reduction:.1f}% ({original_memory:.1f}MB → {optimized_memory:.1f}MB)")
        
        return df_optimized
    
    def convert_datetime(self, df: pd.DataFrame, datetime_columns: List[str], format: str = None) -> pd.DataFrame:
        df_processed = df.copy()
        for col in datetime_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col], format=format, errors='coerce')
                self.log_step(f"Converted {col} to datetime", f"Format: {format or 'inferred'}")
                
                num_nats = df_processed[col].isna().sum()
                if num_nats > 0:
                    self.log_step(f"⚠️ {num_nats} NaT values in {col}", "Some rows couldn't be converted")
            else:
                self.log_step(f"Column {col} not found", "⚠️ Skipped conversion")
        
        return df_processed
    # =================== 5. ENCODING CATEGORICAL VARIABLES ===================
    def _auto_encoding_strategy(self, df: pd.DataFrame, threshold: int) -> Dict[str, str]:
        strategy = {}
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            cardinality = df[column].nunique()
            strategy[column] = 'label' if cardinality == 2 or cardinality > threshold else 'onehot'
        return strategy
    
    def encode_categorical(self, df: pd.DataFrame, encoding_strategy: Dict[str, str] = None,high_cardinality_threshold: int = 10) -> pd.DataFrame:
        df_encoded = df.copy()
        print(f"\n{'=' * 20} 🏷️ ENCODING CATEGORICAL VARIABLES {'=' * 20}")
        categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        if encoding_strategy is None:
            encoding_strategy = self._auto_encoding_strategy(df_encoded, high_cardinality_threshold)
        
        for column, method in encoding_strategy.items():
            if column not in categorical_columns:
                continue
            if method == 'label':
                le = LabelEncoder()
                df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
                self.encoders[f'{column}_label'] = le
                self.log_step(f"Label encoded {column}", f"Classes: {len(le.classes_)}")
            elif method == 'onehot':
                dummies    = pd.get_dummies(df_encoded[column], prefix=column, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(column, axis=1), dummies], axis=1)
                self.log_step(f"One-hot encoded {column}", f"Created {len(dummies.columns)} features")
            elif method == 'target':
                self.log_step(f"Target encoding for {column}", "Skipped - requires target variable")
        
        return df_encoded
    # =================== 6. FEATURE SCALING ===================
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', columns: List[str] = None) -> pd.DataFrame:
        df_scaled = df.copy()
        print(f"\n{'=' * 20} SCALING FEATURES {'=' * 20}")
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if method   == 'standard':
            scaler  = StandardScaler()
        elif method == 'minmax':
            scaler  = MinMaxScaler()
        elif method == 'robust':
            scaler  = RobustScaler()
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.scalers[method] = scaler
        self.log_step(f"{method.title()} scaling applied", f"Scaled {len(columns)} features")
        
        return df_scaled
    # =================== 7. FEATURE ENGINEERING ===================
    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2,interaction_only: bool = False) -> pd.DataFrame:
        df_poly = df.copy()
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_features      = poly.fit_transform(df[columns])
        poly_feature_names = poly.get_feature_names_out(columns)
        for i, name in enumerate(poly_feature_names):
            if name not in columns:
                df_poly[name] = poly_features[:, i]
        self.log_step("Polynomial features created", f"Degree: {degree}, New features: {len(poly_feature_names) - len(columns)}")
        return df_poly
    
    def create_datetime_features(self, df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
        df_dt = df.copy()
        for col in datetime_columns:
            if col in df_dt.columns and pd.api.types.is_datetime64_any_dtype(df_dt[col]):
                df_dt[f'{col}_year']       = df_dt[col].dt.year
                df_dt[f'{col}_month']      = df_dt[col].dt.month
                df_dt[f'{col}_day']        = df_dt[col].dt.day
                df_dt[f'{col}_dayofweek']  = df_dt[col].dt.dayofweek
                df_dt[f'{col}_hour']       = df_dt[col].dt.hour
                df_dt[f'{col}_is_weekend'] = (df_dt[col].dt.dayofweek >= 5).astype(int)
                df_dt[f'{col}_quarter']    = df_dt[col].dt.quarter
                df_dt[f'{col}_season']     = df_dt[col].dt.month % 12 // 3 + 1
                self.log_step(f"DateTime features created for {col}", "8 new features")
        return df_dt
    
    def create_binned_features(self, df: pd.DataFrame, column: str, bins: Union[int, List], labels: List[str] = None) -> pd.DataFrame:
        df_binned  = df.copy()
        binned_col = f'{column}_binned'
        df_binned[binned_col] = pd.cut(df_binned[column], bins=bins, labels=labels)
        self.log_step(f"Created binned feature {binned_col}", f"Bins: {bins}")
        return df_binned
    # =================== 8. FEATURE SELECTION ===================
    def select_features(self, df: pd.DataFrame, target: str, method: str = 'correlation', k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        print(f"\n{'=' * 20} 🎯 FEATURE SELECTION {'=' * 20}")
        X = df.drop(columns=[target])
        y = df[target]
        
        if method == 'correlation':
            corr_matrix = X.corr().abs()
            upper_tri   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop     = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            selected_features = [col for col in X.columns if col not in to_drop][:k]
            
        elif method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=k)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'f_classif':     # ANOVA F-Value
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'lasso':
            lasso = Lasso(alpha=0.01, random_state=42)
            selector = SelectFromModel(lasso, max_features=k)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        else:
            raise ValueError("Method must be one of: 'correlation', 'chi2', 'f_classif', 'rfe', 'lasso'")
        selected_df = df[selected_features + [target]]
        self.log_step(f"Feature selection ({method})", f"Selected {len(selected_features)} features")
        return selected_df, selected_features
    # =================== 9. DATA SPLITTING ===================
    def split_data(self, df: pd.DataFrame, target: str, test_size: float = 0.2, val_size: Optional[float] = None,
                stratify: bool = True, time_based: bool = False, date_column: Optional[str] = None,random_state: int = 42) -> Union[Tuple, Dict]:
        print(f"\n{'=' * 20} DATA SPLITTING {'=' * 20}")
        X = df.drop(columns=[target])
        y = df[target]
        
        if time_based:
            if date_column is None or date_column not in df.columns:
                raise ValueError("date_column must be specified for time-based splitting")
            df_sorted = df.sort_values(date_column)
            X_sorted  = df_sorted.drop(columns=[target])
            y_sorted  = df_sorted[target]
            test_idx  = int(len(df_sorted) * (1 - test_size))
            
            if val_size:
                val_idx = int(len(df_sorted) * (1 - test_size - val_size))
                X_train = X_sorted.iloc[:val_idx]
                y_train = y_sorted.iloc[:val_idx]
                X_val   = X_sorted.iloc[val_idx:test_idx]
                y_val   = y_sorted.iloc[val_idx:test_idx]
                X_test  = X_sorted.iloc[test_idx:]
                y_test  = y_sorted.iloc[test_idx:]
                self.log_step("Time-based split with validation", f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
                return {
                    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                    'y_train': y_train, 'y_val': y_val, 'y_test': y_test
                }
            
            else:
                X_train = X_sorted.iloc[:test_idx]
                y_train = y_sorted.iloc[:test_idx]
                X_test  = X_sorted.iloc[test_idx:]
                y_test  = y_sorted.iloc[test_idx:]
                self.log_step("Time-based split", f"Train: {len(X_train)}, Test: {len(X_test)}")
                return X_train, X_test, y_train, y_test
        
        stratify_param = y if stratify and len(y.unique()) > 1 else None
        if val_size:
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_param, random_state=random_state)
            val_size_adjusted   = val_size / (1 - test_size)
            stratify_temp       = y_temp if stratify and len(y_temp.unique()) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, stratify=stratify_temp, random_state=random_state)
            self.log_step("Three-way split", f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            return {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test
            }
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_param, random_state=random_state)
        self.log_step("Train-test split", f"Train: {len(X_train)}, Test: {len(X_test)}, Stratified: {stratify}")
        return X_train, X_test, y_train, y_test
    
    def create_cross_validation_folds(self, X: pd.DataFrame, y: pd.Series, cv_type: str = 'kfold',n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> object:
        if cv_type   == 'stratified':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif cv_type == 'timeseries':
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.log_step(f"{cv_type.title()} CV created", f"{n_splits} folds")
        return cv
    # =================== 10. CLASS BALANCING ===================
    def balance_classes(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote',sampling_strategy: Union[str, dict] = 'auto', random_state: int = 42,**kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 20} ⚖️ CLASS BALANCING {'=' * 20}")
        original_dist = Counter(y)
        print(f"Original distribution: {dict(original_dist)}")
        
        samplers = {
            'smote'             : SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs),
            'adasyn'            : ADASYN(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs),
            'borderline_smote'  : BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs),
            'random_over'       : RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs),
            'random_under'      : RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs),
            'tomek'             : TomekLinks(sampling_strategy=sampling_strategy, **kwargs),
            'enn'               : EditedNearestNeighbours(sampling_strategy=sampling_strategy, **kwargs),
            'smote_tomek'       : SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs),
            'smote_enn'         : SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)}
        
        if method not in samplers:
            raise ValueError(f"Unknown balancing method: {method}")
        X_balanced, y_balanced = samplers[method].fit_resample(X, y)
        
        if isinstance(X, pd.DataFrame):
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        
        if isinstance(y, pd.Series):
            y_balanced = pd.Series(y_balanced, name=y.name)
        
        new_dist = Counter(y_balanced)
        print(f"Balanced distribution: {dict(new_dist)}")
        
        self.balancing_info = {
            'method': method,
            'original_distribution': dict(original_dist),
            'balanced_distribution': dict(new_dist),
            'original_size': len(y),
            'balanced_size': len(y_balanced)}
        self.log_step(f"Class balancing ({method})", f"Size: {len(y)} → {len(y_balanced)}")
        return X_balanced, y_balanced
    
    def create_balanced_pipeline(self, balancing_method: str = 'smote', classifier=None,**balance_params) -> ImbPipeline:
        if classifier is None:
            classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
        samplers = {
            'smote'         : SMOTE(random_state=42, **balance_params),
            'adasyn'        : ADASYN(random_state=42, **balance_params),
            'random_over'   : RandomOverSampler(random_state=42, **balance_params)
        }
        sampler  = samplers.get(balancing_method, SMOTE(random_state=42, **balance_params))
        pipeline = ImbPipeline([('balancer', sampler), ('classifier', classifier)])
        self.log_step("Balanced pipeline created", f"Method: {balancing_method}, Classifier: {type(classifier).__name__}")
        return pipeline
    
    def evaluate_balanced_model(self, pipeline: ImbPipeline, X: pd.DataFrame, y: pd.Series,
                                cv_folds: int = 5, scoring: List[str] = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']) -> Dict:
        results = {}
        for metric in scoring:
            scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=metric)
            results[metric] = {
                'mean'  : scores.mean(),
                'std'   : scores.std(),
                'scores': scores
            }
            print(f"{metric.upper()}: {scores.mean():.4f} (±{scores.std():.4f})")
        self.log_step("Model evaluation completed", f"Metrics: {', '.join(scoring)}")
        return results
    
    def _get_balancing_recommendation(self, imbalance_ratio: float, total_samples: int) -> str:
        if imbalance_ratio <= 1.5:
            return "No balancing needed"
        elif imbalance_ratio <= 4:
            return "Consider class weights or light oversampling"
        elif imbalance_ratio <= 10:
            return "Use SMOTE or ADASYN"
        elif total_samples < 1000:
            return "Use SMOTE with careful validation"
        else:
            return "Use advanced techniques like SMOTE + undersampling"
    
    def detect_class_imbalance(self, y: pd.Series, imbalance_threshold: float = 0.1) -> Dict:
        print(f"\n {'=' * 20} 📊 CLASS IMBALANCE ANALYSIS {'=' * 20}")
        
        class_counts  = Counter(y)
        total_samples = len(y)
        class_proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        min_proportion    = min(class_proportions.values())
        max_proportion    = max(class_proportions.values())
        is_imbalanced     = min_proportion < imbalance_threshold
        imbalance_ratio   = max_proportion / min_proportion if min_proportion > 0 else float('inf')
        analysis = {
            'is_imbalanced'         : is_imbalanced,
            'class_counts'          : dict(class_counts),
            'class_proportions'     : class_proportions,
            'min_class_proportion'  : min_proportion,
            'max_class_proportion'  : max_proportion,
            'imbalance_ratio'       : imbalance_ratio,
            'recommendation'        : self._get_balancing_recommendation(imbalance_ratio, total_samples)}
        print(f"Classes         : {len(class_counts)}")
        print(f"Total samples   : {total_samples}")
        print(f"Imbalanced      : {'Yes' if is_imbalanced else 'No'}")
        print(f"Imbalance ratio : {imbalance_ratio:.2f}:1")
        print(f"Recommendation  : {analysis['recommendation']}")
        return analysis
    # =================== UTILITY METHODS ===================
    def get_preprocessing_summary(self) -> pd.DataFrame:
        steps_data = []
        
        if not self.steps_log:
            return pd.DataFrame(columns=['Step', 'Details'])
        
        for i, step in enumerate(self.steps_log, 1):
            parts     = step.replace('✅ ', '').split(': ', 1)
            step_name = parts[0] if len(parts) > 0 else f"Step {i}"
            details   = parts[1] if len(parts) > 1 else "No details"
            steps_data.append({
                'Step_Number': i,
                'Step_Name'  : step_name,
                'Details'    : details
            })
        return pd.DataFrame(steps_data)
    
    def export_preprocessing_config(self, filepath: str):
        config = {
            'preprocessing_steps': self.steps_log,
            'encoders'           : {k: str(type(v)) for k, v in self.encoders.items()},
            'scalers'            : {k: str(type(v)) for k, v in self.scalers.items()},
            'feature_names'      : self.feature_names,
            'balancing_info'     : self.balancing_info
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.log_step("Configuration exported", f"Saved to {filepath}")
    
    def reset_preprocessor(self):
        self.steps_log      = []
        self.encoders       = {}
        self.scalers        = {}
        self.feature_names  = []
        self.balancing_info = {}
        print("🔄 Preprocessor reset successfully")