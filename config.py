# Application Settings
APP_CONFIG = {
    "title"             : "ML Studio",
    "version"           : "2.0.0",
    "page_icon"         : "🚀",
    "layout"            : "wide",
    "github_url"        : "https://github.com/Abdelrhman941/ml-preprocessing-guide.git",
    "documentation_url" : "#"
}

# Model Configuration
MODEL_CONFIG = {
    "available_models"     : ["Random Forest", "XGBoost", "LightGBM"],
    "default_test_size"    : 0.2,
    "default_random_state" : 42,
    "default_cv_folds"     : 5,
    "max_features_display" : 15
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "missing_value_strategies": [
        "Mean imputation", 
        "Median imputation", 
        "Mode imputation", 
        "Drop rows", 
        "Drop columns"
    ],
    "encoding_methods"          : ["Label Encoding", "One-Hot Encoding"],
    "scaling_methods"           : ["StandardScaler", "MinMaxScaler", "RobustScaler"],
    "feature_selection_methods" : [
        "Manual Selection", 
        "Correlation Threshold", 
        "Feature Importance"
    ],
    "drop_threshold"            : 0.7,
    "correlation_threshold"     : 0.8,
    "outlier_detection_methods" : ["IQR", "Z-Score"],
    "balancing_methods"         : ["smote", "random_over", "random_under", "smote_tomek"]
}

# Sample Datasets Configuration
SAMPLE_DATASETS = [
    "Iris (Classification)",
    "Wine (Classification)", 
    "Breast Cancer (Classification)",
    "Diabetes (Regression)",
    "California Housing (Regression)"
]

# UI Configuration
UI_CONFIG = {
    "max_log_entries"           : 100,
    "default_sample_size"       : 5,
    "chart_height"              : 400,
    "max_categories_display"    : 20,
    "progress_update_interval"  : 1.0
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "enable_parallel_processing": True,
    "max_workers"     : -1,     # -1 means use all available cores
    "memory_limit_mb" : 1024,   # Memory limit for large datasets
    "chunk_size"      : 10000   # Chunk size for processing large datasets
}

# Color Schemes
COLOR_SCHEMES = {
    "primary"         : "#5bc0be",
    "secondary"       : "#3a506b", 
    "success"         : "#28a745",
    "warning"         : "#ffc107",
    "danger"          : "#dc3545",
    "info"            : "#17a2b8",
    "gradient_colors" : ["#3a506b", "#5bc0be", "#1c2541", "#b2bec3"]
}

# Hyperparameter Grids for Models
HYPERPARAMETER_GRIDS = {
    "Random Forest": {
        'n_estimators'      : [50, 100, 200],
        'max_depth'         : [None, 5, 10, 20],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf'  : [1, 2, 4]
    },
    "XGBoost": {
        'n_estimators'  : [50, 100, 200],
        'max_depth'     : [3, 6, 10],
        'learning_rate' : [0.01, 0.1, 0.2],
        'subsample'     : [0.8, 0.9, 1.0]
    },
    "LightGBM": {
        'n_estimators'  : [50, 100, 200],
        'max_depth'     : [3, 6, 10],
        'learning_rate' : [0.01, 0.1, 0.2],
        'num_leaves'    : [31, 50, 100]
    }
}

# Evaluation Metrics Configuration
METRICS_CONFIG = {
    "classification_metrics"        : ["accuracy", "f1", "roc_auc"],
    "regression_metrics"            : ["mse", "mae", "r2"],
    "default_classification_metric" : "accuracy",
    "default_regression_metric"     : "mse"
}

# Export Configuration
EXPORT_CONFIG = {
    "model_file_format"     : "pkl",
    "results_file_format"   : "json",
    "data_export_formats"   : ["csv", "xlsx", "json"],
    "default_export_folder" : "exports"
}
