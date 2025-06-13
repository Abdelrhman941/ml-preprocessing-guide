import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error,
    roc_curve, auc, classification_report, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config.settings import HYPERPARAMETER_GRIDS, MODEL_CONFIG, COLOR_SCHEMES

def detect_task_type(y):
    """
    → Automatically detect if the target variable is for classification or regression.
    
    Args    → y  : Target variable (pandas Series or array-like)
    
    Returns → str: 'classification' or 'regression'
    """
    # Convert to pandas Series if it isn't already
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Check if target is numeric
    is_numeric = pd.api.types.is_numeric_dtype(y)
    
    if not is_numeric:
        # Non-numeric data is likely classification
        return 'classification'
    
    # For numeric data, check characteristics
    unique_values = y.nunique()
    total_values = len(y)
    
    # If all values are integers and there are relatively few unique values
    if y.dtype in ['int64', 'int32'] and unique_values <= 20:
        return 'classification'
    
    # If there are very few unique values relative to total (less than 5% and max 50)
    if unique_values <= max(10, total_values * 0.05) and unique_values <= 50:
        return 'classification'
    
    # Check if all values are whole numbers (could be classification)
    if all(y == y.astype(int)) and unique_values <= 100:
        return 'classification'
    
    # Otherwise, assume regression
    return 'regression'

# Load sample datasets for testing and demonstration
def load_sample_dataset(dataset_name: str):
    datasets = {
        "Iris (Classification)"           : (load_iris, 'classification'),
        "Wine (Classification)"           : (load_wine, 'classification'), 
        "Breast Cancer (Classification)"  : (load_breast_cancer, 'classification'),
        "Diabetes (Regression)"           : (load_diabetes, 'regression'),
        "California Housing (Regression)" : (None, 'regression')  # Will be handled separately
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not supported")
    
    loader, task_type = datasets[dataset_name]
    
    if dataset_name == "California Housing (Regression)":
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df, task_type
        except ImportError:
            st.error("California Housing dataset not available")
            return None, None
    else:
        data = loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, task_type

# Get hyperparameter options for each model
def get_model_params(model_name: str):
    return HYPERPARAMETER_GRIDS.get(model_name, {})

# Create model instance with given parameters
def create_model(model_name: str, params: dict, task_type: str = 'classification'):
    random_state = MODEL_CONFIG["default_random_state"]
    
    if model_name == "Random Forest":
        if task_type == 'classification':
            return RandomForestClassifier(**params, random_state=random_state)
        else:
            return RandomForestRegressor(**params, random_state=random_state)
    
    elif model_name == "XGBoost":
        if task_type == 'classification':
            return xgb.XGBClassifier(**params, random_state=random_state)
        else:
            return xgb.XGBRegressor(**params, random_state=random_state)
    
    elif model_name == "LightGBM":
        if task_type == 'classification':
            return lgb.LGBMClassifier(**params, random_state=random_state, verbose=-1)
        else:
            return lgb.LGBMRegressor(**params, random_state=random_state, verbose=-1)
    
    raise ValueError(f"Model '{model_name}' not supported")

# Evaluate model using cross-validation
def evaluate_model(model, X, y, task_type: str = 'classification', metric: str = 'accuracy'):
    try:
        if task_type == 'classification':
            if metric == 'accuracy':
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            elif metric == 'f1':
                scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            elif metric == 'roc_auc':
                scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc_ovr_weighted')
            else:
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        else:
            if metric == 'mse':
                scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            elif metric == 'mae':
                scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            elif metric == 'r2':
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            else:
                scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        
        return scores.mean(), scores.std()
    except Exception as e:
        st.error(f"Error evaluating model: {e}")
        return 0.0, 0.0

# Create interactive confusion matrix (legacy function - redirects to enhanced version)
def plot_confusion_matrix(y_true, y_pred, labels=None):
    return plot_confusion_matrix_enhanced(y_true, y_pred, labels, normalize=False)

# Create feature importance plot
def plot_feature_importance(model, feature_names, top_n: int = 15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        st.warning("Model doesn't have feature importance")
        return None
    
    # Get top features
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig = px.bar(
        x=top_importances,
        y=top_features,
        orientation='h',
        title=f"Top {top_n} Feature Importances",
        labels={'x': 'Importance', 'y': 'Features'} )
    
    fig.update_layout(
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'} )
    
    return fig

# Plot ROC curves for binary and multiclass classification
def plot_roc_curve(model, X, y):
    return plot_roc_curve_multiclass(model, X, y)

# Plot actual vs predicted values for regression
def plot_regression_results(y_true, y_pred):
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x      = y_true,
        y      = y_pred,
        mode   = 'markers',
        name   = 'Predictions',
        marker = dict(color='blue', size=8, opacity=0.6)
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title       = 'Actual vs Predicted Values',
        xaxis_title = 'Actual Values',
        yaxis_title = 'Predicted Values',
        height      = 400,
        showlegend  = True
    )
    
    return fig

# Add message to training logs
def add_log_message(message: str):
    if 'training_logs' not in st.session_state:
        st.session_state.training_logs = []
    
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.training_logs.append(log_entry)
    
    # Keep only last 100 log entries to prevent memory issues
    if len(st.session_state.training_logs) > 100:
        st.session_state.training_logs = st.session_state.training_logs[-100:]

# Get comprehensive model performance metrics
def get_model_metrics_summary(model, X_test, y_test, task_type: str):
    y_pred = model.predict(X_test)
    
    if task_type == 'classification':
        metrics = {
            'Accuracy'  : accuracy_score(y_test, y_pred),
            'F1 Score'  : f1_score(y_test, y_pred, average='weighted'),
            'Precision' : precision_score(y_test, y_pred, average='weighted'),
            'Recall'    : recall_score(y_test, y_pred, average='weighted')
        }
        
        # Add ROC AUC for binary and multiclass classification
        try:
            unique_classes = np.unique(y_test)
            if len(unique_classes) == 2:
                # Binary classification
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    metrics['ROC AUC'] = roc_auc_score(y_test, y_pred_proba)
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                    metrics['ROC AUC'] = roc_auc_score(y_test, y_scores)
            elif len(unique_classes) > 2:
                # Multiclass classification
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    metrics['ROC AUC (OvR)'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
    else:
        metrics = {
            'R² Score' : r2_score(y_test, y_pred),
            'MSE'      : mean_squared_error(y_test, y_pred),
            'MAE'      : mean_absolute_error(y_test, y_pred),
            'RMSE'     : np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return metrics

def get_classification_report(y_test, y_pred, target_names=None):
    """
    → Generate comprehensive classification report.
    
    Args:
        y_test       : True labels
        y_pred       : Predicted labels
        target_names : List of target class names
    
    Returns → str: Formatted classification report
    """
    return classification_report(y_test, y_pred, target_names=target_names)

def plot_confusion_matrix_enhanced(y_true, y_pred, labels=None, normalize=False):
    """
    → Create enhanced confusion matrix with better styling.
    
    Args:
        y_true    : True labels
        y_pred    : Predicted labels
        labels    : Class labels
        normalize : Whether to normalize the confusion matrix
    
    Returns → Plotly figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
        text_template = ".2%"
    else:
        title = "Confusion Matrix"
        text_template = "d"
    
    if labels is None:
        labels = [f"Class {i}" for i in range(len(cm))]
    
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        text_auto=True,
        title=title,
        aspect="auto"
    )
    
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        width=500
    )
    
    return fig

def plot_roc_curve_multiclass(model, X_test, y_test, class_names=None):
    """
    → Plot ROC curves for multiclass classification.
    
    Args:
        model       : Trained model with predict_proba method
        X_test      : Test features
        y_test      : Test labels
        class_names : List of class names
    
    Returns → Plotly figure
    """
    try:
        y_pred_proba = model.predict_proba(X_test)
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                showlegend=True
            )
            
        else:
            # Multiclass classification - One vs Rest
            y_test_binarized = label_binarize(y_test, classes=unique_classes)
            
            fig = go.Figure()
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = class_names[i] if class_names else f'Class {unique_classes[i]}'
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{class_name} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='black', dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curves (One vs Rest)',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                showlegend=True
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating ROC curve: {e}")
        return None

# Create a formatted display of model metrics with color coding and explanations
def create_metrics_display(metrics: dict):
    # Color legend
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem; padding: 0.5rem; border-radius: 0.25rem; background-color: rgba(255,255,255,0.05);">
        <span style="color: green; font-weight: bold;">🟢 Excellent ≥ 0.8</span> &nbsp;&nbsp;
        <span style="color: orange; font-weight: bold;">🟡 Moderate ≥ 0.6</span> &nbsp;&nbsp;
        <span style="color: red; font-weight: bold;">🔴 Poor < 0.6</span>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(metrics))
    
    for i, (metric_name, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
                
                # Color coding based on metric value
                if metric_name.lower() in ['accuracy', 'f1 score', 'precision', 'recall', 'roc auc', 'r² score']:
                    # Higher is better metrics
                    if value >= 0.8:
                        score_color = "green"
                        performance = "🟢 Excellent"
                        emoji = "🟢"
                    elif value >= 0.6:
                        score_color = "orange"
                        performance = "🟡 Moderate"
                        emoji = "🟡"
                    else:
                        score_color = "red"
                        performance = "🔴 Poor"
                        emoji = "🔴"
                elif metric_name.lower() in ['mse', 'mae', 'rmse']:
                    # Lower is better metrics - neutral display for regression
                    score_color = "blue"
                    performance = "📊 Metric"
                    emoji = "📊"
                else:
                    score_color = "blue"
                    performance = "📊 Metric"
                    emoji = "📊"
                
                # Enhanced metric display with color and emoji
                st.markdown(f"""
                    <div style="
                        padding: 1rem;
                        border-radius: 0.5rem;
                        border: 2px solid {score_color};
                        background-color: rgba(255,255,255,0.05);
                        text-align: center;
                        margin-bottom: 0.5rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <div style="
                            font-size: 2rem;
                            font-weight: bold;
                            color: {score_color};
                            margin-bottom: 0.25rem;
                        ">{formatted_value}</div>
                        <div style="
                            font-size: 0.9rem;
                            color: #666;
                            margin-bottom: 0.25rem;
                            font-weight: 500;
                        ">{metric_name}</div>
                        <div style="
                            font-size: 0.8rem;
                            color: {score_color};
                            font-weight: bold;
                        ">{performance}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                formatted_value = str(value)
                # Simple display for non-numeric values
                st.metric(metric_name, formatted_value)

def detect_and_remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    → Detect and remove duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame
    
    Returns → Tuple of (cleaned_df, num_duplicates_removed)
    """
    initial_rows = len(df)
    
    # Check for duplicates
    duplicates_mask = df.duplicated()
    num_duplicates  = duplicates_mask.sum()
    
    if num_duplicates > 0:
        # Remove duplicates
        df_cleaned    = df.drop_duplicates()
        removed_count = initial_rows - len(df_cleaned)
        return df_cleaned, removed_count
    else:
        return df, 0

def plot_learning_curves(model, X, y, task_type: str = 'classification', cv: int = 5):
    """
    → Generate learning curves for model performance evaluation.
    
    Args:
        model    : Trained model
        X        : Feature matrix
        y        : Target vector
        task_type: 'classification' or 'regression'
        cv       : Number of cross-validation folds
    
    Returns → Plotly figure with learning curves
    """
    try:
        # Define scoring metric based on task type
        scoring = 'accuracy' if task_type == 'classification' else 'neg_mean_squared_error'
        
        # Generate learning curve data
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, 
            cv=cv, 
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate means and standard deviations
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)
        
        # For regression, convert negative MSE to positive
        if task_type == 'regression':
            train_mean = -train_mean
            val_mean = -val_mean
        
        # Create the plot
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=COLOR_SCHEMES['primary'], width=2),
            marker=dict(size=6)
        ))
        
        # Training score confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor=f"rgba(91, 192, 190, 0.2)",
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Training ±1 std'
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=COLOR_SCHEMES['secondary'], width=2),
            marker=dict(size=6)
        ))
        
        # Validation score confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
            fill='toself',
            fillcolor=f"rgba(58, 80, 107, 0.2)",
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Validation ±1 std'
        ))
        
        # Update layout
        metric_name = 'Accuracy' if task_type == 'classification' else 'Mean Squared Error'
        fig.update_layout(
            title=f'Learning Curves - {metric_name}',
            xaxis_title='Training Set Size',
            yaxis_title=metric_name,
            height=500,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error generating learning curves: {e}")
        return None

def create_enhanced_preprocessing_summary(steps: list, dataset_info: dict):
    """
    → Create an enhanced preprocessing summary with detailed information.
    
    Args:
        steps       : List of preprocessing steps
        dataset_info: Dictionary with dataset information
    
    Returns → Formatted summary for display
    """
    summary = {
        'preprocessing_steps': steps,
        'dataset_shape'      : dataset_info.get('shape', 'Unknown'),
        'missing_values'     : dataset_info.get('missing_values', 0),
        'duplicates_removed' : dataset_info.get('duplicates_removed', 0),
        'categorical_encoded': dataset_info.get('categorical_encoded', []),
        'features_scaled'    : dataset_info.get('features_scaled', []),
        'features_selected'  : dataset_info.get('features_selected', 'All')
    }
    
    return summary

# Display metric explanations in expandable sections
def create_metrics_explanations():
    st.markdown("---")
    st.markdown("### 📊 Understanding Your Metrics")
    
    explanation_cols = st.columns(2)
    
    with explanation_cols[0]:
        with st.expander("📊 What does Accuracy mean?"):
            st.info("""
            **Accuracy**: Proportion of correctly classified samples over all samples.
            
            Formula: (True Positives + True Negatives) / Total Samples
            
            - **Best for**: Balanced datasets
            - **Range**: 0.0 to 1.0 (higher is better)
            """)
        
        with st.expander("🎯 What does Precision mean?"):
            st.info("""
            **Precision**: Of all positive predictions, how many were actually correct?
            
            Formula: True Positives / (True Positives + False Positives)
            
            - **Best for**: When false positives are costly
            - **Range**: 0.0 to 1.0 (higher is better)
            """)
    
    with explanation_cols[1]:
        with st.expander("🔍 What does Recall mean?"):
            st.info("""
            **Recall**: Of all actual positive cases, how many did we correctly identify?
            
            Formula: True Positives / (True Positives + False Negatives)
            
            - **Best for**: When false negatives are costly
            - **Range**: 0.0 to 1.0 (higher is better)
            """)
        
        with st.expander("⚖️ What does F1 Score mean?"):
            st.info("""
            **F1 Score**: Harmonic mean of Precision and Recall.
            
            Formula: 2 × (Precision × Recall) / (Precision + Recall)
            
            - **Best for**: When you need balance between precision and recall
            - **Range**: 0.0 to 1.0 (higher is better)
            """)
    
    with st.expander("📈 What does ROC AUC mean?"):
        st.info("""
        **ROC AUC**: Area Under the Receiver Operating Characteristic Curve.
        
        Measures the model's ability to distinguish between classes.
        
        - **0.5**: Random guessing
        - **1.0**: Perfect classifier
        - **Best for**: Binary classification problems
        """)