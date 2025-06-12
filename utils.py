import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, 
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error,
    roc_curve, auc)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP for advanced feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Import configuration
from config import HYPERPARAMETER_GRIDS, MODEL_CONFIG, COLOR_SCHEMES


def load_sample_dataset(dataset_name: str):
    """Load sample datasets for testing and demonstration."""
    datasets = {
        "Iris": (load_iris, 'classification'),
        "Wine": (load_wine, 'classification'), 
        "Breast Cancer": (load_breast_cancer, 'classification'),
        "Diabetes (Regression)": (load_diabetes, 'regression'),
        "California Housing (Regression)": (None, 'regression')  # Will be handled separately
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


def get_model_params(model_name: str):
    """Get hyperparameter options for each model."""
    return HYPERPARAMETER_GRIDS.get(model_name, {})


def create_model(model_name: str, params: dict, task_type: str = 'classification'):
    """Create model instance with given parameters."""
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


def evaluate_model(model, X, y, task_type: str = 'classification', metric: str = 'accuracy'):
    """Evaluate model using cross-validation."""
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


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Create interactive confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f"Class {i}" for i in range(len(cm))]
    
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        text_auto=True,
        title="Confusion Matrix"
    )
    
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    return fig


def plot_feature_importance(model, feature_names, top_n: int = 15):
    """Create feature importance plot."""
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
        labels={'x': 'Importance', 'y': 'Features'}
    )
    
    fig.update_layout(
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_roc_curve(model, X, y):
    """Plot ROC curve for classification models."""
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
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
            height=400,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating ROC curve: {e}")
        return None


def plot_regression_results(y_true, y_pred):
    """Plot actual vs predicted values for regression."""
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', size=8, opacity=0.6)
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
        title='Actual vs Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=400,
        showlegend=True
    )
    
    return fig


def add_log_message(message: str):
    """Add message to training logs."""
    if 'training_logs' not in st.session_state:
        st.session_state.training_logs = []
    
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.training_logs.append(log_entry)
    
    # Keep only last 100 log entries to prevent memory issues
    if len(st.session_state.training_logs) > 100:
        st.session_state.training_logs = st.session_state.training_logs[-100:]


def get_model_metrics_summary(model, X_test, y_test, task_type: str):
    """Get comprehensive model performance metrics."""
    y_pred = model.predict(X_test)
    
    if task_type == 'classification':
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Add ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['ROC AUC'] = roc_auc_score(y_test, y_pred_proba)
            except:
                pass
    else:
        metrics = {
            'R² Score': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return metrics


def create_metrics_display(metrics: dict):
    """Create a formatted display of model metrics."""
    cols = st.columns(len(metrics))
    
    for i, (metric_name, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{formatted_value}</div>
                    <div class="metric-label">{metric_name}</div>
                </div>
            """, unsafe_allow_html=True)


def detect_and_remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Detect and remove duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (cleaned_df, num_duplicates_removed)
    """
    initial_rows = len(df)
    
    # Check for duplicates
    duplicates_mask = df.duplicated()
    num_duplicates = duplicates_mask.sum()
    
    if num_duplicates > 0:
        # Remove duplicates
        df_cleaned = df.drop_duplicates()
        removed_count = initial_rows - len(df_cleaned)
        return df_cleaned, removed_count
    else:
        return df, 0


def plot_learning_curves(model, X, y, task_type: str = 'classification', cv: int = 5):
    """
    Generate learning curves for model performance evaluation.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        task_type: 'classification' or 'regression'
        cv: Number of cross-validation folds
        
    Returns:
        Plotly figure with learning curves
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
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
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
    Create an enhanced preprocessing summary with detailed information.
    
    Args:
        steps: List of preprocessing steps
        dataset_info: Dictionary with dataset information
        
    Returns:
        Formatted summary for display
    """
    summary = {
        'preprocessing_steps': steps,
        'dataset_shape': dataset_info.get('shape', 'Unknown'),
        'missing_values': dataset_info.get('missing_values', 0),
        'duplicates_removed': dataset_info.get('duplicates_removed', 0),
        'categorical_encoded': dataset_info.get('categorical_encoded', []),
        'features_scaled': dataset_info.get('features_scaled', []),
        'features_selected': dataset_info.get('features_selected', 'All')
    }
    
    return summary
