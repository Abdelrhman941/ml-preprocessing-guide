# Core Libraries
import os
import time
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Streamlit for Web Apps
import streamlit as st

# Machine Learning Models & Utilities
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error, roc_curve, auc, precision_recall_curve )
import xgboost as xgb
import lightgbm as lgb

# Preprocessing (Custom)
from gui_code import MLPreprocessor

# Automated ML and Explainability
import h2o
from h2o.automl import H2OAutoML
import shap
################################################################################################### streamlit configuration
st.set_page_config(
    page_title="🚀 ML Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3a506b, #5bc0be, #1c2541, #b2bec3);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /*---------------------*/
    .metric-bar {
        width: 100%;
        height: 36px;
        background: linear-gradient(90deg, #5bc0be 0%, #3a506b 100%);
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    .metric-num {
        font-size: 1.7rem;
        font-weight: bold;
        color: #f4f4f9;
        letter-spacing: 1px;
    }
    .metric-label {
        text-align: center;
        font-size: 1rem;
        color: #e0e1dd;
        opacity: 0.85;
    }
    
    .metric-card {
        background: rgba(58, 80, 107, 0.13);
        padding: 1.5rem;
        border-radius: 25px;
        color: #e0e1dd;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.10);
        backdrop-filter: blur(4px);
        border: 2px solid rgba(91, 192, 190, 0.18);
        margin: 0.5rem 0;
    }
    /*---------------------*/

    .success-card {
        background: linear-gradient(135deg, #5bc0be 0%, #3a506b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #f4f4f9;
        margin: 1rem 0;
    }

    .info-card {
        background: rgba(44, 62, 80, 0.09);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #5bc0be;
        margin: 1rem 0;
        color: #e0e1dd;
    }

    .stSelectbox > div > div > select {
        background-color: #232931;
        color: #e0e1dd;
        border: 1px solid #5bc0be;
    }

    .stSlider > div > div > div > div {
        background-color: #5bc0be;
    }

    .sidebar-section {
        background: rgba(44, 62, 80, 0.07);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(91, 192, 190, 0.10);
    }

    .training-log {
        background: #232931;
        color: #e0e1dd;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #5bc0be;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_params' not in st.session_state:
    st.session_state.best_params = {}
if 'training_results' not in st.session_state:
    st.session_state.training_results = {}
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = MLPreprocessor()
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'confusion_matrix' not in st.session_state:
    st.session_state.confusion_matrix = None
if 'preprocessing_steps' not in st.session_state:
    st.session_state.preprocessing_steps = []
################################################################################################### Helper functions
# Add message to training logs
def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.training_logs.append(f"[{timestamp}] {message}")

# Get hyperparameter options for each model
def get_model_params(model_name):
    params = {
        'Random Forest': {
            'n_estimators': {'type': 'slider', 'min': 10, 'max': 500, 'default': 100, 'step': 10},
            'max_depth': {'type': 'slider', 'min': 1, 'max': 30, 'default': 10, 'step': 1},
            'min_samples_split': {'type': 'slider', 'min': 2, 'max': 20, 'default': 2, 'step': 1},
            'min_samples_leaf': {'type': 'slider', 'min': 1, 'max': 20, 'default': 1, 'step': 1},
            'max_features': {'type': 'selectbox', 'options': ['sqrt', 'log2', 'auto'], 'default': 'sqrt'}
        },
        'XGBoost': {
            'n_estimators': {'type': 'slider', 'min': 10, 'max': 500, 'default': 100, 'step': 10},
            'max_depth': {'type': 'slider', 'min': 1, 'max': 15, 'default': 6, 'step': 1},
            'learning_rate': {'type': 'slider', 'min': 0.01, 'max': 0.3, 'default': 0.1, 'step': 0.01},
            'subsample': {'type': 'slider', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'step': 0.1},
            'colsample_bytree': {'type': 'slider', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'step': 0.1}
        },
        'LightGBM': {
            'n_estimators': {'type': 'slider', 'min': 10, 'max': 500, 'default': 100, 'step': 10},
            'max_depth': {'type': 'slider', 'min': 1, 'max': 15, 'default': -1, 'step': 1},
            'learning_rate': {'type': 'slider', 'min': 0.01, 'max': 0.3, 'default': 0.1, 'step': 0.01},
            'num_leaves': {'type': 'slider', 'min': 10, 'max': 300, 'default': 31, 'step': 1},
            'feature_fraction': {'type': 'slider', 'min': 0.5, 'max': 1.0, 'default': 1.0, 'step': 0.1}
        }
    }
    return params.get(model_name, {})

# Create model instance with given parameters
def create_model(model_name, params, task_type='classification'):
    if model_name == 'Random Forest':
        if task_type == 'classification':
            return RandomForestClassifier(**params, random_state=42)
        else:
            return RandomForestRegressor(**params, random_state=42)
    elif model_name == 'XGBoost':
        if task_type == 'classification':
            return xgb.XGBClassifier(**params, random_state=42)
        else:
            return xgb.XGBRegressor(**params, random_state=42)
    elif model_name == 'LightGBM':
        if task_type == 'classification':
            return lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        else:
            return lgb.LGBMRegressor(**params, random_state=42, verbose=-1)

# Evaluate model using cross-validation
def evaluate_model(model, X, y, task_type='classification', metric='accuracy'):
    scoring_map = {
        'accuracy': 'accuracy',
        'f1': 'f1_macro',
        'auc': 'roc_auc',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    
    scoring = scoring_map.get(metric, 'accuracy')
    scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
    
    if metric in ['mse', 'mae']:
        scores = -scores  # Convert back to positive
    
    return scores.mean(), scores.std()

# Create interactive confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f'Class {i}' for i in range(len(cm))]
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        annotation_text=cm,
        colorscale='Viridis',
        showscale=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        width=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Fix for heatmap layout
    fig.update_xaxes(side="bottom")
    
    return fig

# Create feature importance plot
def plot_feature_importance(model, feature_names, top_n=15):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return None
        
        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig = px.bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'},
            title=f'Top {top_n} Feature Importance'
        )
        
        fig.update_layout(
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    except Exception as e:
        st.error(f"Error plotting feature importance: {e}")
        return None

# Plot ROC curve for classification models
def plot_roc_curve(model, X, y):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # For binary classification
    if len(np.unique(y)) == 2:
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC = {roc_auc:.3f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    # For multi-class classification
    else:
        y_bin = label_binarize(y, classes=np.unique(y))
        n_classes = y_bin.shape[1]
        
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        for i in range(n_classes):
            y_score = model.predict_proba(X)[:, i]
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score)
            roc_auc = auc(fpr, tpr)
            name = f'Class {i} (AUC = {roc_auc:.3f})'
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        
        fig.update_layout(
            title='ROC Curve (Multi-class)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700, height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig

# Plot actual vs predicted values for regression
def plot_regression_results(y_true, y_pred):
    fig = px.scatter(
        x=y_true, y=y_pred,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title='Actual vs Predicted Values'
    )
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        height=500,
        width=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

# Plot learning curve to show model performance with varying training set sizes
def plot_learning_curve(model, X, y, cv=5):
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy' if st.session_state.task_type == 'classification' else 'neg_mean_squared_error'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    
    # Add training scores
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores_mean,
        mode='lines+markers',
        name='Training score',
        line=dict(color='#4ECDC4'),
        error_y=dict(type='data', array=train_scores_std, visible=True)
    ))
    
    # Add test scores
    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_scores_mean,
        mode='lines+markers',
        name='Cross-validation score',
        line=dict(color='#FF6B6B'),
        error_y=dict(type='data', array=test_scores_std, visible=True)
    ))
    
    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Training examples',
        yaxis_title='Score',
        height=500,
        width=700,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

# Load a sample dataset for demonstration
@st.cache_data
def load_sample_dataset(dataset_name):
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, fetch_california_housing
    
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task_type = 'classification'
    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task_type = 'classification'
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task_type = 'classification'
    elif dataset_name == "Diabetes (Regression)":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task_type = 'regression'
    elif dataset_name == "California Housing (Regression)":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task_type = 'regression'
    
    return df, task_type

@st.cache_data
def create_histogram(_df, column):
    fig = px.histogram(
        _df, x=column,
        marginal="box",
        title=f"Distribution of {column}"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white' if st.session_state.theme == 'dark' else 'black')
    )
    return fig

def plot_shap_summary(model, X):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        fig = plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        return fig
    except Exception as e:
        st.error(f"Error generating SHAP plot: {e}")
        return None
################################################################################################### Main Navigation
# Create slide-like navigation at the top
def create_navigation_slides():
    pages = [
        ("🏠", "Home", "home"),
        ("📊", "Data Exploration", "data_exploration"),
        ("🔧", "Preprocessing", "preprocessing"),
        ("🧠", "Model Training", "training"),
        ("📈", "Evaluation", "evaluation"),
        ("⚙️", "Settings", "settings")
    ]
    
    current_page_index = next((i for i, (_, _, page_id) in enumerate(pages) if page_id == st.session_state.page), 0)
    
    # Add custom CSS for slide navigation
    st.markdown("""
    <style>
        .slide-nav {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
            padding: 10px;
            background: rgba(0,0,0,0.1);
            border-radius: 15px;
            gap: 10px;
        }
        .nav-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 10px 15px;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 120px;
            text-align: center;
        }
        .nav-step.active {
            background: linear-gradient(135deg, #4ECDC4, #45B7D1);
            color: white;
            box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);
        }
        .nav-step.completed {
            background: rgba(76, 175, 80, 0.2);
            border: 2px solid #4CAF50;
        }
        .nav-step.disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .nav-step:hover:not(.disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .step-number {
            font-size: 24px;
            margin-bottom: 5px;
        }
        .step-title {
            font-size: 12px;
            font-weight: bold;
        }
        .progress-line {
            height: 3px;
            background: linear-gradient(90deg, #4ECDC4, #45B7D1);
            border-radius: 2px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create navigation HTML
    nav_html = '<div class="slide-nav">'
    
    for i, (icon, title, page_id) in enumerate(pages):
        is_active = page_id == st.session_state.page
        is_completed = i < current_page_index
        is_disabled = False  # You can add logic here to disable certain steps
        
        class_name = "nav-step"
        if is_active:
            class_name += " active"
        elif is_completed:
            class_name += " completed"
        elif is_disabled:
            class_name += " disabled"
        
        nav_html += f'''
        <div class="{class_name}" onclick="navigate_to_page('{page_id}')">
            <div class="step-number">{icon}</div>
            <div class="step-title">{title}</div>
        </div>
        '''
        
        # Add connecting line between steps (except for the last one)
        if i < len(pages) - 1:
            nav_html += '<div style="width: 30px; height: 2px; background: rgba(255,255,255,0.3); margin: 0 10px;"></div>'
    
    nav_html += '</div>'
    
    # Add progress bar
    progress_percentage = (current_page_index / (len(pages) - 1)) * 100
    nav_html += f'<div class="progress-line" style="width: {progress_percentage}%;"></div>'
    
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Add JavaScript for navigation
    st.markdown("""
    <script>
    function navigate_to_page(page_id) {
        // This will be handled by Streamlit buttons below
        window.parent.postMessage({type: 'navigate', page: page_id}, '*');
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Create invisible buttons for navigation (Streamlit way)
    cols = st.columns(len(pages))
    for i, (icon, title, page_id) in enumerate(pages):
        with cols[i]:
            if st.button(f"{icon} {title}", key=f"nav_btn_{page_id}", 
                        help=f"Navigate to {title}", 
                        use_container_width=True):
                st.session_state.page = page_id
                st.rerun()

# Create quick settings in a collapsible section
def create_quick_settings():
    with st.expander("⚙️ Quick Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Dataset**")
            dataset_option = st.radio("Choose dataset source", ["Upload CSV", "Sample Dataset"], key="quick_dataset")
            
            if dataset_option == "Upload CSV":
                uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="quick_csv_uploader")
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.dataset = df
                        st.success(f"Dataset loaded: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
            else:
                sample_dataset = st.selectbox(
                    "Select Sample Dataset",
                    ["Iris", "Wine", "Breast Cancer", "Diabetes (Regression)", "California Housing (Regression)"],
                    key="quick_sample_select"
                )
                if st.button("Load Sample Dataset", key="quick_load_sample"):
                    df, task_type = load_sample_dataset(sample_dataset)
                    st.session_state.dataset = df
                    st.session_state.task_type = task_type
                    st.success(f"Sample dataset loaded")
        
        with col2:
            st.markdown("**🎯 Target & Task**")
            if st.session_state.dataset is not None:
                target_col = st.selectbox(
                    "Select target variable",
                    st.session_state.dataset.columns.tolist(),
                    key="quick_target_select"
                )
                if target_col:
                    st.session_state.target = target_col
                    
                    # Auto-detect task type
                    if st.session_state.task_type is None:
                        unique_values = st.session_state.dataset[target_col].nunique()
                        st.session_state.task_type = 'classification' if unique_values <= 10 else 'regression'
                    
                    task_type = st.radio(
                        "Task type",
                        ["Classification", "Regression"],
                        index=0 if st.session_state.task_type == 'classification' else 1,
                        key="quick_task_type"
                    )
                    st.session_state.task_type = task_type.lower()
        
        with col3:
            st.markdown("**🤖 Model Settings**")
            if st.session_state.dataset is not None and st.session_state.target is not None:
                model_name = st.selectbox(
                    "Select model",
                    ["Random Forest", "XGBoost", "LightGBM"],
                    key="quick_model_select"
                )
                
                # Quick start training button
                if st.button("🚀 Quick Start Training", key="quick_train_button", type="primary"):
                    # Store default model settings
                    st.session_state.model_name = model_name
                    st.session_state.model_params = {}
                    st.session_state.metric = 'accuracy' if st.session_state.task_type == 'classification' else 'mse'
                    st.session_state.tuning_method = 'None'
                    st.session_state.use_cv = True
                    
                    # Navigate to training page
                    st.session_state.page = 'training'
                    st.rerun()

def create_navigation_buttons():
    """Create Previous/Next navigation buttons at the bottom of each page"""
    pages = [
        ("🏠", "Home", "home"),
        ("📊", "Data Exploration", "data_exploration"),
        ("🔧", "Preprocessing", "preprocessing"),
        ("🧠", "Model Training", "training"),
        ("📈", "Evaluation", "evaluation"),
        ("⚙️", "Settings", "settings")
    ]
    
    current_page_index = next((i for i, (_, _, page_id) in enumerate(pages) if page_id == st.session_state.page), 0)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_page_index > 0:
            prev_page = pages[current_page_index - 1]
            if st.button(f"⬅️ {prev_page[1]}", key="prev_btn", use_container_width=True):
                st.session_state.page = prev_page[2]
                st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 10px;'><strong>Step {current_page_index + 1} of {len(pages)}</strong></div>", unsafe_allow_html=True)
    
    with col3:
        if current_page_index < len(pages) - 1:
            next_page = pages[current_page_index + 1]
            if st.button(f"{next_page[1]} ➡️", key="next_btn", use_container_width=True):
                st.session_state.page = next_page[2]
                st.rerun()

# Call the navigation function
create_navigation_slides()
create_quick_settings()
################################################################################################### Page rendering based on selected page
def render_home_page():
    st.markdown("<h1 class='main-header'>🚀 Machine Learning Studio</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Welcome to ML Studio")
        st.markdown("""
        This application guides you through the complete machine learning workflow:
        
        
        1. **Data Loading**     - Upload your CSV or use sample datasets
        2. **Data Exploration** - Understand your data with visualizations
        3. **Preprocessing**    - Clean and transform your data
        4. **Model Training**   - Train and tune machine learning models
        5. **Evaluation**       - Assess model performance with metrics and visualizations
        6. **Export Results**   - Save your model and findings
        
        To get started, upload a dataset using the sidebar or load a sample dataset.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Quick Start")
        st.markdown("""
        1. Select a dataset source in the sidebar
        2. Choose your target variable
        3. Select a model and configure parameters
        4. Click 'Start Training'
        5. Explore results in the Evaluation page
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Supported Models")
        st.markdown("""
        - **Random Forest**
        - **XGBoost**
        - **LightGBM**
        """)
        st.markdown("</div>", unsafe_allow_html=True)
      # Show dataset preview if loaded
    if st.session_state.dataset is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Dataset Preview")
        st.dataframe(st.session_state.dataset.head())
        st.markdown(f"Shape: {st.session_state.dataset.shape[0]} rows, {st.session_state.dataset.shape[1]} columns")
        st.markdown("</div>", unsafe_allow_html=True)
    
    create_navigation_buttons()

def render_data_exploration_page():
    st.markdown("<h1 class='main-header'>📊 Data Exploration</h1>", unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("Please load a dataset first.")
        return
    
    df = st.session_state.dataset
    
    # Dataset overview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-bar">
                <span class="metric-num">{df.shape[0]:,}</span>
            </div>
            <div class="metric-label">Rows</div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-bar">
            <span class="metric-num">{df.shape[1]:,}</span>
        </div>
        <div class="metric-label">Columns</div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-bar">
            <span class="metric-num">{df.isna().sum().sum():,}</span>
        </div>
        <div class="metric-label">Missing Values</div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-bar">
            <span class="metric-num">{df.duplicated().sum():,}</span>
        </div>
        <div class="metric-label">Duplicates</div>""", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Data preview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Data Preview")
    st.dataframe(df.head(10))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Data types
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Data Types")
    
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isna().sum(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    st.dataframe(dtypes_df)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Data Visualizations")
    
    viz_tabs = st.tabs(["Distribution", "Correlation", "Missing Values", "Target Analysis"])
    
    with viz_tabs[0]:
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column for histogram", numeric_cols)
                fig = px.histogram(
                    df, x=selected_col,
                    marginal="box",
                    title=f"Distribution of {selected_col}"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                selected_cat_col = st.selectbox("Select column for bar chart", categorical_cols)
                fig = px.bar(
                    df[selected_cat_col].value_counts().reset_index(),
                    x='index', y=selected_cat_col,
                    title=f"Counts of {selected_cat_col}"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        # Correlation heatmap
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Viridis',
                title="Correlation Heatmap"
            )
            fig.update_layout(
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for correlation analysis.")
    
    with viz_tabs[2]:
        # Missing values visualization
        if df.isna().sum().sum() > 0:
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': df.isna().sum(),
                'Percentage': (df.isna().sum() / len(df) * 100).round(2)
            }).sort_values('Missing Values', ascending=False)
            
            fig = px.bar(
                missing_df,
                x='Column', y='Percentage',
                title="Missing Values by Column (%)",
                color='Percentage',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(missing_df)
        else:
            st.success("No missing values in the dataset!")
    
    with viz_tabs[3]:
        # Target variable analysis
        if st.session_state.target:
            target_col = st.session_state.target
            
            if st.session_state.task_type == 'classification':
                # Classification target distribution
                fig = px.bar(
                    df[target_col].value_counts().reset_index(),
                    x='index', y=target_col,
                    title=f"Distribution of Target Variable: {target_col}",
                    color='index'
                )
                fig.update_layout(
                    xaxis_title="Class",
                    yaxis_title="Count",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Class balance analysis
                class_counts = df[target_col].value_counts()
                min_class = class_counts.min()
                max_class = class_counts.max()
                imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
                
                st.markdown(f"**Class Imbalance Ratio**: {imbalance_ratio:.2f}:1")
                if imbalance_ratio > 10:
                    st.warning("Severe class imbalance detected. Consider using class balancing techniques.")
                elif imbalance_ratio > 3:
                    st.info("Moderate class imbalance detected. Consider using class weights or sampling techniques.")
            else:
                # Regression target distribution
                fig = px.histogram(
                    df, x=target_col,
                    title=f"Distribution of Target Variable: {target_col}",
                    marginal="box"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Target statistics
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness'],
                    'Value': [
                        df[target_col].mean(),
                        df[target_col].median(),
                        df[target_col].std(),
                        df[target_col].min(),
                        df[target_col].max(),
                        df[target_col].skew()
                    ]
                })
                st.dataframe(stats_df)
        else:            st.info("Please select a target variable in the sidebar.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    create_navigation_buttons()

def render_preprocessing_page():
    st.markdown("<h1 class='main-header'>🔧 Preprocessing</h1>", unsafe_allow_html=True)
    if st.session_state.dataset is None:
        st.warning("Please load a dataset first.")
        return
    df = st.session_state.dataset.copy()
    preprocessor = st.session_state.preprocessor
    
    # Preprocessing options
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Preprocessing Pipeline")
    
    preprocessing_tabs = st.tabs(["Missing Values", "Outliers", "Feature Engineering", "Encoding", "Scaling", "Feature Selection", "Data Splitting"])
    
    with preprocessing_tabs[0]:
        st.markdown("#### Handle Missing Values")
        missing_cols = df.columns[df.isna().any()].tolist()
        if missing_cols:
            st.write(f"Columns with missing values: {', '.join(missing_cols)}")
            missing_strategy_tooltips = {
                'Drop rows': 'Remove rows with missing values. Use when missing data is minimal.',
                'Drop columns': 'Remove columns with missing values. Use when columns are not critical.',
                'Mean imputation': 'Replace missing values with the column mean. Suitable for numerical data.',
                'Median imputation': 'Replace missing values with the column median. Robust to outliers.',
                'Mode imputation': 'Replace missing values with the column mode. Suitable for categorical data.',
                'KNN imputation': 'Use nearest neighbors to impute missing values. More accurate but slower.'
            }
            missing_strategy = st.selectbox(
                "Missing values strategy",
                ["Drop rows", "Drop columns", "Mean imputation", "Median imputation", "Mode imputation", "KNN imputation"],
                help="Select a strategy to handle missing values."
            )
            st.markdown(f"""
            <div class='tooltip'>Selected Strategy: {missing_strategy}
                <span class='tooltiptext'>{missing_strategy_tooltips.get(missing_strategy, 'Choose a strategy to handle missing data.')}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Apply Missing Values Strategy"):
                try:
                    if missing_strategy in ["Mean imputation", "Median imputation"] and not df[missing_cols].select_dtypes(include=['number']).columns.tolist():
                        st.error("Mean or Median imputation requires numerical columns with missing values.")
                        return
                    if missing_strategy == "Drop rows" and df[missing_cols].isna().sum().sum() > len(df) * 0.5:
                        st.warning("Dropping rows will remove more than 50% of the data. Consider imputation instead.")
                    if missing_strategy == "Drop rows":
                        df = preprocessor.handle_missing_data(df, strategy='drop_rows')
                    elif missing_strategy == "Drop columns":
                        df = preprocessor.handle_missing_data(df, strategy='drop_columns')
                    elif missing_strategy == "Mean imputation":
                        df = preprocessor.handle_missing_data(df, strategy='mean')
                    elif missing_strategy == "Median imputation":
                        df = preprocessor.handle_missing_data(df, strategy='median')
                    elif missing_strategy == "Mode imputation":
                        df = preprocessor.handle_missing_data(df, strategy='mode')
                    elif missing_strategy == "KNN imputation":
                        df = preprocessor.handle_missing_data(df, strategy='knn')
                    
                    st.session_state.dataset = df
                    st.success(f"Applied {missing_strategy} to handle missing values")
                    st.session_state.preprocessing_steps.append(f"Applied {missing_strategy} to handle missing values")
                except Exception as e:
                    st.error(f"Error applying missing values strategy: {e}")
        else:
            st.info("No missing values found in the dataset.")
    
    with preprocessing_tabs[1]:
        # Outlier handling
        st.markdown("#### Handle Outliers")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            selected_cols = st.multiselect(
                "Select columns for outlier detection",
                numeric_cols
            )
            
            if selected_cols:
                outlier_method = st.selectbox(
                    "Outlier detection method",
                    ["IQR", "Z-Score", "Modified Z-Score"]
                )
                
                outlier_treatment = st.selectbox(
                    "Outlier treatment method",
                    ["None", "Remove", "Winsorize", "Cap", "Log transform"]
                )
                
                if st.button("Detect and Handle Outliers"):
                    try:
                        # Detect outliers
                        detection_method = outlier_method.lower() if outlier_method != "IQR" else "iqr"
                        outliers = preprocessor.detect_outliers(df, method=detection_method, columns=selected_cols)
                        
                        # Display outlier counts
                        outlier_counts = {col: len(indices) for col, indices in outliers.items()}
                        st.write("Outlier counts:", outlier_counts)
                        
                        # Handle outliers if treatment selected
                        if outlier_treatment != "None":
                            treatment_method = outlier_treatment.lower()
                            df = preprocessor.handle_outliers(
                                df, 
                                method=treatment_method,
                                detection_method=detection_method,
                                columns=selected_cols
                            )
                            st.session_state.dataset = df
                            st.success(f"Applied {outlier_treatment} to handle outliers")
                            st.session_state.preprocessing_steps.append(f"Applied {outlier_treatment} to handle outliers in {', '.join(selected_cols)}")
                    except Exception as e:
                        st.error(f"Error handling outliers: {e}")
            else:
                st.info("Please select at least one column for outlier detection.")
        else:
            st.info("No numeric columns available for outlier detection.")
    
    with preprocessing_tabs[2]:
        # Feature Engineering
        st.markdown("#### Feature Engineering")
        
        st.markdown("Select operations to create new features or transform existing ones.")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        feature_eng_options = st.multiselect(
            "Select feature engineering operations",
            ["Polynomial Features", "Interaction Features", "Binning", "Log Transform", "Extract Date Components"]
        )
        
        if "Polynomial Features" in feature_eng_options:
            st.markdown("##### Polynomial Features")
            poly_cols = st.multiselect("Select columns for polynomial features", numeric_cols)
            poly_degree = st.slider("Polynomial degree", 2, 5, 2)
            
            if poly_cols and st.button("Create Polynomial Features"):
                try:
                    df = preprocessor.create_polynomial_features(df, poly_cols, degree=poly_degree)
                    st.session_state.dataset = df
                    st.success(f"Created polynomial features (degree {poly_degree}) for {', '.join(poly_cols)}")
                    st.session_state.preprocessing_steps.append(f"Created polynomial features (degree {poly_degree})")
                except Exception as e:
                    st.error(f"Error creating polynomial features: {e}")
        
        if "Interaction Features" in feature_eng_options:
            st.markdown("##### Interaction Features")
            interaction_cols = st.multiselect("Select columns for interaction features", numeric_cols)
            
            if len(interaction_cols) >= 2 and st.button("Create Interaction Features"):
                try:
                    df = preprocessor.create_interaction_features(df, interaction_cols)
                    st.session_state.dataset = df
                    st.success(f"Created interaction features for {', '.join(interaction_cols)}")
                    st.session_state.preprocessing_steps.append("Created interaction features")
                except Exception as e:
                    st.error(f"Error creating interaction features: {e}")
        
        if "Binning" in feature_eng_options:
            st.markdown("##### Binning")
            bin_col = st.selectbox("Select column for binning", numeric_cols)
            bin_method = st.selectbox("Binning method", ["Equal Width", "Equal Frequency", "Custom"])
            num_bins = st.slider("Number of bins", 2, 20, 5)
            
            if bin_col and st.button("Apply Binning"):
                try:
                    df = preprocessor.bin_feature(df, bin_col, method=bin_method.lower().replace(" ", "_"), bins=num_bins)
                    st.session_state.dataset = df
                    st.success(f"Applied {bin_method} binning to {bin_col} with {num_bins} bins")
                    st.session_state.preprocessing_steps.append(f"Binned {bin_col} into {num_bins} bins")
                except Exception as e:
                    st.error(f"Error applying binning: {e}")
        
        if "Log Transform" in feature_eng_options:
            st.markdown("##### Log Transform")
            log_cols = st.multiselect("Select columns for log transform", numeric_cols)
            
            if log_cols and st.button("Apply Log Transform"):
                try:
                    df = preprocessor.apply_log_transform(df, log_cols)
                    st.session_state.dataset = df
                    st.success(f"Applied log transform to {', '.join(log_cols)}")
                    st.session_state.preprocessing_steps.append(f"Applied log transform to {len(log_cols)} columns")
                except Exception as e:
                    st.error(f"Error applying log transform: {e}")
        
        if "Extract Date Components" in feature_eng_options and datetime_cols:
            st.markdown("##### Extract Date Components")
            date_col = st.selectbox("Select datetime column", datetime_cols)
            date_components = st.multiselect(
                "Select date components to extract",
                ["Year", "Month", "Day", "Weekday", "Quarter", "Week of Year", "Day of Year"]
            )
            
            if date_col and date_components and st.button("Extract Date Components"):
                try:
                    df = preprocessor.extract_datetime_features(df, date_col, components=date_components)
                    st.session_state.dataset = df
                    st.success(f"Extracted {', '.join(date_components)} from {date_col}")
                    st.session_state.preprocessing_steps.append(f"Extracted date components from {date_col}")
                except Exception as e:
                    st.error(f"Error extracting date components: {e}")
    
    with preprocessing_tabs[3]:
        # Encoding
        st.markdown("#### Encode Categorical Features")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            encoding_method = st.selectbox(
                "Encoding method",
                ["One-Hot Encoding", "Label Encoding", "Target Encoding", "Frequency Encoding", "Binary Encoding"]
            )
            
            selected_cols = st.multiselect("Select columns to encode", categorical_cols)
            
            if selected_cols and st.button("Apply Encoding"):
                try:
                    if encoding_method == "One-Hot Encoding":
                        df = preprocessor.encode_categorical(df, selected_cols, method='onehot')
                    elif encoding_method == "Label Encoding":
                        df = preprocessor.encode_categorical(df, selected_cols, method='label')
                    elif encoding_method == "Target Encoding":
                        if st.session_state.target:
                            df = preprocessor.encode_categorical(df, selected_cols, method='target', target=st.session_state.target)
                        else:
                            st.error("Target encoding requires a target variable to be selected.")
                            return
                    elif encoding_method == "Frequency Encoding":
                        df = preprocessor.encode_categorical(df, selected_cols, method='frequency')
                    elif encoding_method == "Binary Encoding":
                        df = preprocessor.encode_categorical(df, selected_cols, method='binary')
                    
                    st.session_state.dataset = df
                    st.success(f"Applied {encoding_method} to {', '.join(selected_cols)}")
                    st.session_state.preprocessing_steps.append(f"Applied {encoding_method} to {len(selected_cols)} columns")
                except Exception as e:
                    st.error(f"Error encoding categorical features: {e}")
        else:
            st.info("No categorical columns found in the dataset.")
    
    with preprocessing_tabs[4]:
        # Scaling
        st.markdown("#### Scale Numerical Features")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            scaling_method = st.selectbox(
                "Scaling method",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "MaxAbsScaler"]
            )
            
            selected_cols = st.multiselect("Select columns to scale", numeric_cols)
            
            if selected_cols and st.button("Apply Scaling"):
                try:
                    if scaling_method == "StandardScaler":
                        df = preprocessor.scale_features(df, selected_cols, method='standard')
                    elif scaling_method == "MinMaxScaler":
                        df = preprocessor.scale_features(df, selected_cols, method='minmax')
                    elif scaling_method == "RobustScaler":
                        df = preprocessor.scale_features(df, selected_cols, method='robust')
                    elif scaling_method == "Normalizer":
                        df = preprocessor.scale_features(df, selected_cols, method='normalizer')
                    elif scaling_method == "MaxAbsScaler":
                        df = preprocessor.scale_features(df, selected_cols, method='maxabs')
                    
                    st.session_state.dataset = df
                    st.success(f"Applied {scaling_method} to {', '.join(selected_cols)}")
                    st.session_state.preprocessing_steps.append(f"Applied {scaling_method} to {len(selected_cols)} columns")
                except Exception as e:
                    st.error(f"Error scaling features: {e}")
        else:
            st.info("No numerical columns found in the dataset.")
    
    with preprocessing_tabs[5]:
        # Feature Selection
        st.markdown("#### Feature Selection")
        
        if st.session_state.target:
            selection_method = st.selectbox(
                "Feature selection method",
                ["Correlation-based", "Variance Threshold", "Feature Importance", "Recursive Feature Elimination", "SelectKBest"]
            )
            
            if selection_method == "Correlation-based":
                corr_threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.8, 0.01)
                if st.button("Apply Correlation-based Selection"):
                    try:
                        df, selected_features = preprocessor.select_features(df, method='correlation', threshold=corr_threshold, target=st.session_state.target)
                        st.session_state.dataset = df
                        st.success(f"Selected {len(selected_features)} features based on correlation threshold {corr_threshold}")
                        st.write("Selected features:", selected_features)
                        st.session_state.preprocessing_steps.append(f"Selected {len(selected_features)} features using correlation-based method")
                    except Exception as e:
                        st.error(f"Error selecting features: {e}")
            
            elif selection_method == "Variance Threshold":
                var_threshold = st.slider("Variance threshold", 0.0, 1.0, 0.1, 0.01)
                if st.button("Apply Variance Threshold Selection"):
                    try:
                        df, selected_features = preprocessor.select_features(df, method='variance', threshold=var_threshold)
                        st.session_state.dataset = df
                        st.success(f"Selected {len(selected_features)} features based on variance threshold {var_threshold}")
                        st.write("Selected features:", selected_features)
                        st.session_state.preprocessing_steps.append(f"Selected {len(selected_features)} features using variance threshold method")
                    except Exception as e:
                        st.error(f"Error selecting features: {e}")
            
            elif selection_method == "Feature Importance":
                n_features = st.slider("Number of features to select", 1, len(df.columns) - 1, min(5, len(df.columns) - 1))
                if st.button("Apply Feature Importance Selection"):
                    try:
                        df, selected_features = preprocessor.select_features(df, method='importance', n_features=n_features, target=st.session_state.target)
                        st.session_state.dataset = df
                        st.success(f"Selected top {len(selected_features)} features based on importance")
                        st.write("Selected features:", selected_features)
                        st.session_state.preprocessing_steps.append(f"Selected {len(selected_features)} features using feature importance method")
                    except Exception as e:
                        st.error(f"Error selecting features: {e}")
            
            elif selection_method == "Recursive Feature Elimination":
                n_features = st.slider("Number of features to select", 1, len(df.columns) - 1, min(5, len(df.columns) - 1))
                if st.button("Apply RFE Selection"):
                    try:
                        df, selected_features = preprocessor.select_features(df, method='rfe', n_features=n_features, target=st.session_state.target)
                        st.session_state.dataset = df
                        st.success(f"Selected {len(selected_features)} features using RFE")
                        st.write("Selected features:", selected_features)
                        st.session_state.preprocessing_steps.append(f"Selected {len(selected_features)} features using RFE method")
                    except Exception as e:
                        st.error(f"Error selecting features: {e}")
            
            elif selection_method == "SelectKBest":
                n_features = st.slider("Number of features to select", 1, len(df.columns) - 1, min(5, len(df.columns) - 1))
                score_func = st.selectbox("Scoring function", ["f_classif", "chi2", "mutual_info_classif", "f_regression", "mutual_info_regression"])
                if st.button("Apply SelectKBest Selection"):
                    try:
                        df, selected_features = preprocessor.select_features(df, method='kbest', n_features=n_features, score_func=score_func, target=st.session_state.target)
                        st.session_state.dataset = df
                        st.success(f"Selected {len(selected_features)} features using SelectKBest with {score_func}")
                        st.write("Selected features:", selected_features)
                        st.session_state.preprocessing_steps.append(f"Selected {len(selected_features)} features using SelectKBest method")
                    except Exception as e:
                        st.error(f"Error selecting features: {e}")
        else:
            st.info("Please select a target variable in the sidebar first.")
    
    with preprocessing_tabs[6]:
        # Data splitting
        st.markdown("#### Split Data for Training")
        
        if st.session_state.target:
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
            validation_split = st.checkbox("Use validation set", value=False)
            
            if validation_split:
                val_size = st.slider("Validation set size (%)", 10, 30, 15) / 100
            
            stratify = st.checkbox("Stratify split (for classification)", value=True if st.session_state.task_type == 'classification' else False)
            time_based = st.checkbox("Time-based split", value=False)
            
            if time_based:
                datetime_cols = []
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        datetime_cols.append(col)
                
                if datetime_cols:
                    time_col = st.selectbox("Select datetime column for time-based split", datetime_cols)
                else:
                    st.warning("No datetime columns available for time-based split.")
                    time_based = False
            
            if st.button("Split Data"):
                try:
                    if validation_split:
                        if time_based and 'time_col' in locals():
                            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
                                df,
                                target=st.session_state.target,
                                test_size=test_size,
                                val_size=val_size,
                                stratify=stratify,
                                time_based=True,
                                time_col=time_col
                            )
                        else:
                            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
                                df,
                                target=st.session_state.target,
                                test_size=test_size,
                                val_size=val_size,
                                stratify=stratify
                            )
                        
                        st.session_state.X_train = X_train
                        st.session_state.X_val = X_val
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_val = y_val
                        st.session_state.y_test = y_test
                        
                        st.success(f"Data split into train ({len(X_train)} samples), validation ({len(X_val)} samples), and test ({len(X_test)} samples) sets")
                    else:
                        if time_based and 'time_col' in locals():
                            X_train, X_test, y_train, y_test = preprocessor.split_data(
                                df,
                                target=st.session_state.target,
                                test_size=test_size,
                                stratify=stratify,
                                time_based=True,
                                time_col=time_col
                            )
                        else:
                            X_train, X_test, y_train, y_test = preprocessor.split_data(
                                df,
                                target=st.session_state.target,
                                test_size=test_size,
                                stratify=stratify
                            )
                        
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.X_val = None
                        st.session_state.y_val = None
                        
                        st.success(f"Data split into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets")
                    
                    # Check for class imbalance in classification tasks
                    if st.session_state.task_type == 'classification':
                        imbalance_detected, imbalance_ratio, recommendation = preprocessor.detect_class_imbalance(y_train)
                        
                        if imbalance_detected:
                            st.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}:1). {recommendation}")
                    
                    st.session_state.preprocessing_steps.append(f"Split data into train/test sets (test size: {test_size})")
                except Exception as e:
                    st.error(f"Error splitting data: {e}")
        else:
            st.info("Please select a target variable in the sidebar first.")
      # Display preprocessing steps
    if st.session_state.preprocessing_steps:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Preprocessing Steps Applied")
        for i, step in enumerate(st.session_state.preprocessing_steps, 1):
            st.markdown(f"{i}. {step}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    create_navigation_buttons()

def render_training_page():
    st.markdown("<h1 class='main-header'>🧠 Model Training</h1>", unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("Please load a dataset first.")
        return
    
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Please split your data in the Preprocessing page first.")
        return
    
    # Training section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model Training")
    
    # Display training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Configuration")
        st.write(f"**Model:** {st.session_state.model_name}")
        st.write(f"**Task Type:** {st.session_state.task_type.capitalize()}")
        st.write(f"**Evaluation Metric:** {st.session_state.metric}")
        st.write(f"**Hyperparameter Tuning:** {st.session_state.tuning_method}")
        st.write(f"**Cross-Validation:** {'Enabled' if st.session_state.use_cv else 'Disabled'}")
    
    with col2:
        st.markdown("#### Dataset Information")
        st.write(f"**Training Samples:** {len(st.session_state.X_train)}")
        if st.session_state.X_val is not None:
            st.write(f"**Validation Samples:** {len(st.session_state.X_val)}")
        st.write(f"**Test Samples:** {len(st.session_state.X_test)}")
        st.write(f"**Features:** {st.session_state.X_train.shape[1]}")
        
        if st.session_state.task_type == 'classification':
            n_classes = len(np.unique(st.session_state.y_train))
            st.write(f"**Classes:** {n_classes}")
    
    # Training log
    st.markdown("#### Training Log")
    log_placeholder = st.empty()
    
    # Start training button
    if st.button("🚀 Start Training", key="start_training"):
        with st.spinner("Training in progress..."):
            try:
                # Clear previous logs
                st.session_state.training_logs = []
                add_log(f"Starting training with {st.session_state.model_name}...")
                
                # Get model parameters
                model_params = st.session_state.model_params
                add_log(f"Model parameters: {model_params}")
                
                # Create base model
                base_model = create_model(
                    st.session_state.model_name,
                    model_params,
                    task_type=st.session_state.task_type
                )
                
                # Hyperparameter tuning
                if st.session_state.tuning_method != "None":
                    add_log(f"Starting hyperparameter tuning with {st.session_state.tuning_method}...")
                    
                    # Define parameter grid based on model
                    if st.session_state.model_name == "Random Forest":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20, 30],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                    elif st.session_state.model_name == "XGBoost":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 6, 9],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'subsample': [0.8, 1.0],
                            'colsample_bytree': [0.8, 1.0]
                        }
                    elif st.session_state.model_name == "LightGBM":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 6, 9],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'num_leaves': [31, 50, 100],
                            'feature_fraction': [0.8, 1.0]
                        }
                    
                    # Create tuner
                    if st.session_state.tuning_method == "Grid Search":
                        tuner = GridSearchCV(
                            base_model,
                            param_grid,
                            cv=5 if st.session_state.use_cv else None,
                            scoring=st.session_state.metric,
                            n_jobs=-1
                        )
                    else:  # Random Search
                        tuner = RandomizedSearchCV(
                            base_model,
                            param_grid,
                            n_iter=10,
                            cv=5 if st.session_state.use_cv else None,
                            scoring=st.session_state.metric,
                            n_jobs=-1,
                            random_state=42
                        )
                    
                    # Fit tuner
                    start_time = time.time()
                    tuner.fit(st.session_state.X_train, st.session_state.y_train)
                    training_time = time.time() - start_time
                    
                    # Get best model and parameters
                    best_model = tuner.best_estimator_
                    best_params = tuner.best_params_
                    best_score = tuner.best_score_
                    
                    add_log(f"Hyperparameter tuning completed in {training_time:.2f} seconds")
                    add_log(f"Best parameters: {best_params}")
                    add_log(f"Best CV score: {best_score:.4f}")
                    
                    # Store results
                    st.session_state.best_model = best_model
                    st.session_state.best_params = best_params
                    st.session_state.training_results = {
                        'best_score': best_score,
                        'training_time': training_time,
                        'tuned_params_count': len(best_params),
                        'all_results': tuner.cv_results_
                    }
                else:
                    # Train single model
                    add_log("Training single model...")
                    
                    start_time = time.time()
                    
                    if st.session_state.use_cv:
                        # Evaluate with cross-validation
                        cv_score, cv_std = evaluate_model(
                            base_model,
                            st.session_state.X_train,
                            st.session_state.y_train,
                            task_type=st.session_state.task_type,
                            metric=st.session_state.metric
                        )
                        add_log(f"Cross-validation score: {cv_score:.4f} (±{cv_std:.4f})")
                    
                    # Fit on full training data
                    base_model.fit(st.session_state.X_train, st.session_state.y_train)
                    training_time = time.time() - start_time
                    
                    # Make predictions on test set
                    y_pred = base_model.predict(st.session_state.X_test)
                    
                    # Calculate metrics
                    if st.session_state.task_type == 'classification':
                        accuracy = accuracy_score(st.session_state.y_test, y_pred)
                        f1 = f1_score(st.session_state.y_test, y_pred, average='weighted')
                        
                        add_log(f"Training completed in {training_time:.2f} seconds")
                        add_log(f"Test accuracy: {accuracy:.4f}")
                        add_log(f"Test F1 score: {f1:.4f}")
                        
                        # Store results
                        st.session_state.best_model = base_model
                        st.session_state.best_params = model_params
                        st.session_state.training_results = {
                            'accuracy': accuracy,
                            'f1_score': f1,
                            'training_time': training_time
                        }
                        
                        # Generate confusion matrix
                        st.session_state.confusion_matrix = confusion_matrix(st.session_state.y_test, y_pred)
                    else:  # Regression
                        mse = mean_squared_error(st.session_state.y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(st.session_state.y_test, y_pred)
                        r2 = r2_score(st.session_state.y_test, y_pred)
                        
                        add_log(f"Training completed in {training_time:.2f} seconds")
                        add_log(f"Test MSE: {mse:.4f}")
                        add_log(f"Test RMSE: {rmse:.4f}")
                        add_log(f"Test MAE: {mae:.4f}")
                        add_log(f"Test R²: {r2:.4f}")
                        
                        # Store results
                        st.session_state.best_model = base_model
                        st.session_state.best_params = model_params
                        st.session_state.training_results = {
                            'mse': mse,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'training_time': training_time
                        }
                
                # Generate feature importance
                if hasattr(st.session_state.best_model, 'feature_importances_') or hasattr(st.session_state.best_model, 'coef_'):
                    feature_names = st.session_state.X_train.columns.tolist()
                    st.session_state.feature_importance = plot_feature_importance(
                        st.session_state.best_model,
                        feature_names)
                
                add_log("Training completed successfully!")
                
                # Set page to evaluation
                st.session_state.page = 'evaluation'
                st.rerun()
            
            except Exception as e:
                add_log(f"Error during training: {e}")
                st.error(f"Training failed: {e}")
      # Display training logs
    if st.session_state.training_logs:
        log_content = "\n".join(st.session_state.training_logs)
        log_placeholder.markdown(f"<div class='training-log'>{log_content}</div>", unsafe_allow_html=True)
    else:
        log_placeholder.info("Training logs will appear here once training starts.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    create_navigation_buttons()

def render_evaluation_page():
    st.markdown("<h1 class='main-header'>📈 Evaluation</h1>", unsafe_allow_html=True)
    if st.session_state.best_model is None:
        st.warning("Please train a model first.")
        return
    
    # Results overview
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Information")
        st.write(f"**Model Type:** {st.session_state.model_name}")
        st.write(f"**Task Type:** {st.session_state.task_type.capitalize()}")
        
        if 'training_time' in st.session_state.training_results:
            training_time = st.session_state.training_results['training_time']
            st.write(f"**Training Time:** {training_time:.2f} seconds")
        
        if st.session_state.tuning_method != "None":
            st.write(f"**Tuning Method:** {st.session_state.tuning_method}")
            if 'tuned_params_count' in st.session_state.training_results:
                st.write(f"**Tuned Parameters:** {st.session_state.training_results['tuned_params_count']}")
    
    with col2:
        st.markdown("#### Performance Metrics")
        
        if st.session_state.task_type == 'classification':
            if 'accuracy' in st.session_state.training_results:
                st.write(f"**Accuracy:** {st.session_state.training_results['accuracy']:.4f}")
            if 'f1_score' in st.session_state.training_results:
                st.write(f"**F1 Score:** {st.session_state.training_results['f1_score']:.4f}")
            if 'best_score' in st.session_state.training_results:
                st.write(f"**Best CV Score:** {st.session_state.training_results['best_score']:.4f}")
        else:  # Regression
            if 'mse' in st.session_state.training_results:
                st.write(f"**MSE:** {st.session_state.training_results['mse']:.4f}")
            if 'rmse' in st.session_state.training_results:
                st.write(f"**RMSE:** {st.session_state.training_results['rmse']:.4f}")
            if 'mae' in st.session_state.training_results:
                st.write(f"**MAE:** {st.session_state.training_results['mae']:.4f}")
            if 'r2' in st.session_state.training_results:
                st.write(f"**R²:** {st.session_state.training_results['r2']:.4f}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Best hyperparameters
    if st.session_state.best_params:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Best Hyperparameters")
        
        params_df = pd.DataFrame({
            'Parameter': list(st.session_state.best_params.keys()),
            'Value': list(st.session_state.best_params.values())
        })
        st.dataframe(params_df)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Visualizations")
    
    viz_tabs = st.tabs(["Feature Importance", "Confusion Matrix", "ROC Curve", "Learning Curve", "SHAP Explanation"])
    
    with viz_tabs[0]:
        # Feature importance
        if st.session_state.feature_importance is not None:
            st.plotly_chart(st.session_state.feature_importance, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")
    
    with viz_tabs[1]:
        # Confusion matrix for classification
        if st.session_state.task_type == 'classification':
            if st.session_state.best_model is not None:
                y_pred = st.session_state.best_model.predict(st.session_state.X_test)
                cm_fig = plot_confusion_matrix(st.session_state.y_test, y_pred)
                st.plotly_chart(cm_fig, use_container_width=True)
        else:
            st.info("Confusion matrix is only available for classification tasks.")
    
    with viz_tabs[2]:
        # ROC curve for classification or Actual vs Predicted for regression
        if st.session_state.task_type == 'classification':
            if st.session_state.best_model is not None:
                roc_fig = plot_roc_curve(st.session_state.best_model, st.session_state.X_test, st.session_state.y_test)
                st.plotly_chart(roc_fig, use_container_width=True)
        else:  # Regression
            if st.session_state.best_model is not None:
                y_pred = st.session_state.best_model.predict(st.session_state.X_test)
                reg_fig = plot_regression_results(st.session_state.y_test, y_pred)
                st.plotly_chart(reg_fig, use_container_width=True)
    
    with viz_tabs[3]:
        # Learning curve
        if st.session_state.best_model is not None:
            with st.spinner("Generating learning curve..."):
                try:
                    lc_fig = plot_learning_curve(
                        st.session_state.best_model,
                        st.session_state.X_train,
                        st.session_state.y_train
                    )
                    st.plotly_chart(lc_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating learning curve: {e}")
    
    with viz_tabs[4]:
        st.markdown("### SHAP Model Explanation")
        if st.session_state.best_model is not None:
            with st.spinner("Generating SHAP summary..."):
                shap_fig = plot_shap_summary(st.session_state.best_model, st.session_state.X_test)
                if shap_fig:
                    st.pyplot(shap_fig)
        else:
            st.info("SHAP explanation not available for this model.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Export section
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export model
        if st.button("Export Model (.pkl)"):
            try:
                model_bytes = pickle.dumps(st.session_state.best_model)
                model_name = f"{st.session_state.model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                
                st.download_button(
                    label="Download Model",
                    data=model_bytes,
                    file_name=model_name,
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Error exporting model: {e}")
    
    with col2:
        # Export hyperparameters
        if st.button("Export Hyperparameters (.json)"):
            try:
                params_json = json.dumps(st.session_state.best_params, indent=4)
                params_name = f"hyperparameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                st.download_button(
                    label="Download Hyperparameters",
                    data=params_json,
                    file_name=params_name,
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error exporting hyperparameters: {e}")
    
    with col3:
        # Export results report
        if st.button("Export Results Report (.json)"):
            try:
                # Create comprehensive report
                report = {
                    'model_name': st.session_state.model_name,
                    'task_type': st.session_state.task_type,
                    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'hyperparameters': st.session_state.best_params,
                    'performance_metrics': st.session_state.training_results,
                    'preprocessing_steps': st.session_state.preprocessing_steps
                }
                
                report_json = json.dumps(report, indent=4)
                report_name = f"ml_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                st.download_button(
                    label="Download Report",
                    data=report_json,
                    file_name=report_name,
                    mime="application/json"                )
            except Exception as e:
                st.error(f"Error exporting report: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    create_navigation_buttons()

def render_settings_page():
    st.markdown("<h1 class='main-header'>⚙️ Settings & About</h1>", unsafe_allow_html=True)
    
    # About section (collapsible)
    with st.expander("ℹ️ About ML Studio", expanded=False):
        st.markdown("""
        <div class='card'>
        <strong>ML Studio</strong> is a professional machine learning platform built with Streamlit, offering an end-to-end workflow:
        <ul>
            <li>Data loading, exploration, and preprocessing</li>
            <li>Feature engineering and selection</li>
            <li>Model training, tuning, and evaluation</li>
            <li>Interactive visualizations and export options</li>
        </ul>
        <hr>
        <b>Version:</b> 1.0.0 &nbsp; | &nbsp; <b>Streamlit:</b> 1.24.0 &nbsp; | &nbsp; <b>Python:</b> 3.9+<br>
        <div style="margin-top: 1em; padding: 1em; background: linear-gradient(90deg, #374151 0%, #4ECDC4 100%); border-radius: 10px; color: #e0f7fa; font-size: 1.1em;">
            📚 <strong>See more details and source code in the official repo:</strong><br>
            <a href="https://github.com/Abdelrhman941/ml-preprocessing-guide.git" target="_blank" style="color: #b2f5ea; text-decoration: underline; font-weight: bold;">
                https://github.com/Abdelrhman941/ml-preprocessing-guide.git
            </a>
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Reset application
    st.markdown("#### Reset Application")
    if st.button("Reset All Data", key="reset_app"):
        # Reset all session state variables
        for key in list(st.session_state.keys()):
            if key != 'page':  # Keep the current page
                del st.session_state[key]
        
        # Reinitialize necessary session state variables
        st.session_state.dataset = None
        st.session_state.target = None
        st.session_state.task_type = None
        st.session_state.training_logs = []
        st.session_state.best_model = None
        st.session_state.best_params = {}
        st.session_state.training_results = {}
        st.session_state.preprocessor = MLPreprocessor()
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        st.session_state.feature_importance = None
        st.session_state.confusion_matrix = None
        st.session_state.preprocessing_steps = []
        
        st.success("Application has been reset. All data and models have been cleared.")
        st.session_state.page = 'home'  # Go back to home page
        st.rerun()
    
    create_navigation_buttons()


# Render the appropriate page based on session state
if st.session_state.page == 'home':
    render_home_page()
elif st.session_state.page == 'data_exploration':
    render_data_exploration_page()
elif st.session_state.page == 'preprocessing':
    render_preprocessing_page()
elif st.session_state.page == 'training':
    render_training_page()
elif st.session_state.page == 'evaluation':
    render_evaluation_page()
elif st.session_state.page == 'settings':
    render_settings_page()
##################################################################################################