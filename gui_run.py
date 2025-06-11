import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import json
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your preprocessing class
from gui_code import MLPreprocessor

# Configure page
st.set_page_config(
    page_title="🚀 ML Hyperparameter Tuning Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
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
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 0.5rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div > select {
        background-color: #2b2b2b;
        color: white;
        border: 1px solid #4ECDC4;
    }
    
    .stSlider > div > div > div > div {
        background-color: #4ECDC4;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .training-log {
        background: #1e1e1e;
        color: #00ff00;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

def add_log(message):
    """Add message to training logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.training_logs.append(f"[{timestamp}] {message}")

def get_model_params(model_name):
    """Get hyperparameter options for each model"""
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

def create_model(model_name, params, task_type='classification'):
    """Create model instance with given parameters"""
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

def evaluate_model(model, X, y, task_type='classification', metric='accuracy'):
    """Evaluate model using cross-validation"""
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

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Create interactive confusion matrix"""
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
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    """Create feature importance plot"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            return None
        
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[feature_names[i] for i in indices],
                y=[importances[i] for i in indices],
                marker=dict(
                    color=importances[indices],
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importances',
            xaxis_title='Features',
            yaxis_title='Importance',
            font=dict(color='white'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickangle=45)
        )
        
        return fig
    except:
        return None

# Main header
st.markdown('<h1 class="main-header">🚀 ML Hyperparameter Tuning Studio</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Dataset Configuration")
    
    # Dataset upload/selection
    data_source = st.radio("Data Source", ["Upload File", "Sample Dataset"])
    
    df = None
    if data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        sample_datasets = {
            "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "Tips": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
            "Titanic": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
        }
        selected_dataset = st.selectbox("Select Sample Dataset", list(sample_datasets.keys()))
        if st.button("Load Sample Dataset"):
            try:
                df = pd.read_csv(sample_datasets[selected_dataset])
                st.success(f"✅ {selected_dataset} dataset loaded!")
            except:
                st.error("Failed to load sample dataset")

    if df is not None:
        st.markdown("### 🎯 Model Configuration")
        
        # Target variable selection
        target_column = st.selectbox(
            "Select Target Variable",
            df.columns.tolist(),
            help="Choose the column you want to predict"
        )
        
        # Task type detection
        if target_column:
            unique_values = df[target_column].nunique()
            if unique_values <= 10 and df[target_column].dtype in ['object', 'int64']:
                task_type = 'classification'
                st.info(f"🏷️ Detected: **Classification** task ({unique_values} classes)")
            else:
                task_type = 'regression'
                st.info(f"📈 Detected: **Regression** task")
        
        # Model selection
        if task_type == 'classification':
            available_models = ['Random Forest', 'XGBoost', 'LightGBM']
        else:
            available_models = ['Random Forest', 'XGBoost', 'LightGBM']
        
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose the machine learning algorithm"
        )
        
        # Hyperparameter configuration
        st.markdown("### ⚙️ Hyperparameters")
        model_params = get_model_params(selected_model)
        user_params = {}
        
        for param_name, param_config in model_params.items():
            if param_config['type'] == 'slider':
                if isinstance(param_config['default'], float):
                    user_params[param_name] = st.slider(
                        param_name.replace('_', ' ').title(),
                        param_config['min'],
                        param_config['max'],
                        param_config['default'],
                        param_config['step']
                    )
                else:
                    user_params[param_name] = st.slider(
                        param_name.replace('_', ' ').title(),
                        param_config['min'],
                        param_config['max'],
                        param_config['default'],
                        param_config['step']
                    )
            elif param_config['type'] == 'selectbox':
                user_params[param_name] = st.selectbox(
                    param_name.replace('_', ' ').title(),
                    param_config['options'],
                    index=param_config['options'].index(param_config['default'])
                )
        
        # Performance metric selection
        st.markdown("### 📏 Evaluation Metric")
        if task_type == 'classification':
            metrics = ['accuracy', 'f1', 'auc', 'precision', 'recall']
        else:
            metrics = ['mse', 'mae', 'r2']
        
        selected_metric = st.selectbox(
            "Performance Metric",
            metrics,
            help="Choose the metric to optimize"
        )
        
        # Training configuration
        st.markdown("### 🚀 Training Options")
        use_grid_search = st.checkbox("Use Grid Search", help="Automatically find best hyperparameters")
        
        if use_grid_search:
            search_type = st.radio("Search Type", ["Grid Search", "Random Search"])
            if search_type == "Random Search":
                n_iter = st.slider("Number of Iterations", 10, 100, 20)
        
        # Start training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state.training_logs = []
            add_log("Training started...")
            st.rerun()

# Main content area
if df is not None and target_column:
    # Data preprocessing
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Dataset Overview")
        
        # Dataset info
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.markdown(f'<div class="metric-card"><h3>{df.shape[0]}</h3><p>Rows</p></div>', unsafe_allow_html=True)
        with info_col2:
            st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Columns</p></div>', unsafe_allow_html=True)
        with info_col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.markdown(f'<div class="metric-card"><h3>{missing_pct:.1f}%</h3><p>Missing</p></div>', unsafe_allow_html=True)
        with info_col4:
            duplicates = df.duplicated().sum()
            st.markdown(f'<div class="metric-card"><h3>{duplicates}</h3><p>Duplicates</p></div>', unsafe_allow_html=True)
        
        # Data preview
        st.markdown("#### 🔍 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Target distribution
        st.markdown("#### 🎯 Target Distribution")
        if task_type == 'classification':
            target_counts = df[target_column].value_counts()
            fig = px.bar(
                x=target_counts.index,
                y=target_counts.values,
                title="Target Variable Distribution",
                color=target_counts.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(
                df,
                x=target_column,
                title="Target Variable Distribution",
                color_discrete_sequence=['#4ECDC4']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📝 Training Logs")
        if st.session_state.training_logs:
            log_text = "\n".join(st.session_state.training_logs[-15:])  # Show last 15 logs
            st.markdown(f'<div class="training-log">{log_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="training-log">No training logs yet. Click "Start Training" to begin.</div>', unsafe_allow_html=True)
        
        # Auto-refresh logs during training
        if len(st.session_state.training_logs) > 0 and "Training completed" not in st.session_state.training_logs[-1]:
            time.sleep(1)
            st.rerun()

    # Training execution
    if st.session_state.training_logs and "Training started" in st.session_state.training_logs[-1]:
        # Prepare data
        add_log("Preparing data...")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Basic preprocessing
        add_log("Applying preprocessing...")
        # Handle missing values
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target if classification
        if task_type == 'classification' and y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
        
        add_log("Data preprocessing completed")
        
        # Model training
        add_log(f"Training {selected_model} model...")
        
        if not use_grid_search:
            # Single model training
            model = create_model(selected_model, user_params, task_type)
            mean_score, std_score = evaluate_model(model, X, y, task_type, selected_metric)
            
            # Fit the model for feature importance
            model.fit(X, y)
            
            st.session_state.best_model = model
            st.session_state.best_params = user_params
            st.session_state.training_results = {
                'mean_score': mean_score,
                'std_score': std_score,
                'metric': selected_metric
            }
            
            add_log(f"Training completed. {selected_metric}: {mean_score:.4f} (±{std_score:.4f})")
        
        else:
            # Grid/Random search
            add_log(f"Starting {search_type.lower()}...")
            
            # Define parameter grid (simplified for demo)
            param_grid = {}
            for param_name, param_config in model_params.items():
                if param_config['type'] == 'slider':
                    if isinstance(param_config['default'], float):
                        param_grid[param_name] = np.linspace(
                            param_config['min'], 
                            param_config['max'], 
                            5
                        ).tolist()
                    else:
                        param_grid[param_name] = list(range(
                            param_config['min'], 
                            param_config['max'] + 1, 
                            max(1, (param_config['max'] - param_config['min']) // 4)
                        ))
                elif param_config['type'] == 'selectbox':
                    param_grid[param_name] = param_config['options']
            
            base_model = create_model(selected_model, {}, task_type)
            
            if search_type == "Grid Search":
                search = GridSearchCV(
                    base_model, 
                    param_grid, 
                    cv=5, 
                    scoring=selected_metric if selected_metric != 'auc' else 'roc_auc',
                    n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    base_model, 
                    param_grid, 
                    cv=5, 
                    scoring=selected_metric if selected_metric != 'auc' else 'roc_auc',
                    n_iter=n_iter,
                    n_jobs=-1,
                    random_state=42
                )
            
            search.fit(X, y)
            
            st.session_state.best_model = search.best_estimator_
            st.session_state.best_params = search.best_params_
            st.session_state.training_results = {
                'mean_score': search.best_score_,
                'std_score': 0,  # GridSearchCV doesn't provide std
                'metric': selected_metric
            }
            
            add_log(f"{search_type} completed. Best {selected_metric}: {search.best_score_:.4f}")
        
        add_log("Training completed successfully!")
        st.rerun()

    # Results section
    if st.session_state.best_model is not None:
        st.markdown("### 🏆 Training Results")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            score = st.session_state.training_results['mean_score']
            metric = st.session_state.training_results['metric'].upper()
            st.markdown(f'<div class="success-card"><h2>{score:.4f}</h2><p>Best {metric} Score</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="success-card"><h2>{selected_model}</h2><p>Best Model</p></div>', unsafe_allow_html=True)
        
        with col3:
            param_count = len(st.session_state.best_params)
            st.markdown(f'<div class="success-card"><h2>{param_count}</h2><p>Tuned Parameters</p></div>', unsafe_allow_html=True)
        
        # Best parameters
        st.markdown("#### ⚙️ Best Hyperparameters")
        params_df = pd.DataFrame(
            list(st.session_state.best_params.items()),
            columns=['Parameter', 'Value']
        )
        st.dataframe(params_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature importance
            if hasattr(st.session_state.best_model, 'feature_importances_'):
                st.markdown("#### 🎯 Feature Importance")
                fig = plot_feature_importance(st.session_state.best_model, X.columns.tolist())
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model predictions for confusion matrix (classification only)
            if task_type == 'classification':
                st.markdown("#### 📊 Confusion Matrix")
                y_pred = st.session_state.best_model.predict(X)
                fig = plot_confusion_matrix(y, y_pred)
                st.plotly_chart(fig, use_container_width=True)
        
        # Export section
        st.markdown("### 💾 Export Model")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Model", use_container_width=True):
                model_data = pickle.dumps(st.session_state.best_model)
                st.download_button(
                    label="Download Pickle File",
                    data=model_data,
                    file_name=f"best_{selected_model.lower().replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📋 Download Parameters", use_container_width=True):
                params_json = json.dumps(st.session_state.best_params, indent=2)
                st.download_button(
                    label="Download JSON File",
                    data=params_json,
                    file_name=f"best_params_{selected_model.lower().replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("📊 Download Results", use_container_width=True):
                results_data = {
                    'model': selected_model,
                    'task_type': task_type,
                    'metric': selected_metric,
                    'score': st.session_state.training_results['mean_score'],
                    'parameters': st.session_state.best_params,
                    'dataset_shape': df.shape,
                    'timestamp': datetime.now().isoformat()
                }
                results_json = json.dumps(results_data, indent=2)
                st.download_button(
                    label="Download Report",
                    data=results_json,
                    file_name=f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

else:
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Welcome to ML Hyperparameter Tuning Studio!</h3>
        <p>Get started by:</p>
        <ol>
            <li>📊 Upload your dataset or select a sample dataset from the sidebar</li>
            <li>🎯 Choose your target variable</li>
            <li>🤖 Select a machine learning model</li>
            <li>⚙️ Tune hyperparameters using the sliders</li>
            <li>🚀 Click "Start Training" to begin!</li>
        </ol>
        <p>This studio supports both classification and regression tasks with automated preprocessing and model evaluation.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "Built by Abdelrhman Ezzat 🫡",
    help="ML Hyperparameter Tuning Studio v1.0"
)