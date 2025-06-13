import numpy as np
from sklearn.model_selection import  GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import configurations
from config.settings import HYPERPARAMETER_GRIDS, MODEL_CONFIG

def create_model(model_name: str, task_type: str, **params):
    """
    Create a model instance based on model name and task type
    
    Args:
        model_name (str): Name of the model ('Random Forest', 'XGBoost', 'LightGBM')
        task_type (str): 'classification' or 'regression'
        **params: Model parameters
        
    Returns:
        Configured model instance
    """
    if task_type == 'classification':
        if model_name == 'Random Forest':
            return RandomForestClassifier(random_state=42, **params)
        elif model_name == 'XGBoost':
            return xgb.XGBClassifier(random_state=42, eval_metric='logloss', **params)
        elif model_name == 'LightGBM':
            return lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
    else:  # regression
        if model_name == 'Random Forest':
            return RandomForestRegressor(random_state=42, **params)
        elif model_name == 'XGBoost':
            return xgb.XGBRegressor(random_state=42, eval_metric='rmse', **params)
        elif model_name == 'LightGBM':
            return lgb.LGBMRegressor(random_state=42, verbose=-1, **params)
    
    raise ValueError(f"Unsupported model: {model_name} for task: {task_type}")

def train_model_with_tuning(X_train, X_test, y_train, y_test, model_name, task_type, 
                           tuning_method='None', use_cv=True, **model_params):
    """
    Train model with optional hyperparameter tuning
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        model_name (str): Name of the model
        task_type (str): 'classification' or 'regression'
        tuning_method (str): 'None', 'Grid Search', or 'Random Search'
        use_cv (bool): Whether to use cross-validation
        **model_params: Additional model parameters
        
    Returns:
        dict: Training results containing model, scores, and parameters
    """
    results = {
        'model': None,
        'best_params': {},
        'train_score': 0,
        'test_score': 0,
        'cv_scores': [],
        'model_name': model_name,
        'task_type': task_type
    }
    
    try:
        # Create base model
        if tuning_method == 'None':
            model = create_model(model_name, task_type, **model_params)
            model.fit(X_train, y_train)
            results['model'] = model
            results['best_params'] = model_params
        else:
            # Hyperparameter tuning
            base_model = create_model(model_name, task_type)
            param_grid = HYPERPARAMETER_GRIDS.get(model_name, {})
            
            if tuning_method == 'Grid Search':
                search = GridSearchCV(base_model, param_grid, cv=5, 
                                    scoring='accuracy' if task_type == 'classification' else 'neg_mean_squared_error',
                                    n_jobs=-1)
            else:  # Random Search
                search = RandomizedSearchCV(base_model, param_grid, cv=5, n_iter=20,
                                        scoring='accuracy' if task_type == 'classification' else 'neg_mean_squared_error',
                                        n_jobs=-1, random_state=42)
            
            search.fit(X_train, y_train)
            results['model'] = search.best_estimator_
            results['best_params'] = search.best_params_
        
        # Calculate scores
        model = results['model']
        
        if task_type == 'classification':
            results['train_score'] = accuracy_score(y_train, model.predict(X_train))
            results['test_score'] = accuracy_score(y_test, model.predict(X_test))
            
            if use_cv:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                results['cv_scores'] = cv_scores.tolist()
        else:
            results['train_score'] = r2_score(y_train, model.predict(X_train))
            results['test_score'] = r2_score(y_test, model.predict(X_test))
            
            if use_cv:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                results['cv_scores'] = cv_scores.tolist()
        
        return results
        
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return results

def evaluate_model(model, X_test, y_test, task_type):
    """
    Evaluate trained model and return comprehensive metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        task_type (str): 'classification' or 'regression'
        
    Returns:
        dict: Evaluation metrics
    """
    try:
        predictions = model.predict(X_test)
        
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'f1_score': f1_score(y_test, predictions, average='weighted'),
            }
            
            # Add probability-based metrics if available
            if hasattr(model, 'predict_proba'):
                from sklearn.metrics import roc_auc_score
                try:
                    if len(np.unique(y_test)) == 2:  # Binary classification
                        proba = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, proba)
                    else:  # Multi-class
                        proba = model.predict_proba(X_test)
                        metrics['roc_auc'] = roc_auc_score(y_test, proba, multi_class='ovr')
                except:
                    metrics['roc_auc'] = None
        else:
            from sklearn.metrics import mean_absolute_error
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")
        return {}

def add_log_message(message: str):
    """
    Add a message to the training logs
    
    Args:
        message (str): Log message to add
    """
    if 'training_logs' not in st.session_state:
        st.session_state.training_logs = []
    
    st.session_state.training_logs.append(message)
    
    # Keep only last 100 logs to prevent memory issues
    if len(st.session_state.training_logs) > 100:
        st.session_state.training_logs = st.session_state.training_logs[-100:]
