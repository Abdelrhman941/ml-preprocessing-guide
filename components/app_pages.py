import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

import streamlit as st
from preprocessing.preprocessor import MLPreprocessor
from utils.navigation import create_navigation_buttons, display_dataset_overview
from utils.helpers import (
    get_model_params, create_model, evaluate_model, add_log_message, plot_feature_importance,
    plot_regression_results, get_model_metrics_summary, create_metrics_display, create_metrics_explanations,
    detect_and_remove_duplicates, plot_learning_curves, detect_task_type,
    get_classification_report, plot_confusion_matrix_enhanced, plot_roc_curve_multiclass
)

# ------ Render the home page with welcome information ------
def render_home_page():
    st.markdown("<h1 class='main-header'>🚀 Machine Learning Studio</h1>", unsafe_allow_html=True)
    
    # Add pipeline status indicators
    st.markdown("---")
    st.markdown("### 📋 Pipeline Status")
    
    # Status indicators
    status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
    
    with status_col1:
        data_status = "✅ Loaded" if st.session_state.dataset is not None else "⏳ Pending"
        status_color = "green" if st.session_state.dataset is not None else "orange"
        st.markdown(f"**📁 Data:** <span style='color:{status_color}'>{data_status}</span>", unsafe_allow_html=True)
    
    with status_col2:
        target_status = "✅ Selected" if st.session_state.target else "⏳ Pending"
        status_color = "green" if st.session_state.target else "orange"
        st.markdown(f"**🎯 Target:** <span style='color:{status_color}'>{target_status}</span>", unsafe_allow_html=True)
    
    with status_col3:
        preprocessing_status = "✅ Applied" if st.session_state.preprocessing_steps else "⏳ Pending"
        status_color = "green" if st.session_state.preprocessing_steps else "orange"
        st.markdown(f"**🔧 Preprocessing:** <span style='color:{status_color}'>{preprocessing_status}</span>", unsafe_allow_html=True)
    
    with status_col4:
        model_status = "✅ Trained" if st.session_state.best_model else "⏳ Pending"
        status_color = "green" if st.session_state.best_model else "orange"
        st.markdown(f"**🤖 Model:** <span style='color:{status_color}'>{model_status}</span>", unsafe_allow_html=True)
    
    with status_col5:
        evaluation_status = "✅ Complete" if st.session_state.training_results else "⏳ Pending"
        status_color = "green" if st.session_state.training_results else "orange"
        st.markdown(f"**📊 Evaluation:** <span style='color:{status_color}'>{evaluation_status}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Welcome to ML Studio v2.0")
        st.markdown("""
        **Enhanced Features:**
        - 🔧 **Fixed Training Errors**: Automatic task detection (classification/regression)
        - 📊 **Comprehensive Metrics**: Color-coded performance indicators
        - 🎨 **Beautiful Visualizations**: Interactive plots and charts
        - ⚡ **Improved UX**: Real-time feedback and status updates
        
        **Complete ML Workflow:**
        1. **📁 Data Loading** - Upload CSV or use sample datasets
        2. **🔍 Data Exploration** - Comprehensive data analysis
        3. **🔧 Preprocessing** - Advanced data cleaning and feature engineering
        4. **🤖 Model Training** - Automated hyperparameter tuning
        5. **📈 Evaluation** - Detailed performance analysis
        
        **Get Started:** Upload a dataset using Quick Settings above! 👆
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🚀 Quick Actions")
        
        # Quick action buttons
        if st.session_state.dataset is None:
            st.markdown("**Next Step:** Load your data")
            if st.button("📂 Go to Data Exploration", use_container_width=True):
                st.session_state.page = 'data_exploration'
                st.rerun()
        elif not st.session_state.target:
            st.markdown("**Next Step:** Select target variable")
            if st.button("🎯 Go to Preprocessing", use_container_width=True):
                st.session_state.page = 'preprocessing'
                st.rerun()
        elif not st.session_state.best_model:
            st.markdown("**Next Step:** Train your model")
            if st.button("🤖 Go to Training", use_container_width=True):
                st.session_state.page = 'training'
                st.rerun()
        else:
            st.markdown("**All set!** View results")
            if st.button("📊 Go to Evaluation", use_container_width=True):
                st.session_state.page = 'evaluation'
                st.rerun()
        
        st.markdown("---")
        
        # Dataset info if available
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            st.markdown("**📈 Current Dataset:**")
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", len(df.columns))
            if st.session_state.target:
                st.metric("Target", st.session_state.target)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions and Tips
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("💡 Quick Tips")
    
    tip_cols = st.columns(3)
    
    with tip_cols[0]:
        st.markdown("""
        **📊 Data Quality**
        - Upload clean, well-structured CSV files
        - Ensure consistent data types
        - Handle missing values appropriately
        """)
    
    with tip_cols[1]:
        st.markdown("""
        **🎯 Target Selection**
        - Choose meaningful target variables
        - Verify classification vs regression tasks
        - Balance your dataset if needed
        """)
    
    with tip_cols[2]:
        st.markdown("""
        **🚀 Model Performance**
        - Try different algorithms
        - Use hyperparameter tuning
        - Validate with cross-validation
        """)

# ------ Render the data exploration page with comprehensive data analysis ------
def render_data_exploration_page():
    st.markdown("<h1 class='main-header'>📊 Data Exploration</h1>", unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.warning("⚠️ Please load a dataset first using the Quick Settings above.")
        st.markdown("</div>", unsafe_allow_html=True)
        create_navigation_buttons()
        return
    
    with st.spinner("⏳ Loading dataset analysis..."):
        df = st.session_state.dataset
        
        # Dataset overview
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📋 Dataset Overview")
        display_dataset_overview(df)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data preview
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data types and info
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Data Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("#### Column Details")
        dtypes_df = pd.DataFrame({
            'Column'        : df.columns,
            'Type'          : df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count'    : df.isna().sum(),
            'Unique Values' : [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True)
    
    with info_col2:
        st.markdown("#### Statistical Summary")
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns for statistical summary")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Data Visualizations")
    
    viz_tabs = st.tabs(["📊 Distributions", "🔗 Correlations", "❓ Missing Values", "🎯 Target Analysis"])
    
    with viz_tabs[0]:
        _render_distribution_plots(df)
    
    with viz_tabs[1]:
        _render_correlation_analysis(df)
    
    with viz_tabs[2]:
        _render_missing_values_analysis(df)
    
    with viz_tabs[3]:
        _render_target_analysis(df)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    create_navigation_buttons()

# ------ Render distribution plots for numeric and categorical variables ------
def _render_distribution_plots(df):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Numeric Distributions")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select column for histogram", numeric_cols, key="hist_col")
            if selected_col:
                fig = px.histogram(
                    df, x=selected_col,
                    marginal="box",
                    title=f"Distribution of {selected_col}",
                    color_discrete_sequence=['#5bc0be']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available")
    
    with col2:
        st.markdown("#### Categorical Distributions")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            selected_cat_col = st.selectbox("Select column for bar chart", categorical_cols, key="bar_col")
            if selected_cat_col:
                value_counts = df[selected_cat_col].value_counts().head(20)  # Limit to top 20
                fig = px.bar(
                    x=value_counts.index, y=value_counts.values,
                    title=f"Counts of {selected_cat_col}",
                    color_discrete_sequence=['#3a506b']
                )
                fig.update_layout(
                    xaxis_title=selected_cat_col,
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available")

# ------ Render correlation analysis for numeric variables ------
def _render_correlation_analysis(df):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] < 2:
        st.info("Need at least 2 numeric columns for correlation analysis")
        return
    
    corr = numeric_df.corr()
    
    # Correlation heatmap
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # High correlation pairs
    st.markdown("#### High Correlation Pairs (|r| > 0.7)")
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_val = corr.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append({
                    'Variable 1': corr.columns[i],
                    'Variable 2': corr.columns[j], 
                    'Correlation': corr_val
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        st.dataframe(high_corr_df, use_container_width=True)
    else:
        st.info("No high correlation pairs found")

# ------ Render missing values analysis and visualization ------
def _render_missing_values_analysis(df):
    missing_data = df.isna().sum()
    
    if missing_data.sum() == 0:
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.success("🎉 No missing values in the dataset!")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Missing values summary
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
    }).query('`Missing Count` > 0').sort_values('Missing Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Values Summary")
        st.dataframe(missing_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Missing Values Visualization")
        if not missing_df.empty:
            fig = px.bar(
                missing_df,
                x='Column', y='Missing Percentage',
                title="Missing Values by Column (%)",
                color='Missing Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ------ Render target variable analysis ------
def _render_target_analysis(df):
    if not st.session_state.target:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.info("Please select a target variable in the Quick Settings above.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    target_col = st.session_state.target
    
    if st.session_state.task_type == 'classification':
        _render_classification_target_analysis(df, target_col)
    else:
        _render_regression_target_analysis(df, target_col)

# ------ Render analysis for classification target variable ------
def _render_classification_target_analysis(df, target_col):
    target_counts = df[target_col].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Target Distribution")
        fig = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title=f"Distribution of {target_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Class Balance Analysis")
        min_class = target_counts.min()
        max_class = target_counts.max()
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
            st.warning("Severe class imbalance detected. Consider using balancing techniques.")
            st.markdown("</div>", unsafe_allow_html=True)
        elif imbalance_ratio > 3:
            st.markdown("<div class='info-box'>", unsafe_allow_html=True)
            st.info("Moderate class imbalance. Consider using class weights.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='success-box'>", unsafe_allow_html=True)
            st.success("Classes are relatively balanced.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Class counts table
        st.dataframe(target_counts.to_frame('Count'), use_container_width=True)

# ------ Render analysis for regression target variable ------
def _render_regression_target_analysis(df, target_col):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Target Distribution")
        fig = px.histogram(
            df, x=target_col,
            title=f"Distribution of {target_col}",
            marginal="box",
            color_discrete_sequence=['#5bc0be']        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Target Statistics")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{df[target_col].mean():.4f}",
                f"{df[target_col].median():.4f}",
                f"{df[target_col].std():.4f}",
                f"{df[target_col].min():.4f}",
                f"{df[target_col].max():.4f}",
                f"{df[target_col].skew():.4f}",
                f"{df[target_col].kurtosis():.4f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

# ------ Render the preprocessing page with data cleaning options ------
def render_preprocessing_page():
    st.markdown("<h1 class='main-header'>🔧 Preprocessing</h1>", unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.warning("⚠️ Please load a dataset first.")
        st.markdown("</div>", unsafe_allow_html=True)
        create_navigation_buttons()
        return
    
    with st.spinner("⏳ Loading preprocessing options..."):
        df = st.session_state.dataset.copy()
        # Initialize preprocessor if not exists
        if 'preprocessor' not in st.session_state:
            st.session_state.preprocessor = MLPreprocessor()
        preprocessor = st.session_state.preprocessor
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔧 Preprocessing Pipeline")
    preprocessing_tabs = st.tabs([
        "🔧 Missing Values", 
        "️ Feature Engineering",
        "📊 Encoding", 
        "⚖️ Scaling", 
        "✂️ Feature Selection",
        "📈 Summary"
    ])
    
    with preprocessing_tabs[0]:
        _render_missing_values_handling(df, preprocessor)
    
    with preprocessing_tabs[1]:
        _render_feature_engineering(df, preprocessor)
    
    with preprocessing_tabs[2]:
        _render_encoding_options(df)
    
    with preprocessing_tabs[3]:
        _render_scaling_options(df)
    with preprocessing_tabs[4]:
        _render_feature_selection_options(df)
    
    with preprocessing_tabs[5]:
        _render_preprocessing_summary()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        create_navigation_buttons()

# ------ Render missing values handling options ------
def _render_missing_values_handling(df, preprocessor):
    st.markdown("#### Handle Missing Values")
    
    missing_cols = df.columns[df.isna().any()].tolist()
    
    if not missing_cols:
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.success("🎉 No missing values to handle!")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    st.markdown(f"**Columns with missing values:** {', '.join(missing_cols)}")
    
    # Strategy selection
    strategy = st.selectbox(
        "Select missing value strategy",
        ["Mean imputation", "Median imputation", "Mode imputation", "Drop rows", "Drop columns"],
        help="Choose how to handle missing values"    )
    
    if st.button("Apply Missing Value Treatment", type="primary"):
        try:
            with st.spinner("⏳ Processing missing values..."):
                # Create strategy dictionary
                if strategy == "Drop rows":
                    rows_before = len(df)
                    processed_df = df.dropna()
                    rows_dropped = rows_before - len(processed_df)
                    st.session_state.preprocessing_steps.append(f"Dropped {rows_dropped} rows with missing values")
                    st.success(f"✅ Dropped {rows_dropped} rows with missing values.")
                    st.toast("🗑️ Rows with missing values removed!", icon="✅")
                elif strategy == "Drop columns":
                    processed_df = df.drop(columns=missing_cols)
                    st.session_state.preprocessing_steps.append(f"Dropped columns: {', '.join(missing_cols)}")
                    st.success(f"✅ Dropped {len(missing_cols)} columns with missing values.")
                    st.toast("🗑️ Columns with missing values removed!", icon="✅")
                else:
                    # Use the preprocessor class
                    strategy_map = {
                        'Mean imputation'  : 'mean',
                        'Median imputation': 'median', 
                        'Mode imputation'  : 'mode'
                    }
                    processed_df = preprocessor.handle_missing_data(
                        df, 
                        strategy={col: strategy_map[strategy] for col in missing_cols}
                    )
                    st.session_state.preprocessing_steps.append(f"Applied {strategy} to {len(missing_cols)} columns")                    
                    st.toast(f"🔧 {strategy} applied successfully!", icon="✅")
                    st.success(f"✅ {strategy} applied to {len(missing_cols)} columns with missing values.")
                
                # Ensure Arrow compatibility before storing
                processed_df = preprocessor.ensure_arrow_compatibility(processed_df)
                st.session_state.dataset = processed_df
                st.markdown("---")
                st.rerun()
        
        except Exception as e:
            st.error(f"Error processing missing values: {e}")

# ------ Render feature engineering options ------
def _render_feature_engineering(df, preprocessor):
    st.markdown("#### Feature Engineering")
    
    engineering_method = st.selectbox(
        "Select feature engineering method",
        ["Datetime Features", "Mathematical Features", "Binning Features"],
        help="Choose the type of features to create"
    )
    
    if engineering_method == "Datetime Features":
        datetime_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
        
        if not datetime_cols:
            st.info("No datetime columns found. Convert columns to datetime type first.")
            return
        
        selected_datetime_cols = st.multiselect(
            "Select datetime columns",
            datetime_cols,
            help="Choose datetime columns to extract features from"
        )
        
        if selected_datetime_cols:            
            feature_options = st.multiselect(
                "Select features to create",                ['year', 'month', 'day', 'weekday', 'is_weekend', 'quarter', 'hour', 'minute'],
                default=['year', 'month', 'weekday', 'is_weekend'],
                help="Choose which datetime features to extract"
            )
            
            if st.button("Create Datetime Features", type="primary"):
                try:
                    with st.spinner("⏳ Creating datetime features... Please wait."):                        
                        df_engineered = preprocessor.create_datetime_features(
                        df, selected_datetime_cols, feature_options)
                        
                        # Ensure Arrow compatibility before storing
                        df_engineered = preprocessor.ensure_arrow_compatibility(df_engineered)
                        st.session_state.dataset = df_engineered
                        
                        # Add to preprocessing steps for summary (no UI message shown)
                        new_features = [col for col in df_engineered.columns if col not in df.columns]
                        step_msg = f"Created {len(new_features)} datetime features from {len(selected_datetime_cols)} columns"
                        st.session_state.preprocessing_steps.append(step_msg)
                        
                        st.success(f"✅ Created {len(new_features)} datetime features successfully!")
                        st.toast(f"📅 Datetime features created!", icon="✅")
                        
                        st.markdown("---")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error creating datetime features: {e}")
    
    elif engineering_method == "Mathematical Features":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for mathematical operations")
            return
        
        st.markdown("##### Create Mathematical Features")
        # Simple interface for common operations
        col1, col2 = st.columns(2)
        
        with col1:
            feature_name = st.text_input("New feature name", placeholder="e.g., price_per_sqft")
            operation = st.selectbox("Operation", ["ratio", "difference", "product", "sum"])
        
        with col2:
            col1_select = st.selectbox("First column", numeric_cols, key="math_col1")
            col2_select = st.selectbox("Second column", numeric_cols, key="math_col2")
        
        if feature_name and st.button("Create Mathematical Feature", type="primary"):
            try:
                with st.spinner("Creating mathematical feature..."):
                    feature_operations = {
                        feature_name: {
                            'operation': operation,
                            'columns': [col1_select, col2_select]
                        }                    }
                    
                    df_engineered = preprocessor.create_mathematical_features(df, feature_operations)
                    # Ensure Arrow compatibility before storing
                    df_engineered = preprocessor.ensure_arrow_compatibility(df_engineered)
                    st.session_state.dataset = df_engineered
                    
                    # Add to preprocessing steps for summary (no UI message shown)
                    step_msg = f"Created mathematical feature '{feature_name}' using {operation} of {col1_select} and {col2_select}"
                    st.session_state.preprocessing_steps.append(step_msg)
                    
                    st.success(f"✅ Created mathematical feature '{feature_name}' successfully!")
                    st.toast(f"🧮 Mathematical feature created!", icon="✅")
                    
                    st.markdown("---")
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error creating mathematical feature: {e}")
    
    elif engineering_method == "Binning Features":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns found for binning")
            return
        
        st.markdown("##### Create Binned Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_col = st.selectbox("Select column to bin", numeric_cols)
            feature_name = st.text_input("Binned feature name", placeholder=f"{source_col}_binned")            
            method = st.selectbox("Binning method", ["equal_width", "equal_freq", "custom"])
        
        with col2:
            if method in ["equal_width", "equal_freq"]:
                n_bins = st.slider("Number of bins", 2, 10, 5)
                st.info(f"💡 For {n_bins} bins, provide {n_bins} labels (or leave empty for automatic)")
                labels = st.text_input("Labels (optional, comma-separated)", placeholder=f"Low,Medium,High,Very High" if n_bins == 4 else "Low,Medium,High")
            elif method == "custom":
                bin_edges_str = st.text_input("Bin edges (comma-separated)", placeholder="0,25,50,75,100")
                if bin_edges_str:
                    try:
                        edges_count = len([float(x.strip()) for x in bin_edges_str.split(',')])
                        st.info(f"💡 For {edges_count} bin edges, provide {edges_count-1} labels (or leave empty)")
                    except:
                        st.warning("⚠️ Please provide valid numeric bin edges")
                labels = st.text_input("Labels (optional, comma-separated)", placeholder="Very Low,Low,Medium,High")
        
        if feature_name and source_col and st.button("Create Binned Feature", type="primary"):
            try:
                with st.spinner("Creating binned feature..."):
                    binning_config = {
                        feature_name: {
                            'column': source_col,
                            'method': method
                        }
                    }
                    
                    if method in ["equal_width", "equal_freq"]:
                        binning_config[feature_name]['bins'] = n_bins
                        if labels:
                            binning_config[feature_name]['labels'] = [l.strip() for l in labels.split(',')]
                    elif method == "custom":
                        if 'bin_edges_str' in locals() and bin_edges_str:
                            bin_edges = [float(x.strip()) for x in bin_edges_str.split(',')]
                            binning_config[feature_name]['bin_edges'] = bin_edges
                            if labels:
                                binning_config[feature_name]['labels'] = [l.strip() for l in labels.split(',')]
                    
                    df_engineered, success_messages, error_messages = preprocessor.create_binning_features(df, binning_config)
                    
                    # Display messages in Streamlit UI
                    if error_messages:
                        for error_msg in error_messages:
                            st.error(error_msg)
                        # Don't proceed if there were errors
                        return
                    
                    if success_messages:
                        for success_msg in success_messages:
                            if success_msg.startswith("✅"):
                                st.success(success_msg)
                            elif success_msg.startswith("📝"):
                                st.info(success_msg)
                    
                    # Ensure Arrow compatibility before storing
                    df_engineered = preprocessor.ensure_arrow_compatibility(df_engineered)
                    st.session_state.dataset = df_engineered
                    
                    # Add to preprocessing steps for summary
                    if success_messages:
                        step_msg = f"Created binned feature '{feature_name}' from {source_col} using {method} method"
                        st.session_state.preprocessing_steps.append(step_msg)
                        st.toast(f"🎯 Feature '{feature_name}' created!", icon="✅")
                    
                    st.markdown("---")
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error creating binned feature: {e}")

# ------ Render categorical encoding options ------
def _render_encoding_options(df):
    st.markdown("#### Categorical Encoding")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.info("No categorical columns found for encoding")
        return
    
    st.markdown(f"**Categorical columns:** {', '.join(categorical_cols)}")
    
    encoding_method = st.selectbox(
        "Select encoding method",
        ["Label Encoding", "One-Hot Encoding"],        help="Choose how to encode categorical variables"
    )
    
    selected_cols = st.multiselect(
        "Select columns to encode",
        categorical_cols,
        default=categorical_cols
    )
    
    if selected_cols and st.button("Apply Encoding", type="primary"):
        try:
            with st.spinner("Encoding categorical variables..."):
                df_encoded = df.copy()
                
                if encoding_method == "Label Encoding":
                    for col in selected_cols:
                        le = LabelEncoder()
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    
                    step_msg = f"Applied Label Encoding to: {', '.join(selected_cols)}"
                    st.success(f"✅ Label encoding applied to {len(selected_cols)} columns.")
                
                else:  # One-Hot Encoding
                    original_cols = len(df_encoded.columns)
                    df_encoded = pd.get_dummies(df_encoded, columns=selected_cols, prefix=selected_cols)
                    new_cols = len(df_encoded.columns) - original_cols
                    step_msg = f"Applied One-Hot Encoding to: {', '.join(selected_cols)}"
                    st.success(f"✅ One-hot encoding applied to {len(selected_cols)} columns, creating {new_cols} new features.")                
                # Ensure Arrow compatibility before storing
                from preprocessing.preprocessor import MLPreprocessor
                if 'preprocessor' not in st.session_state:
                    st.session_state.preprocessor = MLPreprocessor()
                df_encoded = st.session_state.preprocessor.ensure_arrow_compatibility(df_encoded)
                
                st.session_state.dataset = df_encoded
                st.session_state.preprocessing_steps.append(step_msg)
                st.toast(f"🏷️ {encoding_method} applied!", icon="✅")
                st.markdown("---")
                st.rerun()
        
        except Exception as e:
            st.error(f"Error encoding categorical variables: {e}")

# ------ Render feature scaling options ------
def _render_scaling_options(df):
    st.markdown("#### Feature Scaling")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target column if selected
    if st.session_state.target and st.session_state.target in numeric_cols:
        numeric_cols.remove(st.session_state.target)
    
    if not numeric_cols:
        st.info("No numeric columns found for scaling")
        return
    
    st.markdown(f"**Numeric columns (excluding target):** {', '.join(numeric_cols)}")
    scaling_method = st.selectbox(
        "Select scaling method",
        ["StandardScaler", "MinMaxScaler", "RobustScaler"],
        help="Choose how to scale numeric features"
    )
    
    if st.button("Apply Scaling", type="primary"):
        try:
            with st.spinner("⏳ Scaling features... Please wait."):
                from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                
                df_scaled = df.copy()
                
                # Calculate statistics before scaling
                original_stats = df[numeric_cols].describe()
                
                # Select scaler
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                    scaler_description = "Standardizes features to have mean=0 and std=1"
                elif scaling_method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                    scaler_description = "Scales features to a fixed range [0, 1]"
                else:
                    scaler = RobustScaler()
                    scaler_description = "Uses median and IQR, robust to outliers"
                # Apply scaling
                df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
                
                # Calculate statistics after scaling                scaled_stats = df_scaled[numeric_cols].describe()
                
                # Ensure Arrow compatibility before storing
                from preprocessing.preprocessor import MLPreprocessor
                if 'preprocessor' not in st.session_state:
                    st.session_state.preprocessor = MLPreprocessor()
                df_scaled = st.session_state.preprocessor.ensure_arrow_compatibility(df_scaled)
                
                st.session_state.dataset = df_scaled
                step_msg = f"Applied {scaling_method} to {len(numeric_cols)} numeric features"
                st.session_state.preprocessing_steps.append(step_msg)
                
                st.success(f"✅ {scaling_method} applied to {len(numeric_cols)} features!")
                st.toast(f"⚖️ Features scaled using {scaling_method}!", icon="✅")
                
                # Update summary (no UI success message shown)
                st.markdown("---")
                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error scaling features: {e}")
            st.exception(e)

# ------ Render feature selection options ------
def _render_feature_selection_options(df):
    st.markdown("#### Feature Selection")
    
    if not st.session_state.target:
        st.info("Please select a target variable first")
        return
    
    feature_cols = [col for col in df.columns if col != st.session_state.target]
    if len(feature_cols) < 2:
        st.info("Need at least 2 features for selection")
        return
    
    st.markdown(f"**Available features:** {len(feature_cols)} columns")
    
    selection_method = st.selectbox(
        "Select feature selection method",
        ["Manual Selection", "Correlation Threshold", "Feature Importance", "Outlier Detection"],
        help="Choose how to select features"
    )
    
    if selection_method == "Manual Selection":
        selected_features = st.multiselect(
            "Select features to keep",
            feature_cols,
            default=feature_cols[:10]  # Default to first 10
        )
        
        if selected_features and st.button("Apply Manual Selection", type="primary"):
            with st.spinner("⏳ Applying feature selection... Please wait."):
                df_selected = df[selected_features + [st.session_state.target]]
                st.session_state.dataset = df_selected
                original_features = len(feature_cols)
                
                step_msg = f"Manual feature selection: {len(selected_features)} out of {original_features} features"
                st.session_state.preprocessing_steps.append(step_msg)
                
                # Enhanced success message
                st.markdown("---")
                st.success(f"✅ Selected top {len(selected_features)} features based on manual selection.")
                
                # Display selection metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Features", original_features)
                with col2:
                    st.metric("Selected Features", len(selected_features))
                with col3:
                    reduction_pct = ((original_features - len(selected_features)) / original_features * 100)
                    st.metric("Reduction", f"{reduction_pct:.1f}%")
                with col4:
                    st.metric("Target", st.session_state.target)
                
                # Show selected features in expander
                with st.expander("📋 Selected Features", expanded=False):
                    features_df = pd.DataFrame({
                        "Feature": selected_features,
                        "Data Type": [str(df[col].dtype) for col in selected_features]
                    })
                    st.dataframe(features_df, use_container_width=True)
                
                st.markdown("---")
                st.rerun()
    elif selection_method == "Correlation Threshold":
        threshold = st.slider("Correlation threshold", 0.5, 0.95, 0.8, 0.05)
        
        if st.button("Apply Correlation Filtering", type="primary"):
            try:
                numeric_features = df.select_dtypes(include=['number']).columns.tolist()
                if st.session_state.target in numeric_features:
                    numeric_features.remove(st.session_state.target)
                
                if len(numeric_features) > 1:
                    corr_matrix = df[numeric_features].corr().abs()
                    upper_triangle = corr_matrix.where(
                        np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    )
                    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
                    remaining_features = [col for col in feature_cols if col not in to_drop]
                    df_selected = df[remaining_features + [st.session_state.target]]
                    
                    st.session_state.dataset = df_selected
                    step_msg = f"Correlation filtering: removed {len(to_drop)} highly correlated features (threshold={threshold})"
                    st.session_state.preprocessing_steps.append(step_msg)
                    # Enhanced confirmation message with detailed column information
                    st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                    st.success(f"✅ Feature Selection Applied Successfully!")
                    st.markdown(f"""
                    **Selection Details:**
                    - Method: Correlation Threshold Filtering
                    - Threshold: {threshold}
                    - Features removed: {len(to_drop)}
                    - Features remaining: {len(remaining_features)}
                      **Removed features:** {', '.join(to_drop) if to_drop else 'None'}
                    **Remaining features:** {', '.join(remaining_features[:5])}{'...' if len(remaining_features) > 5 else ''}
                    """)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.rerun()
                else:
                    st.warning("Need at least 2 numeric features for correlation filtering")
            
            except Exception as e:
                st.error(f"Error in correlation filtering: {e}")
    
    elif selection_method == "Outlier Detection":
        st.markdown("##### Outlier Detection & Removal")
        st.markdown("Detect and remove outliers using the IQR (Interquartile Range) method.")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if st.session_state.target in numeric_cols:
            numeric_cols.remove(st.session_state.target)
        
        if not numeric_cols:
            st.info("No numeric columns found for outlier detection")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_cols = st.multiselect(
                "Select columns for outlier detection",
                numeric_cols,
                default=numeric_cols,
                help="Choose numeric columns to detect outliers in"
            )
        
        with col2:
            iqr_multiplier = st.slider(
                "IQR Multiplier", 
                min_value=1.0, 
                max_value=3.0, 
                value=1.5, 
                step=0.1,
                help="Higher values are more tolerant to outliers"
            )
        
        if selected_cols and st.button("🧹 Remove Outliers", type="primary"):
            try:
                with st.spinner("Detecting and removing outliers..."):
                    df_cleaned = df.copy()
                    total_outliers = 0
                    
                    for col in selected_cols:
                        Q1 = df_cleaned[col].quantile(0.25)
                        Q3 = df_cleaned[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - iqr_multiplier * IQR
                        upper_bound = Q3 + iqr_multiplier * IQR
                        
                        # Count outliers in this column
                        outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                        col_outliers = outliers_mask.sum()
                        total_outliers += col_outliers
                        
                        # Remove outliers
                        df_cleaned = df_cleaned[~outliers_mask]
                    
                    st.session_state.dataset = df_cleaned
                    step_msg = f"Removed {total_outliers} outliers from {len(selected_cols)} columns using IQR method"
                    st.session_state.preprocessing_steps.append(step_msg)
                    
                    # Show contextual success message
                    st.success(f"✅ Removed {total_outliers} outliers using IQR method.")
                    if total_outliers > 0:
                        st.info("ℹ️ Summary updated.")
                    
                    st.markdown("---")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error in outlier detection: {e}")

# ------ Render preprocessing summary and reset option ------
def _render_preprocessing_summary():
    st.markdown("#### Preprocessing Summary")
    
    if not st.session_state.preprocessing_steps:
        st.info("No preprocessing steps applied yet")
        return
    
    st.markdown("**Applied Steps:**")
    for i, step in enumerate(st.session_state.preprocessing_steps, 1):
        st.markdown(f"{i}. {step}")
    
    # Current dataset info
    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Rows", f"{df.shape[0]:,}")
            st.metric("Missing Values", f"{df.isna().sum().sum():,}")
        
        with col2:
            st.metric("Current Columns", f"{df.shape[1]:,}")
            st.metric("Data Types", f"{df.dtypes.nunique()}")
    
    # Reset preprocessing button
    if st.button("🔄 Reset All Preprocessing", type="secondary"):
        st.session_state.preprocessing_steps = []
        st.success("Preprocessing steps reset")
        st.rerun()

# ------ Render the model training page ------
def render_training_page():
    st.markdown("<h1 class='main-header'>🧠 Model Training</h1>", unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.warning("Please load a dataset first.")
        st.markdown("</div>", unsafe_allow_html=True)
        create_navigation_buttons()
        return
    
    if not st.session_state.target:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.warning("Please select a target variable first.")
        st.markdown("</div>", unsafe_allow_html=True)
        create_navigation_buttons()
        return
    
    df = st.session_state.dataset
    
    # Training configuration
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Model Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        model_name = st.selectbox(
            "Select Model",
            ["Random Forest", "XGBoost", "LightGBM"],
            key="training_model_select"
        )
        st.session_state.model_name = model_name
    
    with config_col2:
        tuning_method = st.selectbox(
            "Hyperparameter Tuning",
            ["None", "Grid Search", "Random Search"],
            help="Choose hyperparameter optimization method"
        )
        st.session_state.tuning_method = tuning_method
    
    with config_col3:
        use_cv = st.checkbox("Use Cross-Validation", value=True)
        st.session_state.use_cv = use_cv
    
    # Model parameters
    if tuning_method == "None":
        st.markdown("#### Model Parameters")
        param_options = get_model_params(model_name)
        model_params = {}
        
        param_cols = st.columns(min(len(param_options), 4))        
        for i, (param_name, param_values) in enumerate(param_options.items()):
            with param_cols[i % len(param_cols)]:
                if isinstance(param_values[0], int):
                    model_params[param_name] = st.selectbox(f"{param_name}", param_values, key=f"param_{param_name}")
                elif isinstance(param_values[0], float):
                    model_params[param_name] = st.selectbox(f"{param_name}", param_values, key=f"param_{param_name}")
                else:
                    model_params[param_name] = st.selectbox(f"{param_name}", param_values, key=f"param_{param_name}")
        
        st.session_state.model_params = model_params
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Training execution
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Training Control")
    train_col1, train_col2 = st.columns(2)
    
    with train_col1:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner("⏳ Training model... Please wait."):
                success = _execute_training(df)
                if success:
                    st.toast("🎉 Training completed successfully!", icon="✅")
    
    with train_col2:
        if st.button("🧹 Clear Training Logs", use_container_width=True):
            st.session_state.training_logs = []
            st.rerun()
    
    # Training logs with enhanced styling
    if st.session_state.training_logs:
        st.markdown("---")
        st.markdown("#### 📜 Training Logs")
        
        # Add log controls
        log_col1, log_col2 = st.columns([3, 1])
        with log_col1:
            show_all_logs = st.checkbox("Show all logs", value=False)
        with log_col2:
            if st.button("📋 Export Logs", help="Copy logs to clipboard"):
                log_text = "\n".join(st.session_state.training_logs)
                st.code(log_text, language="text")
        
        # Display logs in a styled container
        log_container = st.container()
        with log_container:
            logs_to_show = st.session_state.training_logs if show_all_logs else st.session_state.training_logs[-10:]
            
            # Enhanced log styling
            st.markdown("""            <div style="
                background-color: rgba(0,0,0,0.05);
                border-radius: 0.5rem;
                padding: 1rem;
                max-height: 300px;
                overflow-y: auto;
                font-family: monospace;
            ">
            """, unsafe_allow_html=True)
            
            for log in logs_to_show:
                # Color code log entries based on content
                if "✅" in log or "🎉" in log:
                    log_color = "#28a745"  # Green for success
                elif "⚠️" in log or "❌" in log:
                    log_color = "#dc3545"  # Red for warnings/errors
                elif "🚀" in log or "🔍" in log:
                    log_color = "#5bc0be"  # Blue for process start
                else:
                    log_color = "#6c757d"  # Gray for info
                
                st.markdown(f'<p style="color: {log_color}; margin: 0.2rem 0;">{log}</p>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    create_navigation_buttons()

# ------ Execute the model training process ------
def _execute_training(df):
    try:
        add_log_message("🚀 Starting training process...")
        
        # Check for and remove duplicates before training
        add_log_message("🔍 Checking for duplicate rows...")
        df_cleaned, num_duplicates = detect_and_remove_duplicates(df)
        
        if num_duplicates > 0:
            add_log_message(f"🗑️ Removed {num_duplicates} duplicate rows")
            st.session_state.dataset = df_cleaned
            df = df_cleaned
            
            # Display duplicate removal success message
            st.success(f"✅ Detected and removed {num_duplicates} duplicate rows.")
            st.info("ℹ️ Summary updated.")
        else:
            add_log_message("✅ No duplicate rows found")
        
        # Prepare data
        add_log_message("📊 Preparing data...")
        target_col = st.session_state.target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Validate and detect task type automatically
        add_log_message("🔍 Detecting task type...")
        detected_task_type = detect_task_type(y)
        
        # Validate task type consistency
        if st.session_state.task_type != detected_task_type:
            add_log_message(f"⚠️ Task type mismatch detected. Updating from {st.session_state.task_type} to {detected_task_type}")
            st.session_state.task_type = detected_task_type
            st.warning(f"Task type updated to {detected_task_type} based on target variable analysis")
        
        add_log_message(f"✅ Task type confirmed: {st.session_state.task_type}")
        
        # Basic preprocessing for any remaining issues
        add_log_message("🔧 Applying basic preprocessing...")
        
        # Handle any remaining missing values
        for col in X.select_dtypes(include=['number']).columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        for col in X.select_dtypes(include=['object']).columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Handle any remaining categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            add_log_message(f"🏷️ Encoding categorical variables: {', '.join(categorical_cols)}")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Ensure target variable is properly formatted for task type
        if st.session_state.task_type == 'classification':
            # For classification, ensure target is properly encoded
            if y.dtype == 'object' or (y.dtype in ['float64', 'float32'] and not all(y == y.astype(int))):
                add_log_message("🏷️ Encoding target variable for classification...")
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
                st.session_state.target_encoder = le_target
        else:
            # For regression, ensure target is numeric
            if y.dtype == 'object':
                add_log_message("🔢 Converting target variable to numeric for regression...")
                try:
                    y = pd.to_numeric(y, errors='coerce')
                    if y.isna().any():
                        raise ValueError("Target variable contains non-numeric values that cannot be converted")
                except:
                    raise ValueError("Target variable must be numeric for regression tasks")
        
        # Split data
        test_size = 0.2
        random_state = 42
        
        if st.session_state.task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        add_log_message(f"📈 Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        # Store splits in session state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        
        # Create and train model
        model_name = st.session_state.model_name
        add_log_message(f"🤖 Creating {model_name} model...")
        
        if st.session_state.tuning_method == "None":
            # Use specified parameters
            model_params = st.session_state.model_params
            model = create_model(model_name, model_params, st.session_state.task_type)
            
            add_log_message("🏋️ Training model...")
            model.fit(X_train, y_train)
            
            best_model = model
            best_params = model_params
        
        else:
            # Hyperparameter tuning
            add_log_message(f"🎯 Starting {st.session_state.tuning_method}...")
            param_grid = get_model_params(model_name)
            base_model = create_model(model_name, {}, st.session_state.task_type)
            
            scoring = 'accuracy' if st.session_state.task_type == 'classification' else 'neg_mean_squared_error'
            
            if st.session_state.tuning_method == "Grid Search":
                search = GridSearchCV(
                    base_model, param_grid, cv=3, scoring=scoring, n_jobs=-1
                )
            else:  # Random Search
                search = RandomizedSearchCV(
                    base_model, param_grid, cv=3, scoring=scoring, n_jobs=-1, n_iter=20
                )
            
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            add_log_message(f"✅ Best parameters: {best_params}")
        
        # Evaluate model
        add_log_message("📊 Evaluating model performance...")
        
        if st.session_state.use_cv:
            cv_mean, cv_std = evaluate_model(
                best_model, X_train, y_train, 
                st.session_state.task_type, st.session_state.metric
            )
            add_log_message(f"📈 CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        # Test set evaluation
        y_pred = best_model.predict(X_test)
        metrics = get_model_metrics_summary(best_model, X_test, y_test, st.session_state.task_type)
        
        for metric_name, value in metrics.items():
            add_log_message(f"🎯 {metric_name}: {value:.4f}")
        
        # Store results
        st.session_state.best_model = best_model
        st.session_state.best_params = best_params
        st.session_state.training_results = {
            'model_name': model_name,
            'best_params': best_params,
            'metrics': metrics,
            'y_pred': y_pred
        }
        add_log_message("🎉 Training completed successfully!")
        
        # Auto-navigate to evaluation page
        st.session_state.page = 'evaluation'
        st.rerun()
        return True
    
    except Exception as e:
        add_log_message(f"❌ Training failed: {str(e)}")
        st.error(f"Training failed: {e}")
        return False

# ------ Render the model evaluation page ------
def render_evaluation_page():
    st.markdown("<h1 class='main-header'>📈 Model Evaluation</h1>", unsafe_allow_html=True)
    
    if st.session_state.best_model is None:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.warning("⚠️ No trained model found. Please train a model first.")
        st.markdown("</div>", unsafe_allow_html=True)
        create_navigation_buttons()
        return
    
    with st.spinner("⏳ Loading model evaluation results..."):
        # Model performance metrics
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📈 Model Performance Metrics")
        
        if 'training_results' in st.session_state and st.session_state.training_results:
            results = st.session_state.training_results
            metrics = results.get('metrics', {})
            
            # Display enhanced metrics with color coding
            create_metrics_display(metrics)
            
            # Add metrics explanations
            create_metrics_explanations()
            
            st.markdown("---")
            
            # Model details
            st.subheader("🔍 Model Details")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.markdown(f"**🤖 Model Type:** {results.get('model_name', 'Unknown')}")
                st.markdown(f"**🎯 Task Type:** {st.session_state.task_type.title()}")
            
            with details_col2:
                st.markdown(f"**📊 Training Samples:** {st.session_state.X_train.shape[0] if st.session_state.X_train is not None else 'Unknown'}")
                st.markdown(f"**🧪 Test Samples:** {st.session_state.X_test.shape[0] if st.session_state.X_test is not None else 'Unknown'}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    # Enhanced Visualizations
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Advanced Model Analysis")
    viz_tabs = st.tabs([
        "📊 Performance", 
        "📈 Learning Curves", 
        "🔍 Feature Importance", 
        "📋 Predictions"
    ])
    with viz_tabs[0]:
        _render_performance_visualizations()
    
    with viz_tabs[1]:
        _render_learning_curves_analysis()
    
    with viz_tabs[2]:
        _render_feature_importance_analysis()
    
    with viz_tabs[3]:
        _render_prediction_analysis()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model export
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Export Results")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("💾 Download Model", use_container_width=True):
            _export_model()
    
    with export_col2:
        if st.button("📄 Download Results", use_container_width=True):
            _export_results()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    create_navigation_buttons()

# ------ Render prediction analysis tab ------
def _render_performance_visualizations():
    if st.session_state.task_type == 'classification':
        _render_classification_performance()
    else:
        _render_regression_performance()

# ------ Render classification performance visualizations ------
def _render_classification_performance():
    if st.session_state.y_test is None or 'training_results' not in st.session_state:
        st.info("No test results available")
        return
    
    y_true = st.session_state.y_test
    y_pred = st.session_state.training_results.get('y_pred')
    
    if y_pred is None:
        st.info("No predictions available")
        return
    
    # Performance visualizations in tabs
    perf_tabs = st.tabs(["🔢 Confusion Matrix", "📈 ROC Curve", "📊 Classification Report"])
    
    with perf_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Standard Confusion Matrix")
            fig_cm = plot_confusion_matrix_enhanced(y_true, y_pred, normalize=False)
            if fig_cm:
                st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("##### Normalized Confusion Matrix")
            fig_cm_norm = plot_confusion_matrix_enhanced(y_true, y_pred, normalize=True)
            if fig_cm_norm:
                st.plotly_chart(fig_cm_norm, use_container_width=True)
    with perf_tabs[1]:
        st.markdown("##### ROC Curve Analysis")
        unique_classes = np.unique(y_true)
        
        if len(unique_classes) == 2:
            # Binary classification
            fig_roc = plot_roc_curve_multiclass(st.session_state.best_model, st.session_state.X_test, y_true)
            if fig_roc:
                st.plotly_chart(fig_roc, use_container_width=True)
        elif len(unique_classes) > 2:
            # Multiclass classification
            fig_roc = plot_roc_curve_multiclass(st.session_state.best_model, st.session_state.X_test, y_true)
            if fig_roc:
                st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curve requires at least 2 classes")
    
    with perf_tabs[2]:
        st.markdown("##### 📋 Classification Report")
        try:
            # Get class names if available
            unique_classes = sorted(np.unique(y_true))
            class_names = [f"Class {cls}" for cls in unique_classes]
            
            # Get classification report as dictionary for DataFrame formatting
            from sklearn.metrics import classification_report
            report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            
            # Convert to DataFrame for better presentation
            report_df = pd.DataFrame(report_dict).transpose()
            
            # Style the DataFrame with conditional formatting
            styled_df = report_df.style.format({
                'precision': '{:.3f}',
                'recall': '{:.3f}',
                'f1-score': '{:.3f}',
                'support': '{:.0f}'
            }).background_gradient(
                cmap='RdYlGn', 
                subset=['precision', 'recall', 'f1-score']
            ).highlight_max(
                axis=0, 
                color='lightgreen',
                subset=['precision', 'recall', 'f1-score']
            )
            
            # Display the styled DataFrame
            st.dataframe(styled_df, use_container_width=True)
            
            # Add interpretation note
            st.markdown("""
            **📊 Report Interpretation:**
            - **Precision**: Of all predicted positives, how many were actually positive
            - **Recall**: Of all actual positives, how many were correctly predicted
            - **F1-Score**: Harmonic mean of precision and recall
            - **Support**: Number of actual occurrences of each class
            - 🟢 Green highlighting indicates best performance per metric
            """)
            
        except Exception as e:
            st.error(f"Error generating classification report: {e}")
            # Fallback to text report
            try:
                report = get_classification_report(y_true, y_pred, target_names=class_names)
                st.text(report)
            except:
                st.error("Could not generate classification report")

# ------ Render regression performance visualizations ------
def _render_regression_performance():
    if st.session_state.y_test is None or 'training_results' not in st.session_state:
        st.info("No test results available")
        return
    
    y_true = st.session_state.y_test
    y_pred = st.session_state.training_results.get('y_pred')
    
    if y_pred is None:
        st.info("No predictions available")
        return
    
    # Performance visualizations in tabs
    perf_tabs = st.tabs(["📊 Prediction Analysis", "📈 Residual Analysis", "📉 Error Distribution"])
    
    with perf_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Actual vs Predicted")
            fig_pred = plot_regression_results(y_true, y_pred)
            if fig_pred:
                st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            st.markdown("##### Prediction Statistics")
            pred_stats = {
                "Mean Actual": np.mean(y_true),
                "Mean Predicted": np.mean(y_pred),
                "Std Actual": np.std(y_true),
                "Std Predicted": np.std(y_pred),
                "Min Error": np.min(y_true - y_pred),
                "Max Error": np.max(y_true - y_pred)
            }
            
            for stat, value in pred_stats.items():
                st.metric(stat, f"{value:.4f}")
    
    with perf_tabs[1]:
        residuals = y_true - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Residuals vs Predicted")
            fig_residuals = px.scatter(
                x=y_pred, y=residuals,
                title="Residuals vs Predicted Values",
                labels={'x': 'Predicted Values', 'y': 'Residuals'}
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        with col2:
            st.markdown("##### Residuals vs Actual")
            fig_residuals_actual = px.scatter(
                x=y_true, y=residuals,
                title="Residuals vs Actual Values",
                labels={'x': 'Actual Values', 'y': 'Residuals'}
            )
            fig_residuals_actual.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals_actual, use_container_width=True)
    
    with perf_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Error Distribution")
            fig_error_dist = px.histogram(
                x=np.abs(y_true - y_pred),
                title="Absolute Error Distribution",
                labels={'x': 'Absolute Error', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_error_dist, use_container_width=True)
        
        with col2:
            st.markdown("##### Q-Q Plot of Residuals")
            from scipy import stats
            
            fig_qq = go.Figure()
            
            # Generate theoretical quantiles
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            
            fig_qq.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode='markers',
                name='Residuals'
            ))
            
            # Add reference line
            min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
            max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
            fig_qq.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Distribution',
                line=dict(dash='dash', color='red')
            ))
            
            fig_qq.update_layout(
                title="Q-Q Plot of Residuals",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )
            
            st.plotly_chart(fig_qq, use_container_width=True)

# ------ Render learning curves analysis ------
def _render_learning_curves_analysis():
    if st.session_state.best_model is None or st.session_state.X_train is None:
        st.info("No trained model available for learning curve analysis")
        return
    # Create tabs for learning curves only
    st.markdown("#### Learning Curves Analysis")
    st.markdown("""
    Learning curves show how model performance changes with training set size, 
    helping to diagnose bias and variance issues.
    """)
    
    with st.expander("📊 Generate Learning Curves", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            cv_folds = st.slider("Cross-validation folds", 3, 10, 5, help="Number of CV folds for learning curves")
        
        with col2:
            if st.button("🔄 Generate Learning Curves", type="primary"):
                with st.spinner("Generating learning curves..."):
                    try:
                        # Combine training and test data for learning curve analysis
                        X_combined = pd.concat([st.session_state.X_train, st.session_state.X_test])
                        y_combined = pd.concat([st.session_state.y_train, st.session_state.y_test])
                        
                        fig = plot_learning_curves(
                            st.session_state.best_model,
                            X_combined,
                            y_combined,
                            st.session_state.task_type,
                            cv_folds
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Learning curve interpretation
                            st.markdown("##### 📋 Learning Curve Interpretation")
                            st.markdown("""
                            - **High Bias (Underfitting)**: Both training and validation scores are low and converge to a low value
                            - **High Variance (Overfitting)**: Large gap between training and validation scores
                            - **Good Fit**: Training and validation scores converge to a high value with small gap
                            - **More Data Needed**: Validation score is still improving with more training samples
                            """)
                        else:
                            st.error("Failed to generate learning curves")
                    except Exception as e:
                        st.error(f"Error generating learning curves: {e}")

# ------ Render feature importance analysis tab ------
def _render_feature_importance_analysis():
    if st.session_state.best_model is None or st.session_state.X_train is None:
        st.info("No trained model available for feature importance analysis")
        return
    
    st.markdown("#### Feature Importance Analysis")
    
    # Check if model has built-in feature importance
    if hasattr(st.session_state.best_model, 'feature_importances_'):
        feature_names = list(st.session_state.X_train.columns)
        fig = plot_feature_importance(st.session_state.best_model, feature_names)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Top features summary
            importances = st.session_state.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            st.markdown("##### Top 10 Most Important Features")
            st.dataframe(feature_importance_df.head(10), use_container_width=True)
        else:
            st.error("Failed to generate feature importance plot")
    else:
        st.info("This model doesn't provide built-in feature importance scores")

# ------ Render prediction analysis tab ------
def _render_prediction_analysis():
    if st.session_state.y_test is None or 'training_results' not in st.session_state:
        st.info("No test predictions available")
        return
    
    y_true = st.session_state.y_test
    y_pred = st.session_state.training_results.get('y_pred')
    
    if y_pred is None:
        st.info("No predictions available")
        return
    
    st.markdown("#### Prediction Analysis")
      # Create prediction comparison table
    comparison_df = pd.DataFrame({
        'Actual': y_true[:100],  # Show first 100 for performance
        'Predicted': y_pred[:100],
    })
    
    if st.session_state.task_type == 'regression':
        comparison_df['Error'] = comparison_df['Actual'] - comparison_df['Predicted']
        comparison_df['Absolute Error'] = np.abs(comparison_df['Error'])
    else:
        comparison_df['Correct'] = comparison_df['Actual'] == comparison_df['Predicted']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Prediction Sample (First 100)")
        st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        st.markdown("##### Prediction Statistics")
        if st.session_state.task_type == 'regression':
            stats_df = pd.DataFrame({
                'Metric': ['Mean Error', 'Mean Absolute Error', 'Max Error', 'Min Error'],
                'Value': [
                    f"{comparison_df['Error'].mean():.4f}",
                    f"{comparison_df['Absolute Error'].mean():.4f}",
                    f"{comparison_df['Error'].max():.4f}",
                    f"{comparison_df['Error'].min():.4f}"
                ]
            })
        else:
            accuracy = (comparison_df['Correct']).mean()
            stats_df = pd.DataFrame({
                'Metric': ['Accuracy (Sample)', 'Correct Predictions', 'Total Predictions'],
                'Value': [f"{accuracy:.3f}", str(comparison_df['Correct'].sum()), str(len(comparison_df))]
            })
        
        st.dataframe(stats_df, use_container_width=True)

# ------ Export trained model ------
def _export_model():
    try:
        import pickle
        import io
        
        model_data = {
            'model'         : st.session_state.best_model,
            'params'        : st.session_state.best_params,
            'feature_names' : list(st.session_state.X_train.columns) if st.session_state.X_train is not None else [],
            'task_type'     : st.session_state.task_type,
            'target_column' : st.session_state.target
        }
        
        # Serialize model
        buffer = io.BytesIO()
        pickle.dump(model_data, buffer)
        buffer.seek(0)
        
        st.download_button(
            label     = "Download model.pkl",
            data      = buffer.getvalue(),
            file_name = f"{st.session_state.model_name}_model.pkl",
            mime      = "application/octet-stream"
        )
        
        st.success("✅ Model exported successfully!")
    
    except Exception as e:
        st.error(f"Error exporting model: {e}")

# ------ Export results to JSON ------
def _export_results():
    try:
        if 'training_results' not in st.session_state:
            st.warning("No results to export")
            return
        
        results = st.session_state.training_results
        
        # Create results summary
        results_summary = {
            'model_name'          : results.get('model_name', ''),
            'task_type'           : st.session_state.task_type,
            'best_parameters'     : results.get('best_params', {}),
            'performance_metrics' : results.get('metrics', {}),
            'preprocessing_steps' : st.session_state.preprocessing_steps,
            'training_logs'       : st.session_state.training_logs[-20:]  # Last 20 logs
        }
        
        # Convert to JSON
        results_json = json.dumps(results_summary, indent=2, default=str)
        
        st.download_button(
            label     = "Download results.json",
            data      = results_json,
            file_name = "training_results.json",
            mime      = "application/json"
        )
        
        st.success("✅ Results exported successfully!")
    
    except Exception as e:
        st.error(f"Error exporting results: {e}")
