import streamlit as st

# ------ Initialize all session state variables ------
def initialize_session_state():
    from preprocessing.preprocessor import MLPreprocessor
    
    default_values = {
        'page'               : 'home',
        'dataset'            : None,
        'target'             : None,
        'task_type'          : None,
        'training_logs'      : [],
        'best_model'         : None,
        'best_params'        : {},
        'training_results'   : {},
        'X_train'            : None,
        'X_test'             : None,
        'y_train'            : None,
        'y_test'             : None,
        'feature_importance' : None,
        'confusion_matrix'   : None,        
        'preprocessing_steps': [],
        'model_name'         : None,
        'model_params'       : {},
        'metric'             : 'accuracy',
        'tuning_method'      : 'None',
        'use_cv'             : True,
        'preprocessor'       : MLPreprocessor()
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ------ Apply custom CSS for styling ------
def apply_custom_css():
    st.markdown("""    <style>
        /* COMPLETELY HIDE STREAMLIT SIDEBAR AND DEFAULT NAVIGATION */
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        .css-1d391kg {
            display: none !important;
        }
        .css-1lcbmhc {
            display: none !important;
        }
        div[data-testid="stSidebarNav"] {
            display: none !important;
        }
        .css-1629p8f {
            display: none !important;
        }
        .css-6qob1r {
            display: none !important;
        }
        /* Hide any page navigation elements */
        nav[role="navigation"] {
            display: none !important;
        }
        .main .block-container {
            padding-top: 1rem !important;
        }
        
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
        .metric-card {
            background: rgba(58, 80, 107, 0.05);
            padding: 1.5rem;
            border-radius: 12px;
            color: #e0e1dd;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #5bc0be;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #b2bec3;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .card {
            background: rgba(58, 80, 107, 0.05);
            padding: 2rem;
            border-radius: 12px;
            color: #e0e1dd;
            margin-bottom: 2rem;
        }
        .slide-nav {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
            padding: 10px;
            background: rgba(0,0,0,0.03);
            border-radius: 12px;
            gap: 10px;
        }
        
        .nav-step {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 10px 15px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 120px;
            text-align: center;
        }
        
        .nav-step.active {
            background: linear-gradient(135deg, #4ECDC4, #45B7D1);
            color: white;
        }
        
        .nav-step.completed {
            background: rgba(76, 175, 80, 0.15);
        }
        
        .nav-step:hover:not(.disabled) {
            background: rgba(78, 205, 196, 0.1);
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
            height: 2px;
            background: linear-gradient(90deg, #4ECDC4, #45B7D1);
            border-radius: 1px;
            margin: 10px 0;
        }
        
        .training-log {
            background: #232931;
            color: #e0e1dd;
            padding: 1rem;
            border-radius: 10px;            font-family: 'Courier New', monospace;
            max-height: 300px;
            overflow-y: auto;
            margin: 1rem 0;
        }          .info-box {
            background: rgba(91, 192, 190, 0.05);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .warning-box {
            background: rgba(255, 193, 7, 0.05);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }          .success-box {
            background: rgba(40, 167, 69, 0.05);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# ------ Create navigation slides with progress bar ------
def create_navigation_slides():
    pages = [
        ("🏠", "Home", "home"),
        ("📊", "Data Exploration", "data_exploration"),
        ("🔧", "Preprocessing", "preprocessing"),
        ("🧠", "Model Training", "training"),
        ("📈", "Evaluation", "evaluation")
    ]
    
    current_page_index = next((i for i, (_, _, page_id) in enumerate(pages) if page_id == st.session_state.page), 0)
    
    # Create navigation HTML
    nav_html = '<div class="slide-nav">'
    
    for i, (icon, title, page_id) in enumerate(pages):
        is_active = page_id == st.session_state.page
        is_completed = i < current_page_index
        
        class_name = "nav-step"
        if is_active:
            class_name += " active"
        elif is_completed:
            class_name += " completed"
        
        nav_html += f'''
        <div class="{class_name}">
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
    
    # Create navigation buttons
    cols = st.columns(len(pages))
    for i, (icon, title, page_id) in enumerate(pages):
        with cols[i]:
            if st.button(f"{icon} {title}", key=f"nav_btn_{page_id}", 
                        help=f"Navigate to {title}", 
                        use_container_width=True):
                st.session_state.page = page_id
                st.rerun()

# ------ Create quick settings in a collapsible section ------
def create_quick_settings():
    import pandas as pd
    from utils.helpers import load_sample_dataset
    
    with st.expander("⚙️ Quick Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Dataset**")
            dataset_option = st.radio(
                "Choose dataset source", 
                ["Upload CSV", "Sample Dataset"], 
                key="quick_dataset"
            )
            
            if dataset_option == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "Upload CSV file", 
                    type=["csv"], 
                    key="quick_csv_uploader"
                )
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
                    ["Iris (Classification)", "Wine (Classification)", "Breast Cancer (Classification)", "Diabetes (Regression)", "California Housing (Regression)"],
                    key="quick_sample_select"
                )
                if st.button("Load Sample Dataset", key="quick_load_sample"):
                    df, task_type = load_sample_dataset(sample_dataset)
                    if df is not None:
                        st.session_state.dataset = df
                        st.session_state.task_type = task_type
                        st.success(f"Sample dataset loaded: {sample_dataset}")
        
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

# ------ Create Previous/Next navigation buttons at the bottom of each page ------
def create_navigation_buttons():
    pages = [
        ("🏠", "Home", "home"),
        ("📊", "Data Exploration", "data_exploration"),
        ("🔧", "Preprocessing", "preprocessing"),
        ("🧠", "Model Training", "training"),
        ("📈", "Evaluation", "evaluation")
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

# ------ Display dataset overview metrics ------
def display_dataset_overview(df):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[0]:,}</div>
                <div class="metric-label">Rows</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1]:,}</div>
                <div class="metric-label">Columns</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.isna().sum().sum():,}</div>
                <div class="metric-label">Missing Values</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.duplicated().sum():,}</div>
                <div class="metric-label">Duplicates</div>
            </div>
        """, unsafe_allow_html=True)