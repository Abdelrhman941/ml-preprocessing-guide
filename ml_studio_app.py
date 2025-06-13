import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.navigation import initialize_session_state, apply_custom_css, create_navigation_slides, create_quick_settings
from components.app_pages import (
    render_home_page, 
    render_data_exploration_page, 
    render_preprocessing_page,
    render_training_page, 
    render_evaluation_page
)

# Configure Streamlit page
st.set_page_config(
    page_title  = "ML Studio",
    page_icon   = "🚀",
    layout      = "wide",
    initial_sidebar_state = "collapsed"  # Hide default sidebar
)

# Force hide sidebar completely
st.markdown("""
<style>
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Clean up Streamlit default styling */
    .stAlert {
        border: none !important;
        box-shadow: none !important;
    }
    
    .stSelectbox > div > div {
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: none !important;
    }
    
    .stButton > button {
        border: none !important;
        box-shadow: none !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border: none !important;
        background-color: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
    }
    
    /* Remove shadows from cards and containers */
    .element-container {
        box-shadow: none !important;
    }
    
    .stExpander {
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state and apply styling
    initialize_session_state()
    apply_custom_css()
    
    # Create navigation and quick settings
    create_navigation_slides()
    create_quick_settings()
    
    # Route to appropriate page based on session state
    page_routes = {
        'home'              : render_home_page,
        'data_exploration'  : render_data_exploration_page,
        'preprocessing'     : render_preprocessing_page,
        'training'          : render_training_page,
        'evaluation'        : render_evaluation_page
    }
    
    # Get current page from session state
    current_page = st.session_state.get('page', 'home')
    
    # Render the selected page
    if current_page in page_routes:
        page_routes[current_page]()
    else:
        # Default to home page if invalid page
        st.session_state.page = 'home'
        render_home_page()
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; opacity: 0.7;">
        <p>🚀 ML Studio v2.0.0 | Built with ❤️ using Streamlit | 
        <a href="https://github.com/Abdelrhman941/ml-preprocessing-guide.git" target="_blank">GitHub</a> 
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()