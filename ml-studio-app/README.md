# 🚀 ML Studio - Professional Machine Learning Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)

> **A comprehensive, production-ready machine learning studio built with Streamlit for end-to-end ML workflows**

## 🎯 **Project Overview**

ML Studio is a professional-grade web application that provides a complete machine learning pipeline from data exploration to model deployment. Built with modern software engineering practices, it offers an intuitive interface for both technical and non-technical users to perform sophisticated ML tasks.

### ✨ **Key Features**

- 📊 **Interactive Data Exploration** - Comprehensive dataset analysis and visualization
- 🔧 **Advanced Preprocessing** - Missing data handling, scaling, encoding, feature engineering
- 🤖 **Model Training** - Support for Random Forest, XGBoost, and LightGBM
- 📈 **Model Evaluation** - Detailed metrics, confusion matrices, ROC curves
- 🎯 **Automated Task Detection** - Automatic classification vs regression detection
- 📱 **Professional UI** - Modern, responsive interface built with Streamlit

---

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Abdelrhman941/ml-preprocessing-guide.git
   cd ml-preprocessing-guide
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run ml_studio_app.py
   ```
   Or use the automated deployment script:
   ```bash
   python deploy.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

---

## 🛠️ **Technology Stack**

### **Core Technologies**
- **Frontend**: Streamlit 1.25+ (Interactive web interface)
- **Backend**: Python 3.8+ (Core logic and ML pipeline)
- **Visualization**: Plotly, Matplotlib (Interactive charts and plots)

### **Machine Learning Libraries**
- **Models**: scikit-learn, XGBoost, LightGBM
- **Preprocessing**: pandas, numpy, imbalanced-learn
- **Evaluation**: scipy (statistical analysis)

### **Data Processing**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: ML algorithms and preprocessing

---

## 📊 **Features Deep Dive**

### **1. Data Exploration**
- **Dataset Overview**: Shape, types, missing values, duplicates
- **Statistical Analysis**: Comprehensive descriptive statistics
- **Visualizations**: Distribution plots, correlation matrices, missing value patterns
- **Target Analysis**: Classification/regression-specific insights

### **2. Preprocessing Pipeline**
- **Missing Data Handling**: Multiple imputation strategies
- **Feature Engineering**: Datetime, mathematical, and binning features
- **Encoding**: Label and one-hot encoding for categorical variables
- **Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Feature Selection**: Correlation-based and importance-based selection

### **3. Model Training**
- **Algorithms**: Random Forest, XGBoost, LightGBM
- **Hyperparameter Tuning**: Grid Search and Random Search
- **Cross-Validation**: K-fold validation with detailed metrics
- **Real-time Logging**: Comprehensive training progress tracking

### **4. Model Evaluation**
- **Classification Metrics**: Accuracy, F1-score, ROC-AUC, Precision, Recall
- **Regression Metrics**: MSE, MAE, R², RMSE
- **Visualizations**: Confusion matrices, ROC curves, residual plots
- **Feature Importance**: Model interpretability and feature ranking

---

## 📁 **Project Structure**

```
ml-studio/
├── assets/dataset/          # Sample datasets (heart.csv)
├── config/                  # Configuration files and settings
├── guide/                   # Documentation and user guides
│   ├── USER_GUIDE.md        # End-user documentation
│   └── EXPLAIN_PROCESS.md   # Process explanation
├── pages/                   # Streamlit page components
├── preprocessing/           # Data preprocessing modules
├── training/                # ML model training modules
├── utils/                   # Utility functions and helpers
├── tests/                   # Test files and validation
├── ml_studio_app.py         # Main application entry point
├── deploy.py                # Automated deployment script
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── LICENSE                  # MIT License
```

---

## 🧪 **Development**

### **Running Tests**
```bash
python run_tests.py                    # Complete test suite
python tests/validate_setup.py         # Quick validation
python tests/test_functionality.py     # Feature tests
python tests/test_app.py               # Import tests
```

---

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

### **Development Setup**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following our coding standards
4. Test your changes: `python run_tests.py`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

---

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 **Links**

- **User Guide**: [guide/USER_GUIDE.md](guide/USER_GUIDE.md) - For end users

---

## 📧 **Support**

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Bug Reports**: Use GitHub Issues for bug reports
- 📖 **Documentation**: Check guide/USER_GUIDE.md for usage instructions

---

**By Abdelrhman Ezzat 🫡**
