# 🚀 **ML Studio v2.0 - Complete Machine Learning Pipeline**

> **MAJOR UPDATE**: Fixed critical training errors and enhanced with comprehensive evaluation metrics!

## ✨ **Latest Improvements (v2.0)**

### 🔧 **Critical Fixes**
- ✅ **Fixed Training Error**: Resolved "Unknown label type: continuous" error with automatic task detection
- ✅ **Enhanced Task Detection**: Automatic classification vs regression detection based on target analysis
- ✅ **Robust Validation**: Comprehensive input validation and error handling

### 📊 **Enhanced Evaluation Metrics**

#### Classification Metrics:
- ✅ Accuracy Score with multi-class support
- ✅ F1 Score (weighted average for multi-class)
- ✅ Precision & Recall scores
- ✅ ROC-AUC with proper label binarization
- ✅ Detailed Classification Report
- ✅ Enhanced Confusion Matrix (normalized & standard)
- ✅ Multi-class ROC Curves (One-vs-Rest)

#### Regression Metrics:
- ✅ Mean Squared Error (MSE)
- ✅ Mean Absolute Error (MAE) 
- ✅ R² Score
- ✅ Root Mean Squared Error (RMSE)
- ✅ Comprehensive Residual Analysis
- ✅ Q-Q Plot for normality testing
- ✅ Error distribution visualization

### 🎯 **Advanced Features**
- ✅ **Validation Curves**: Interactive hyperparameter tuning visualization
- ✅ **Learning Curves**: Bias-variance analysis with interpretation guides
- ✅ **Enhanced Preprocessing**: Comprehensive data balancing and scaling
- ✅ **Professional Visualizations**: Seaborn/Matplotlib/Plotly integration

---

# ✅ **Complete Machine Learning Preprocessing Guide**

<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 1. Data Understanding & Inspection </summary>

**Purpose**: Get familiar with your dataset structure and identify potential issues early.

**Key Steps**:
- Load the dataset using appropriate libraries (`pandas`, `numpy`)
- Understand the structure: `.shape`, `.info()`, `.describe()`
- Visual inspection: `.head()`, `.tail()`, `.sample()`
- Identify data types (numerical, categorical, datetime, text, etc.)
- Check for duplicate records

**Best Practices**:
- 👉 Always start with data profiling
- 👉 Document your findings
- 👉 Look for patterns in missing data
- 👉 Check data consistency across columns

**Code Example**:
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('dataset.csv')

# Basic inspection
print(f"Dataset shape    : {df.shape}")
print(f"Dataset info     : {df.info()}")
print(f"Basic statistics :\n{df.describe()}")
print(f"Random sample    :\n{df.sample(5)}\n")
print(f"Missing values   :\n{df.isnull().sum()}")
print(f"Duplicated rows  : {df.duplicated().sum()}")
display(df.head())
display(df.tail())
display(df.sample(5))     # Random sample
```

</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 2. Handling Missing Data </summary>

**Purpose**: Deal with incomplete data that can negatively impact model performance.

**Detection Methods**:
- `.isnull().sum()` - Count missing values
- `.isnull().sum().sum()` - Total missing values
- Visualize missing patterns with `missingno` library

**Strategies**:
**1️⃣ Deletion Approach**:
- **Drop rows**: When missing data is random and dataset is large ⚠️
- **Drop columns**: When >70% values are missing ⚠️

**2️⃣ Imputation Approach**:
- **Numerical Data**:
  - Mean (for normal distribution)
  - Median (for skewed data, robust to outliers)
  - Mode (for categorical-like numerical data)
- **Categorical Data**:
  - Mode (most frequent value)
  - Create "Unknown" category
- **Advanced Methods**:
  - KNN Imputation
  - Iterative Imputation
  - Forward/Backward fill (for time series)

**Code Example**:
```python
from sklearn.impute import SimpleImputer, KNNImputer
import missingno as msno

# visualize missing values
msno.matrix(df)

# Simple imputation
imputer    = SimpleImputer(strategy='median')
df_numeric = imputer.fit_transform(df.select_dtypes(include=[np.number]))

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = knn_imputer.fit_transform(df_numeric)
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 3. Handling Outliers </summary>

**Purpose**: Identify and handle extreme values that can skew model performance.

**Detection Methods**:
**1️⃣ Visual Methods**:
- Box plots: `sns.boxplot()`
- Scatter plots: `plt.scatter()`
- Histograms: `df.hist()`

**2️⃣ Statistical Methods**:
- **IQR Method**: Values beyond $Q1-1.5×IQR$ or $Q3+1.5×IQR$
- **Z-Score**: Values with $|z-score| > 3$
- **Modified Z-Score**: Using median absolute deviation

**3️⃣ Treatment Options**:
- **Remove**: Delete outlier records
- **Cap/Floor (Winsorization)**: Set to percentile limits
- **Transform**: Log, square root, Box-Cox transformation
- **Binning**: Convert to categorical ranges

**Code Example**:
```python
# IQR method
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_clean = df[(df['column'] >= lower_bound) & (df['column'] <= upper_bound)]

# Winsorization
df['column'] = np.clip(df['column'], lower_bound, upper_bound)

from scipy import stats
z_scores = np.abs(stats.zscore(df[numeric_columns]))
df = df[(z_scores < 3).all(axis=1)]
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 4. Data Type Conversion </summary>

**Purpose**: Ensure data types are appropriate for analysis and modeling.

**Common Conversions**:
- Convert `object` to `category` for categorical data (saves memory)
- Convert `strings` to `datetime` for temporal data
- Convert `categorical text labels` to `numerical codes`
- Convert `boolean strings` to `actual boolean type`

**Benefits**:
- ✅ Improved memory efficiency
- ✅ Better performance in operations
- ✅ Enables appropriate statistical operations

**Code Example**:
```python
# Convert to category 
df['category_col'] = df['category_col'].astype('category')

# Convert to datetime
df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce')

# Convert boolean strings
df['bool_col'] = df['bool_col'].map({'True': True, 'False': False})

# Optimize numeric types
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 5. Encoding Categorical Variables </summary>

**Purpose**: Convert categorical data into numerical format for machine learning algorithms.

**Encoding Methods**:
**1️⃣ Label Encoding**:
```md
- Best for : Ordinal features (with natural order)
- Creates  : Single column with integer values
- Example  : Education level (High School=0, Bachelor=1, Master=2, PhD=3)
```
**2️⃣ One-Hot Encoding**:
```md
- Best for : Nominal features (no natural order)
- Creates  : Multiple binary columns
- Example  : Color (Red, Blue, Green) → 3 binary columns
```
**3️⃣ Target/Mean Encoding**:
```md
- Best for : High cardinality categorical features
- Risk     : Data leakage if not done properly
- Use with : Cross-validation and regularization
```
```bash
pip install category_encoders
```

**4️⃣ Binary Encoding**:
```md
- Best for: High cardinality features (more efficient than one-hot)
- Creates: Log2(n) binary columns
```
**Code Example**:
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import pandas as pd

# Label Encoding
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')

# Or using sklearn
ohe = OneHotEncoder(sparse=False, drop='first')
encoded_features = ohe.fit_transform(df[['color']])

# Target Encoding
from category_
encoder = ce.TargetEncoder()
df['encoded'] = encoder.fit_transform(df['feature'], df['target'])
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 6. Feature Scaling </summary>

**Purpose**: Normalize feature ranges to prevent algorithms from being biased toward features with larger scales.

**When Needed**:
```md
- Distance-based algorithms : KNN, K-Means, SVM
- Gradient-based algorithms : Neural Networks, Logistic Regression
- Regularized algorithms    : Ridge, Lasso, Elastic Net
```
**When NOT Needed**:
```md
- Tree-based algorithms: Random Forest, Decision Trees, XGBoost
```
**Scaling Methods**:
**1️⃣ MinMaxScaler**:
- Range: $[0, 1]$
- Formula: $(x - min) / (max - min)$
- Best for: Bounded data, when you know min/max

**2️⃣ StandardScaler (Z-score)**:
- Range: Mean=0, Std=1
- Formula: $(x - mean) / std$
- Best for: Normally distributed data

**3️⃣ RobustScaler**:
- Uses: Median and IQR instead of mean and std
- Best for: Data with outliers

> ### ⚠️ when you use **MinMaxScaler** or **StandardScaler**, you should use `fit` on train data and use `transform` on test data.

**Code Example**:
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# MinMax Scaling
scaler    = MinMaxScaler()
df_scaled = scaler.fit_transform(df[numeric_columns])

# Standard Scaling
std_scaler = StandardScaler()
df_std     = std_scaler.fit_transform(df[numeric_columns])

# Robust Scaling
robust_scaler = RobustScaler()
df_robust     = robust_scaler.fit_transform(df[numeric_columns])

## it's better to use fit on train data and transform on test data
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 7. Feature Engineering </summary>

**Purpose**: Create new features from existing ones to improve model performance.

**Common Techniques**:
**1️⃣ Mathematical Operations**:
- Ratios: income/expense, price/sqft
- Differences: current_price - previous_price
- Products: length × width for area

**2️⃣ Date/Time Features**:
- Extract: year, month, day, hour, day_of_week
- Create: is_weekend, is_holiday, days_since_event

**3️⃣ Text Features**:
- Length: character count, word count
- Patterns: email domains, phone area codes

**4️⃣ Binning/Discretization**:
- Age groups: 0-18, 19-35, 36-50, 50+
- Income brackets: Low, Medium, High

**5️⃣ Polynomial Features**:
- $x², x³, x₁×x₂$ (interaction terms)

> ### 👍 You can also use `Featuretools` for automatic feature engineering

**Code Example**:
```python
from sklearn.preprocessing import PolynomialFeatures

# Date feature engineering
df['year']       = df['date'].dt.year
df['month']      = df['date'].dt.month
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# Mathematical operations
df['bmi'] = df['weight'] / (df['height'] ** 2)
df['price_per_sqft'] = df['price'] / df['area']

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], 
                        labels=['Child', 'Young', 'Adult', 'Senior'])

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 8. Feature Selection </summary>

**Purpose**: Select the most relevant features to improve model performance and reduce overfitting.

**Benefits**:
- ✅ Reduces overfitting
- ✅ Improves model interpretability
- ✅ Decreases training time
- ✅ Reduces storage requirements

**Selection Methods**:
**1️⃣ Filter Methods** (Statistical):
- **Correlation Matrix**: Remove highly correlated features (>0.95)
- **Chi-square Test**: For categorical features vs categorical target
- **ANOVA F-test**: For numerical features vs categorical target
- **Mutual Information**: Measures dependency between features and target

**2️⃣ Wrapper Methods**:
- **Recursive Feature Elimination (RFE)**: Iteratively remove features
- **Forward/Backward Selection**: Add/remove features stepwise

**3️⃣ Embedded Methods**:
- **L1 Regularization (Lasso)**: Automatically selects features
- **Tree-based Feature Importance**: From Random Forest, XGBoost

> ### you can also use `SelectFromModel` with **Lasso**

**Code Example**:
```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)

# Remove highly correlated features
high_corr = np.where(np.abs(corr_matrix) > 0.95)
high_corr_features = [corr_matrix.columns[x] for x in high_corr[0]]

# Chi-square for categorical features
chi2_selector = SelectKBest(chi2, k=10)
chi2_features = chi2_selector.fit_transform(X_categorical, y)

# RFE with Random Forest
rf  = RandomForestClassifier()
rfe = RFE(rf, n_features_to_select=10)
rfe_features = rfe.fit_transform(X, y)

# selecting features with L1 regularization
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.01)
lasso_selector = SelectFromModel(lasso)
lasso_features = lasso_selector.fit_transform(X, y)
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 9. Text Preprocessing (if applicable) </summary>

**Purpose**: Clean and prepare text data for NLP and machine learning tasks.

**Common Steps**:
**1️⃣ Basic Cleaning**:
- Lowercasing: Convert all text to lowercase
- Remove punctuation and special characters
- Remove extra whitespace and newlines
- Handle encoding issues

**2️⃣ Tokenization**:
- Split text into individual words/tokens
- Handle contractions (don't → do not)

**3️⃣ Stopwords Removal**:
- Remove common words (the, and, or, etc.)
- Language-specific stopword lists

**4️⃣ Normalization**:
- **Stemming**: Reduce words to root form (running → run)
- **Lemmatization**: Reduce to dictionary form (better → good)

**5️⃣ Vectorization**:
- **Bag of Words**: Count frequency of words
- **TF-IDF**: Term frequency-inverse document frequency
- **Word Embeddings**: Word2Vec, GloVe, FastText

> 👉 you can also use `Contractions` for fix contractions in text
> 👉 you can also use `spaCy` for more advanced text processing

**Code Example**:
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, use_lemmatizer=False):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and stem
    if use_lemmatizer:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    else:
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Vectorization
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
text_features = tfidf.fit_transform(df['cleaned_text'])

import contractions
text = contractions.fix("Don't do this!")  # Output: "Do not do this!"

import spacy
nlp    = spacy.load('en_core_web_sm')
doc    = nlp("This is an example.")
tokens = [token.lemma_ for token in doc if not token.is_stop]
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 10. Data Splitting </summary>

**Purpose**: Separate data into training and testing sets to evaluate model performance on unseen data.

**Common Split Ratios**:
- 80/20 (Train/Test)
- 70/30 (Train/Test)
- 60/20/20 (Train/Validation/Test)

**Types of Splitting**:
**1️⃣ Random Split**:
- Good for: Independent observations
- Use: `train_test_split()`

**2️⃣ Stratified Split**:
- Good for: Imbalanced datasets
- Maintains: Class distribution in both sets

**3️⃣ Time-based Split**:
- Good for: Time series data
- Rule: Train on past, test on future

**4️⃣ Cross-validation**:
- `K-fold`: Split data into k folds
- `Stratified K-fold`: Maintains class distribution
- `Time series split`: Respects temporal order

> ⚠️ Use `cross_val_score` or `cross_validate` for model evaluation across folds.

**Code Example**:
```python
from sklearn.model_selection import train_test_split, StratifiedKFold, TimeSeriesSplit

# Basic random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stratified split for imbalanced data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Time series split
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```
</details>

---
<details> <summary style="font-size: 20px; font-weight: bold; cursor: pointer;"> 11. Balancing the Dataset (if needed) </summary>

**Purpose**: Address class imbalance that can lead to biased model predictions.

**When to Use**:
- 👉 Imbalanced classification problems
- 👉 Minority class < 10–20% of total data
- 👉 When **accuracy alone** is not sufficient (e.g., fraud detection, medical diagnosis)

**Techniques**:
**1️⃣ Oversampling**:
- **SMOTE** (Synthetic Minority Oversampling Technique): Creates synthetic samples from the minority class.
- **ADASYN**: Adaptive version of SMOTE, focuses more on difficult examples.
- **Random Oversampling**: Duplicates existing samples from the minority class.
- **BorderlineSMOTE**: Oversamples near the decision boundary.

**2️⃣ Undersampling**:
- **Random Undersampling**: Removes samples from the majority class.
- **Tomek Links**: Removes majority samples that are borderline.
- **Edited Nearest Neighbors**: Removes noisy or ambiguous samples.

**3️⃣ Algorithmic Approaches**:
- **Class Weights**: Increase penalty for misclassifying minority class.
- **Cost-sensitive Learning**: Custom loss functions for imbalance.
- **Ensemble Methods**: Use balanced subsets in ensemble models (e.g., BalancedRandomForest).
---
### 🔄 Avoiding Data Leakage with imblearn.pipeline

When oversampling is done **before** splitting data (train/test), it leaks information from the test set into training.
To prevent this, use `imblearn.pipeline.Pipeline` to perform oversampling **inside the cross-validation loop**:

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Build pipeline with oversampling and classifier
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))])

# Cross-validation with safe oversampling inside folds
scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
print("F1 Scores:", scores)
print("Average F1:", scores.mean())
```
---
### Code Example (Basic Sampling)

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from collections import Counter

# Original distribution
print("Original    :", Counter(y))

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)
print("After SMOTE :", Counter(y_sm))

# Random undersampling
undersample = RandomUnderSampler(random_state=42)
X_us, y_us = undersample.fit_resample(X, y)
print("After Undersampling:", Counter(y_us))

# Class weights in classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
```
</details>

---
## 📊 **Evaluation Metrics for Imbalanced Data**
<details> <summary> Click to expand evaluation metrics</summary>

**Beyond Accuracy**:
- **Precision**: $TP / (TP + FP)$ - How many positive predictions were correct?
- **Recall (Sensitivity)**: $TP / (TP + FN)$ - How many actual positives were found?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **PR-AUC**: Area under the precision-recall curve (better for imbalanced data)

> ### ✅ Use `PR-AUC` instead of `ROC-AUC` when the dataset is **highly imbalanced**.

**Code Example**:
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Make predictions
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Comprehensive evaluation
print(classification_report(y_test, y_pred))
print("ROC-AUC          :", roc_auc_score(y_test, y_pred_proba))
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred))
```
</details>

---
## 📈 **Summary Workflow Diagram**
```md
📊 Data Loading & Understanding
           ↓
🔍 Missing Values Detection & Handling
           ↓
📈 Outlier Detection & Treatment
           ↓
🔄 Data Type Conversion & Optimization
           ↓
🏷️ Categorical Variable Encoding
           ↓
⚖️ Feature Scaling & Normalization
           ↓
🛠️ Feature Engineering & Creation
           ↓
🎯 Feature Selection & Reduction
           ↓
📝 Text Preprocessing (if applicable)
           ↓
✂️ Data Splitting (Train/Validation/Test)
           ↓
⚖️ Class Balancing (if needed)
           ↓
🤖 Model Training & Evaluation
           ↓
🔁 Iterative Improvement & Hyperparameter Tuning
```
---
## 🔗 **Useful Libraries & Resources**

**Essential Libraries**:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning preprocessing and algorithms
- `matplotlib` & `seaborn` - Data visualization
- `imbalanced-learn` - Handling imbalanced datasets

**Advanced Libraries**:
- `feature-engine` - Advanced feature engineering
- `category_encoders` - Specialized categorical encoding
- `missingno` - Missing data visualization
- `yellowbrick` - Machine learning visualization

**Remember 🤔**: The best preprocessing pipeline depends on your specific dataset, problem type, and chosen algorithms. Always validate your preprocessing decisions with domain expertise and cross-validation!

<div style="background: linear-gradient(135deg,rgb(37, 127, 201), #8b000e); 
            color: #ffffff; 
            width: 100%; 
            height: 30px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 39px; 
            margin: 15px 0; 
            font-size: 24px; 
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);">
    By Abdelrhman Ezzat 🫡
</div>