# 🚀 **Complete Machine Learning Preprocessing Guide**
*An Interactive Tutorial for Data Scientists*

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 20px 0; 
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
    <h2 style="margin: 0; font-size: 1.8em;">📊 Master Data Preprocessing Like a Pro</h2>
    <p style="margin: 5px 0 0 0; opacity: 0.9;">Click on each section below to explore comprehensive preprocessing techniques</p>
</div>

---

## 📚 **Table of Contents**
<div style="background: rgba(0,123,255,0.1); 
            border-left: 4px solid #007bff; 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 0 10px 10px 0;">

1. **[Data Understanding & Inspection](#1-data-understanding--inspection)** 🔍
2. **[Handling Missing Data](#2-handling-missing-data)** 🕳️
3. **[Handling Outliers](#3-handling-outliers)** 📈
4. **[Data Type Conversion](#4-data-type-conversion)** 🔄
5. **[Encoding Categorical Variables](#5-encoding-categorical-variables)** 🏷️
6. **[Feature Scaling](#6-feature-scaling)** ⚖️
7. **[Feature Engineering](#7-feature-engineering)** 🛠️
8. **[Feature Selection](#8-feature-selection)** 🎯
9. **[Text Preprocessing](#9-text-preprocessing)** 📝
10. **[Data Splitting](#10-data-splitting)** ✂️
11. **[Balancing the Dataset](#11-balancing-the-dataset)** ⚖️

</div>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #28a745, #20c997); color: white; border-radius: 8px; margin: 10px 0;">
    🔍 1. Data Understanding & Inspection
</summary>

<div style="padding: 20px; background: rgba(40, 167, 69, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose** 
Get familiar with your dataset structure and identify potential issues early.

### **🎯 Key Steps**
- Load the dataset using appropriate libraries (`pandas`, `numpy`)
- Understand the structure: `.shape`, `.info()`, `.describe()`
- Visual inspection: `.head()`, `.tail()`, `.sample()`
- Identify data types (numerical, categorical, datetime, text, etc.)
- Check for duplicate records

### **✅ Best Practices**
<div style="background: rgba(255, 193, 7, 0.1); border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0;">

- 👉 **Always start with data profiling**
- 👉 **Document your findings**
- 👉 **Look for patterns in missing data**
- 👉 **Check data consistency across columns**

</div>

### **💻 Code Example**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('dataset.csv')

# Basic inspection
print(f"Dataset shape    : {df.shape}")
print(f"Dataset info     : {df.info()}")
print(f"Basic statistics :\n{df.describe()}")
print(f"Random sample    :\n{df.sample(5)}")
print(f"Missing values   :\n{df.isnull().sum()}")
print(f"Duplicated rows  : {df.duplicated().sum()}")

# Visual inspection
display(df.head())
display(df.tail())
display(df.sample(5))  # Random sample

# Quick visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
df.hist(bins=20, ax=axes.flatten()[:len(df.select_dtypes(include=[np.number]).columns)])
plt.tight_layout()
plt.show()
```

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #dc3545, #fd7e14); color: white; border-radius: 8px; margin: 10px 0;">
    🕳️ 2. Handling Missing Data
</summary>

<div style="padding: 20px; background: rgba(220, 53, 69, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Deal with incomplete data that can negatively impact model performance.

### **🔍 Detection Methods**
```python
# Count missing values
df.isnull().sum()
df.isnull().sum().sum()  # Total missing values

# Visualize missing patterns
import missingno as msno
msno.matrix(df)
msno.heatmap(df)
```

### **🛠️ Strategies**

#### **1️⃣ Deletion Approach**
<div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; padding: 10px; margin: 10px 0;">

- **Drop rows**: When missing data is random and dataset is large ⚠️
- **Drop columns**: When >70% values are missing ⚠️

</div>

#### **2️⃣ Imputation Approach**

**📊 Numerical Data:**
- **Mean** (for normal distribution)
- **Median** (for skewed data, robust to outliers)
- **Mode** (for categorical-like numerical data)

**🏷️ Categorical Data:**
- **Mode** (most frequent value)
- **Create "Unknown" category**

**🚀 Advanced Methods:**
- **KNN Imputation**
- **Iterative Imputation**
- **Forward/Backward fill** (for time series)

### **💻 Implementation**
```python
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import missingno as msno

# Visualize missing values
msno.matrix(df)
plt.show()

# Simple imputation for numerical data
numerical_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Simple imputation for categorical data
categorical_cols = df.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# KNN imputation (more sophisticated)
knn_imputer = KNNImputer(n_neighbors=5)
df_numeric_knn = knn_imputer.fit_transform(df[numerical_cols])

# Iterative imputation
iterative_imputer = IterativeImputer(random_state=42)
df_iterative = iterative_imputer.fit_transform(df[numerical_cols])
```

<div style="background: rgba(13, 202, 240, 0.1); border-left: 4px solid #0dcaf0; padding: 15px; margin: 15px 0;">

**💡 Pro Tip:** Always analyze the **missing data pattern** before choosing a strategy. Random missing data can be imputed, but systematic missing patterns might indicate data collection issues.

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #6f42c1, #e83e8c); color: white; border-radius: 8px; margin: 10px 0;">
    📈 3. Handling Outliers
</summary>

<div style="padding: 20px; background: rgba(111, 66, 193, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Identify and handle extreme values that can skew model performance.

### **🔍 Detection Methods**

#### **1️⃣ Visual Methods**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Box plots
plt.figure(figsize=(12, 8))
df.boxplot()
plt.xticks(rotation=45)
plt.show()

# Individual box plots
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot: {col}')
    plt.show()

# Scatter plots
sns.pairplot(df[numerical_cols])
plt.show()

# Histograms
df[numerical_cols].hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()
```

#### **2️⃣ Statistical Methods**

**📐 IQR Method:** Values beyond Q1-1.5×IQR or Q3+1.5×IQR
**📊 Z-Score:** Values with |z-score| > 3
**🎯 Modified Z-Score:** Using median absolute deviation

### **🛠️ Treatment Options**

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">

<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px;">
<strong>🗑️ Remove</strong><br>
Delete outlier records
</div>

<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px;">
<strong>🔒 Cap/Floor</strong><br>
Set to percentile limits
</div>

<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px;">
<strong>🔄 Transform</strong><br>
Log, square root, Box-Cox
</div>

<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px;">
<strong>📦 Binning</strong><br>
Convert to categorical ranges
</div>

</div>

### **💻 Implementation**
```python
from scipy import stats
from scipy.stats import boxcox
import numpy as np

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(df[column]))
    outliers = df[z_scores > threshold]
    return outliers

# Example: Handle outliers in a specific column
column = 'price'  # Replace with your column name

# Method 1: IQR Method
outliers, lower, upper = detect_outliers_iqr(df, column)
print(f"Outliers detected: {len(outliers)}")
print(f"Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")

# Remove outliers
df_clean = df[(df[column] >= lower) & (df[column] <= upper)]

# Method 2: Winsorization (Cap/Floor)
df[column] = np.clip(df[column], lower, upper)

# Method 3: Z-Score Method
z_scores = np.abs(stats.zscore(df[numerical_cols]))
df_zscore = df[(z_scores < 3).all(axis=1)]

# Method 4: Log transformation for skewed data
df[f'{column}_log'] = np.log1p(df[column])  # log1p = log(1+x)

# Method 5: Box-Cox transformation
df[f'{column}_boxcox'], lambda_param = boxcox(df[column] + 1)  # +1 to handle zeros
```

<div style="background: rgba(255, 193, 7, 0.1); border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0;">

**⚠️ Important:** Don't automatically remove all outliers! Some might represent important patterns or rare but valid cases. Always investigate the context before deciding on treatment.

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #17a2b8, #6610f2); color: white; border-radius: 8px; margin: 10px 0;">
    🔄 4. Data Type Conversion
</summary>

<div style="padding: 20px; background: rgba(23, 162, 184, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Ensure data types are appropriate for analysis and modeling.

### **🔄 Common Conversions**

<div style="background: rgba(23, 162, 184, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- Convert `object` to `category` for categorical data (saves memory)
- Convert `strings` to `datetime` for temporal data
- Convert `categorical text labels` to `numerical codes`
- Convert `boolean strings` to `actual boolean type`

</div>

### **✅ Benefits**
- ✅ **Improved memory efficiency**
- ✅ **Better performance in operations**
- ✅ **Enables appropriate statistical operations**

### **💻 Implementation**
```python
import pandas as pd
import numpy as np

# Before optimization - check current memory usage
print("Memory usage before optimization:")
print(df.info(memory_usage='deep'))

# 1. Convert to category (saves memory for repeated strings)
categorical_columns = ['gender', 'city', 'product_category']
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

# 2. Convert to datetime
date_columns = ['purchase_date', 'birth_date', 'registration_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 3. Convert boolean strings
boolean_mappings = {
    'is_premium': {'True': True, 'False': False, 'true': True, 'false': False},
    'is_active': {'Yes': True, 'No': False, 'Y': True, 'N': False}
}

for col, mapping in boolean_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# 4. Optimize numeric types
# Downcast integers
int_columns = df.select_dtypes(include=['int64']).columns
for col in int_columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

# Downcast floats
float_columns = df.select_dtypes(include=['float64']).columns
for col in float_columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

# 5. Handle mixed data types
def smart_convert(series):
    """Intelligently convert series to appropriate data type"""
    # Try numeric first
    try:
        return pd.to_numeric(series)
    except:
        pass
    
    # Try datetime
    try:
        return pd.to_datetime(series)
    except:
        pass
    
    # Try boolean
    if series.nunique() <= 2:
        unique_vals = series.unique()
        if set(unique_vals).issubset({'True', 'False', 'true', 'false', '1', '0', 1, 0}):
            return series.map({'True': True, 'False': False, 'true': True, 'false': False, '1': True, '0': False, 1: True, 0: False})
    
    # Default to category if few unique values
    if series.nunique() / len(series) < 0.5:
        return series.astype('category')
    
    return series

# Apply smart conversion to object columns
object_columns = df.select_dtypes(include=['object']).columns
for col in object_columns:
    df[col] = smart_convert(df[col])

# After optimization - check memory usage
print("\nMemory usage after optimization:")
print(df.info(memory_usage='deep'))

# Create a memory usage comparison function
def compare_memory_usage(df_before, df_after):
    """Compare memory usage before and after optimization"""
    memory_before = df_before.memory_usage(deep=True).sum()
    memory_after = df_after.memory_usage(deep=True).sum()
    reduction = (memory_before - memory_after) / memory_before * 100
    
    print(f"Memory before: {memory_before / 1024**2:.2f} MB")
    print(f"Memory after:  {memory_after / 1024**2:.2f} MB")
    print(f"Reduction:     {reduction:.1f}%")
```

<div style="background: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745; padding: 15px; margin: 15px 0;">

**🚀 Pro Tip:** Proper data type conversion can reduce memory usage by 50-90% for large datasets, significantly improving processing speed!

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #fd7e14, #dc3545); color: white; border-radius: 8px; margin: 10px 0;">
    🏷️ 5. Encoding Categorical Variables
</summary>

<div style="padding: 20px; background: rgba(253, 126, 20, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Convert categorical data into numerical format for machine learning algorithms.

### **🎯 Encoding Methods**

#### **1️⃣ Label Encoding**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Best for:** Ordinal features (with natural order)
- **Creates:** Single column with integer values
- **Example:** Education level (High School=0, Bachelor=1, Master=2, PhD=3)

</div>

#### **2️⃣ One-Hot Encoding**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Best for:** Nominal features (no natural order)
- **Creates:** Multiple binary columns
- **Example:** Color (Red, Blue, Green) → 3 binary columns

</div>

#### **3️⃣ Target/Mean Encoding**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Best for:** High cardinality categorical features
- **Risk:** Data leakage if not done properly
- **Use with:** Cross-validation and regularization

</div>

#### **4️⃣ Binary Encoding**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Best for:** High cardinality features (more efficient than one-hot)
- **Creates:** Log₂(n) binary columns

</div>

### **📦 Installation**
```bash
pip install category_encoders
```

### **💻 Implementation**
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce
import pandas as pd

# Sample data
df = pd.DataFrame({
    'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor'],
    'color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Boston'],
    'target': [0, 1, 1, 0, 1]
})

# 1. Label Encoding (for ordinal data)
le = LabelEncoder()
education_order = ['High School', 'Bachelor', 'Master', 'PhD']
df['education_encoded'] = df['education'].map({v: i for i, v in enumerate(education_order)})

# Alternative using LabelEncoder
le = LabelEncoder()
df['education_le'] = le.fit_transform(df['education'])
print("Label Encoding mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 2. One-Hot Encoding
# Method 1: Using pandas get_dummies
df_onehot = pd.get_dummies(df, columns=['color'], prefix='color', drop_first=True)

# Method 2: Using sklearn OneHotEncoder
ohe = OneHotEncoder(sparse=False, drop='first')
color_encoded = ohe.fit_transform(df[['color']])
color_columns = [f'color_{cat}' for cat in ohe.categories_[0][1:]]  # Skip first due to drop='first'
df_ohe = pd.concat([df, pd.DataFrame(color_encoded, columns=color_columns)], axis=1)

# 3. Target Encoding (use with caution - potential data leakage)
target_encoder = ce.TargetEncoder()
df['city_target_encoded'] = target_encoder.fit_transform(df['city'], df['target'])

# 4. Binary Encoding (for high cardinality)
binary_encoder = ce.BinaryEncoder()
df_binary = binary_encoder.fit_transform(df['city'])
df = pd.concat([df, df_binary], axis=1)

# 5. Advanced: Frequency Encoding
def frequency_encoding(series):
    """Encode categories by their frequency"""
    frequency_map = series.value_counts().to_dict()
    return series.map(frequency_map)

df['city_frequency'] = frequency_encoding(df['city'])

# 6. Advanced: Mean Encoding with Cross-Validation (safer)
from sklearn.model_selection import KFold

def safe_target_encoding(X, y, column, n_splits=5):
    """Target encoding with cross-validation to prevent leakage"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        train_mean = y.iloc[train_idx].groupby(X[column].iloc[train_idx]).mean()
        encoded[val_idx] = X[column].iloc[val_idx].map(train_mean)
    
    return encoded

# Example usage
df['city_safe_target'] = safe_target_encoding(df, df['target'], 'city')

print("Encoding Results:")
print(df.head())
```

### **📊 Comparison Table**

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Label Encoding** | Ordinal data | Simple, memory efficient | Assumes order, may mislead algorithms |
| **One-Hot Encoding** | Nominal data (low cardinality) | No false relationships | High dimensionality, sparse matrices |
| **Target Encoding** | High cardinality | Compact, captures target relationship | Risk of overfitting, leakage |
| **Binary Encoding** | High cardinality | More compact than one-hot | Less interpretable |

<div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0;">

**⚠️ Critical:** Always apply the same encoding transformations to both training and test data using the same fitted encoder to avoid data leakage!

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #20c997, #28a745); color: white; border-radius: 8px; margin: 10px 0;">
    ⚖️ 6. Feature Scaling
</summary>

<div style="padding: 20px; background: rgba(32, 201, 151, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Normalize feature ranges to prevent algorithms from being biased toward features with larger scales.

### **🎯 When Needed**
<div style="background: rgba(32, 201, 151, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

**✅ Scaling Required:**
- **Distance-based algorithms:** KNN, K-Means, SVM
- **Gradient-based algorithms:** Neural Networks, Logistic Regression
- **Regularized algorithms:** Ridge, Lasso, Elastic Net

**❌ Scaling NOT Needed:**
- **Tree-based algorithms:** Random Forest, Decision Trees, XGBoost

</div>

### **📏 Scaling Methods**

#### **1️⃣ MinMaxScaler**
- **Range:** [0, 1]
- **Formula:** (x - min) / (max - min)
- **Best for:** Bounded data, when you know min/max

#### **2️⃣ StandardScaler (Z-score)**
- **Range:** Mean=0, Std=1
- **Formula:** (x - mean) / std
- **Best for:** Normally distributed data

#### **3️⃣ RobustScaler**
- **Uses:** Median and IQR instead of mean and std
- **Best for:** Data with outliers

### **⚠️ Critical Rule**
<div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0;">

When using **MinMaxScaler** or **StandardScaler**, you should use `fit` on **train data only** and use `transform` on **test data** to prevent data leakage!

</div>

### **💻 Implementation**
```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data preparation
X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select numerical columns for scaling
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# 1. MinMax Scaling
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X_train[numerical_cols])  # Fit only on training data

X_train_minmax = X_train.copy()
X_test_minmax = X_test.copy()
X_train_minmax[numerical_cols] = minmax_scaler.transform(X_train[numerical_cols])
X_test_minmax[numerical_cols] = minmax_scaler.transform(X_test[numerical_cols])

# 2. Standard Scaling
std_scaler = StandardScaler()
std_scaler.fit(X_train[numerical_cols])

X_train_std = X_train.copy()
X_test_std = X_test.copy()
X_train_std[numerical_cols] = std_scaler.transform(X_train[numerical_cols])
X_test_std[numerical_cols] = std_scaler.transform(X_test[numerical_cols])

# 3. Robust Scaling
robust_scaler = RobustScaler()
robust_scaler.fit(X_train[numerical_cols])

X_train_robust = X_train.copy()
X_test_robust = X_test.copy()
X_train_robust[numerical_cols] = robust_scaler.transform(X_train[numerical_cols])
X_test_robust[numerical_cols] = robust_scaler.transform(X_test[numerical_cols])

# Visualization function
def plot_scaling_comparison(original, minmax, standard, robust, feature_name):
    """Compare different scaling methods visually"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original
    axes[0,0].hist(original, bins=30, alpha=0.7)
    axes[0,0].set_title(f'Original {feature_name}')
    axes[0,0].set_ylabel('Frequency')
    
    # MinMax
    axes[0,1].hist(minmax, bins=30, alpha=0.7, color='orange')
    axes[0,1].set_title(f'MinMax Scaled {feature_name}')
    
    # Standard
    axes[1,0].hist(standard, bins=30, alpha=0.7, color='green')
    axes[1,0].set_title(f'Standard Scaled {feature_name}')
    axes[1,0].set_ylabel('Frequency')
    
    # Robust
    axes[1,1].hist(robust, bins=30, alpha=0.7, color='red')
    axes[1,1].set_title(f'Robust Scaled {feature_name}')
    
    plt.tight_layout()
    plt.show()

# Example visualization (replace 'feature_name' with actual column)
if len(numerical_cols) > 0:
    feature = numerical_cols[0]
    plot_scaling_comparison(
        X_train[feature],
        X_train_minmax[feature],
        X_train_std[feature],
        X_train_robust[feature],
        feature
    )

# Statistical comparison
def scaling_stats(original, scaled, method_name):
    """Print statistics for scaled data"""
    print(f"\n{method_name} Scaling Statistics:")
    print(f"Original - Mean: {original.mean():.3f}, Std: {original.std():.3f}")
    print(f"Scaled   - Mean: {scaled.mean():.3f}, Std: {scaled.std():.3f}")
    print(f"Scaled   - Min: {scaled.min():.3f}, Max: {scaled.max():.3f}")

# Compare all scaling methods
for feature in numerical_cols[:1]:  # Compare first numerical feature
    scaling_stats(X_train[feature], X_train_minmax[feature], "MinMax")
    scaling_stats(X_train[feature], X_train_std[feature], "Standard")
    scaling_stats(X_train[feature], X_train_robust[feature], "Robust")
```

### **🎯 Choosing the Right Scaler**

<div style="background: rgba(23, 162, 184, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

| **Scaler** | **Best For** | **Pros** | **Cons** |
|------------|-------------|----------|----------|
| **MinMaxScaler** | Bounded features, uniform distribution | Fixed range [0,1], preserves relationships | Sensitive to outliers |
| **StandardScaler** | Normal distribution, no outliers | Centers data, unit variance | Assumes normal distribution |
| **RobustScaler** | Features with outliers | Robust to outliers | May not scale to [0,1] |

</div>

<div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0;">

**⚠️ Remember:** Always fit the scaler on training data only, then transform both training and test data to prevent data leakage!

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #fd7e14, #e83e8c); color: white; border-radius: 8px; margin: 10px 0;">
    🛠️ 7. Feature Engineering
</summary>

<div style="padding: 20px; background: rgba(253, 126, 20, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Create new features from existing ones to improve model performance and capture domain knowledge.

### **🎯 Common Techniques**

#### **1️⃣ Mathematical Operations**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Ratios:** income/expense, price/sqft
- **Differences:** current_price - previous_price  
- **Products:** length × width for area
- **Powers:** x², √x for non-linear relationships

</div>

#### **2️⃣ Date/Time Features**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Extract:** year, month, day, hour, day_of_week
- **Create:** is_weekend, is_holiday, days_since_event
- **Cyclical:** sin/cos transformations for hours, months

</div>

#### **3️⃣ Text Features**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Length:** character count, word count
- **Patterns:** email domains, phone area codes
- **Sentiment:** positive/negative scoring

</div>

#### **4️⃣ Binning/Discretization**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Age groups:** 0-18, 19-35, 36-50, 50+
- **Income brackets:** Low, Medium, High
- **Performance tiers:** A, B, C grades

</div>

#### **5️⃣ Polynomial Features**
<div style="background: rgba(253, 126, 20, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Powers:** x², x³ for non-linear patterns
- **Interactions:** x₁×x₂ for feature combinations

</div>

### **🔧 Advanced Tools**
<div style="background: rgba(23, 162, 184, 0.1); border-left: 4px solid #17a2b8; padding: 15px; margin: 15px 0;">

👍 You can also use **Featuretools** for automatic feature engineering

</div>

### **💻 Implementation**
```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

# Date feature engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['quarter'] = df['date'].dt.quarter

# Cyclical features for time
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Mathematical operations
df['bmi'] = df['weight'] / (df['height'] ** 2)
df['price_per_sqft'] = df['price'] / df['area']
df['income_expense_ratio'] = df['income'] / df['expense']

# Text features
df['text_length'] = df['description'].str.len()
df['word_count'] = df['description'].str.split().str.len()
df['email_domain'] = df['email'].str.split('@').str[1]

# Binning
df['age_group'] = pd.cut(df['age'], 
                        bins=[0, 18, 35, 50, 100], 
                        labels=['Child', 'Young', 'Adult', 'Senior'])

df['income_bracket'] = pd.qcut(df['income'], 
                              q=3, 
                              labels=['Low', 'Medium', 'High'])

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])
poly_feature_names = poly.get_feature_names_out(['feature1', 'feature2'])

# Create polynomial dataframe
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
```

### **💡 Feature Engineering Tips**

<div style="background: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745; padding: 15px; margin: 15px 0;">

**✅ Best Practices:**
- **Domain Knowledge:** Use business understanding to create meaningful features
- **Validation:** Always validate new features improve model performance  
- **Correlation Check:** Remove highly correlated engineered features
- **Feature Importance:** Use tree-based models to identify valuable features

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #6f42c1, #e83e8c); color: white; border-radius: 8px; margin: 10px 0;">
    🎯 8. Feature Selection
</summary>

<div style="padding: 20px; background: rgba(111, 66, 193, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Select the most relevant features to improve model performance and reduce overfitting.

### **✅ Benefits**
<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- ✅ **Reduces overfitting**
- ✅ **Improves model interpretability**  
- ✅ **Decreases training time**
- ✅ **Reduces storage requirements**

</div>

### **🔍 Selection Methods**

#### **1️⃣ Filter Methods (Statistical)**
<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Correlation Matrix:** Remove highly correlated features (>0.95)
- **Chi-square Test:** For categorical features vs categorical target
- **ANOVA F-test:** For numerical features vs categorical target  
- **Mutual Information:** Measures dependency between features and target

</div>

#### **2️⃣ Wrapper Methods**
<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Recursive Feature Elimination (RFE):** Iteratively remove features
- **Forward/Backward Selection:** Add/remove features stepwise

</div>

#### **3️⃣ Embedded Methods**
<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **L1 Regularization (Lasso):** Automatically selects features
- **Tree-based Feature Importance:** From Random Forest, XGBoost

</div>

### **🔧 Advanced Tip**
<div style="background: rgba(23, 162, 184, 0.1); border-left: 4px solid #17a2b8; padding: 15px; margin: 15px 0;">

You can also use **SelectFromModel** with **Lasso** for automatic feature selection

</div>

### **💻 Implementation**
```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
import seaborn as sns
import numpy as np

# 1. Correlation Analysis
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Remove highly correlated features
def remove_correlated_features(df, threshold=0.95):
    """Remove features with correlation > threshold"""
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_features = [column for column in upper_triangle.columns 
                         if any(upper_triangle[column] > threshold)]
    
    return df.drop(columns=high_corr_features), high_corr_features

df_reduced, removed_features = remove_correlated_features(df[numerical_cols])
print(f"Removed highly correlated features: {removed_features}")

# 2. Chi-square for categorical features
chi2_selector = SelectKBest(chi2, k=10)
X_chi2 = chi2_selector.fit_transform(X_categorical, y)
chi2_scores = chi2_selector.scores_
chi2_features = X_categorical.columns[chi2_selector.get_support()]

print("Top Chi-square features:")
feature_scores = list(zip(chi2_features, chi2_scores[chi2_selector.get_support()]))
for feature, score in sorted(feature_scores, key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.2f}")

# 3. ANOVA F-test for numerical features
f_selector = SelectKBest(f_classif, k=10)
X_f = f_selector.fit_transform(X_numerical, y)
f_scores = f_selector.scores_
f_features = X_numerical.columns[f_selector.get_support()]

# 4. RFE with Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=10, step=1)
X_rfe = rfe.fit_transform(X, y)

selected_features_rfe = X.columns[rfe.support_]
feature_ranking = rfe.ranking_

print("RFE Selected Features:")
for i, feature in enumerate(selected_features_rfe):
    print(f"{feature}: Rank {feature_ranking[X.columns.get_loc(feature)]}")

# 5. Lasso Feature Selection
lasso = Lasso(alpha=0.01, random_state=42)
lasso_selector = SelectFromModel(lasso)
X_lasso = lasso_selector.fit_transform(X, y)

selected_features_lasso = X.columns[lasso_selector.get_support()]
print(f"Lasso selected {len(selected_features_lasso)} features:")
print(selected_features_lasso.tolist())

# 6. Feature Importance from Random Forest
rf.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(feature_importance.head(10))

# Visualization
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, y='feature', x='importance')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

### **📊 Feature Selection Comparison**

<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

| **Method** | **Type** | **Best For** | **Pros** | **Cons** |
|------------|----------|--------------|----------|----------|
| **Correlation** | Filter | Linear relationships | Fast, simple | Misses non-linear relationships |
| **Chi-square** | Filter | Categorical features | Statistical significance | Only for categorical |
| **ANOVA F-test** | Filter | Numerical features | Statistical foundation | Assumes normal distribution |
| **RFE** | Wrapper | Any algorithm | Algorithm-specific | Computationally expensive |
| **Lasso** | Embedded | Linear models | Automatic selection | Linear assumptions |
| **Tree Importance** | Embedded | Tree models | Handles non-linearity | Model-specific |

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #17a2b8, #6610f2); color: white; border-radius: 8px; margin: 10px 0;">
    📝 9. Text Preprocessing (if applicable)
</summary>

<div style="padding: 20px; background: rgba(23, 162, 184, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Clean and prepare text data for NLP and machine learning tasks.

### **🔄 Common Steps**

#### **1️⃣ Basic Cleaning**
<div style="background: rgba(23, 162, 184, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Lowercasing:** Convert all text to lowercase
- **Remove punctuation:** Clean special characters  
- **Remove extra whitespace:** Handle spacing issues
- **Handle encoding:** Fix UTF-8, ASCII issues

</div>

#### **2️⃣ Tokenization**
<div style="background: rgba(23, 162, 184, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Split text:** Into individual words/tokens
- **Handle contractions:** don't → do not

</div>

#### **3️⃣ Stopwords Removal**
<div style="background: rgba(23, 162, 184, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Remove common words:** the, and, or, etc.
- **Language-specific:** Use appropriate stopword lists

</div>

#### **4️⃣ Normalization**
<div style="background: rgba(23, 162, 184, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Stemming:** Reduce words to root form (running → run)
- **Lemmatization:** Reduce to dictionary form (better → good)

</div>

#### **5️⃣ Vectorization**
<div style="background: rgba(23, 162, 184, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Bag of Words:** Count frequency of words
- **TF-IDF:** Term frequency-inverse document frequency  
- **Word Embeddings:** Word2Vec, GloVe, FastText

</div>

### **🔧 Advanced Tools**
<div style="background: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745; padding: 15px; margin: 15px 0;">

👉 You can also use **Contractions** for fixing contractions in text  
👉 You can also use **spaCy** for more advanced text processing

</div>

### **💻 Implementation**
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import contractions
import spacy

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, use_lemmatizer=False):
    """Comprehensive text preprocessing function"""
    # Handle contractions
    text = contractions.fix(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs, emails, and special patterns
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and apply stemming/lemmatization
    if use_lemmatizer:
        tokens = [lemmatizer.lemmatize(word) for word in tokens 
                 if word not in stop_words and len(word) > 2]
    else:
        tokens = [stemmer.stem(word) for word in tokens 
                 if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, use_lemmatizer=True))

# Advanced preprocessing with spaCy
nlp = spacy.load('en_core_web_sm')

def spacy_preprocess(text):
    """Advanced preprocessing using spaCy"""
    doc = nlp(text)
    
    # Extract lemmatized tokens, remove stop words and punctuation
    tokens = [token.lemma_.lower() for token in doc 
             if not token.is_stop and not token.is_punct 
             and not token.is_space and len(token.text) > 2]
    
    return ' '.join(tokens)

df['spacy_cleaned'] = df['text'].apply(spacy_preprocess)

# Vectorization
# 1. Bag of Words
bow_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))
bow_features = bow_vectorizer.fit_transform(df['cleaned_text'])

# 2. TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), 
                                  min_df=2, max_df=0.95)
tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# Get feature names
bow_feature_names = bow_vectorizer.get_feature_names_out()
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"Bag of Words features: {bow_features.shape}")
print(f"TF-IDF features: {tfidf_features.shape}")

# Feature analysis
def analyze_text_features(vectorizer, features, feature_names, top_n=10):
    """Analyze most important text features"""
    # Sum features across all documents
    feature_sums = np.array(features.sum(axis=0)).flatten()
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_sums
    }).sort_values('importance', ascending=False)
    
    print(f"Top {top_n} most frequent features:")
    print(feature_importance.head(top_n))
    
    return feature_importance

# Analyze features
bow_importance = analyze_text_features(bow_vectorizer, bow_features, bow_feature_names)
tfidf_importance = analyze_text_features(tfidf_vectorizer, tfidf_features, tfidf_feature_names)
```

### **📊 Text Preprocessing Comparison**

<div style="background: rgba(23, 162, 184, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

| **Method** | **Purpose** | **Pros** | **Cons** |
|------------|-------------|----------|----------|
| **Stemming** | Root form reduction | Fast, simple | Can be aggressive, lose meaning |
| **Lemmatization** | Dictionary form | Maintains meaning | Slower, requires POS tags |
| **Bag of Words** | Word frequency | Simple, interpretable | Loses word order, sparse |
| **TF-IDF** | Weighted frequency | Reduces common word impact | Still loses context |
| **Word Embeddings** | Dense representations | Captures semantics | Requires pre-training |

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #dc3545, #fd7e14); color: white; border-radius: 8px; margin: 10px 0;">
    ✂️ 10. Data Splitting
</summary>

<div style="padding: 20px; background: rgba(220, 53, 69, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Separate data into training and testing sets to evaluate model performance on unseen data.

### **📊 Common Split Ratios**
<div style="background: rgba(220, 53, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **80/20** (Train/Test)
- **70/30** (Train/Test)  
- **60/20/20** (Train/Validation/Test)

</div>

### **🔄 Types of Splitting**

#### **1️⃣ Random Split**
<div style="background: rgba(220, 53, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Good for:** Independent observations
- **Use:** `train_test_split()`

</div>

#### **2️⃣ Stratified Split**
<div style="background: rgba(220, 53, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Good for:** Imbalanced datasets
- **Maintains:** Class distribution in both sets

</div>

#### **3️⃣ Time-based Split**
<div style="background: rgba(220, 53, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Good for:** Time series data
- **Rule:** Train on past, test on future

</div>

#### **4️⃣ Cross-validation**
<div style="background: rgba(220, 53, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **K-fold:** Split data into k folds
- **Stratified K-fold:** Maintains class distribution
- **Time series split:** Respects temporal order

</div>

### **⚠️ Important Note**
<div style="background: rgba(255, 193, 7, 0.1); border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0;">

Use `cross_val_score` or `cross_validate` for model evaluation across folds.

</div>

### **💻 Implementation**
```python
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                   TimeSeriesSplit, cross_val_score, 
                                   GridSearchCV)
import numpy as np

# 1. Basic random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 2. Stratified split for imbalanced data
X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Check class distribution
print("Original distribution:", np.bincount(y) / len(y))
print("Training distribution:", np.bincount(y_train_strat) / len(y_train_strat))
print("Test distribution:", np.bincount(y_test_strat) / len(y_test_strat))

# 3. Three-way split (Train/Validation/Test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)  # 0.25 * 0.8 = 0.2

print(f"Training: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 4. Time series split
if 'date' in df.columns:
    # Sort by date
    df_sorted = df.sort_values('date')
    
    # Manual time-based split
    train_size = int(0.8 * len(df_sorted))
    train_data = df_sorted[:train_size]
    test_data = df_sorted[train_size:]
    
    print(f"Training period: {train_data['date'].min()} to {train_data['date'].max()}")
    print(f"Test period: {test_data['date'].min()} to {test_data['date'].max()}")

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Fold {fold+1}: Train size={len(X_train_fold)}, Test size={len(X_test_fold)}")

# 5. K-Fold Cross-Validation
from sklearn.ensemble import RandomForestClassifier

# Standard K-Fold
cv_scores = cross_val_score(RandomForestClassifier(random_state=42), 
                           X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_scores = cross_val_score(RandomForestClassifier(random_state=42), 
                              X, y, cv=skf, scoring='accuracy')
print(f"Stratified CV Accuracy: {strat_scores.mean():.3f} (+/- {strat_scores.std() * 2:.3f})")

# 6. Cross-validation with multiple metrics
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(RandomForestClassifier(random_state=42), 
                           X, y, cv=5, scoring=scoring)

for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.upper()}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### **📊 Cross-Validation Strategies**

<div style="background: rgba(220, 53, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

| **Strategy** | **Use Case** | **Pros** | **Cons** |
|--------------|--------------|----------|----------|
| **K-Fold** | General purpose | Robust estimates | May not preserve class distribution |
| **Stratified K-Fold** | Imbalanced data | Preserves class ratios | Requires categorical target |
| **Time Series Split** | Temporal data | Respects time order | Smaller training sets in early folds |
| **Leave-One-Out** | Small datasets | Maximum training data | Computationally expensive |

</div>

</div>
</details>

---

<details>
<summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; padding: 10px; background: linear-gradient(45deg, #28a745, #20c997); color: white; border-radius: 8px; margin: 10px 0;">
    ⚖️ 11. Balancing the Dataset (if needed)
</summary>

<div style="padding: 20px; background: rgba(40, 167, 69, 0.05); border-radius: 10px; margin: 10px 0;">

### **Purpose**
Address class imbalance that can lead to biased model predictions.

### **🎯 When to Use**
<div style="background: rgba(40, 167, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- 👉 **Imbalanced classification problems**
- 👉 **Minority class < 10–20% of total data**
- 👉 **When accuracy alone is not sufficient** (e.g., fraud detection, medical diagnosis)

</div>

### **🔄 Techniques**

#### **1️⃣ Oversampling**
<div style="background: rgba(40, 167, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **SMOTE** (Synthetic Minority Oversampling Technique): Creates synthetic samples from the minority class
- **ADASYN**: Adaptive version of SMOTE, focuses more on difficult examples
- **Random Oversampling**: Duplicates existing samples from the minority class
- **BorderlineSMOTE**: Oversamples near the decision boundary

</div>

#### **2️⃣ Undersampling**
<div style="background: rgba(40, 167, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Random Undersampling**: Removes samples from the majority class
- **Tomek Links**: Removes majority samples that are borderline
- **Edited Nearest Neighbors**: Removes noisy or ambiguous samples

</div>

#### **3️⃣ Algorithmic Approaches**
<div style="background: rgba(40, 167, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Class Weights**: Increase penalty for misclassifying minority class
- **Cost-sensitive Learning**: Custom loss functions for imbalance
- **Ensemble Methods**: Use balanced subsets in ensemble models (e.g., BalancedRandomForest)

</div>

### **🔄 Avoiding Data Leakage with imblearn.pipeline**

<div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0;">

When oversampling is done **before** splitting data (train/test), it leaks information from the test set into training.  
To prevent this, use `imblearn.pipeline.Pipeline` to perform oversampling **inside the cross-validation loop**:

</div>

### **💻 Implementation**
```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter
import numpy as np

# Check original distribution
print("Original distribution:", Counter(y))

# 1. SMOTE Oversampling
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
print("After SMOTE:", Counter(y_smote))

# 2. ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
print("After ADASYN:", Counter(y_adasyn))

# 3. Random Oversampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)
print("After Random Oversampling:", Counter(y_ros))

# 4. Borderline SMOTE
borderline_smote = BorderlineSMOTE(random_state=42)
X_borderline, y_borderline = borderline_smote.fit_resample(X, y)
print("After Borderline SMOTE:", Counter(y_borderline))

# 5. Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)
print("After Random Undersampling:", Counter(y_rus))

# 6. Tomek Links
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X, y)
print("After Tomek Links:", Counter(y_tomek))

# 7. Safe Pipeline with Cross-Validation (RECOMMENDED)
# Build pipeline with oversampling and classifier
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Cross-validation with safe oversampling inside folds
scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
print(f"Pipeline F1 Scores: {scores}")
print(f"Average F1: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# 8. Class Weights Approach (Alternative to sampling)
# This doesn't change the data size, just adjusts algorithm behavior
rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)
balanced_scores = cross_val_score(rf_balanced, X, y, cv=5, scoring='f1_macro')
print(f"Balanced RF F1: {balanced_scores.mean():.3f} (+/- {balanced_scores.std() * 2:.3f})")

# 9. Custom class weights
class_weights = {0: 1, 1: 10}  # Give 10x weight to minority class
rf_custom = RandomForestClassifier(class_weight=class_weights, random_state=42)
custom_scores = cross_val_score(rf_custom, X, y, cv=5, scoring='f1_macro')
print(f"Custom Weighted RF F1: {custom_scores.mean():.3f} (+/- {custom_scores.std() * 2:.3f})")

# 10. Comparison of methods
methods = {
    'Original': (X, y),
    'SMOTE': (X_smote, y_smote),
    'ADASYN': (X_adasyn, y_adasyn),
    'Random Oversample': (X_ros, y_ros),
    'Random Undersample': (X_rus, y_rus)
}

results = {}
for name, (X_method, y_method) in methods.items():
    if name == 'Original':
        clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    else:
        clf = RandomForestClassifier(random_state=42)
    
    scores = cross_val_score(clf, X_method, y_method, cv=5, scoring='f1_macro')
    results[name] = scores.mean()

# Display results
print("\nMethod Comparison (F1-Score):")
for method, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{method:20}: {score:.3f}")
```

### **📊 Sampling Methods Comparison**

<div style="background: rgba(40, 167, 69, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

| **Method** | **Type** | **Pros** | **Cons** |
|------------|----------|----------|----------|
| **SMOTE** | Oversample | Creates realistic synthetic samples | May create noise in complex datasets |
| **ADASYN** | Oversample | Focuses on difficult cases | Can increase class overlap |
| **Random Oversample** | Oversample | Simple, preserves all information | Risk of overfitting |
| **Random Undersample** | Undersample | Reduces training time | Loss of potentially useful information |
| **Tomek Links** | Undersample | Removes borderline cases | May remove useful boundary information |
| **Class Weights** | Algorithmic | No data modification | Algorithm-dependent effectiveness |

</div>

</div>
</details>

---

## 📊 **Evaluation Metrics for Imbalanced Data**

<details>
<summary style="font-size: 1.2em; font-weight: bold; cursor: pointer; padding: 8px; background: rgba(108, 117, 125, 0.1); border-radius: 6px; margin: 8px 0;">
    Click to expand evaluation metrics
</summary>

<div style="padding: 15px; background: rgba(108, 117, 125, 0.05); border-radius: 8px; margin: 10px 0;">

### **Beyond Accuracy**
<div style="background: rgba(108, 117, 125, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **Precision**: $TP / (TP + FP)$ - How many positive predictions were correct?
- **Recall (Sensitivity)**: $TP / (TP + FN)$ - How many actual positives were found?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **PR-AUC**: Area under the precision-recall curve (better for imbalanced data)

</div>

### **✅ Key Recommendation**
<div style="background: rgba(40, 167, 69, 0.1); border-left: 4px solid #28a745; padding: 15px; margin: 15px 0;">

Use **PR-AUC** instead of **ROC-AUC** when the dataset is **highly imbalanced**.

</div>

### **💻 Code Example**
```python
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, average_precision_score,
                           precision_recall_curve, roc_curve)
import matplotlib.pyplot as plt

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Comprehensive evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"PR-AUC: {average_precision_score(y_test, y_pred_proba):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
axes[1].plot(recall, precision, label=f'PR Curve (AUC = {average_precision_score(y_test, y_pred_proba):.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.tight_layout()
plt.show()
```

</div>
</details>

---

## 📈 **Summary Workflow Diagram**

<div style="background: rgba(0,123,255,0.1); 
            border-left: 4px solid #007bff; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 0 10px 10px 0;
            font-family: 'Courier New', monospace;">

```
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

</div>

---

## 🔗 **Useful Libraries & Resources**

<div style="background: rgba(111, 66, 193, 0.05); 
            padding: 20px; 
            border-radius: 10px; 
            margin: 20px 0;">

### **Essential Libraries**
<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **`pandas`** - Data manipulation and analysis
- **`numpy`** - Numerical computing  
- **`scikit-learn`** - Machine learning preprocessing and algorithms
- **`matplotlib` & `seaborn`** - Data visualization
- **`imbalanced-learn`** - Handling imbalanced datasets

</div>

### **Advanced Libraries**
<div style="background: rgba(111, 66, 193, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">

- **`feature-engine`** - Advanced feature engineering
- **`category_encoders`** - Specialized categorical encoding
- **`missingno`** - Missing data visualization  
- **`yellowbrick`** - Machine learning visualization
- **`featuretools`** - Automated feature engineering

</div>

### **Installation Commands**
```bash
# Essential packages
pip install pandas numpy scikit-learn matplotlib seaborn

# Imbalanced data
pip install imbalanced-learn

# Advanced preprocessing
pip install feature-engine category_encoders missingno

# Text processing
pip install nltk spacy contractions
python -m spacy download en_core_web_sm

# Visualization
pip install yellowbrick plotly

# Automated feature engineering
pip install featuretools
```

</div>

---

<div style="background: rgba(255, 193, 7, 0.1); 
            border: 2px solid #ffc107; 
            padding: 20px; 
            border-radius: 15px; 
            margin: 25px 0; 
            text-align: center;">

### **🤔 Remember**

**The best preprocessing pipeline depends on your specific dataset, problem type, and chosen algorithms. Always validate your preprocessing decisions with domain expertise and cross-validation!**

</div>

---

<div style="background: linear-gradient(135deg, rgb(37, 127, 201), #8b000e); 
            color: #ffffff; 
            width: 100%; 
            height: 50px; 
            text-align: center; 
            font-weight: bold; 
            line-height: 50px; 
            margin: 25px 0; 
            font-size: 24px; 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);">
    By Abdelrhman Ezzat 🫡
</div>