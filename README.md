# Adaptive Feature Selection and Projection (AFSP)

AFSP is a robust preprocessing pipeline for adaptive feature selection and dimensionality reduction, designed especially for imbalanced datasets with multiple classes. It standardizes data, removes noisy and redundant features, selects the most informative attributes using mutual information, and dynamically tunes the number of features to optimize model performance.

---

## 🔍 Features

- **Data Standardization:** Applies Z-score normalization to ensure all features are on a common scale.
- **Missing Data Handling:** Replaces missing values with the median of the corresponding feature.
- **Low-Variance Feature Removal:** Eliminates features with minimal variation to reduce noise.
- **High-Correlation Filtering:** Discards highly correlated features to prevent redundancy and mitigate overfitting.
- **Mutual Information-Based Feature Selection:** Selects features based on their information contribution to the target labels.
- **Adaptive Feature Tuning:** Dynamically adjusts the number of selected features (ensuring at least 10 features are retained) to balance dimensionality reduction with predictive performance.

---

## 📦 Installation

### Prerequisites

Ensure you have Python 3.x installed. Required libraries:

- `numpy`
- `pandas`
- `scikit-learn`

Install dependencies using pip:

```bash
pip install numpy pandas scikit-learn
```

### Clone the repository

```bash
git clone https://github.com/yourusername/afsp.git
cd afsp
```

Install in editable mode:

```bash
pip install -e .
```

---

## 🚀 Getting Started

### Example Usage

```python
import pandas as pd
from afsp import AFSP

# Load dataset
X = pd.read_csv("features.csv")               # Features (DataFrame)
y = pd.read_csv("labels.csv").values.ravel()  # Target labels

# Initialize and run AFSP
afsp = AFSP()
X_selected = afsp.fit_transform(X, y)

# Print selected features
print(X_selected.head())

# Display info
afsp.get_feature_info()
```

---

## 📘 API Reference

### `AFSP(top_rho=0.8, tau_corr=0.85, tau_var=0.01, max_retries=3)`

- `top_rho`: Retention ratio for MI-based feature selection.
- `tau_corr`: Correlation threshold to remove redundant features.
- `tau_var`: Variance threshold to remove uninformative features.
- `max_retries`: Number of retries if too few features are retained.

---

### Methods

#### `fit(X, y)`
Fits the AFSP model to input `X` and target `y`.

#### `transform(X)`
Transforms the dataset using previously selected features.

#### `fit_transform(X, y)`
Applies fit and transform in a single step.

#### `get_feature_info()`
Prints a summary of selected and removed features, and number of features at each step.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Pham Ngoc Van**  
📧 [phamngocvan.kma@gmail.com](mailto:phamngocvan.kma@gmail.com)  
🎓 MSc in Information Security  
🏛 Vietnam Academy of Cryptography Techniques  
🏦 Cybersecurity Analyst at MBBank  

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repository and submit a pull request.

---

## 📌 Acknowledgements

This project is inspired by real-world challenges in malware classification, anomaly detection, and imbalanced data learning, where effective preprocessing is critical.
