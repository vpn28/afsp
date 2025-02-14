# Adaptive Feature Selection and Projection (AFSP)

AFSP is a robust preprocessing pipeline for adaptive feature selection and dimensionality reduction, designed especially for imbalanced datasets with multiple classes. It standardizes data, removes noisy and redundant features, selects the most informative attributes using mutual information, and dynamically tunes the number of features to optimize model performance.

## Features

- **Data Standardization:** Applies Z-score normalization to ensure all features are on a common scale.
- **Missing Data Handling:** Replaces missing values with the median of the corresponding feature.
- **Low-Variance Feature Removal:** Eliminates features with minimal variation to reduce noise.
- **High-Correlation Filtering:** Discards highly correlated features to prevent redundancy and mitigate overfitting.
- **Mutual Information-Based Feature Selection:** Selects features based on their information contribution to the target labels.
- **Adaptive Feature Tuning:** Dynamically adjusts the number of selected features (ensuring at least 10 features are retained) to balance dimensionality reduction with predictive performance.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed. The following Python libraries are required:
- `numpy`
- `pandas`
- `scikit-learn`

Install the dependencies using pip:

```bash
pip install numpy pandas scikit-learn
