"""
afsp.py

Adaptive Feature Selection and Projection (AFSP) Algorithm.

Author: Pham Ngoc Van
Email: phamngocvan.kma@gmail.com
Description:
    This module implements the AFSP algorithm to preprocess high-dimensional datasets by removing
    low-variance and highly-correlated features, followed by selecting the most informative ones
    using Mutual Information (MI).

Dependencies:
    - numpy
    - pandas
    - sklearn

Usage:
    from afsp import AFSP

    afsp = AFSP()
    X_reduced = afsp.fit_transform(X, y)
    afsp.get_feature_info()
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif


class AFSP:
    """
    Adaptive Feature Selection and Projection (AFSP) Algorithm
    """

    def __init__(self, top_rho=0.8, tau_corr=0.85, tau_var=0.01, max_retries=3):
        """
        Initialize the AFSP algorithm with custom parameters.

        Parameters:
        - top_rho: float
            Retention ratio for selecting top-ranked features based on Mutual Information.
        - tau_corr: float
            Correlation threshold for removing redundant features.
        - tau_var: float
            Threshold for removing low-variance features.
        - max_retries: int
            Maximum retries if too few features are selected.
        """
        self.top_rho = top_rho
        self.tau_corr = tau_corr
        self.tau_var = tau_var
        self.max_retries = max_retries
        self.scaler = StandardScaler()
        self.selected_features = None
        self.removed_features = []

        # Internal tracking for reporting
        self.initial_features = 0
        self.features_after_variance = 0
        self.features_after_correlation = 0

    def fit(self, X, y):
        """
        Fit the AFSP model to the data.

        Parameters:
        - X: pd.DataFrame
            Feature matrix.
        - y: pd.Series or np.array
            Target labels.
        """
        try:
            self.removed_features = []
            self.initial_features = X.shape[1]

            # Handle missing values
            X_filled = X.fillna(X.median())

            # Standardize the data
            X_scaled = self.scaler.fit_transform(X_filled)
            zero_std_mask = self.scaler.scale_ == 0
            if np.any(zero_std_mask):
                X_scaled[:, zero_std_mask] = 0
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

            # Remove low-variance features
            selector = VarianceThreshold(threshold=self.tau_var)
            X_reduced = selector.fit_transform(X_scaled_df)
            X_reduced_df = pd.DataFrame(X_reduced, columns=X_scaled_df.columns[selector.get_support()])
            self.features_after_variance = X_reduced_df.shape[1]
            self.removed_features.extend(list(set(X_scaled_df.columns) - set(X_reduced_df.columns)))

            # Remove highly correlated features
            corr_matrix = X_reduced_df.corr(method='spearman').abs()
            to_drop = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.tau_corr:
                        to_drop.add(corr_matrix.columns[j])
            X_reduced_df = X_reduced_df.drop(columns=list(to_drop), axis=1)
            self.features_after_correlation = X_reduced_df.shape[1]
            self.removed_features.extend(list(to_drop))

            # Compute mutual information
            retry_count = 0
            min_features = 10
            top_rho = self.top_rho

            while retry_count < self.max_retries:
                mi_scores = mutual_info_classif(X_reduced_df, y, discrete_features='auto')
                num_features = max(min_features, int(len(mi_scores) * top_rho))
                selected_indices = np.argsort(mi_scores)[-num_features:]
                self.selected_features = X_reduced_df.columns[selected_indices]

                if len(self.selected_features) >= min_features:
                    break
                top_rho = min(top_rho + 0.05, 1.0)
                retry_count += 1

            if len(self.selected_features) < min_features:
                print("Warning: Fewer than minimum desired features retained.")

            # Track removed features not selected by MI
            self.removed_features.extend([col for col in X_reduced_df.columns if col not in self.selected_features])

        except Exception as e:
            print(f"[AFSP.fit] Error: {e}")

    def transform(self, X):
        """
        Transform input data using selected features.

        Parameters:
        - X: pd.DataFrame
            Data to transform.

        Returns:
        - pd.DataFrame
            Transformed data.
        """
        try:
            if self.selected_features is None:
                raise ValueError("AFSP has not been fitted. Call fit() first.")
            return X[self.selected_features].copy()
        except Exception as e:
            print(f"[AFSP.transform] Error: {e}")
            return None

    def fit_transform(self, X, y):
        """
        Fit and transform the dataset.

        Parameters:
        - X: pd.DataFrame
        - y: pd.Series

        Returns:
        - pd.DataFrame
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_info(self):
        """
        Print detailed information about the feature selection process.
        """
        print("\n[AFSP] Feature Selection Summary:")
        print(f"Initial number of features: {self.initial_features}")
        print(f"After variance filtering: {self.features_after_variance}")
        print(f"After correlation filtering: {self.features_after_correlation}")
        print(f"Number of features selected by Mutual Information: {len(self.selected_features) if self.selected_features is not None else 0}")
        print(f"Number of removed features: {len(self.removed_features)}")
        print(f"List of removed features: {self.removed_features}")
