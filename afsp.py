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
        :param top_rho: The retention ratio for selecting top-ranked features based on Mutual Information (MI).
        :param tau_corr: The correlation threshold to remove redundant features.
        :param tau_var: The threshold for removing low-variance features.
        :param max_retries: Maximum number of retries if too few features are selected.
        """
        self.top_rho = top_rho
        self.tau_corr = tau_corr
        self.tau_var = tau_var
        self.max_retries = max_retries
        self.scaler = StandardScaler()
        self.selected_features = None
        self.removed_features = []
    
    def fit(self, X, y):
        """
        Train the AFSP algorithm on the input dataset.
        :param X: Input data (DataFrame, excluding labels).
        :param y: Corresponding labels.
        """
        try:
            self.removed_features = []
            initial_features = X.shape[1]

            # Step 1: Standardize the data
            # Handle missing data (NaN) by replacing them with the median
            X_filled = X.fillna(X.median())

            # Normalize the data
            X_scaled = self.scaler.fit_transform(X_filled)

            # Handle the case where the standard deviation is 0 (to avoid division by zero)
            zero_std_mask = self.scaler.scale_ == 0
            if np.any(zero_std_mask):
                X_scaled[:, zero_std_mask] = 0  # Set all values to 0 if the standard deviation is 0

            # Convert back to a DataFrame to retain column names
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


            # Step 2: Remove low-variance features
            selector = VarianceThreshold(threshold=self.tau_var)
            X_reduced = selector.fit_transform(X_scaled_df)
            X_reduced_df = pd.DataFrame(X_reduced, columns=X_scaled_df.columns[selector.get_support()])
            removed_low_variance = list(set(X_scaled_df.columns) - set(X_reduced_df.columns))
            self.removed_features.extend(removed_low_variance)

            # Step 3: Compute the Spearman correlation matrix & remove redundant features
            correlation_matrix = X_reduced_df.corr(method='spearman').abs()
            to_drop = set()

            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > self.tau_corr:
                        to_drop.add(correlation_matrix.columns[j])

            X_reduced = X_reduced_df.drop(columns=list(to_drop), axis=1)
            self.removed_features.extend(list(to_drop))

            # Step 4: Compute Mutual Information (MI) to select important features
            retry_count = 0
            min_features = 10
            top_rho = self.top_rho

            while retry_count < self.max_retries:
                mi_scores = mutual_info_classif(X_reduced, y, discrete_features='auto')
                num_features = max(min_features, int(len(mi_scores) * top_rho))  
                selected_indices = np.argsort(mi_scores)[-num_features:]
                self.selected_features = X_reduced.columns[selected_indices]

                if len(self.selected_features) >= min_features:
                    break  # Enough features selected, exit loop
                else:
                    print(f"Too few features selected ({len(self.selected_features)}). Retrying with higher top_rho...")
                    top_rho = min(top_rho + 0.05, 1.0)  # Increase retention ratio
                    retry_count += 1

            if len(self.selected_features) < min_features:
                print("Warning: Could not retain at least 10 features after maximum retries.")

            # Store the list of features removed due to low MI
            self.removed_features.extend([col for col in X_reduced.columns if col not in self.selected_features])

        except Exception as e:
            print(f"Error during AFSP fitting: {e}")

    def transform(self, X):
        """
        Transform the dataset based on the selected features.
        :param X: Input dataset to transform.
        :return: Reduced DataFrame containing only important features.
        """
        try:
            if self.selected_features is None:
                raise ValueError("AFSP has not been trained. Call fit() first.")

            return X[self.selected_features].copy()
        except Exception as e:
            print(f"Error during AFSP transformation: {e}")
            return None

    def fit_transform(self, X, y):
        """
        Train AFSP and return the reduced dataset.
        :param X: Input data.
        :param y: Corresponding labels.
        :return: Reduced DataFrame.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_info(self):
        """
        Provide information about the feature selection process.
        """
        print("\nFeature Selection Information:")
        print(f"Initial number of features: {self.initial_features}")
        print(f"Number of features after variance filtering: {self.features_after_variance}")
        print(f"Number of features after correlation filtering: {self.features_after_correlation}")
        print(f"Number of features retained after MI selection: {len(self.selected_features) if self.selected_features is not None else 0}")
        print(f"Total number of removed features: {len(self.removed_features)}")
        print(f"List of removed features: {self.removed_features}")
