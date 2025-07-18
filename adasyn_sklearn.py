import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

def split_data(data, label_column='Label', test_size=0.2, random_state=42):
    """
    Step 1: Split the dataset into training and testing sets using stratified sampling.

    Parameters:
        data (pd.DataFrame): Input dataset including features and label.
        label_column (str): Name of the label column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Feature and label sets for training and testing.
    """
    X = data.drop(columns=[label_column])
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def balance_data_adasyn(X_train, y_train, ratio=0.5, min_samples_to_generate=100, n_neighbors=5):
    """
    Step 2–14: Perform adaptive oversampling using ADASYN to balance minority classes.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training label set.
        ratio (float): Desired ratio of minority class size to majority class size.
        min_samples_to_generate (int): Minimum number of samples to trigger oversampling.
        n_neighbors (int): Number of neighbors used in ADASYN.

    Returns:
        X_train_balanced, y_train_balanced: The balanced training feature and label sets.
    """
    class_counts = y_train.value_counts()
    majority_class = class_counts.idxmax()
    max_class_count = class_counts.max()

    # Step 3–4: Identify target number of samples for each class
    sampling_strategy = {}
    for cls, count in class_counts.items():
        target_count = int(max_class_count * ratio)
        samples_needed = target_count - count
        if samples_needed >= min_samples_to_generate:
            sampling_strategy[cls] = target_count  # Step 5–6

    # Step 7: If no class meets the criteria, skip balancing
    if not sampling_strategy:
        return X_train, y_train

    # Step 8–9: Adjust n_neighbors if any class has fewer samples than required
    min_class_count = class_counts.min()
    if min_class_count <= n_neighbors:
        n_neighbors = max(1, min_class_count - 1)

    # Step 10–13: Perform ADASYN oversampling
    try:
        adasyn = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=n_neighbors, random_state=42)
        X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)
    except ValueError:
        # Step 14: In case of failure, return original data
        return X_train, y_train

    return X_train_balanced, y_train_balanced
