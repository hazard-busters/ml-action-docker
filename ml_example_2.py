from tslearn.datasets import UCR_UEA_datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load ECGFiveDays dataset
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("ECGFiveDays")

# Flatten time series to 2D for sklearn (n_samples, n_features)
n_samples_train, series_len = X_train.shape
X_train_flat = X_train.reshape(n_samples_train, series_len)

n_samples_test = X_test.shape[0]
X_test_flat = X_test.reshape(n_samples_test, series_len)

# Train Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_flat, y_train)

# Predict & evaluate
y_pred = rf.predict(X_test_flat)
print( "RandomForest accuracy: {accuracy_score(y_test, y_pred):.4f}")
