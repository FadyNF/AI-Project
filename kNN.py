import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load datasets
app_record = hf.load_file('datasets/application_record.csv')
credit_record = hf.load_file('datasets/credit_record.csv')

# ---------------- Data Preprocessing ---------------- #
print("Data Preprocessing")

app_record['OCCUPATION_TYPE'] = app_record['OCCUPATION_TYPE'].fillna('Unknown')
app_record['AGE'] = abs(app_record['DAYS_BIRTH']) // 365
app_record['DAYS_EMPLOYED'] = app_record['DAYS_EMPLOYED'].replace(365243, 0)
app_record = app_record.drop(columns=['FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_EMAIL', 'CODE_GENDER', 'DAYS_BIRTH'], errors='ignore')

label_encoder = LabelEncoder()
categorical_cols = app_record.select_dtypes(include=['object']).columns
for col in categorical_cols:
    app_record[col] = label_encoder.fit_transform(app_record[col])

scaler = StandardScaler()
numeric_cols = ['AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AGE', 'DAYS_EMPLOYED']
app_record[numeric_cols] = scaler.fit_transform(app_record[numeric_cols])

status_mapping = {'C': 0, 'X': 0, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}
credit_record['STATUS'] = credit_record['STATUS'].map(status_mapping)
credit_record = credit_record.groupby('ID', as_index=False)['STATUS'].max()

merged_dataset = app_record.merge(credit_record, on='ID', how='inner')

# ---------------- Feature Selection ---------------- #
X = merged_dataset.drop(columns=['ID', 'STATUS'], errors='ignore')
y = merged_dataset['STATUS']

# ---------------- kNN Model Training and Evaluation ---------------- #

# ---------------- Data Preparation ---------------- #
# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, y)

# Splitting the balanced dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)

# ---------------- Hyperparameter Tuning for kNN ---------------- #
# Define the model
knn = KNeighborsClassifier()

# Define hyperparameters to tune
param_grid = {
    'n_neighbors': [3, 5, 7],  # Fewer neighbors to reduce computation time
    'weights': ['uniform', 'distance'],  # Weighting strategy
    'metric': ['euclidean', 'manhattan']  # Distance metrics
}

# GridSearchCV to find the best parameters with fewer folds for faster performance
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Best parameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# ---------------- Train the kNN Model ---------------- #
# Train with the best parameters
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, Y_train)

# ---------------- Evaluate the Model ---------------- #
Y_pred = best_knn.predict(X_test)

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

# Classification Report
print("Classification Report:")
print(classification_report(Y_test, Y_pred))

# ---------------- Visualize Results ---------------- #
sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
