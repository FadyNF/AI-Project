import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load datasets
app_record = hf.load_file('datasets/application_record.csv')
credit_record = hf.load_file('datasets/credit_record.csv')

# ---------------- Data Exploration and Statistical Analysis ---------------- #
print("Data Exploration and Statistical Analysis")
print('First Rows of the Datasets')
print(app_record.head())
print(credit_record.head())
print('---------------------------------')

print("Dataset Information")
print(app_record.info())  # Info about application_record
print(credit_record.info())  # Info about credit_record
print('---------------------------------')

print("Checking for Null Values")
print(app_record.isnull().sum())
print(credit_record.isnull().sum())
print('---------------------------------')

print("Dataset Description")
print(app_record.describe())  # Statistical summary of application_record
print(credit_record.describe())  # Statistical summary of credit_record
print('---------------------------------')

print("Checking for Duplicates")
print(f"Duplicate rows in application_record: {app_record.duplicated().sum()}")
print(f"Duplicate rows in credit_record: {credit_record.duplicated().sum()}")
print('---------------------------------')

print("Value Counts for 'STATUS'")
print(credit_record['STATUS'].value_counts())  # Check distribution of STATUS values

# Visualize class distribution in credit_record
sns.countplot(x='STATUS', data=credit_record)
plt.title("Class Distribution of STATUS")
plt.show()

# ---------------- Data Preprocessing ---------------- #
print("Data Preprocessing")

# Map 'STATUS' values to binary classification
status_mapping = {'C': 0, 'X': 0, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}
credit_record['STATUS'] = credit_record['STATUS'].map(status_mapping)

# Merge the datasets on 'ID'
merged_data = app_record.merge(credit_record, on='ID', how='inner')
print(f"Merged dataset shape: {merged_data.shape}")

# Handle missing values
missing_data_strategies = {
    'OCCUPATION_TYPE': 'Unknown',  # fill with 'Unknown'
}
merged_data = hf.handle_missing_data(merged_data, missing_data_strategies)

# Transform 'DAYS_BIRTH' to 'AGE' before removing outliers
merged_data['AGE'] = abs(merged_data['DAYS_BIRTH']) / 365
merged_data.drop(columns=['DAYS_BIRTH'], inplace=True)

# Create child-to-family ratio
merged_data['CHILD_TO_FAMILY_RATIO'] = merged_data['CNT_CHILDREN'] / (merged_data['CNT_FAM_MEMBERS'] + 1)

# Remove outliers for relevant columns
columns_to_check_for_outliers = ['AMT_INCOME_TOTAL', 'AGE', 'DAYS_EMPLOYED', 'CHILD_TO_FAMILY_RATIO']
merged_data = hf.remove_outliers(merged_data, columns_to_check_for_outliers)

# Drop irrelevant columns
merged_data.drop(columns=['ID'], inplace=True)

# Transform DAYS_EMPLOYED
merged_data['DAYS_EMPLOYED'] = merged_data['DAYS_EMPLOYED'].apply(lambda x: np.nan if x > 0 else abs(x))

# Encode categorical variables
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
                    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
merged_data = pd.get_dummies(merged_data, columns=categorical_cols)

# Feature scaling for numerical columns
scaler = StandardScaler()
numerical_cols = ['AMT_INCOME_TOTAL', 'AGE', 'DAYS_EMPLOYED', 'CHILD_TO_FAMILY_RATIO']
merged_data[numerical_cols] = scaler.fit_transform(merged_data[numerical_cols])

# Check class imbalance after preprocessing
sns.countplot(x='STATUS', data=merged_data)
plt.title("Class Distribution After Preprocessing")
plt.show()

print("Preprocessing Complete")
