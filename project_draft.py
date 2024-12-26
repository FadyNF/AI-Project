import numpy as np
import matplotlib as plt
import seaborn as sns
import pandas as pd

app_record = pd.read_csv('datasets/application_record.csv')
credit_record = pd.read_csv('datasets/credit_record.csv')


# Data Exploration and Preprocessing
print("Data Exploration and Preprocessing")
print('First Rows of the Datasets')
print(app_record.head()) #Print the first 5 rows of the dataset
print(credit_record.head()) #Print the first 5 rows of the dataset
print('---------------------------------')

print(app_record.info()) # Print the information of the dataset
print(credit_record.info()) # Print the information of the dataset
print('---------------------------------')

print(app_record.isnull().sum()) # Print the null values of the dataset
print(credit_record.isnull().sum()) # Print the null values of the dataset
print('---------------------------------')

print(app_record.describe()) # Print the description of the dataset
print(credit_record.describe()) # Print the description of the dataset
print('---------------------------------')


# print(credit_record['STATUS'].value_counts()) # Print the value counts of the dataset