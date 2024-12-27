import os
import pandas as pd

def load_file(file_path):
    if os.path.exists(file_path):
            return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"File {file_path} not found.")
    
    
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def handle_missing_data(df, column_strategies):
    for column, strategy in column_strategies.items():
        if strategy == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif strategy == 'median':
            df[column] = df[column].fillna(df[column].median())
        elif strategy == 'mode':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(strategy)
    return df   


