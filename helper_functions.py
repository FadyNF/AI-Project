
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


def remove_columns_above_threshold(df, threshold=50):
    # Calculate the null percentage for each column
    null_percentage = (df.isnull().sum() / df.shape[0]) * 100

    # Identify columns with null percentage above the threshold
    cols_to_remove = null_percentage[null_percentage > threshold].index

    # Drop those columns
    df_cleaned = df.drop(columns=cols_to_remove)

    # Print the columns that were removed
    if len(cols_to_remove) > 0:
        print(f"Columns removed due to null percentage > {threshold}: {list(cols_to_remove)}")
    else:
        print(f"No columns removed (no column exceeded {threshold}% null values)")

    return df_cleaned
