
import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from deap import base, creator, tools, algorithms
from concurrent.futures import ThreadPoolExecutor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform


# Load datasets
app_record = hf.load_file('datasets/application_record.csv')
credit_record = hf.load_file('datasets/credit_record.csv')

# ---------------- Data Exploration and Statistical Analysis ---------------- #
print("Data Exploration and Statistical Analysis")
print('First Rows of the Datasets')
print(app_record.head())
print(credit_record.head())
print('-----------------------------------------------------------------')

print("Dataset Information")
print(app_record.info())  # Info about application_record
print(credit_record.info())  # Info about credit_record
print('-----------------------------------------------------------------')

print("Checking for Null Values")
print(app_record.isnull().sum())
print(credit_record.isnull().sum())
print('-----------------------------------------------------------------')

print("Dataset Description")
print(app_record.describe())  # Statistical summary of application_record
print(credit_record.describe())  # Statistical summary of credit_record
print('-----------------------------------------------------------------')

print("Checking for Duplicates")
print(f"Duplicate rows in application_record: {app_record.duplicated().sum()}")
print(f"Duplicate rows in credit_record: {credit_record.duplicated().sum()}")
print('-----------------------------------------------------------------')

print("Value Counts for 'STATUS'")
print(credit_record['STATUS'].value_counts())  # Check distribution of STATUS values

# Visualize class distribution in credit_record
# sns.countplot(x='STATUS', data=credit_record)
# plt.title("Class Distribution of STATUS")
# plt.show()

# ---------------- Data Preprocessing ---------------- #
print("Data Preprocessing")


print("---------------------------")
print("Removing columns with missing values above the threshold")
threshold = 50  # Set threshold for column removal
app_record = hf.remove_columns_above_threshold(app_record, threshold)
credit_record = hf.remove_columns_above_threshold(credit_record, threshold)
print("-----------------------------------------------------------------")

print("Filling missing values in OCCUPATION_TYPE with 'Unknown'")
app_record['OCCUPATION_TYPE'] = app_record['OCCUPATION_TYPE'].fillna('Unknown')
print("-----------------------------------------------------------------")

print("Calculating AGE from DAYS_BIRTH")
app_record['AGE'] = abs(app_record['DAYS_BIRTH']) // 365
print("-----------------------------------------------------------------")

print("Replacing anomalous values in DAYS_EMPLOYED")
app_record['DAYS_EMPLOYED'] = app_record['DAYS_EMPLOYED'].replace(365243, 0)
print("-----------------------------------------------------------------")

print("Dropping unnecessary columns")
app_record = app_record.drop(columns=['FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_EMAIL', 'CODE_GENDER', 'DAYS_BIRTH'], errors='ignore')
print("-----------------------------------------------------------------")

print("Encoding categorical columns")
label_encoder = LabelEncoder()
categorical_cols = app_record.select_dtypes(include=['object']).columns
for col in categorical_cols:
    app_record[col] = label_encoder.fit_transform(app_record[col])
print("-----------------------------------------------------------------")

print("Scaling numeric columns")
scaler = StandardScaler()
numeric_cols = ['AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AGE', 'DAYS_EMPLOYED']
app_record[numeric_cols] = scaler.fit_transform(app_record[numeric_cols])
print("-----------------------------------------------------------------")

print("Mapping credit status to binary values")
status_mapping = {'C': 0, 'X': 0, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1}
credit_record['STATUS'] = credit_record['STATUS'].map(status_mapping)
credit_record = credit_record.groupby('ID', as_index=False)['STATUS'].max()
print("-----------------------------------------------------------------")

print("Merging application and credit records")
merged_dataset = app_record.merge(credit_record, on='ID', how='inner')
print("-----------------------------------------------------------------")

print("Removing outliers from selected columns")
columns_to_check_for_outliers = ['AMT_INCOME_TOTAL', 'AGE', 'DAYS_EMPLOYED', 'CNT_CHILDREN']
merged_dataset = hf.remove_outliers(merged_dataset, columns_to_check_for_outliers)
print("-----------------------------------------------------------------")

# Check class imbalance after preprocessing
# sns.countplot(x='STATUS', data=merged_data)
# plt.title("Class Distribution After Preprocessing")
# plt.show()

print("Preprocessing Complete")

# ---------------- Feature Selection ---------------- #
# Split features and target
X = merged_dataset.drop(columns=['ID', 'STATUS'], errors='ignore')
y = merged_dataset['STATUS']

# Genetic Algorithm Functions
def evaluate(individual):
    selected_features = [f for i, f in enumerate(X.columns) if individual[i] == 1]
    if len(selected_features) == 0:
        return 0,

    X_selected = X[selected_features]
    X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.30, random_state=42)
    X_val, _, y_val, _ = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred),


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Parallelized Evaluation
def evaluate_population(population):
    with ThreadPoolExecutor() as executor:
        fitnesses = list(executor.map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

def feature_selection_ga():
    population = toolbox.population(n=20)
    generations = 20
    cx_prob = 0.7
    mut_prob = 0.2

    for gen in range(generations):
        start_time = time.time()
        
        print(f"Generation {gen}:")
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        evaluate_population(invalid_individuals)

        population[:] = offspring
        
        best_ind = tools.selBest(population, 1)[0]
        print(f"  Best Fitness: {best_ind.fitness.values[0]}")
        print(f"  Time Taken: {time.time() - start_time:.2f} seconds")

    return tools.selBest(population, 1)[0]

best_features = feature_selection_ga()
selected_columns = [f for i, f in enumerate(X.columns) if best_features[i] == 1]
print("Best Selected Features:", selected_columns)

# ---------------- Data Splitting ---------------- #
print("-----------------------------------------------------------------")
print("Splitting data into train, validation, and test sets")
X_selected = X[selected_columns]

# Initial split: 85% train+validation and 15% test
print("-----------------------------------------------------------------")
print("Initial split: 85% train+validation and 15% test")
X_train_val, X_test, y_train_val, y_test = train_test_split(X_selected, y, test_size=0.15, random_state=42)

# Split train+validation into 70% train and 15% validation
print("-----------------------------------------------------------------")
print("Splitting train+validation into 70% train and 15% validation")
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)  # 0.1765 * 85% = 15%

# ---------------- Decision Tree Classifier ---------------- #
print("-----------------------------------------------------------------")
print("Training Decision Tree Classifier")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate on validation data for fitness function
print("-----------------------------------------------------------------")
print("Evaluating Decision Tree on validation data")
val_accuracy = dt_model.score(X_val, y_val)
print(f"Validation Accuracy for Decision Tree: {val_accuracy:.4f}")

# Final evaluation on test data
print("-----------------------------------------------------------------")
print("Final evaluation of Decision Tree on test data")
dt_y_pred = dt_model.predict(X_test)
print("Classification Report for Decision Tree:")
print(classification_report(y_test, dt_y_pred))

# ---------------- Random Search for Decision Tree ---------------- #
print("-----------------------------------------------------------------")
print("Random Search for Decision Tree")

# Define the hyperparameter grid for Decision Tree
print("Defining hyperparameter grid for Decision Tree")
dt_param_dist = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': [None, 'sqrt', 'log2'],
}

# Initialize the Decision Tree Classifier
print("-----------------------------------------------------------------")
print("Initializing Decision Tree Classifier")
dt_model = DecisionTreeClassifier(random_state=42)

# Perform RandomizedSearchCV
print("-----------------------------------------------------------------")
print("Performing RandomizedSearchCV for Decision Tree")
dt_random_search = RandomizedSearchCV(
    dt_model,
    param_distributions=dt_param_dist,
    n_iter=50,
    scoring='accuracy',
    n_jobs=-1,
    cv=5,
    random_state=42
)

dt_random_search.fit(X_train, y_train)

# Best parameters and performance
print("-----------------------------------------------------------------")
print("Best Parameters for Decision Tree:", dt_random_search.best_params_)
print("Best Cross-Validation Score:", dt_random_search.best_score_)

# Evaluate on the test set
print("-----------------------------------------------------------------")
print("Evaluating optimized Decision Tree on test data")
dt_best_model = dt_random_search.best_estimator_
dt_y_pred = dt_best_model.predict(X_test)
print("-----------------------------------------------------------------")
print("Classification Report for Optimized Decision Tree:")
print(classification_report(y_test, dt_y_pred))

# ---------------- MLP Classifier ---------------- #
print("-----------------------------------------------------------------")
print("Training MLP Classifier")
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

# Evaluate on validation data for fitness function
print("-----------------------------------------------------------------")
print("Evaluating MLP on validation data")
val_accuracy = mlp_model.score(X_val, y_val)
print(f"Validation Accuracy for MLP: {val_accuracy:.4f}")

# Final evaluation on test data
print("-----------------------------------------------------------------")
print("Final evaluation of MLP on test data")
mlp_y_pred = mlp_model.predict(X_test)
print("Classification Report for MLP Classifier:")
print(classification_report(y_test, mlp_y_pred, zero_division=1))

# ---------------- kNN Classifier with SMOTE ---------------- #
print("-----------------------------------------------------------------")
print("Applying SMOTE to handle class imbalance")
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_selected, y)

# Define kNN classifier and hyperparameter grid
print("-----------------------------------------------------------------")
print("Defining kNN classifier and hyperparameter grid")
knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7],  
    'weights': ['uniform', 'distance'], 
    'metric': ['euclidean', 'manhattan']  
}

# Perform Grid Search with resampled data
print("-----------------------------------------------------------------")
print("Performing Grid Search for kNN with resampled data")
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
grid_search.fit(X_resampled, Y_resampled)

# Output best parameters and score
print("-----------------------------------------------------------------")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

# Best model from grid search
print("-----------------------------------------------------------------")
print("Evaluating best kNN model on original test set")
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

# Display results
print("-----------------------------------------------------------------")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("-----------------------------------------------------------------")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Visualize confusion matrix
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()