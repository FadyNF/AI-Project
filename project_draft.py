import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split, cross_val_score
from deap import base, creator, tools, algorithms
from concurrent.futures import ThreadPoolExecutor


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
# sns.countplot(x='STATUS', data=credit_record)
# plt.title("Class Distribution of STATUS")
# plt.show()

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

# Remove outliers from selected columns
columns_to_check_for_outliers = ['AMT_INCOME_TOTAL', 'AGE', 'DAYS_EMPLOYED', 'CNT_CHILDREN']
merged_dataset = hf.remove_outliers(merged_dataset, columns_to_check_for_outliers)

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


# ---------------- Decision Tree Classifier ---------------- #
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Split the data into training and testing sets using the selected features
X_selected = X[selected_columns]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.30, random_state=42)

# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree Classifier
dt_model.fit(X_train, y_train)

# Predict on the test set
dt_y_pred = dt_model.predict(X_test)

# Print the classification report for Decision Tree Classifier
print("Classification Report for Decision Tree:")
print(classification_report(y_test, dt_y_pred))
# ---------------- MLP Classifier ---------------- #
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Split the data into training and testing sets using the selected features
X_selected = X[selected_columns]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.30, random_state=42)

# Initialize the MLP Classifier
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the MLP Classifier
mlp_model.fit(X_train, y_train)

# Predict on the test set
mlp_y_pred = mlp_model.predict(X_test)

# Print the classification report for MLP Classifier
print("Classification Report for MLP Classifier:")
print(classification_report(y_test, mlp_y_pred))


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# ---------------- Random Search for Decision Tree ---------------- #
print("Random Search for Decision Tree")

# Define the hyperparameter grid for Decision Tree
dt_param_dist = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'max_features': [None, 'sqrt', 'log2'],
}

# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Perform RandomizedSearchCV
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
print("Best Parameters for Decision Tree:", dt_random_search.best_params_)
print("Best Cross-Validation Score:", dt_random_search.best_score_)

# Evaluate on the test set
dt_best_model = dt_random_search.best_estimator_
dt_y_pred = dt_best_model.predict(X_test)
print("Classification Report for Optimized Decision Tree:")
print(classification_report(y_test, dt_y_pred))


from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from scipy.stats import uniform

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from scipy.stats import uniform

# # ---------------- NOT WORKING Random Search for MLP ---------------- #
# print("Random Search for MLP Classifier")

# # Optimized parameter grid
# mlp_param_dist = {
#     'hidden_layer_sizes': [(50,), (100,), (100, 50)],
#     'activation': ['relu', 'tanh'],  # Simplified activations
#     'solver': ['adam'],             # Focus on 'adam'
#     'alpha': uniform(0.0001, 0.01), # Narrow regularization range
#     'learning_rate': ['constant'],  # Fixed learning rate schedule
#     'max_iter': [1500],             # Single iteration limit
# }

# # Subset the training data
# X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=42)

# # Initialize RandomizedSearchCV
# mlp_random_search = RandomizedSearchCV(
#     estimator=MLPClassifier(random_state=42),
#     param_distributions=mlp_param_dist,
#     n_iter=20,        # Reduced iterations
#     scoring='accuracy',
#     n_jobs=2,         # Limited parallelism
#     cv=3,             # Fewer cross-validation folds
#     random_state=42
# )

# # Fit the random search to the subset of the training data
# mlp_random_search.fit(X_train_subset, y_train_subset)

# # Best parameters and performance
# print("Best Parameters for MLP Classifier:", mlp_random_search.best_params_)
# print("Best Cross-Validation Score:", mlp_random_search.best_score_)

# # Use the best model to predict on the test set
# mlp_best_model = mlp_random_search.best_estimator_
# mlp_y_pred = mlp_best_model.predict(X_test)

# # Print the classification report for the optimized MLP Classifier
# print("Classification Report for Optimized MLP Classifier:")
# print(classification_report(y_test, mlp_y_pred))



from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# ---------------- Grid Search for MLP ---------------- #
print("Grid Search for MLP Classifier")

# Define the parameter grid
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # Few reasonable configurations
    'activation': ['relu', 'tanh'],                  # Common activations
    'alpha': [0.0001, 0.001, 0.01],                  # Regularization strength
    'solver': ['adam'],                              # Focus on 'adam'
    'max_iter': [1000],                              # Fixed iteration limit
}

# Subset the training data for quicker execution
X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, train_size=0.3, random_state=42)

# Initialize GridSearchCV
mlp_grid_search = GridSearchCV(
    estimator=MLPClassifier(random_state=42),
    param_grid=mlp_param_grid,
    scoring='accuracy',
    n_jobs=-1,  # Use all available processors
    cv=3,       # 3-fold cross-validation
    verbose=2   # Show progress
)

# Fit the grid search to the subset of training data
mlp_grid_search.fit(X_train_small, y_train_small)

# Best parameters and performance
print("Best Parameters for MLP Classifier:", mlp_grid_search.best_params_)
print("Best Cross-Validation Score:", mlp_grid_search.best_score_)

# Use the best model to predict on the test set
mlp_best_model = mlp_grid_search.best_estimator_
mlp_y_pred = mlp_best_model.predict(X_test)

# Print the classification report for the best MLP Classifier
print("Classification Report for Optimized MLP Classifier:")
print(classification_report(y_test, mlp_y_pred))
