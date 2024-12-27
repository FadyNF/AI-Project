import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier 
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from deap import base, creator, tools, algorithms

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

# Remove irrelevant columns
irrelevant_columns = ['FLAG_MOBIL', 'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_EMAIL']
merged_data.drop(columns=irrelevant_columns, errors='ignore', inplace=True)

# Handle extreme values of DAYS_EMPLOYED
merged_data['DAYS_EMPLOYED'] = merged_data['DAYS_EMPLOYED'].replace(365243, 0)

# Remove outliers for relevant columns
columns_to_check_for_outliers = ['AMT_INCOME_TOTAL', 'AGE', 'DAYS_EMPLOYED', 'CHILD_TO_FAMILY_RATIO']
merged_data = hf.remove_outliers(merged_data, columns_to_check_for_outliers)

# Drop ID column as it's not required for analysis
merged_data.drop(columns=['ID'], inplace=True)

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
# sns.countplot(x='STATUS', data=merged_data)
# plt.title("Class Distribution After Preprocessing")
# plt.show()

print("Preprocessing Complete")

# ---------------- Feature Selection ---------------- #
print("Feature Selection")
X = merged_data.drop(columns=['STATUS'])
Y = merged_data['STATUS']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

POP_SIZE = 10  # Population size
GENS = 10  # Number of generations
CXPB = 0.7  # Crossover probability
MUTPB = 0.2  # Mutation probability

# Fitness function for the Genetic Algorithm
def fitness_function(individual):
    selected_features = [index for index, value in enumerate(individual) if value == 1]
    if len(selected_features) == 0:  # Avoid empty feature selection
        return 0,
    X_selected = X.iloc[:, selected_features]

    # Split data dynamically into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_selected, Y, test_size=0.3, random_state=42)

    # Train a Decision Tree classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, Y_train)

    # Evaluate accuracy on the validation set
    Y_pred = dt_model.predict(X_val)
    accuracy = accuracy_score(Y_val, Y_pred)

    return accuracy,

# Create types for DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize toolbox
toolbox = base.Toolbox()

# Attribute generator for binary representation (0 or 1)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])  # n=number of features
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)

# Initialize population
population = toolbox.population(n=POP_SIZE)

# Evaluate initial population
fitnesses = map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Begin evolution
for gen in range(GENS):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:  # Crossover probability
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:  # Mutation probability
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate invalid individuals
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring
    best_ind = tools.selBest(population, k=1)[0]
    print(f"Generation {gen}: Best Fitness = {best_ind.fitness.values[0]}")

# Final selected features
selected_features = [index for index, value in enumerate(best_ind) if value == 1]
print("Selected Features:", X.columns[selected_features].tolist())


#Decision Tree Model
# Train the final model using selected features
print("Training Decision Tree Classifier...")
X_selected = X.iloc[:, selected_features]
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

# Make predictions and evaluate the model
Y_pred = model.predict(X_test)
print("Decision Tree Classification Report:")
print(classification_report(Y_test, Y_pred))



from sklearn.neural_network import MLPClassifier

# Train the MLP model using the selected features
print("Training MLP Classifier...")
X_selected = X.iloc[:, selected_features]
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.3, random_state=42)

# Initialize the MLP model (you can adjust parameters as needed)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=150, random_state=42)
mlp_model.fit(X_train, Y_train)

# Make predictions and evaluate the MLP model
Y_pred_mlp = mlp_model.predict(X_test)

# Evaluate the MLP model
print("MLP Classification Report:")
print(classification_report(Y_test, Y_pred_mlp))
