# -*- coding: utf-8 -*-
"""Assignment_4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hO_gND6hNEjqIGojgVvnN4n64Wc23IjZ
"""

import pandas as pd

df = pd.read_excel("Homeowner_Claim_History.xlsx", sheet_name="HOCLAIMDATA")

# Calculate the Frequency by dividing num_claims by exposure
df["Frequency"] = df["num_claims"] / df["exposure"]

# Create a function to categorize policies into frequency groups
def categorize_frequency(frequency):
    if frequency == 0:
        return 0
    elif 0 < frequency <= 1:
        return 1
    elif 1 < frequency <= 2:
        return 2
    elif 2 < frequency <= 3:
        return 3
    else:
        return 4

# Apply the categorization function to create the Frequency Group column
df["Frequency Group"] = df["Frequency"].apply(categorize_frequency)

# Now, the DataFrame df contains the Frequency and Frequency Group information

from sklearn.model_selection import train_test_split

def categorize_partition(policy_identifier):
    if policy_identifier[0] in ['A', 'G', 'P']:
        return "Training"
    else:
        return "Testing"

# Apply the partition categorization function to create the "Partition" column
df["Partition"] = df["policy"].apply(categorize_partition)

# Split the data into training and testing partitions
train_data = df[df["Partition"] == "Training"]
test_data = df[df["Partition"] == "Testing"]

# Drop the "Partition" column from both training and testing data
train_data = train_data.drop(columns=["Partition"])
test_data = test_data.drop(columns=["Partition"])

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import numpy as np
import statsmodels.api as sm




# Drop rows with missing values in the "Frequency Group" column
df.dropna(subset=["Frequency Group"], inplace=True)

# Define the categorical predictors
categorical_features = ["f_aoi_tier", "f_fire_alarm_type", "f_marital", "f_mile_fire_station", "f_primary_age_tier", "f_primary_gender", "f_residence_location"]

# Split the data into training and testing partitions
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Define a pipeline for preprocessing and model training
pipeline = Pipeline([
    ("encoder", OneHotEncoder(sparse=False, drop='first')),
    ("feature_selection", SelectKBest(chi2, k=10)),  # Adjust k as needed
    ("classifier", LogisticRegression(multi_class='multinomial', solver='newton-cg'))
])

# Define hyperparameters to search
param_grid = {
    'feature_selection__k': [10, 20, 30],  # Number of top features to select
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],  # Regularization parameter
    'classifier__max_iter': [100, 200, 300]  # Maximum number of iterations
}

# Create StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare the training and testing data
X_train = train_data[train_data.columns.intersection(categorical_features)].values
X_test = test_data[test_data.columns.intersection(categorical_features)].values

y_train = train_data["Frequency Group"]
y_test = test_data["Frequency Group"]

# Create a custom scoring function for Grid Search using F1 score
scoring = make_scorer(f1_score, average='weighted')

# Create a dictionary to store AIC and BIC values for each hyperparameter combination
aic_bic_scores = {}

# Perform hyperparameter tuning using Grid Search
for k in param_grid['feature_selection__k']:
    for C in param_grid['classifier__C']:
        for max_iter in param_grid['classifier__max_iter']:
            # Fit the model with the selected hyperparameters
            model = pipeline
            model.set_params(feature_selection__k=k, classifier__C=C, classifier__max_iter=max_iter)

            # Encode categorical features and convert them to a numerical format
            X_train_encoded = model.named_steps['encoder'].fit_transform(X_train)

            # Fit the logistic regression model
            model.named_steps['classifier'].fit(X_train_encoded, y_train)

            # Convert the target variable to integers
            y_train = y_train.astype(int)

            # Calculate AIC and BIC on the training partition
            n = len(X_train_encoded)
            log_likelihood = -model.named_steps['classifier'].score(X_train_encoded, y_train)
            k_params = len(model.named_steps['classifier'].coef_[0]) + 1  # Number of parameters including intercept
            aic = 2 * k_params - 2 * log_likelihood
            bic = n * np.log(n) - 2 * log_likelihood

            # Store AIC and BIC scores for this hyperparameter combination
            aic_bic_scores[(k, C, max_iter)] = (aic, bic)

# Get the hyperparameters with the lowest AIC and BIC
best_aic_hyperparams = min(aic_bic_scores, key=lambda x: aic_bic_scores[x][0])
best_bic_hyperparams = min(aic_bic_scores, key=lambda x: aic_bic_scores[x][1])

# Fit the model with the best hyperparameters based on AIC or BIC
best_hyperparams = best_aic_hyperparams if aic_bic_scores[best_aic_hyperparams][0] < aic_bic_scores[best_bic_hyperparams][0] else best_bic_hyperparams
best_model = pipeline
best_model.set_params(feature_selection__k=best_hyperparams[0], classifier__C=best_hyperparams[1], classifier__max_iter=best_hyperparams[2])

# Encode categorical features in the test data
X_test_encoded = best_model.named_steps['encoder'].transform(X_test)

# Make predictions on the test partition
y_pred = best_model.named_steps['classifier'].predict(X_test_encoded)

# Calculate accuracy and F1 score on the testing partition
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Best Model Hyperparameters (Based on AIC/BIC):", best_hyperparams)
print("Accuracy (Testing):", accuracy)
print("F1 Score (Testing):", f1)

#**********************Question 1.a***************************
# Calculate the counts of policies in each Frequency Group for the Training partition
training_group_counts = train_data["Frequency Group"].value_counts().sort_index()

# Calculate the counts of policies in each Frequency Group for the Testing partition
testing_group_counts = test_data["Frequency Group"].value_counts().sort_index()

print("Training Partition - Frequency Group Counts:")
print(training_group_counts)

print("\nTesting Partition - Frequency Group Counts:")
print(testing_group_counts)

#***********************Question 1.b******************
#***********************Question 1.c******************
# Calculate the lowest AIC value on the Training partition
lowest_aic = min(aic_bic_scores, key=lambda x: aic_bic_scores[x][0])
lowest_aic_value = aic_bic_scores[lowest_aic][0]

# Calculate the lowest BIC value on the Training partition
lowest_bic = min(aic_bic_scores, key=lambda x: aic_bic_scores[x][1])
lowest_bic_value = aic_bic_scores[lowest_bic][1]

print("Lowest AIC on Training Partition:", lowest_aic_value)
print("Model producing Lowest AIC:", f"K={lowest_aic[0]}, C={lowest_aic[1]}, Max Iter={lowest_aic[2]}")

print("\nLowest BIC on Training Partition:", lowest_bic_value)
print("Model producing Lowest BIC:", f"K={lowest_bic[0]}, C={lowest_bic[1]}, Max Iter={lowest_bic[2]}")

#
##***********************Question 1.d******************
# Create a dictionary to store Accuracy values for each model
accuracy_scores = {}

# Iterate through models and calculate accuracy on the Testing partition
for k in param_grid['feature_selection__k']:
    for C in param_grid['classifier__C']:
        for max_iter in param_grid['classifier__max_iter']:
            model = pipeline
            model.set_params(feature_selection__k=k, classifier__C=C, classifier__max_iter=max_iter)

            # Fit the model on the training data
            X_train_encoded = model.named_steps['encoder'].fit_transform(X_train)
            model.named_steps['classifier'].fit(X_train_encoded, y_train)

            # Encode categorical features in the test data
            X_test_encoded = model.named_steps['encoder'].transform(X_test)

            # Make predictions on the test partition
            y_pred = model.named_steps['classifier'].predict(X_test_encoded)

            # Calculate accuracy on the Testing partition
            accuracy = accuracy_score(y_test, y_pred)

            # Store accuracy score for this model
            accuracy_scores[(k, C, max_iter)] = accuracy

# Calculate the highest Accuracy value on the Testing partition
highest_accuracy = max(accuracy_scores, key=lambda x: accuracy_scores[x])

print("Highest Accuracy on Testing Partition:", accuracy_scores[highest_accuracy])
print("Model producing Highest Accuracy:", f"K={highest_accuracy[0]}, C={highest_accuracy[1]}, Max Iter={highest_accuracy[2]}")

#************************Question 1.e*************
from sklearn.metrics import mean_squared_error

# Create a dictionary to store RASE values for each model
rase_scores = {}

# Iterate through models and calculate RASE on the Testing partition
for k in param_grid['feature_selection__k']:
    for C in param_grid['classifier__C']:
        for max_iter in param_grid['classifier__max_iter']:
            model = pipeline
            model.set_params(feature_selection__k=k, classifier__C=C, classifier__max_iter=max_iter)

            # Fit the model on the training data
            X_train_encoded = model.named_steps['encoder'].fit_transform(X_train)
            model.named_steps['classifier'].fit(X_train_encoded, y_train)

            # Encode categorical features in the test data
            X_test_encoded = model.named_steps['encoder'].transform(X_test)

            # Make predictions on the test partition
            y_pred = model.named_steps['classifier'].predict(X_test_encoded)

            # Calculate RASE on the Testing partition
            rase = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store RASE score for this model
            rase_scores[(k, C, max_iter)] = rase

# Calculate the lowest RASE value on the Testing partition
lowest_rase = min(rase_scores, key=lambda x: rase_scores[x])

print("Lowest RASE on Testing Partition:", rase_scores[lowest_rase])
print("Model producing Lowest RASE:", f"K={lowest_rase[0]}, C={lowest_rase[1]}, Max Iter={lowest_rase[2]}")

# Import necessary libraries
#*******************************************Question 2********************************************

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np


train_data = pd.read_csv("WineQuality_Train.csv")
test_data = pd.read_csv("WineQuality_Test.csv")

features = ["alcohol", "citric_acid", "free_sulfur_dioxide", "residual_sugar", "sulphates"]
X_train = train_data[features]
y_train = train_data["quality_grp"]
X_test = test_data[features]

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=2023484)

# Perform grid search to select the best network structure
param_grid = {
    'activation': ['logistic', 'identity', 'relu'],  # Hyperbolic Tangent, Identity, and ReLU activation functions
    'hidden_layer_sizes': [(neurons,)*layers for neurons in range(2, 11, 2) for layers in range(1, 11)],
}

grid_search = GridSearchCV(MLPClassifier(max_iter=10000, random_state=2023484), param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train_split, y_train_split)

# Get the best neural network model from grid search
best_model = grid_search.best_estimator_

# Train the best model on the full training data
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
c = np.mean(y_train)

threshold = 1.5 * c
y_pred_thresholded = (y_pred >= threshold).astype(int)

# Calculate accuracy on the test data
accuracy = accuracy_score(y_pred_thresholded, test_data["quality_grp"])

# Print the accuracy and best model parameters
print("Best Model Parameters:", grid_search.best_params_)
print("Accuracy on Test Data:", accuracy)

#***********************************Question 2*****************************8

#Before we calculate the best functions, plot the training loss for the activation functions

import matplotlib.pyplot as plt

# Define the activation functions to test
activation_functions = ['logistic', 'identity', 'relu']
activation_function_labels = ['Hyperbolic Tangent', 'Identity', 'Linear Rectifier (ReLU)']

# Lists to store training loss values for each activation function
training_losses = []

# Loop through each activation function
for activation_function in activation_functions:
    mlp = MLPClassifier(hidden_layer_sizes=(5, 2), activation=activation_function, max_iter=10000, random_state=2023484)
    mlp.fit(X_train, y_train)
    training_losses.append(mlp.loss_curve_)

# Create a plot to visualize training loss for each activation function
plt.figure(figsize=(10, 6))
for i in range(len(activation_functions)):
    plt.plot(training_losses[i], label=activation_function_labels[i])

plt.title("Training Loss for Different Activation Functions")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()

import matplotlib.pyplot as plt

# Define the activation functions to test
activation_functions = ['logistic', 'identity', 'relu']
activation_function_labels = ['Hyperbolic Tangent', 'Identity', 'Linear Rectifier (ReLU)']

# Create a separate plot for each activation function
plt.figure(figsize=(10, 6))

for i, activation_function in enumerate(activation_functions):
    mlp = MLPClassifier(hidden_layer_sizes=(5, 2), activation=activation_function, max_iter=10000, random_state=2023484)
    mlp.fit(X_train, y_train)

    plt.subplot(1, 3, i + 1)
    plt.plot(mlp.loss_curve_)
    plt.title(f"Training Loss - {activation_function_labels[i]}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Plot the confusion matrix
conf_matrix = confusion_matrix(test_data["quality_grp"], y_pred_thresholded)
plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
plt.yticks([0, 1], ["Actual 0", "Actual 1"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', color='black')
plt.show()

# Plot a classification report
report = classification_report(test_data["quality_grp"], y_pred_thresholded)
print(report)

# 2a) Calculate the proportion of quality_grp = 1 in the training partition
proportion_quality_1 = (y_train_split == 1).sum() / len(y_train_split)

print(f"Proportion of quality_grp = 1 in the training partition: {proportion_quality_1:.2f}")

#2b) Calculate the proportion of quality_grp = 1 in the testing partition
proportion_quality_1_test = (test_data["quality_grp"] == 1).sum() / len(test_data)

print(f"Proportion of quality_grp = 1 in the testing partition: {proportion_quality_1_test:.2f}")

#2c)

import pandas as pd
import time

# Initialize lists to store the results
results = []
activation_functions = ['logistic', 'identity', 'relu']

# Loop through activation functions and grid search results
for activation_function in activation_functions:
    for neurons in range(2, 11, 2):
        for layers in range(1, 11):
            mlp = MLPClassifier(hidden_layer_sizes=(neurons,)*layers, activation=activation_function, max_iter=10000, random_state=2023484)

            # Record start time
            start_time = time.time()

            mlp.fit(X_train_split, y_train_split)

            # Calculate root mean squared error on the testing partition
            y_pred = mlp.predict(X_test)
            rmse = np.sqrt(((y_pred - test_data["quality_grp"]) ** 2).mean())

            # Calculate misclassification rate
            misclassification_rate = 1 - accuracy_score(y_pred, test_data["quality_grp"])

            # Record end time
            end_time = time.time()

            results.append([activation_function, layers, neurons, mlp.n_iter_, mlp.loss_, rmse, misclassification_rate, end_time - start_time])

# Create a DataFrame to display the results
columns = ['Activation Function', 'Number of Layers', 'Number of Neurons', 'Number of Iterations', 'Best Loss', 'Root Mean Squared Error', 'Misclassification Rate', 'Elapsed Time (s)']
result_df = pd.DataFrame(results, columns=columns)
result_df.to_excel("grid_search_results.xlsx", index=False)
# Display the results as a table
print(result_df)

#2d) Sort the results DataFrame by misclassification rate and, in case of ties, by the number of neurons
sorted_results = result_df.sort_values(by=['Misclassification Rate', 'Number of Neurons'], ascending=[True, True])

# Get the network structure with the lowest misclassification rate
best_network_structure = sorted_results.iloc[0]

# Print the best network structure
print("Best Network Structure:")
print(best_network_structure)

#2e) Sort the results DataFrame by RMSE and, in case of ties, by the number of neurons
sorted_results_rmse = result_df.sort_values(by=['Root Mean Squared Error', 'Number of Neurons'], ascending=[True, True])

# Get the network structure with the lowest RMSE
best_network_structure_rmse = sorted_results_rmse.iloc[0]

# Print the best network structure
print("Best Network Structure (Lowest RMSE):")
print(best_network_structure_rmse)

#2f)


# Make predictions on the testing data using the final model with relu
final_model = MLPClassifier(hidden_layer_sizes=(best_network_structure_rmse['Number of Neurons'],) * best_network_structure_rmse['Number of Layers'],
                            activation='relu', max_iter=10000, random_state=2023484)
final_model.fit(X_train, y_train)
predicted_probabilities = final_model.predict_proba(X_test)[:, 1]

# Group the predicted probabilities by the observed quality_grp categories
test_data['Predicted Probability'] = predicted_probabilities

# Create a grouped boxplot
plt.figure(figsize=(10, 6))
boxplot = test_data.boxplot(column='Predicted Probability', by='quality_grp', patch_artist=True)
plt.title("Grouped Boxplot of Predicted Probabilities for quality_grp = 1")
plt.suptitle("")  # Remove the default title added by boxplot
plt.xlabel("Quality Group")
plt.ylabel("Predicted Probability")

# Add a reference line for Prob(quality_grp = 1) = 1.5c
c = proportion_quality_1_test
plt.axhline(y=1.5 * c, color='r', linestyle='--', label='Reference: 1.5c')

# Display the plot
plt.legend()
plt.show()