import numpy as np
import pandas as pd
import random
import sys

import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()
import io

def submissionDetails():
    print("""Name : Sai Ram Oduri \n Course: CS 484 Introduction to Machine Learning""")

trainData = pd.read_csv(io.BytesIO(uploaded['Fraud.csv']))

import sklearn.tree as tree

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors as kNN

#****************************Question 3*****************************

#****************************Question 3.a***************************
fraud_counts = trainData[trainData['FRAUD']==1].count()['FRAUD']
ttl_frauds = trainData['FRAUD'].count()
fraud_percentage = (fraud_counts / ttl_frauds) * 100
fraud_percentage = float("{0:.4f}".format(fraud_percentage))
print(fraud_counts)
print(fraud_percentage)

#******************************Question 3.b***********************

from sklearn.model_selection import train_test_split

random_seed = 202303484
trainData.dropna(inplace=True)

fraud_data = trainData[trainData['FRAUD']==1]
not_fraud =  trainData[trainData['FRAUD']==0]


train_fraud, test_fraud = train_test_split(fraud_data, test_size=0.2, random_state=random_seed)
train_not_fraud , test_not_fraud = train_test_split(not_fraud, test_size=0.2, random_state=random_seed )
print("No.of Training Frauds are",train_fraud)
print("No. of Test Frauds",test_fraud)
print("No. of Train Not Frauds",train_not_fraud)
print("No. of Test not frauds",test_not_fraud)
num_train_obs = len(train_fraud) + len(train_not_fraud)
num_test_obs = len(test_fraud) + len(test_not_fraud)

print("Total training observation are",num_train_obs)
print("Total testing observation are",num_test_obs)

fraud_data = trainData[trainData['FRAUD']==1]
not_fraud =  trainData[trainData['FRAUD']==0]

def fraud_data_empirical(data):
  fraud_proportion = sum(trainData['FRAUD'] == 1) / len(data)
  rounded = round(fraud_proportion,4)


empirical_rate = fraud_data_empirical(trainData)

misclassification_rates = {}

for close in range(2,8):
  train_preds = []
  test_preds = []
  train_labels = []
  test_labels = []

  train_fraud, test_fraud = train_test_split(fraud_data, test_size=0.2, random_state=random_seed)
  train_not_fraud , test_not_fraud = train_test_split(not_fraud, test_size=0.2, random_state=random_seed )


  train_data = pd.concat([train_fraud,train_not_fraud])
  test_data = pd.concat([test_fraud,test_not_fraud])

  #extract he features and target variables 

  x_train = train_data[['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']]
  y_train = train_data['FRAUD']
  x_test = train_data[['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']]
  y_test = train_data['FRAUD']

  #use the standard scaler on the training data, to normalize our dataset

  scaler = StandardScaler()
  scaled_trainx = scaler.fit_transform(x_train)

  #use the standard scaler on the testing data, to normalize our dataset
  scaled_testx = scaler.fit_transform(x_test)

  #KNN classifier

  knn_classifier = KNeighborsClassifier(n_neighbors=close)
  knn_classifier.fit(scaled_trainx,y_train)

  train_pred = knn_classifier.predict(scaled_trainx)
  test_pred = knn_classifier.predict(scaled_testx) 

  # Append predictions and true labels to respective lists
  train_preds.extend(train_pred)
  test_preds.extend(test_pred)
  train_labels.extend(y_train)
  test_labels.extend(y_test)
    
  # Calculate misclassification rates
  train_misclassification_rate = 1 - accuracy_score(train_labels, train_preds)
  test_misclassification_rate = 1 - accuracy_score(test_labels, test_preds)
    
  misclassification_rates[close] = (train_misclassification_rate, test_misclassification_rate)

# Print misclassification rates for different numbers of neighbors
for k, rates in misclassification_rates.items():
    train_rate, test_rate = rates
    print(f"Neighbors (K={k}):")
    print(f"Training Misclassification Rate: {train_rate:.4f}")
    print(f"Testing Misclassification Rate: {test_rate:.4f}")
    print()


for k in range(2, 8):
    # Initialize lists to store predicted labels and true labels for both Training and Testing partitions
    train_predictions = []
    test_predictions = []
    train_labels = []
    test_labels = []
    
    # Split each stratum into Training and Testing partitions
    train_fraud, test_fraud = train_test_split(fraud_data, test_size=0.2, random_state=random_seed)
    train_not_fraud, test_not_fraud = train_test_split(not_fraud_data, test_size=0.2, random_state=random_seed)
    
    # Concatenate Training and Testing partitions
    train_data = pd.concat([train_fraud, train_not_fraud])
    test_data = pd.concat([test_fraud, test_not_fraud])
    
    # Extract features and target variables
    X_train = train_data[['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']]
    y_train = train_data['FRAUD']
    X_test = test_data[['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']]
    y_test = test_data['FRAUD']
    
    # Initialize and train the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    
    # Make predictions on Training and Testing data
    train_pred = knn_classifier.predict(X_train)
    test_pred = knn_classifier.predict(X_test)
    
    # Append predictions and true labels to respective lists
    train_predictions.extend(train_pred)
    test_predictions.extend(test_pred)
    train_labels.extend(y_train)
    test_labels.extend(y_test)
    
    # Calculate misclassification rates
    train_misclassification_rate = 1 - accuracy_score(train_labels, train_predictions)
    test_misclassification_rate = 1 - accuracy_score(test_labels, test_predictions)
    
    # Store misclassification rates in the dictionary
    misclassification_rates[k] = (train_misclassification_rate, test_misclassification_rate)

# Print misclassification rates for different numbers of neighbors
for k, rates in misclassification_rates.items():
    train_rate, test_rate = rates
    print(f"Neighbors (K={k}):")
    print(f"Training Misclassification Rate: {train_rate:.4f}")
    print(f"Testing Misclassification Rate: {test_rate:.4f}")


best_k = None
lowest_test_misclassification_rate = float('inf')

# Loop through different values of K (neighbors)
for k in range(2, 8):
    # Initialize and fit the StandardScaler on Training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform Testing data using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train_scaled, y_train)
    
    # Make predictions on Testing data
    test_pred = knn_classifier.predict(X_test_scaled)
    
    # Calculate the misclassification rate for Testing data
    test_misclassification_rate = 1 - accuracy_score(y_test, test_pred)
    
    # Check if the current rate is lower than the lowest found so far
    if test_misclassification_rate < lowest_test_misclassification_rate:
        lowest_test_misclassification_rate = test_misclassification_rate
        best_k = k

# Print the best number of neighbors and the corresponding misclassification rate
print(f"The best number of neighbors (K) for the lowest test misclassification rate is: {best_k}")
print(f"The lowest test misclassification rate is: {lowest_test_misclassification_rate:.4f}")
    
for k in range(2, 8):
    # Initialize and fit the StandardScaler on the entire dataset
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']])
    
    # Split each stratum into Training and Testing partitions
    train_fraud, test_fraud = train_test_split(fraud_data, test_size=0.2, random_state=random_seed)
    train_not_fraud, test_not_fraud = train_test_split(not_fraud_data, test_size=0.2, random_state=random_seed)
    
    # Concatenate Training and Testing partitions
    train_data = pd.concat([train_fraud, train_not_fraud])
    test_data = pd.concat([test_fraud, test_not_fraud])
    
    # Extract features and target variables
    X_train = train_data[['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']]
    y_train = train_data['FRAUD']
    X_test = test_data[['DOCTOR_VISITS', 'MEMBER_DURATION', 'NUM_CLAIMS', 'NUM_MEMBERS', 'OPTOM_PRESC', 'TOTAL_SPEND']]
    y_test = test_data['FRAUD']
    
    # Initialize and train the KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    
    # Make predictions on Testing data
    test_pred = knn_classifier.predict(X_test)
    
    # Calculate misclassification rate for Testing data
    test_misclassification_rate = 1 - accuracy_score(y_test, test_pred)
    
    # Check if this is the best result so far
    if test_misclassification_rate < best_misclassification_rate:
        best_misclassification_rate = test_misclassification_rate
        best_k = k

# Print the best number of neighbors and its corresponding misclassification rate
print(f"Best Number of Neighbors (K): {best_k}")
print(f"Testing Misclassification Rate for Best K: {best_misclassification_rate:.4f}")

selected_k = 6 # Replace with the selected K value from Part (d)

knn_classifier = KNeighborsClassifier(n_neighbors=selected_k)
knn_classifier.fit(scaled_data, y)

# Define the feature values of the focal observation
focal_observation = [[8, 178, 0, 2, 1, 16300]]

# Scale the feature values of the focal observation
scaled_focal_observation = scaler.transform(focal_observation)

# Find the indices of the K nearest neighbors in the entire dataset
neighbor_indices = knn_classifier.kneighbors(scaled_focal_observation, return_distance=False)

# Retrieve the neighbors' observation values from the entire dataset
neighbors_data = data.iloc[neighbor_indices[0]]

# Calculate the predicted probability that the focal observation is fraudulent
predicted_probability = sum(neighbors_data['FRAUD'] == 1) / selected_k

# Print the neighbors' observation values and predicted probability
print("Neighbors' Observation Values:")
print(neighbors_data)
print(f"Predicted Probability of Fraud: {predicted_probability:.4f}")