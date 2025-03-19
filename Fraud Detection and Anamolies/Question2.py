# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 10:54:10 2023

@author: saira
Spyder Editor


Name : Sai Ram Oduri
Student ID: A20522183
Course : CS 484 - Introduction to Machine Learning
Semester : Fall 2023

"""

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

trainData = pd.read_csv(io.BytesIO(uploaded['hmeq.csv']))

print("We need to create the Training and Testing partitions from the observations in the hmeq.csv. We will use all observations (including those with missing values in one or more variables) for this task. The Training partition will contain 70% of the observations. The Testing partition will contain the remaining 30% of the observations.")


import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the dataset
data = trainData

# Setting the random seed
random_seed = 202303484

# Split data into train and test

np.random.seed(random_seed)
samples = len(data)
test_samples = int(0.3 * samples)
test_indices = np.random.choice(samples,test_samples, replace=False)

test_data = data.iloc[test_indices]
train_data = data.drop(test_indices)

# Reset the index for both partitions
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)


#****************************Question 2.a************************

print("Q2.a Before we partition the observations, we need a baseline for reference. How many  observations are in the dataset What are the frequency distributions of BAD What are the means and the standard deviations of DEBTINC, LOAN, MORTDUE, and VALUE?")

import pandas as pd

# Load the dataset
data = pd.read_csv('hmeq.csv')

# 1. Total number of observations
total_observations = len(data)

# 2. Frequency distribution of "BAD" (including missing)
bad_distribution = data['BAD'].value_counts(dropna=False)

# 3. Means and standard deviations
means = data[selected_columns].mean()
stddev = data[selected_columns].std()

# Print the results
print(f"Total number of observations: {total_observations}\n")
print("Frequency distribution of BAD:")

print(bad_distribution)

print("\nMeans and standard deviations:")
print(means)
print(stddev)


#**********************Question 2.b*****************************


# Split the data into Training (70%) and Testing (30%) partitions with random sampling
train_data, test_data = train_test_split(data, test_size=0.3, random_state=random_seed)

# 1. Number of observations in each partition
train_observations = len(train_data)
test_observations = len(test_data)

# 2. Frequency distribution of BAD (including missing) in each partition
train_bad_frequency = train_data['BAD'].value_counts(dropna=False)
test_bad_frequency = test_data['BAD'].value_counts(dropna=False)

# 3. Means and standard deviations of selected variables in each partition
selected_variables = ['DEBTINC', 'LOAN', 'MORTDUE', 'VALUE']

#train_data[selected_variables]
means_train = train_data[selected_variables].mean()
std_train = train_data[selected_variables].std()

means_test = test_data[selected_variables].mean()
std_test = test_data[selected_variables].std()

# Print the results
print("Training Partition:")
print(f"Number of observations: {train_observations}\n")
print("Frequency distribution of BAD (including missing):\n", train_bad_frequency)
print("\nMeans and standard deviations of selected variables:")
print(means_train)
print(std_train)


print("Testing Partition:")
print(f"Number of observations: {test_observations}\n")
print("Frequency distribution of BAD (including missing):\n", test_bad_frequency)
print("\nMeans and standard deviations of selected variables:")
print(means_test)
print(std_test)

#**********************Question 2.c*****************************

# Replace missing values in BAD with 99 and in REASON with 'MISSING'
data['BAD'].fillna(99, inplace=True)
data['REASON'].fillna('MISSING', inplace=True)

# Define the strata variable combining BAD and REASON
data['STRATA'] = data['BAD'].astype(str) + '_' + data['REASON']

# Stratified random sampling based on STRATA
train_data, test_data = train_test_split(data, test_size=0.3, random_state=random_seed, stratify=data['STRATA'])

# 1. Frequency distribution of BAD (including missing) in each partition
train_bad_frequency = train_data['BAD'].value_counts(dropna=False)
test_bad_frequency = test_data['BAD'].value_counts(dropna=False)

# 2. Means and standard deviations of selected variables in each partition
selected_variables = ['DEBTINC', 'LOAN', 'MORTDUE', 'VALUE']
train_mean = train_data[selected_variables].mean()
train_std = train_data[selected_variables].std()
test_mean = test_data[selected_variables].mean()
test_std = test_data[selected_variables].std()

# Print the results
print("Training Partition:")
print("Frequency distribution of BAD (including missing):\n", train_bad_frequency)
print("\nMeans and standard deviations of selected variables:")
print(train_mean)
print(train_std)

print("Testing Partition:")
print("Frequency distribution of BAD (including missing):\n", test_bad_frequency)
print("\nMeans and standard deviations of selected variables:")
print(test_mean)
print(test_std)

