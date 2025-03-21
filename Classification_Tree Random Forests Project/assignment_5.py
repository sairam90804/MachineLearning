# -*- coding: utf-8 -*-
"""Assignment_5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/165eOyq8IiHxDI5R_moIn3TnRCFTGo-u4
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

train_data = pd.read_csv("WineQuality_Train.csv")
test_data = pd.read_csv("WineQuality_Test.csv")

# Extracting features and target variable from the datasets
X_train = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = train_data['quality_grp']

X_test = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = test_data['quality_grp']

tree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)

# Train the classifier on the training data
tree_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_tree = tree_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_tree)
conf_matrix = confusion_matrix(y_test, y_pred_tree)
classification_rep = classification_report(y_test, y_pred_tree)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")
plt.figure(figsize=(12, 8))
plot_tree(tree_classifier, filled=True, feature_names=X_train.columns, class_names=['0', '1'])
plt.show()
feature_importances = tree_classifier.feature_importances_

for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import combinations


# Extracting features and target variable from the datasets
X_train = train_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = train_data['quality_grp']

X_test = test_data[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = test_data['quality_grp']

# Add an intercept term to the features
X1_train = sm.add_constant(X_train)
X1_test = sm.add_constant(X_test)

# All-Possible Subset method for feature selection
def get_best_model(features, target):
    best_model = None
    best_aic = float('inf')  # initialize with positive infinity

    for subset in combinations(features.columns, len(features.columns) - 1):
        model_features = features[list(subset)]
        model = sm.Logit(target, model_features)
        result = model.fit()

        if result.aic < best_aic:
            best_aic = result.aic
            best_model = result

    return best_model

best_model = get_best_model(X1_train, y_train)


print(best_model.summary())
y_pred_proba = best_model.predict(X_test)
y_pred_bin = (y_pred_proba > 0.5).astype(int)


accuracy_binary = accuracy_score(y_test, y_pred_bin)
conf_matrix_binary = confusion_matrix(y_test, y_pred_bin)
classification_rep_bin = classification_report(y_test, y_pred_bin)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_custom_rase(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

rase_custom_tree_train = calculate_custom_rase(y_train, y_pred_tree)
rase_custom_tree_test = calculate_custom_rase(y_test_custom, y_test_tree_pred)
rase_custom_logistic_train = calculate_custom_rase(y_train_custom, y_pred_tree_reg)
rase_custom_logistic_test = calculate_custom_rase(y_test_custom, y_test_reg)

print("RASE - Custom Classification Tree (Train):", rase_custom_tree_train)
print("RASE - Custom Classification Tree (Test):", rase_custom_tree_test)
print("RASE - Custom Logistic Regression (Train):", rase_custom_logistic_train)
print("RASE - Custom Logistic Regression (Test):", rase_custom_logistic_test)

from sklearn.metrics import roc_auc_score

y_train_tree_pred_proba = tree_classifier.predict_proba(X_train)[:, 1]
# Calculate AUC for training data
auc_train_tree = roc_auc_score(y_train, y_train_tree_pred_proba)

# Make predictions on the testing data
y_test_tree_pred_proba = tree_classifier.predict_proba(X_test)[:, 1]
# Calculate AUC for testing data
auc_test_tree = roc_auc_score(y_test, y_test_tree_pred_proba)

print(f"AUC (Decision Tree, Training): {auc_train_tree}")
print(f"AUC (Decision Tree, Testing): {auc_test_tree}")

y_train_pred_proba = best_model.predict(X_train)
auc_train = roc_auc_score(y_train, y_train_pred_proba)

# Make predictions on the testing data
y_test_pred_proba = best_model.predict(X_test)
auc_test = roc_auc_score(y_test, y_test_pred_proba)

print(f"AUC (Logistic Regression, Training): {auc_train}")
print(f"AUC (Logistic Regression, Testing): {auc_test}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Decision Tree
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_test_tree_pred_proba)

# Logistic Regression
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)

# Plotting ROC curves
plt.figure(figsize=(8, 8))
plt.plot(fpr_tree, tpr_tree, label='Decision Tree')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve

# Decision Tree
precision_tree, recall_tree, _ = precision_recall_curve(y_test, y_test_tree_pred_proba)

# Logistic Regression
precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)

# Plotting Precision-Recall curves
plt.figure(figsize=(8, 8))
plt.plot(recall_tree, precision_tree, label='Decision Tree')
plt.plot(recall, precision, label='Logistic Regression')
plt.axhline(y=sum(y_test) / len(y_test), color='r', linestyle='--', label='No-Skill')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

c

from sklearn.metrics import f1_score, accuracy_score

# Decision Tree
f1_threshold_tree = 0.167
y_test_pred_tree = (y_test_tree_pred_proba > f1_threshold_tree).astype(int)
misclassification_rate_tree = 1 - accuracy_score(y_test, y_test_pred_tree)

# Logistic Regression
f1_threshold_logistic = 0.5
y_test_pred_logistic = (y_test_pred_proba > f1_threshold_logistic).astype(int)
misclassification_rate_logistic = 1 - accuracy_score(y_test, y_test_pred_logistic)

print(f'Misclassification Rate for Decision Tree: {misclassification_rate_tree}')
print(f'Misclassification Rate for Logistic Regression: {misclassification_rate_logistic}')

import numpy as np
import pandas as pd


sorted_indices_tree = np.argsort(y_test_tree_pred_proba)[::-1]
y_test_sorted_tree = y_test.iloc[sorted_indices_tree]


sorted_indices_logistic = np.argsort(y_test_pred_proba)[::-1]
y_test_sorted_logistic = y_test.iloc[sorted_indices_logistic]

# Calculate cumulative gains and lift for Decision Tree
total_positives_tree = np.sum(y_test)
cumulative_gains_tree = np.cumsum(y_test_sorted_tree) / total_positives_tree
lift_tree = cumulative_gains_tree / (np.arange(len(y_test)) + 1)

# Calculate cumulative gains and lift for Logistic Regression
total_positives_logistic = np.sum(y_test)
cumulative_gains_logistic = np.cumsum(y_test_sorted_logistic) / total_positives_logistic
lift_logistic = cumulative_gains_logistic / (np.arange(len(y_test)) + 1)

# Create DataFrames for cumulative gain and lift
df_gain_lift_tree = pd.DataFrame({
    'Cumulative Gains (Decision Tree)': cumulative_gains_tree,
    'Lift (Decision Tree)': lift_tree
})

df_gain_lift_logistic = pd.DataFrame({
    'Cumulative Gains (Logistic Regression)': cumulative_gains_logistic,
    'Lift (Logistic Regression)': lift_logistic
})

# Save to Excel
with pd.ExcelWriter('cumulative_gain_lift_table.xlsx', engine='xlsxwriter') as writer:
    df_gain_lift_tree.to_excel(writer, sheet_name='Decision_Tree', index=False)
    df_gain_lift_logistic.to_excel(writer, sheet_name='Logistic_Regression', index=False)

print("Cumulative Gain and Lift Table for Decision Tree:")
print(df_gain_lift_tree)

print("\nCumulative Gain and Lift Table for Logistic Regression:")
print(df_gain_lift_logistic)