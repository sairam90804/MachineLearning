"""
@Name: Assignment3_Q1.py
@Course: CS 484 Introduction to Machine Learning
@Creation Date: February 16, 2023
@Author: Ming-Long Lam, Ph.D.
@Organization: Illinois Institute of Technology
(C) All Rights Reserved.

"""
import itertools
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10f}'.format

def plog2p (proportion):
   '''A function that calculates the value p * log2(p).
   If p > 0.0, then result = p * log2(p) where log2() is the base 2 logarithm function
   If p == 0.0, then result = 0.0
   Otherwise, then result is NaN

   Argument:
   1. proportion: a value between 0.0 and 1.0

   Output:
   1. result: the p * log2(p) value
   '''

   if (0.0 < proportion and proportion < 1.0):
      result = proportion * numpy.log2(proportion)
   elif (proportion == 0.0 or proportion == 1.0):
      result = 0.0
   else:
      result = numpy.nan

   return (result)

def NodeEntropy (nodeCount):
   '''A function that calculate the entropy of a node given a row of counts.
   
   Argument:
   1. nodeCount: a Pandas Series of the counts of target values of the node
   
   Output:
   1. nodeTotal: the sum of counts of the node
   2. nodeEntropy: the entropy of the node.
   '''

   nodeTotal = numpy.sum(nodeCount)
   nodeProportion = nodeCount / nodeTotal
   nodeEntropy = - numpy.sum(nodeProportion.apply(plog2p))

   return (nodeTotal, nodeEntropy)

def EntropyCategorySplit (target, catPredictor, splitList):
   '''A function that calculates the split entropy for splitting a categorical predictor into
   two groups by a split value

   Argument:
   1. target: a Pandas Series of target values
   2. catPredictor: a Pandas Series of predictor values
   3. splitList: a list of values for splitting the predictor values into two groups

   Output:
   1. leftStats: a list of statistics about the left branch
   2. rightStats: a list of statistics about the right branch
   3. splitEntropy: the split entropy value

   The list of statistics about a branch:
   [0] number of observations in the branch (zero if the branch is empty)
   [1] counts of target categories
   [2] node entropy
   '''

   branch_indicator = numpy.where(catPredictor.isin(splitList), 'LEFT', 'RIGHT')
   xtab = pandas.crosstab(index = branch_indicator, columns = target, margins = False, dropna = True)

   splitEntropy = 0.0
   tableTotal = 0.0

   leftStats = None
   rightStats = None

   for idx, row in xtab.iterrows():
      rowTotal, rowEntropy = NodeEntropy(row)
      tableTotal = tableTotal + rowTotal
      splitEntropy = splitEntropy + rowTotal * rowEntropy

      if (idx == 'LEFT'):
         leftStats = [rowTotal, row, rowEntropy]
      else:
         rightStats = [rowTotal, row, rowEntropy]

   splitEntropy = splitEntropy / tableTotal
  
   return(leftStats, rightStats, splitEntropy)

def takeEntropy(s):
    return s[1]

def FindBestSplit (branch_data, target, nominal_predictor, ordinal_predictor, debug = 'N'):
   '''A function that finds the categorical predictor that optimally split the branch data
   into two groups.

   Argument:
   1. branch_data: a Pandas DataFrame of observations
   2. target: a Pandas Series of target values
   3. nominal_predictor: a list of names of nominal predictors
   4. ordinal_predictor: a list of names of ordinal predictors whose categories are numeric
   5. debug: a Y/N flag to show debugging information

   Output:
   result_list: a list of these four entities:
   (0) splitPredictor: name of the categorical predictor that split
   (1) splitBranches: a list of the categories in the left and the right branches
   (2) nodeEntropy: a list of the left and the right nodes entropy values
   (3) splitEntropy: the split entropy value
   '''

   target_data = branch_data[target]

   split_summary = []

   # Look at each nominal predictor
   for pred in nominal_predictor:
      predictor_data = branch_data[pred]
      category_set = set(numpy.unique(predictor_data))
      n_category = len(category_set)

      split_list = []
      for size in range(1, ((n_category // 2) + 1)):
         comb_size = itertools.combinations(category_set, size)
         for item in list(comb_size):
            left_branch = list(item)
            right_branch = list(category_set.difference(item))
            leftStats, rightStats, splitEntropy = EntropyCategorySplit (target_data, predictor_data, left_branch)
            if (leftStats is not None and rightStats is not None):
               split_list.append([pred, splitEntropy, left_branch, right_branch, leftStats, rightStats])

      # Determine the optimal split of the current predictor
      split_list.sort(key = takeEntropy, reverse = False)

      if (debug == 'Y'):
          print(split_list[0])

      # Update the split summary
      split_summary.append(split_list[0])

   # Look at each ordinal predictor
   for pred in ordinal_predictor:
      predictor_data = branch_data[pred]
      category_set = numpy.unique(predictor_data)
      n_category = len(category_set)

      split_list = []
      for size in range(1, n_category):
         left_branch = list(category_set[0:size])
         right_branch = list(category_set[size:n_category])
         leftStats, rightStats, splitEntropy = EntropyCategorySplit (target_data, predictor_data, left_branch)
         if (leftStats is not None and rightStats is not None):
            split_list.append([pred, splitEntropy, left_branch, right_branch, leftStats, rightStats])

      # Determine the optimal split of the current predictor
      split_list.sort(key = takeEntropy, reverse = False)

      if (debug == 'Y'):
          print(split_list[0])

      # Update the split summary
      split_summary.append(split_list[0])
 
   if (debug == 'Y'):
      print(split_summary)

   # Determine the optimal predictor
   split_summary.sort(key = takeEntropy, reverse = False)
   splitPredictor = split_summary[0][0]
   splitEntropy = split_summary[0][1]
   splitBranches = split_summary[0][2:4]
   nodeStats = split_summary[0][4:6]

   return ([splitPredictor, splitEntropy, splitBranches, nodeStats])

claim_history = pandas.read_excel('C:\\IIT\\Machine Learning\\Data\\claim_history.xlsx')

n_sample = claim_history.shape[0]

# Part (a) Train a decision tree model
target = 'CAR_USE'
nominal_predictor = ['CAR_TYPE', 'OCCUPATION']
ordinal_predictor = ['EDUCATION']

train_data = claim_history[[target] + nominal_predictor + ordinal_predictor].dropna().reset_index(drop = True)
target_count = train_data[target].value_counts().sort_index(ascending = True)

# Recode the EDUCATION categories into numeric values
train_data['EDUCATION'] = train_data['EDUCATION'].map({'Below High School':0, 'High School':1, 'Bachelors':2, 'Masters':3, 'Doctors':4})
print(train_data['EDUCATION'].value_counts().sort_index(ascending = True))

# Students will develop their codes to complete the Question 1
