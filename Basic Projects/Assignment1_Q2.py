"""
@Name: Assignment1_Q2.py
@Creation Date: February 3, 2023
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import numpy
import pandas
import random
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 7, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7f}'.format

def simpleRandomSample (obsIndex, trainFraction = 0.7):
   '''Generate a simple random sample

   Parameters
   ----------
   obsIndex - a list of indices to the observations
   trainFraction - the fraction of observations assigned to Training partition
                   (a value between 0 and 1)

   Output
   ------
   trainIndex - a list of indices of original observations assigned to Training partition
   '''

   trainIndex = []

   nPopulation = len(obsIndex)
   nSample = round(trainFraction * nPopulation)
   kObs = 0
   iSample = 0
   for oi in obsIndex:
      kObs = kObs + 1
      U = random.random()
      uThreshold = (nSample - iSample) / (nPopulation - kObs + 1)
      if (U < uThreshold):
         trainIndex.append(oi)
         iSample = iSample + 1

      if (iSample == nSample):
         break

   testIndex = list(set(obsIndex) - set(trainIndex))
   return (trainIndex, testIndex)

hmeq_data = pandas.read_csv('C:\\IIT\Machine Learning\\Data\\hmeq.csv')

label = 'BAD'

strata_list = ['BAD', 'REASON']
n_strata_var = len(strata_list)

feature_list = ['DEBTINC', 'LOAN', 'MORTDUE', 'VALUE']
n_feature = len(feature_list)

# Recode missing values in BAD as -9
hmeq_data['BAD'].fillna(99, inplace = True)
hmeq_data['REASON'].fillna('MISSING', inplace = True)

# Part (a)

# Frequency Table of BAD
label_freq = hmeq_data[label].value_counts()
print(label_freq)

# Summary Statistics of Feature List
summary_stat = hmeq_data[feature_list].describe()
print(summary_stat)

# Part (b)

sample_data = hmeq_data[[label] + feature_list]

random.seed(a = 202303484)
train_index, test_index = simpleRandomSample (sample_data.index, trainFraction = 0.7)

X_train = sample_data.loc[train_index][feature_list]
y_train = sample_data.loc[train_index][label]

label_freq_train = y_train.value_counts()
summary_stat_train = X_train[feature_list].describe()

X_test = sample_data.loc[test_index][feature_list]
y_test = sample_data.loc[test_index][label]

label_freq_test = y_test.value_counts()
summary_stat_test = X_test[feature_list].describe()

# Part (c)

sample_data = hmeq_data[strata_list + feature_list]

# Find the observed strata
strata = sample_data.groupby(strata_list).count()

train_index = []
test_index = []

random.seed(a = 202303484)

for stratum in strata.index:
   vBAD = stratum[0]
   vREASON = stratum[1]
   print('BAD = ', vBAD, ' REASON = ', vREASON)
   obsIndex = sample_data[numpy.logical_and(sample_data['BAD'] == vBAD, sample_data['REASON'] == vREASON)].index
   trIndex, ttIndex = simpleRandomSample (obsIndex, trainFraction = 0.7)
   train_index.extend(trIndex)
   test_index.extend(ttIndex)
   
X_train = sample_data.loc[train_index][feature_list]
y_train = sample_data.loc[train_index][label]

label_freq_train = y_train.value_counts()
summary_stat_train = X_train[feature_list].describe()

X_test = sample_data.loc[test_index][feature_list]
y_test = sample_data.loc[test_index][label]

label_freq_test = y_test.value_counts()
summary_stat_test = X_test[feature_list].describe()