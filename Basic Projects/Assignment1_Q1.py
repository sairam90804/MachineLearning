"""
@Name: Assignment1_Q1.py
@Creation Date: February 3, 2023
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 7, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7f}'.format

def shimazaki_criterion (y, d_list, y_nvalid, y_min, y_max, y_mean):

   number_bins = []
   bin_middle = []
   matrix_boundary = []
   criterion = []

   # Loop through the bin width candidates
   for delta in d_list:
      y_middle = delta * numpy.round(y_mean / delta)
      n_bin_left = numpy.ceil((y_middle - y_min) / delta)
      n_bin_right = numpy.ceil((y_max - y_middle) / delta)
      y_low = y_middle - n_bin_left * delta

      # Assign observations to bins starting from 0
      list_boundary = []
      n_bin = n_bin_left + n_bin_right
      bin_index = 0
      bin_boundary = y_low
      list_boundary.append(bin_boundary)
      for i in numpy.arange(n_bin):
         bin_boundary = bin_boundary + delta
         bin_index = numpy.where(y > bin_boundary, i+1, bin_index)
         list_boundary.append(bin_boundary)

      # Count the number of observations in each bins
      uvalue, ucount = numpy.unique(bin_index, return_counts = True)

      # Calculate the average frequency
      mean_ucount = numpy.mean(ucount)
      ssd_ucount = numpy.mean(numpy.power((ucount - mean_ucount), 2))
      crit = (2.0 * mean_ucount - ssd_ucount) / delta / delta

      number_bins.append(n_bin)
      bin_middle.append(y_middle)
      matrix_boundary.append(list_boundary)
      criterion.append(crit)

   return(number_bins, bin_middle, matrix_boundary, criterion)

def get_optimal_histogram (y, min_nbin, max_nbin):
    y_stat = y.describe()
    y_nvalid = y_stat.loc['count']
    y_mean = y_stat.loc['mean']
    y_min = y_stat.loc['min']
    y_max = y_stat.loc['max']

    d_list = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]
    number_bins, bin_middle, matrix_boundary, criterion = shimazaki_criterion (y, d_list, y_nvalid, y_min, y_max, y_mean)

    result_df = pandas.DataFrame([d_list, number_bins, bin_middle, criterion]).transpose()
    result_df.columns = ['Bin Width', 'Number of Bins', 'Bin Middle', 'Criterion']

    feasible = result_df[numpy.logical_and(min_nbin <= result_df['Number of Bins'],result_df['Number of Bins']<=max_nbin)]
    optimal_index = feasible['Criterion'].idxmin()
    optimal_boundary = matrix_boundary[optimal_index]
    optimal_binwidth = feasible.loc[optimal_index]['Bin Width']

    return (optimal_binwidth, optimal_boundary, result_df)

# Read only the column X from the NormalSample.csv file
inputData = pandas.read_csv('C:\\IIT\Machine Learning\\Data\\Gamma4804.csv', delimiter=',', header = 0)

# x is a Pandas Series
x = inputData['x']

# Part (a)
x_summary = x.describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.9])
c = ['{:.7f}'.format(x) for x in x_summary]
print(pandas.Series(c, index = x_summary.index))

# Part (b)
optimal_binwidth, optimal_boundary, result_df = get_optimal_histogram (x, 5, 50)

# Part (c)
fig, ax = plt.subplots(1, 1, figsize = (10,6), dpi = 200)
ax.hist(x, density = True, bins = optimal_boundary, color = 'royalblue', histtype = 'step')
ax.set_title('Histogram of x with a bin width of ' + str(optimal_binwidth))
ax.set_xlabel('x')
ax.set_ylabel('Estimated Density Value')
ax.set_xticks(range(0,210,10))
ax.set_yticks(numpy.arange(0.0, 0.018, 0.001))
ax.grid(axis = 'both', linestyle = 'dotted')
ax.margins(y = 0.1)
plt.show()