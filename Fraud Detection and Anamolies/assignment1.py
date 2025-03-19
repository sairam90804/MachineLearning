 # -*- coding: utf-8 -*-
"""
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

#mport graphviz as gv


#use dataset gamma4804

def submissionDetails():
    print("""Name : Sai Ram Oduri \n Course: CS 484 Introduction to Machine Learning""")

trainData = pd.read_csv('Z:\\IIT\\SEM 3\\Machine Learning\\Assignment1\\Gamma4804.csv',
                             delimiter=',')

   
def numBinsCalc(Y, delta):
   maxY = np.max(Y)
   minY = np.min(Y)
   meanY = np.mean(Y)

   # Round the mean to integral multiples of delta
   middleY = delta * np.round(meanY / delta)

   # Determine the number of bins on both sides of the rounded mean
   nBinRight = np.ceil((maxY - middleY) / delta)
   nBinLeft = np.ceil((middleY - minY) / delta)
   lowY = middleY - nBinLeft * delta

   # Assign observations to bins starting from 0
   m = nBinLeft + nBinRight
   bin_index = 0;
   boundaryY = lowY
   for iBin in np.arange(m):
      boundaryY = boundaryY + delta
      bin_index = np.where(Y > boundaryY, iBin+1, bin_index)

   # Count the number of observations in each bins
   uBin, binFreq = np.unique(bin_index, return_counts = True)

   # Calculate the average frequency
   meanBinFreq = np.sum(binFreq) / m
   ssDevBinFreq = np.sum((Y - meanBinFreq)**2) / m
   CDelta = ((2.0 * meanBinFreq) - ssDevBinFreq) / (delta * delta)
   return(m, middleY, lowY, CDelta, uBin, binFreq)

    
    
def UPcheck(up):
    if up <= 1/2 and up>1/2:
        return True
    else:
        return False

submissionDetails()

#************Question 1*************

gamma_sampleData = pd.read_csv('Gamma4804.csv' ,delimiter=',', usecols=['x'])
gamma_x = gamma_sampleData['x']

#************Question 1.a************

print("\n\n\n(10 points) What are the count, the mean, the standard deviation, the minimum, the 10th percentile, the 25th percentile, the median, the 75th percentile, the 90th percentile, and the maximum of the feature x? Please round your answers to the seventh decimal place ")
gamma_describe = gamma_x.describe()
print(gamma_describe)

#************Question 1.b************

print("\n\n\n Q1.b (10 points) Use the Shimazaki and Shinomoto (2007) method to recommend a bin width.  We will try d = 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, and 100.  What bin width would you recommend if we want the number of bins to be between 5 and 50 inclusively? You need to show your calculations to receive full credit.")

deltas = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]
resdf = pd.DataFrame()
df = pd.DataFrame()

for d in deltas:
    nBin, middleY, lowY, CDelta, uBin, binFreq = numBinsCalc(gamma_x,d)
    
    highY = lowY + nBin * d
    resdf = resdf.append([[d, CDelta, lowY, middleY, highY, nBin, uBin, binFreq]], ignore_index = True)
    
    binMid = lowY + 0.5 * d + np.arange(nBin) * d
    plt.hist(gamma_x, bins = binMid, align='mid')
    plt.title('Delta = ' + str(d))
    plt.ylabel('Number of Observations')
    plt.grid(axis = 'y')
    plt.show()
    
resdf = resdf.rename(columns = {0:'Delta', 1:'C(Delta)', 2:'Low Y', 3:'Middle Y', 4:'High Y', 5:'N Bin', 6:'uBin', 7:'binFreq'})

sortedResult = resdf.sort_values(by=['C(Delta)']).reset_index(drop=True)
print(sortedResult)
recommendedBinWidthSSM = sortedResult['Delta'][0]
print("\nTherefore, Recomended bin-width is, d = {0}\n".format(recommendedBinWidthSSM))
fig1, ax1 = plt.subplots()
ax1.set_title('Box Plot')
ax1.boxplot(gamma_x, labels = ['X'])
ax1.grid(linestyle = '--', linewidth = 1)
plt.show()  


sortresults = resdf.sort_values(by=['C(Delta)']).reset_index(drop=True)
print(sortresults)

recBinWidthSSM = sortresults['Delta'][0]
print("\nThe recommended bin width is,d ={0}\n".format(recBinWidthSSM))
figure1 , axis1 = plt.subplots()
axis1.set_title('Box Plot')
axis1.boxplot(gamma_x, labels = ['X'])
axis1.grid(linestyle = "-",linewidth = 1)
plt.show()

#************Question 1.c*************

print("\n\n\nd)Q1.d)	(5 points) Based on your recommended bin width answer in (c), list the mid-points and the estimated density function values.  Draw the density estimator as a vertical bar chart using the matplotlib.  You need to properly label the graph to receive full credit.\n")
lowY = sortedResult['Low Y'][0]
# recommendedBinWidthSSM = sortedResult['Low Y'][0]
d = sortedResult['Delta'][0]
nBin = sortedResult['N Bin'][0]
binFreq = sortedResult['binFreq'][0]
N = len(gamma_x)
depth = np.arange(nBin) * d
value_add = lowY + 0.5 * d
binMid = value_add + depth
# binMid = lowY + 0.5 * d + np.arange(nBin) * d
p = []
for m_i in binMid:
    u = (gamma_x - m_i) / d
    w = np.where(np.logical_and(u > -1/2,u<= 1/2), 1, 0)
    sum_w = np.sum(w)
    p_i = sum_w / (N * d)
    p.append(p_i)

midPointVSDensity = pd.DataFrame(
    {'Mid-Points':binMid, 'Estimated Density Function Values':p})
print("")
print(midPointVSDensity)
print("")   
# Create Vertical Bar Chart
plt.bar(binMid, p, width=0.8, align='center', label="Bin-width={0}".format(recommendedBinWidthSSM))
plt.legend()
x_tick = np.arange(lowY, max(binMid) + d, 1)
y_tick = np.arange(0, max(p) + 0.025, 0.025)
plt.xticks(x_tick)
plt.yticks(y_tick)
plt.xlabel("Mid-points")
plt.ylabel("Estimated Density Function Values")
plt.title("Mid-points v/s Estimated Density Function Values")



    
    