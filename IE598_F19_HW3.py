#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2-1
import pandas as pd
data = pd.read_csv("HY_Universe_corporate bond.csv") 


# In[3]:


#2-2#import urllib2
import sys
data = pd.read_csv("HY_Universe_corporate bond.csv")
lol = data.values.tolist()
#arrange data into list for labels and list of lists for attributes
xList = []
labels = []
#for line in lol:
#for line in lsl: 
for line in lol: 
#split on comma
    row = line
    xList.append(row)
nrow = len(xList)
ncol = len(xList[1])
type = [0]*3
colCounts = []

for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0]*3
sys.stdout.write("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' + str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1
#here we can see the columns with different types of the 
#values column 9,10,13,15,16,17,18,(20:36) have numeric values
#
datanum910 = data[data.columns[9:10]]
datanum13 = data[data.columns[13]]
datanum1518= data[data.columns[15:18]]
datanum2035 = data[data.columns[20:35]]
#datanum2035


# In[6]:


datX = pd.concat([datanum910, datanum13,datanum1518,datanum2035], axis=1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(datX)
datX_std = sc.transform(datX)


# In[11]:


#since the range for number is very large, we standard it using scaler
from pandas import DataFrame
dataX = DataFrame.from_records(datX_std)


# In[12]:


datanumy = data[data.columns[36]]
#datanumy
dataXy= pd.concat([dataX, datanumy],axis=1)
dataXy


# In[13]:


#2-5print the head and tail of the original dataframe
print(data.head())
print(data.tail())
#2-5print the summary of the original dataframe
summary = data.describe()
print(summary)


# In[16]:


#2-4
import numpy as np
import pylab
import scipy.stats as stats
import sys
colDat = dataXy[2]
stats.probplot(colDat, dist="norm", plot=pylab)
pylab.show()


# In[17]:


c= dataXy["weekly_median_ntrades"]


d= pd.DataFrame(c)


d.stack().value_counts().to_dict()               
               
#print dataXy["weekly_median_ntrades"].value_counts()


# In[19]:


dataXy


# In[22]:


#2-6parallel coordinates plot.
#here we use weekly_median_ntrades as labels
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
#data = pd.read_csv("HY_Universe_corporate bond.csv")

#datanumX_std
#datanumy
#here we only use first 300 instances to make the plot

for i in range(300):
#assign color based on bond_type labels
    if dataXy.iat[i,20] == 1:
       pcolor = "red"
#    if dataXy.iat[i,21] == 2:
#       pcolor = "green"
#    if dataXy.iat[i,21] == 3:
#       pcolor = "yellow" 
#    if dataXy.iat[i,21] == 4:
#       pcolor = "brown"
    else:
       pcolor = "blue"
#plot rows of data as if they were series data
    colindex = []
    
    dataRow = dataXy.iloc[i,0:20]
    dataRow.plot(color=pcolor)
plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()


# In[23]:


#2-7Visualizing Interrelationships between Attributes and Labels

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

#data = pd.read_csv("HY_Universe_corporate bond.csv")
#calculate correlations between real-valued instances
#this is the correlations between two instance, 2nd and 3rd instance
dataRow2 = dataXy.iloc[1,0:21]
dataRow3 = dataXy.iloc[2,0:21]
plot.scatter(dataRow2, dataRow3)
plot.xlabel("2nd Instance")
plot.ylabel(("3rd Instance"))
plot.show()

#this is the correlations between two instance, 2nd and 21st instance
dataRow21 = dataXy.iloc[20,0:21]
plot.scatter(dataRow2, dataRow21)
plot.xlabel("2nd Instance")
plot.ylabel(("21st Instance"))
plot.show()


# In[24]:


#let's see the correlations between 2nd and 3rd attributes
dataCol2 = dataXy.iloc[0:2720,1]
dataCol3 = dataXy.iloc[0:2720,2]
plot.scatter(dataCol2, dataCol3)
plot.xlabel("2nd Attribute")
plot.ylabel(("3rd Attribute"))
plot.show()


# In[26]:


#2-8Correlation between Classification Target and Real Attributes—targetCorr
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
from random import uniform

#the column 22th is the target we are used
target = dataXy.iloc[0:2720,20]

#see 5th attributes
dataCol = dataXy.iloc[0:2720,4]
plot.scatter(dataCol, target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()


# In[27]:


#Listing 2-9: Pearson’s Correlation Calculation for instance 2 versus 3 and 2 versus 12
import pandas as pd
from pandas import DataFrame
from math import sqrt
import sys
#calculate correlations between real-valued attributes
dataCol2 = dataXy.iloc[0:2720,1]
dataCol3 = dataXy.iloc[0:2720,2]
dataCol12 = dataXy.iloc[0:2720,11]
mean2 = 0.0; mean3 = 0.0; mean12 = 0.0

numElt = len(dataCol2)

for i in range(numElt):
    mean2 += dataCol2[i]/numElt
    mean3 += dataCol3[i]/numElt
    mean12 += dataCol12[i]/numElt
    
    
var2 = 0.0; var3 = 0.0; var12 = 0.0
for i in range(numElt):
    var2 += (dataCol2[i] - mean2) * (dataCol2[i] - mean2)/numElt
    var3 += (dataCol3[i] - mean3) * (dataCol3[i] - mean3)/numElt
    var12 += (dataCol12[i] - mean12) * (dataCol12[i] - mean12)/numElt

corr23 = 0.0; corr212 = 0.0
for i in range(numElt):
    corr23 += (dataCol2[i] - mean2) *     (dataCol3[i] - mean3) / (sqrt(var2*var3) * numElt)
    corr212 += (dataCol2[i] - mean2) *      (dataCol12[i] - mean12) / (sqrt(var2*var12) * numElt)

sys.stdout.write("Correlation between attribute 2 and 3 \n")
print(corr23)
sys.stdout.write(" \n")

sys.stdout.write("Correlation between attribute 2 and 12 \n")
print(corr212)
sys.stdout.write(" \n")


# In[28]:


#2-10
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

#calculate correlations between real-valued attributes
corMat = DataFrame(dataXy.corr())
#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()


print("My name is Xuehui Chao")
print("My NetID is: xuehuic2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
######STOP HERE######################





