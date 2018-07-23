# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:32:15 2018

@author: david
"""

#%% Importing Modules!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn import linear_model
import statsmodels.api as sm
import seaborn as sns
#%% Data Import & Preprocessing
xlsFile = 'C:/Users/david/Desktop/Challenge/project/terrorData.xlsx'
data = pd.read_excel(xlsFile)

#%% Compare certain vs Doubtful terrorist attack to understand whether decision
# criteria for terroristic attack definition changed over the years

data_doubt = data[data['doubtterr'] == 1]
data_NOdoubt = data[data['doubtterr'] == 0]
data_unknown = data[data['doubtterr'] == -9]

years = data['iyear'].drop_duplicates()

countXyear_doubt = data_doubt.groupby('iyear')['eventid'].count()
countXyear_NOdoubt = data_NOdoubt.groupby('iyear')['eventid'].count()
countXyear_unkown = data_unknown.groupby('iyear')['eventid'].count()
total_unkn = np.concatenate((countXyear_unkown,np.zeros(46-27)))


datasetN = {'Years': np.array(years), 'N. of Doubtful Terrorist attack': np.array(countXyear_doubt), 'N. of Certain Terrorist attack' : np.array(countXyear_NOdoubt), 'unknown' : np.array(total_unkn)  }
dataY_x_Doubt = pd.DataFrame(data=datasetN)


## PLOT
x = dataY_x_Doubt['Years']
y = dataY_x_Doubt['N. of Certain Terrorist attack']
y2= dataY_x_Doubt['N. of Doubtful Terrorist attack']

f, (ax1, ax2) = plt.subplots(2)
ax1.bar(x,y, width=0.1, color="lightblue",label = 'Certain Terr. Attack.', zorder=0)
sns.regplot(x, y, ax=ax1)
ax1.set_ylim(0, 14000)
ax1.set_title('N of Terrorist Attack par Year')
ax1.legend()

ax2.bar(x,y2, width=0.1, color="orange",label = 'Doubtful Terr. Attack.', zorder=0)
sns.regplot(x, y2, ax=ax2)
ax2.set_ylim(0, 14000)
ax2.legend()

#%% Investigate number of certain attacks par geographic area
# df.loc[(df['column_name'] == some_value) & df['other_column'].isin(some_values)]


data_Europe = data_NOdoubt.loc[data_NOdoubt['region'].isin([8,9])]
data_Americas = data_NOdoubt.loc[data_NOdoubt['region'].isin([1,2,3])]
data_Asia = data_NOdoubt.loc[data_NOdoubt['region'].isin([4,5,6,7])]
data_Africa = data_NOdoubt.loc[data_NOdoubt['region'].isin([10,11])]
data_Australasia = data_NOdoubt.loc[data_NOdoubt['region'].isin([12])]


countXyear_Europe = data_Europe.groupby('iyear')['eventid'].count()
countXyear_Americas = data_Americas.groupby('iyear')['eventid'].count()
countXyear_Asia = data_Asia.groupby('iyear')['eventid'].count()
countXyear_Africa = data_Africa.groupby('iyear')['eventid'].count()
countXyear_Austral = data_Australasia.groupby('iyear')['eventid'].count()





## PLOT
#EUROPE
x_Europe = countXyear_Europe.index.values
y_Europe = countXyear_Europe
#Americas
x_America = countXyear_Americas.index.values
y_America = countXyear_Americas
#Asia
x_Asia = countXyear_Asia.index.values
y_Asia = countXyear_Asia
#Africa
x_Africa = countXyear_Africa.index.values
y_Africa = countXyear_Africa
#Australasia
x_Austral = countXyear_Austral.index.values
y_Austral = countXyear_Austral


# Format Subplots
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.3)
plt.ylabel('N of Attacks')
#First Plot
ax1.bar(x_Europe,y_Europe, width=.5, color="palevioletred",label = 'Europe', alpha=0.5,zorder=0)
sns.regplot(x_Europe, y_Europe,color="violet", ax=ax1)
ax1.set_title('N of Terrorist Attack par Year in Each Region')
ax1.set_ylim(0, None)
ax1.legend()
ax1.set_ylabel('')
ax1.set_xticks([])
#Second Plot
ax2.bar(x_America,y_America, width=0.5,color="navy",label = 'America', zorder=0)
sns.regplot(x_America, y_America,color="mediumblue", ax=ax2)
ax2.set_ylim(0, None)
ax2.legend()
ax2.set_ylabel('')
ax2.set_xticks([])
#Third Plot
ax3.bar(x_Asia,y_Asia, width=0.1,color="orange",label = 'Asia', zorder=0)
sns.regplot(x_Asia, y_Asia, color = 'orange', ax=ax3)
ax3.set_ylim(0, None)
ax3.legend()
ax3.set_ylabel('')
ax3.set_xticks([])
#Fourth Plot
ax4.bar(x_Africa,y_Africa, width=0.1,color="blueviolet",label = 'Africa', zorder=0)
sns.regplot(x_Africa, y_Africa, color = 'darkslateblue', ax=ax4)
ax4.set_ylim(0, None)
ax4.legend()
ax4.set_ylabel('')
ax4.set_xticks([])
#Fifth Plot
ax5.bar(x_Austral,y_Austral, width=0.1,color="firebrick",label = 'Australasia', zorder=0)
sns.regplot(x_Austral, y_Austral, color = 'red', ax=ax5)
ax5.set_ylim(0, None)
ax5.legend()
ax5.set_ylabel('')
ax5.set(xticks = range(1970,2016,5))
f.text(0.04, 0.5, 'N. of Attacks', va='center', rotation='vertical')

