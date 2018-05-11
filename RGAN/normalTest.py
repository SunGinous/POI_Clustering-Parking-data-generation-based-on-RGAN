# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:22:48 2018

@author: qicdz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest,shapiro,normaltest
import seaborn as sns

data = []
for i in range(30):
    if i != 9:
        df = pd.read_excel('parking.xls', usecols=[i])
        df = [n[0] for n in df[1500:1788].values.tolist()] 
        data.append(df)

data_T = np.array(data).T
print(len(data_T))

a = 0
b = 0
for i in data_T:
    b += 1
    if normaltest(i)[1] < 0.05:
        a += 1
print(b,'  ',a,' ',a/b)

sns.set_style('darkgrid')
plt.subplot(211)
for i in range(len(data)):
    if i != 11:
        plt.plot(data[i], 'lightsteelblue',linewidth=2)
        plt.plot(25,data[i][25], '.',color = 'lightcoral')
    else:
        plt.plot(data[i], 'lightskyblue',linewidth=3)
        plt.plot(25,data[i][25], '.',color = 'lightcoral')
plt.yticks(fontsize=19)
plt.xticks(fontsize=19)
plt.xlabel('Time Point', fontsize=20)
plt.ylabel('Unoccupied Parking Space Rate', fontsize=20)
plt.axvline(20)
plt.axvline(30)

plt.subplot(212)
for i in range(len(data)):
    if i != 11:
        plt.plot(data[i], 'lightsteelblue',linewidth=2)
        plt.plot(25,data[i][25], '.',color = 'lightcoral')
plt.yticks(fontsize=19)
plt.xticks(fontsize=19)
plt.xlabel('Time Point', fontsize=20)
plt.ylabel('Unoccupied Parking Space Rate', fontsize=20)
plt.axvline(20)
plt.axvline(30)
fig = plt.gcf()
fig.set_size_inches(15,15)
fig.savefig('compare.png', dpi = 100, bbox_inches='tight')
plt.show()
