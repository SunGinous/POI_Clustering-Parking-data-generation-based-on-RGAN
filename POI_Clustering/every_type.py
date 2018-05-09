# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:38:04 2018

@author: qicdz
"""

import pandas as pd
import xlwt
import sys
import os

try:
    df = pd.read_excel('停车场七维向量.xls')
except FileNotFoundError:
    print('** 先运行POI_Clustering.py生成停车场七维向量.xls **')
    sys.exit(0)

if not os.path.exists('type'):
    os.makedirs('type')

type0 = []
type1 = []
type2 = []
type3 = []
type4 = []
type5 = []
for i in range(304,len(df)):
    if str(df.ix[i,8]) == 'type0':
        type0.append(df.loc[i].tolist())
    if str(df.ix[i,8]) == 'type1':
        type1.append(df.loc[i].tolist())
    if str(df.ix[i,8]) == 'type2':
        type2.append(df.loc[i].tolist())
    if str(df.ix[i,8]) == 'type3':
        type3.append(df.loc[i].tolist())
    if str(df.ix[i,8]) == 'type4':
        type4.append(df.loc[i].tolist())
    if str(df.ix[i,8]) == 'type5':
        type5.append(df.loc[i].tolist())

type_all = [type0,type1,type2,type3,type4,type5]
for t in range(len(type_all)):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('test')
    for i in range(len(type_all[t])):
        sheet.write(i, 0, type_all[t][i][0])
        for j in range(7):
            sheet.write(i, j+1, int(type_all[t][i][j+1]))
        sheet.write(i, 8, type_all[t][i][8])
    wbk.save('type/type'+str(t)+'.xls')
