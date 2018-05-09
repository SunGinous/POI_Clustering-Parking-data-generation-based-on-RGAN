# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:25:57 2017

@author: sunginous

修改：11/3/2017
修改内容：统计每个停车场周围的建筑类型，加上停车场类型，最终得到每个停车场的七维向量，对七维向量进行聚类

"""

import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from pandas.tools.plotting import parallel_coordinates
import xlwt
import xlrd
import os
from xlutils.copy import copy
import seaborn as sns

'''
提取Excel表中的数据绘制某一类型建筑的点状分布图
'''
def figure_poi(name, row, col_lat, col_lng, style):
    df = pd.read_excel(name)
    lat = df.ix[row:, col_lat]
    lng = df.ix[row:, col_lng]
    x = []
    y = []
    for i in lng:
        x.append(float(i))
    for j in lat:
        y.append(float(j))
    sns.set_style('darkgrid')
    plt.plot(x,y,style)  
    plt.grid(True)

figure_poi('data/深圳市罗湖区公交地铁.xls', 0, 3, 4, 'b.')
figure_poi('data/深圳市罗湖区购物.xls', 0, 6, 7, 'g.')
figure_poi('data/深圳市罗湖区酒店.xls', 0, 6, 7, 'c.')
figure_poi('data/深圳市罗湖区旅游景点.xls', 0, 6, 7, 'm.')
figure_poi('data/深圳市罗湖区医疗.xls', 0 ,6, 7, 'r.')
figure_poi('data/深圳市罗湖区餐厅.xls', 0, 6, 7, 'y.')
figure_poi('data/深圳市罗湖区停车场.xls', 0, 6, 7,'k.')
plt.legend(['public transportation','shop','hotel','view spot','hospital & clinic','restaurant','parking lot'], fontsize=19)
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)
plt.title('POIs of Luohu District Shenzhen', fontsize=20)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
fig = plt.gcf()
fig.set_size_inches(15, 15)
fig.savefig('POI of Shenzhen City Luohu District.png', dpi=100, bbox_inches='tight')
plt.show()

'''
将每一种建筑的经纬度与type类型一起构成一个三维向量组
'''
total_spot = []
def add_spot(name, row, col_lat, col_lng, type):
    df = pd.read_excel(name)
    lat = df.ix[row:, col_lat]
    lng = df.ix[row:, col_lng]
    x = []
    y = []
    for i in lat:
        x.append(float(i))
    for j in lng:
        y.append(float(j))
    for k in range(len(x)):
        total_spot.append([x[k], y[k], type])
      
add_spot('data/深圳市罗湖区公交地铁.xls', 0, 3, 4, 1)
add_spot('data/深圳市罗湖区购物.xls', 0, 6, 7, 2)
add_spot('data/深圳市罗湖区酒店.xls', 0, 6, 7, 3)
add_spot('data/深圳市罗湖区旅游景点.xls', 0, 6, 7, 4)
add_spot('data/深圳市罗湖区医疗.xls', 0 ,6, 7, 5)
add_spot('data/深圳市罗湖区餐厅.xls', 0, 6, 7, 6)
add_spot('data/深圳市罗湖区停车场.xls', 0, 6, 7, 7)
print('总的地点数量：', len(total_spot))
print('.........................................................................')

'''
单独统计停车场的经纬度信息
'''
df = pd.read_excel('data/深圳市罗湖区停车场.xls')
lat = df.ix[0:, 6]
lng = df.ix[0:, 7]
x = []
y = []
for i in lat:
    x.append(float(i))
for j in lng:
    y.append(float(j))
parking_lots = []
for k in range(len(x)):
    parking_lots.append([x[k], y[k]])

'''
计算任意两点间的距离
'''        
def count_distance(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance

'''
计算每一个停车场到所有建筑的距离，其中0.002778°约等于380米，
最终每一个停车场都都得到一个六维向量，分别对应380米范围内6种建筑的数目
'''
multi_type = []
for park in parking_lots:
    public = 0
    shop = 0
    hotel = 0
    scenic = 0
    hospital = 0
    resturant = 0
    parkinglots = 0
    for spot in total_spot:
        if count_distance(park[0], park[1], spot[0], spot[1]) <= 0.002778 and count_distance(park[0], park[1], spot[0], spot[1]) != 0:
            if spot[2] == 1:
                public += 1
            elif spot[2] == 2:
                shop += 1
            elif spot[2] == 3:
                hotel += 1
            elif spot[2] == 4:
                scenic += 1
            elif spot[2] == 5:
                hospital += 1
            elif spot[2] == 6:
                resturant += 1
            elif spot[2] == 7:
                parkinglots += 1
    multi_type.append([public, shop, hotel, scenic, hospital, resturant, parkinglots])

print('每个停车场的七维向量是：')
print(multi_type)
print('.........................................................................')

'''
对所有停车场根据七维向量进行聚类
'''
clf = KMeans(n_clusters=6)
y_pred = clf.fit_predict(multi_type)
print('clf:',clf)
print('y_pred:',y_pred)
print('.........................................................................')

'''
将聚类的结果存储到“停车场六维向量.xls”中
'''
wbk = xlwt.Workbook()
sheet = wbk.add_sheet('a test sheet')
title = ['name','public transportation','shop','hotel','view spot','hospital & clinic','restaurant','parking lot','type']
parking_type = [int(i) for i in y_pred]
type_num = [0,0,0,0,0,0]
for i in parking_type:
    type_num[i] += 1
print('0、1、2、3、4、5类停车场数目分别有：',type_num)
df = pd.read_excel('data/深圳市罗湖区停车场.xls')
for index,key in enumerate(title):
    sheet.write(0,index,key)
for i in range(len(multi_type)):
    sheet.write(i+1, 0, df.ix[i,1])
    for j in range(7):
        sheet.write(i+1, j+1, multi_type[i][j])
    sheet.write(i+1, 8, 'type'+str(parking_type[i]))
wbk.save('停车场七维向量.xls')

'''
从刚才保存的“停车场六维向量.xls”表中提取数据绘制平行坐标系
'''
data = pd.read_excel('停车场七维向量.xls',usecols=[1,2,3,4,5,6,7,8])
#plt.subplot(111,axisbg='ghostwhite')#设置图片的背景色
parallel_coordinates(data,'type',color=['#aaaaaa', '#d87a80', '#ffb980','#5ab1ef','#b6a2de','#2ec7c9'],linewidth=3)#灰红黄蓝紫绿
plt.xlabel('POI', fontsize=20)
plt.ylabel('Number of POIs', fontsize=20)
plt.title('POI Clustering  type0:{}  type1:{}  type2:{}  type3:{}  type4:{}  type5:{}'.format(type_num[0],type_num[1],type_num[2],type_num[3],type_num[4],type_num[5]), fontsize=21)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.legend(fontsize=19)
fig = plt.gcf()
fig.set_size_inches(15, 10)
fig.savefig('POI Clustering of 7d.png', dpi=100, bbox_inches='tight')
plt.show()

'''
给停车场加type属性，并统计各个类型停车场的数目
'''
rd = xlrd.open_workbook('data/深圳市罗湖区停车场.xls', formatting_info=True)
wb = copy(rd)
sheet = wb.get_sheet(0)
sheet.write(0,8,'type')
for i in range(len(parking_type)):
    sheet.write(i+1, 8, 'type'+str(parking_type[i]))
os.remove('data/深圳市罗湖区停车场.xls')
wb.save('data/深圳市罗湖区停车场.xls')
print('.........................................................................')
df = pd.read_excel('data/深圳市罗湖区停车场.xls')
type0 = []
type1 = []
type2 = []
type3 = []
type4 = []
type5 = []
for i in range(304,len(df)):
    if str(df.ix[i,8]) == 'type0':
        type0.append(str(df.ix[i,1]))
    if str(df.ix[i,8]) == 'type1':
        type1.append(str(df.ix[i,1]))
    if str(df.ix[i,8]) == 'type2':
        type2.append(str(df.ix[i,1]))
    if str(df.ix[i,8]) == 'type3':
        type3.append(str(df.ix[i,1]))
    if str(df.ix[i,8]) == 'type4':
        type4.append(str(df.ix[i,1]))
    if str(df.ix[i,8]) == 'type5':
        type5.append(str(df.ix[i,1]))
print('type0 : ',len(type0),'个',' : ', type0)
print('type1 : ',len(type1),'个',' : ', type1)
print('type2 : ',len(type2),'个',' : ', type2)
print('type3 : ',len(type3),'个',' : ', type3)
print('type4 : ',len(type4),'个',' : ', type4)
print('type5 : ',len(type5),'个',' : ', type5)
 