# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:06:46 2017

@author: sunginous
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
import pandas as pd
import time
import seaborn as sns
import calendar
import os
import xlwt
import datetime

'''
程序运行统计
'''
flag = 0
error = []

'''
统计每个停车场当月的停车数据
'''
count_parking_num = []

'''
得到一个二维数组，分别存储year年mon月份每一辆车的进、出时间戳
'''
def get_time_list(name, year, mon):
    try:
        df = pd.read_excel('data\\'+name)
        in_time =df.ix[8:, 1]
        out_time = df.ix[8:, 2]
        parking_time = []
        for i in range(8, len(in_time)+8):
            parking_time.append([in_time[i],out_time[i]])
    except:
        error.append([name, '全部', '全部', '全部'])
        return None
    #由于数据集间的差别，这里需要多次使用try except来读取数据，防止出现ValueError    
    stamp1 = "%Y-%m-%d %H:%M:%S"
    stamp2 = "%Y/%m/%d %H:%M:%S"
    time_list = []
    for i in parking_time:
        try:
            try:
                t1 = time.strptime(str(i[0]), stamp1)
            except ValueError:
                try:
                    t1 = time.strptime(str(i[0])[:-7], stamp1)
                except ValueError:
                    t1 = time.strptime(str(i[0])[:-7], stamp2)
     
            if t1[0] == year and  t1[1] == mon:
                in_localtime = float(time.mktime(t1))
                try:
                    t2 = time.strptime(str(i[1]), stamp1)
                except ValueError:
                    try:
                        t2 = time.strptime(str(i[1])[:-7], stamp1)
                    except ValueError:
                        t2 = time.strptime(str(i[1])[:-7], stamp2)                       
                out_localtime = float(time.mktime(t2))
                time_list.append([in_localtime, out_localtime])
        except ValueError:
            print(name, '第'+str(parking_time.index(i))+'行', i)
            error.append([name, parking_time.index(i), i[0], i[1]])
            continue
            
    print(name, len(time_list))
    count_parking_num.append([name, len(time_list)])        
    return time_list

'''
将出、入场数据分别提出生成两个二维数组，入场的第二维数据为1，出场的第二维数据为-1，
最后将两个二维数组合并，按时间升序排列
'''
def get_car(name, year, mon):
    car_in = []
    car_out = []
    time_list = get_time_list(name, year, mon)
    if time_list == None:
        return None
    for i in time_list:
        car_in.append([i[0], 1])
        car_out.append([i[1], -1])
    car = []
    for i in range(len(car_in)):
        car.append(car_in[i])
        car.append(car_out[i]) 
    car.sort()
    return car

'''
将日期转换成时间戳
'''
def to_time_stamp(realtime):
    stamp = "%Y-%m-%d %H:%M:%S"
    time_stamp = time.mktime(time.strptime(realtime, stamp))
    return time_stamp

'''
每300秒做一个时间节点，统计当前时间节点下的停车数量
'''
def get_car_list(name, year, mon):
    car_num = 0
    car_list = []
    car = get_car(name, year, mon)
    if car == None:
        return None
    start = str(year)+'-'+str(mon)+'-01 00:00:00'
    month_range = calendar.monthrange(year, mon)
    end = str(year)+'-'+str(mon)+'-'+str(month_range[1])+' 23:59:59'
    time1 = int(to_time_stamp(start))
    time2 = int(to_time_stamp(end))
    for i in range(time1, time2, 300):
        for j in car:
            if j[0] <= i:
                car_num += j[1]
                car.remove(j)
        car_list.append(car_num)
    return car_list

'''
根据停车数据进行绘图，横坐标为每5分钟一个的时间节点，纵坐标为停车场的空车率
'''
def get_figure(name, year, mon, style):
    car_list = get_car_list(name, year, mon)
    if car_list == None:
        print(name+'有误，程序结束')
        return 
    unoccupied = []
    for i in car_list:
        try:
            percend = (max(car_list)-i) / max(car_list)
            unoccupied.append(percend)
        except ZeroDivisionError:
            unoccupied.append(1)       
    month_range = calendar.monthrange(year, mon)
    x = [i for i in range(288*month_range[1])]
    sns.set_style('darkgrid')
    myfont = FontProperties(fname=r'C:\Windows\Fonts\MSYH.TTC')  
    matplotlib.rcParams['axes.unicode_minus']=False  
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x, unoccupied, style)
    plt.title(name[:-5]+' '+calendar.month_name[mon]+' '+str(year), fontproperties=myfont, fontsize=23)
    plt.ylabel('Unoccupied Parking Space Rate', fontsize=20)
    plt.xlabel('Time Point', fontsize=20)
    fig = plt.gcf()
    fig.set_size_inches(30,10)
    fig.savefig(str(year)+'年'+str(mon)+'月'+name[:-5]+'.png', dpi=100)
    plt.show()
  
def main(year, mon, style):
    global flag
    file_list = os.listdir('data')
    for i in range(len(file_list)):
        get_figure(file_list[i], year, mon, style)
        flag += 1
        print('第 '+str(flag)+' 次运行结束')
    
    for i in count_parking_num:
        print(i)
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('count')
    title = ['parking lot', 'parking data']
    for index,key in enumerate(title):
        sheet.write(0, index, key)
    for i in range(len(count_parking_num)):
        for j in range(2):
            sheet.write(i+1, j, count_parking_num[i][j])
    wbk.save(str(year)+calendar.month_name[mon]+' parking statistic.xls')
    count_parking_num.clear()

if __name__=='__main__':
    begin = datetime.datetime.now()
    for index, color in enumerate('rgbyrgb'):
        main(2016, index+6, color)
    for index, color in enumerate('rgbyrg'):
        main(2017, index+1, color)
    end = datetime.datetime.now()
    
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('count')
    title = ['parking lot', 'col', 'in', 'out']
    for index,key in enumerate(title):
        sheet.write(0, index, key)
    for i in range(len(error)):
        for j in range(4):
            sheet.write(i+1, j, error[i][j])
    wbk.save('error.xls')
    print('运行时间', end-begin)
