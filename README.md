# POI_Clustering-Parking-data-generation-based-on-RGAN

## 2018/5/8上传
- Parking curve中为停车数据和初步数据分析。出于隐私考虑，在该目录下的data文件夹中至给出了深圳市罗湖区百汇大厦停车场的停车数据。原始停车数据是一张数据表，记录了每辆车出入停车场的具体时间，parking_curve.py的运行效果是将停车数据表转换为一维时间序列数据，即以5分钟为时间步长，统计每个时间节点的停车场的停车数，然后将停车数数据转换为空车率数据，即完成数据的归一化。
- parking_curve.py的运行结果包括停车场从2016年6月至2017年6月每个月的停车曲线图及数据总量并以excel形式存储。
- POI_Clustering中对深圳市罗湖区停车场进行聚类，聚类依据是停车场周围308米（约精度1秒）范围内各种POI（Point of Interest）的数量，即一个7维的向量，各种POI的经纬度通过爬虫程序从百度地图上获得。
- 聚类使用K-Means算法，这里只是直接引用了sklearn包，聚类的结果以平行坐标系的形式表现出来。![](https://github.com/SunGinous/POI_Clustering-Parking-data-generation-based-on-RGAN/blob/master/POI_Clustering/POI%20Clustering%20of%207d.png)

## 2018/5/10上传
- 新增RGAN(Recurrent GAN)程序，RGAN是将原始GAN中的生成器和鉴别器替换为LSTM，在时间序列数据生成方面效果显著。
- RGAN文件夹下parking.xls为已做脱敏处理的深圳市部分停车场数据。
- 以任意一条停车数据为样本，训练RGAN，可以生成具有相似特征的新数据。但由于GAN本身的不稳定性，部分生成的数据会存在异常。![](https://github.com/SunGinous/POI_Clustering-Parking-data-generation-based-on-RGAN/blob/master/RGAN/iteration_5000batchsize_288/1.png)![](https://github.com/SunGinous/POI_Clustering-Parking-data-generation-based-on-RGAN/blob/master/RGAN/iteration_5000batchsize_288/9p.png)

## 2018/5/11上传
- 增加对深圳市部分停车场数据的正态性检验，对于数据量较少的情况，使用D'Agostino检验(D检验)效果较好。验证结果是聚类后，同一类型停车场数据具有一定的正态性。 
