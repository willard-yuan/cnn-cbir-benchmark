## PCA使用数据源对CDW的影响

实验过程中，发现PCA使用不同数据源对CDW的影响是比较大的。下面是PCA采用不同数据源对CDW的影响实验，实验在oxford上进行，评鉴准则采用mAP。

实验设置：crop、no query expansion、do PCA

数据源 | 维度 | mAP
---|---|---
oxford | 512 | 60.2155%
oxford | 256 | 64.3746%
oxford | 128 | 66.9665%
oxford | 64 | 64.3458%
oxford | 32 | 58.0331%

数据源 | 维度 | mAP
---|---|---
paris | 512 | **70.8359%**
paris | 256 | 69.6122%
paris | 128 | 64.0718%
paris | 64 | 58.4009%
paris | 32 | 52.5268%

mAP在小数点后又微小浮动。

### no crop对特征的影响

实验设置：no crop、no query expansion, do PCA

数据源 | 维度 | mAP
---|---|---
oxford | 512 | 59.9517%
oxford | 256 | 64.3746%
oxford | 128 | 66.9665%
oxford | 64 | 64.3458%
oxford | 32 | 58.0331%

### qe对特征的影响

实验设置：no crop、query expansion、do PCA

top@K | 维度 | mAP
---|---|--- 
0 | 512 | 59.9517%
1 | 512 | 59.9517%
2 | 512 | 63.7079%
3 | 512 | 65.6768%
4 | 512 | 66.7678%
5 | 512 | 67.4205%
6 | 512 | 68.3001%
7 | 512 | 68.9647%
8 | 512 | 69.5633%
9 | 512 | 69.5831%
10 | 512 | 69.8873%

## 重构代码

重构后的代码完成的功能如下:

- 全图提取特征 or 区域框选提取特征
- do PCA or not
- do query expansion or not
- 特征可视化

重构后的检索精度指标评价：do crop, do qe(top@10), do PCA

| 维度 | mAP
|---|--- 
| 512 | 71.88%
| 256 | 72.04%
| 128 | 70.33%
| 64 | 65.6768%
| 32 | 58.8%

在几百万数据集上获取PCA，然后用在oxford上：

| 维度 | mAP
|---|--- 
| 128 | 59.15%
可能的原因：Oxford查询图片都是地标图像集，而这几百万数据集都是短视频中的一些数据，导致获得的主成分不利于地标数据的表达，所以精度降低。

### HybridNet

do crop, do qe(top@10), do PCA，最高维度256。

| 维度 | mAP
|---|--- 
| 256 | 62.41%

> We chose the HybridNet for several reasons: first, its ar- chitecture is the same as the famous AlexNet [19]; second, the HybridNet has been trained on the ImageNet subset used for ILSVRC competitions (as many others) and the Places Database [29]; last, but not least, experiments conducted on various datasets demonstrate the good transferability of the learning [29, 12, 9]. Originally proposed in [29], Hybrid- Net has been used in [29, 12, 9]. The results reported in [12] show that deep features extracted from the HybridNet outperforms various architectures trained only on ImageNet, on both InriaHolidays and OxforBuilding benchmarks.