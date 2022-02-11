### faiss-2 快速入门

### 数据准备

faiss可以处理**固定维度**`d`的**向量集合，该集合用二维数组表示**。 一般来说，需要两个数组：

* `data`：包含被索引的所有向量元素；

* `query`：索引向量，需要根据索引向量的值返回与向量集中的最近邻元素；

为了对比不同索引方式的差别，在下面的例子中统一使用完全相同的数据，即维数d为512，data包含2000个向量，每个向量符合正态分布。

> **注意**
>
> **faiss需要向量数组中的元素都是32位浮点数格式**， `datatype = 'float32'`



**制作向量集合数据**

```python
import numpy as np
from matplotlib import pyplot as plt

d = 512  # 维数
n_data = 2000  # 向量集合中的向量个数
np.random.seed(0)
data = []
mu = 3
sigma = 0.1

for i in range(n_data):
    data.append(np.random.normal(mu, sigma, d))

# convert into float32
data = np.array(data).astype('float32')

# print(data[0])

# 查看第六个向量是否符合正态分布
plt.hist(data[5])
plt.show()
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAXoAAAD8CAYAAAB5Pm/hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvIxREBQAADjJJREFUeJzt3X2MZfVdx/H3RxaklMryMNkACw5JSSvWWshIaDC1dmukYoEYQpaoXcmaTbRWKka71j+I/WuJprWmpnVTareVUAilAQU1uNBUTbo6PJSnBdlQHhYXdipPrY3WpV//uIdmXIeZu/fcmbvz4/1KJvecc3/nnu83u/OZc3/3nntTVUiS2vVDky5AkrS8DHpJapxBL0mNM+glqXEGvSQ1zqCXpMYZ9JLUOINekhpn0EtS49ZMugCAk046qaanpyddhiStKnffffe3qmpqqXGHRdBPT08zOzs76TIkaVVJ8uQw45y6kaTGGfSS1DiDXpIaZ9BLUuMMeklqnEEvSY0z6CWpcQa9JDXOoJekxh0WV8ZKh6vprbdN7NhPbLtwYsdWWzyjl6TGGfSS1DiDXpIaZ9BLUuMMeklqnEEvSY0z6CWpcQa9JDXOoJekxhn0ktS4JYM+yeeS7E/y4LxtJyS5I8lj3e3x3fYk+bMke5Lcn+Sc5SxekrS0Yc7oPw9ccNC2rcDOqjoT2NmtA7wPOLP72QJ8ejxlSpJGtWTQV9XXgOcP2nwxsKNb3gFcMm/7F2rg68DaJCePq1hJ0qEbdY5+XVXt65afBdZ1y6cCT88bt7fbJkmakN4vxlZVAXWo+yXZkmQ2yezc3FzfMiRJr2HUoH/u1SmZ7nZ/t/0Z4LR549Z32/6fqtpeVTNVNTM1NTViGZKkpYwa9LcCm7rlTcAt87Z/oHv3zXnAS/OmeCRJE7DkN0wluR54N3BSkr3A1cA24MYkm4Engcu64bcDvwDsAb4LXLEMNUuSDsGSQV9Vl7/GXRsWGFvAB/sWJUkaH6+MlaTGGfSS1DiDXpIat+QcvaTJmN5620SO+8S2CydyXC0fz+glqXEGvSQ1zqCXpMYZ9JLUOINekhpn0EtS4wx6SWqcQS9JjTPoJalxBr0kNc6gl6TGGfSS1DiDXpIaZ9BLUuMMeklqnEEvSY0z6CWpcQa9JDXOrxLUqjCpr9WTWuAZvSQ1zqCXpMYZ9JLUOINekhpn0EtS4wx6SWqcQS9JjesV9El+J8lDSR5Mcn2So5OckWRXkj1Jbkhy1LiKlSQdupGDPsmpwG8DM1X1NuAIYCNwDfCJqnoz8AKweRyFSpJG03fqZg3whiRrgGOAfcB7gJu6+3cAl/Q8hiSph5GDvqqeAf4EeIpBwL8E3A28WFUHumF7gVMX2j/JliSzSWbn5uZGLUOStIQ+UzfHAxcDZwCnAG8ELhh2/6raXlUzVTUzNTU1ahmSpCX0mbp5L/DNqpqrqv8BbgbOB9Z2UzkA64FnetYoSeqhT9A/BZyX5JgkATYADwN3AZd2YzYBt/QrUZLUR585+l0MXnS9B3ige6ztwEeAq5LsAU4Erh1DnZKkEfX6PPqquhq4+qDNjwPn9nlcSdL4eGWsJDXOoJekxhn0ktQ4g16SGueXg+uQ+CXd0urjGb0kNc6gl6TGGfSS1DiDXpIaZ9BLUuMMeklqnEEvSY0z6CWpcQa9JDXOoJekxhn0ktQ4g16SGmfQS1LjDHpJapxBL0mNM+glqXEGvSQ1zqCXpMYZ9JLUOINekhpn0EtS4wx6SWqcQS9JjTPoJalxvYI+ydokNyV5JMnuJO9MckKSO5I81t0eP65iJUmHru8Z/SeBv6uqtwI/CewGtgI7q+pMYGe3LkmakJGDPslxwLuAawGq6ntV9SJwMbCjG7YDuKRvkZKk0fU5oz8DmAP+Msm9ST6b5I3Auqra1415FljXt0hJ0uj6BP0a4Bzg01V1NvCfHDRNU1UF1EI7J9mSZDbJ7NzcXI8yJEmL6RP0e4G9VbWrW7+JQfA/l+RkgO52/0I7V9X2qpqpqpmpqakeZUiSFrNm1B2r6tkkTyd5S1U9CmwAHu5+NgHbuttbxlKppBUxvfW2iR37iW0XTuzYLRs56DsfAq5LchTwOHAFg2cJNybZDDwJXNbzGJKkHnoFfVXdB8wscNeGPo8rSRofr4yVpMYZ9JLUOINekhpn0EtS4wx6SWqcQS9JjTPoJalxBr0kNc6gl6TGGfSS1DiDXpIaZ9BLUuMMeklqnEEvSY0z6CWpcQa9JDXOoJekxhn0ktQ4g16SGmfQS1LjDHpJapxBL0mNM+glqXEGvSQ1zqCXpMYZ9JLUOINekhpn0EtS4wx6SWpc76BPckSSe5P8Tbd+RpJdSfYkuSHJUf3LlCSNahxn9FcCu+etXwN8oqreDLwAbB7DMSRJI+oV9EnWAxcCn+3WA7wHuKkbsgO4pM8xJEn99D2j/1Pg94Hvd+snAi9W1YFufS9was9jSJJ6WDPqjkl+EdhfVXcnefcI+28BtgCcfvrpo5bxujS99bZJlyBpFelzRn8+cFGSJ4AvMZiy+SSwNsmrf0DWA88stHNVba+qmaqamZqa6lGGJGkxIwd9Vf1BVa2vqmlgI3BnVf0ycBdwaTdsE3BL7yolSSNbjvfRfwS4KskeBnP21y7DMSRJQxp5jn6+qvoq8NVu+XHg3HE8riSpP6+MlaTGGfSS1DiDXpIaZ9BLUuMMeklqnEEvSY0z6CWpcQa9JDXOoJekxhn0ktQ4g16SGmfQS1LjDHpJapxBL0mNM+glqXEGvSQ1zqCXpMYZ9JLUOINekhpn0EtS4wx6SWqcQS9JjTPoJalxBr0kNc6gl6TGGfSS1DiDXpIaZ9BLUuMMeklqnEEvSY0bOeiTnJbkriQPJ3koyZXd9hOS3JHkse72+PGVK0k6VH3O6A8Av1tVZwHnAR9MchawFdhZVWcCO7t1SdKEjBz0VbWvqu7plr8N7AZOBS4GdnTDdgCX9C1SkjS6sczRJ5kGzgZ2Aeuqal9317PAutfYZ0uS2SSzc3Nz4yhDkrSA3kGf5Fjgy8CHq+rl+fdVVQG10H5Vtb2qZqpqZmpqqm8ZkqTX0CvokxzJIOSvq6qbu83PJTm5u/9kYH+/EiVJffR5102Aa4HdVfXxeXfdCmzqljcBt4xeniSprzU99j0f+FXggST3dds+CmwDbkyyGXgSuKxfiZJeL6a33jaR4z6x7cKJHHeljBz0VfVPQF7j7g2jPq4kaby8MlaSGmfQS1LjDHpJapxBL0mNM+glqXEGvSQ1rs/76F/3JvWeX0k6FJ7RS1LjPKOX9Lo3yWfnK3FVrmf0ktQ4g16SGmfQS1LjDHpJapxBL0mNM+glqXEGvSQ1zqCXpMYZ9JLUOINekhpn0EtS4wx6SWqcQS9JjTPoJalxBr0kNc6gl6TGGfSS1DiDXpIat+q/StAv6JakxXlGL0mNW5agT3JBkkeT7EmydTmOIUkaztiDPskRwJ8D7wPOAi5Pcta4jyNJGs5ynNGfC+ypqser6nvAl4CLl+E4kqQhLEfQnwo8PW99b7dNkjQBE3vXTZItwJZu9TtJHp1390nAt1a+qomw1zbZa3uWpc9c02v3Hx1m0HIE/TPAafPW13fb/o+q2g5sX+gBksxW1cwy1HbYsdc22Wt7VnOfyzF186/AmUnOSHIUsBG4dRmOI0kawtjP6KvqQJLfAv4eOAL4XFU9NO7jSJKGsyxz9FV1O3B7j4dYcEqnUfbaJnttz6rtM1U16RokScvIj0CQpMZNLOiTnJbkriQPJ3koyZULjPm9JPd1Pw8meSXJCZOot48hez0uyV8n+UY35opJ1NrXkL0en+QrSe5P8i9J3jaJWvtKcnRX/6v/Zn+0wJgfTnJD93Egu5JMr3yl/QzZ57uS3JPkQJJLJ1HnOAzZ61Xd/+/7k+xMMtRbHCeqqibyA5wMnNMtvwn4N+CsRca/H7hzUvUud6/AR4FruuUp4HngqEnXvky9/jFwdbf8VmDnpOsesdcAx3bLRwK7gPMOGvObwGe65Y3ADZOue5n6nAbeDnwBuHTSNS9zrz8LHNMt/8Zq+Ded2Bl9Ve2rqnu65W8Du1n8CtrLgetXorZxG7LXAt6UJMCxDIL+wIoWOgZD9noWcGc35hFgOsm6FS10DGrgO93qkd3PwS96XQzs6JZvAjZ0/8arxjB9VtUTVXU/8P2Vrm+chuz1rqr6brf6dQbXCh3WDos5+u7p7NkM/noudP8xwAXAl1euquWxSK+fAn4M+HfgAeDKqlrVvzSL9PoN4Je6MecyuLrvsP9lWUiSI5LcB+wH7qiqg3v9wUeCVNUB4CXgxJWtsr8h+mzGIfa6GfjblalsdBMP+iTHMgjwD1fVy68x7P3AP1fV8ytX2fgt0evPA/cBpwDvAD6V5EdWuMSxWaLXbcDa7pfpQ8C9wCsrXOJYVNUrVfUOBn+ozl2trzcs5fXSJwzfa5JfAWYYTEUe1iYa9EmOZBAG11XVzYsM3cgqnbZ51RC9XgHc3D113AN8k8H89aqzVK9V9XJVXdH9Mn2AwWsSj69wmWNVVS8CdzF45jnfDz4SJMka4DjgP1a2uvFZpM/mLNZrkvcCfwhcVFX/vdK1HapJvusmwLXA7qr6+CLjjgN+BrhlpWobtyF7fQrY0I1fB7yFVRh+w/SaZG338RgAvw58bZFnc4etJFNJ1nbLbwB+DnjkoGG3Apu65UsZvKFgVV28MmSfTRim1yRnA3/BIOT3r3yVh25iF0wl+WngHxnMR786F/1R4HSAqvpMN+7XgAuqauMEyhyLYXpNcgrweQbvWgmwrar+auWr7WfIXt/J4AXKAh4CNlfVCxMot5ckb2fQxxEMTppurKqPJfkYMFtVtyY5Gvgig9cqngc2VtWq+gM+ZJ8/BXwFOB74L+DZqvrxiRU9oiF7/QfgJ4B93W5PVdVFk6l4OF4ZK0mNm/iLsZKk5WXQS1LjDHpJapxBL0mNM+glqXEGvSQ1zqCXpMYZ9JLUuP8F80WBUXsFyawAAAAASUVORK5CYII=)

**制作请求向量**

```python
query = []
n_query = 10  # 请求向量十个
mu = 3
sigma = 0.1
np.random.seed(12)
query = []
for i in range(n_query):
    query.append(np.random.normal(mu, sigma, d))  # 请求向量维度要与向量集合维度一致

# convert into float32
query = np.array(query).astype('float32')
print(query[0])
```



### 精确索引

在使用faiss时，是围绕`index`对象进行的。**index中包含被索引的数据库向量**，**在索引时可以选择不同方式的预处理来提高索引的效率，表现维不同的索引类型**。

**在精确搜索时选择最简单的`IndexFlatL2`索引类型**。`IndexFlatL2`类型**遍历计算查询向量与被查询向量的L2精确距离（faiss中是没有开方运算）**，不需要训练操作（大部分index类型都需要train操作）。
在构建index时要提供相关参数：`向量维数d`，构建完成index之后可以通过`add()`向向量集合中添加向量和`search()`进行请求向量查询。

```python
import sys
import faiss

index = faiss.IndexFlatL2(d)  # 构建index，维度为512
print(index.is_trained)  # False时需要train
index.add(data)  # 添加数据
print(index.ntotal)  # index中向量的个数
```

```
True
2000
```



```python
k = 10  # 返回结果个数
query_self = data[:5]  # 查询本身
dis, ind = index.search(query_self, k)  # 搜索自己的前五个元素
print(dis)  # 升序返回每个查询向量的距离
print(ind)  # 升序返回每个查询向量的k个相似结果
```

```
[[0.        8.007045  8.313329  8.53525   8.560173  8.561645  8.624167
  8.628234  8.70998   8.770039 ]
 [0.        8.2780905 8.355575  8.426064  8.462012  8.468867  8.487029
  8.549965  8.562829  8.599193 ]
 [0.        8.152366  8.156565  8.223303  8.276013  8.376869  8.379269
  8.406124  8.418624  8.443278 ]
 [0.        8.26052   8.336825  8.339297  8.402873  8.46439   8.474661
  8.479044  8.485244  8.526601 ]
 [0.        8.34627   8.407206  8.4628315 8.497226  8.520797  8.597082
  8.600384  8.605135  8.63059  ]]
[[   0  798  879  223  981 1401 1458 1174  919   26]
 [   1  981 1524 1639 1949 1472 1162  923  840  300]
 [   2 1886  375 1351  518 1735 1551 1958  390 1695]
 [   3 1459  331  389  655 1943 1483 1723 1672 1859]
 [   4   13  715 1470  608  459  888  850 1080 1654]]
```

**因为查询向量是数据库向量的子集，所以每个查询向量返回的结果中排序第一的就是其本身，对应的index为其数组下标，L2距离是0.**



**查询随机生成的向量请求**

```python
k = 10
dis, ind = index.search(query, k)
print(dis)
print(ind)
```

```
[[8.61838   8.782156  8.782816  8.832027  8.837635  8.8484955 8.897978
  8.9166355 8.919006  8.937399 ]
 [9.033302  9.038906  9.091706  9.155842  9.164592  9.200113  9.201885
  9.220333  9.279479  9.312859 ]
 [8.063819  8.211029  8.306456  8.373353  8.459253  8.459894  8.498556
  8.546466  8.555407  8.621424 ]
 [8.193895  8.211957  8.34701   8.446963  8.45299   8.45486   8.473572
  8.504771  8.513636  8.530685 ]
 [8.369623  8.549446  8.704066  8.736764  8.760081  8.777317  8.831345
  8.835485  8.858271  8.860057 ]
 [8.299071  8.432397  8.434382  8.457373  8.539217  8.562357  8.579033
  8.618738  8.630859  8.6433935]
 [8.615003  8.615164  8.72604   8.730944  8.762621  8.796932  8.797066
  8.797366  8.813984  8.834725 ]
 [8.377228  8.522776  8.711159  8.724562  8.745737  8.763845  8.7686
  8.7728    8.786858  8.828223 ]
 [8.3429165 8.488056  8.655106  8.662771  8.701336  8.741288  8.7436075
  8.770506  8.786265  8.8490505]
 [8.522163  8.575702  8.684618  8.767246  8.782908  8.850494  8.883732
  8.903692  8.909395  8.917681 ]]
[[1269 1525 1723 1160 1694   48 1075 1028  544  916]
 [1035  259 1279 1116 1398  879  289  882 1420 1927]
 [ 327  345 1401  389 1904 1992 1612  106  981 1179]
 [1259  112  351  804 1412 1987 1377  250 1624  133]
 [1666  854 1135  616   94  280   30   99 1212    3]
 [ 574 1523  366  766 1046   91  456  649   46  896]
 [1945  944  244  655 1686  981  256 1555 1280 1969]
 [ 879 1025  390  269 1115 1662 1831  610   11  191]
 [ 156  154   99   31 1237  289  769 1524   56  661]
 [ 427  182  375 1826  610 1384 1299  750    2 1430]]
```



### 倒排表快速索引

在数据量非常大的时候，需要对数据做预处理来提高索引效率。一种方式是**对数据库向量进行分割，划分为多个d维维诺空间，查询阶段，只需要将查询向量落入的维诺空间中的数据库向量与之比较，返回计算所得的k个最近邻结果即可，大大缩减了索引时间。**

- `nlist`参数控制将数据集向量分为多少个维诺空间；
- `nprobe`参数控制在多少个维诺空间的范围内进行索引；

![img](https://pic3.zhimg.com/80/v2-be228fba82151b5a4f698a459f7779d6_720w.jpg)

```python
nlist = 50  # 将数据库向量分割为多少了维诺空间
k = 10
quantizer = faiss.IndexFlatL2(d)  # 量化器
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
       # METRIC_L2计算L2距离, 或faiss.METRIC_INNER_PRODUCT计算内积(归一化后就是cos距离)
print(index.is_trained)
assert not index.is_trained   # 倒排表索引类型需要训练, index.is_trained is False
index.train(data)  # 训练数据集应该与数据库数据集同分布
assert index.is_trained  # 训练完毕后，index.is_trained is True

index.add(data)
index.nprobe = 50  # 选择nprobe个维诺空间进行索引
dis, ind = index.search(query, k)
print(dis)
print(ind)
```

```
[[8.6183815 8.78215   8.782821  8.832027  8.837631  8.848494  8.897979
  8.916634  8.919007  8.937392 ]
 [9.033305  9.038907  9.0917015 9.155841  9.164595  9.2001095 9.201888
  9.220334  9.279476  9.312865 ]
 [8.063819  8.21103   8.306449  8.373355  8.459251  8.459897  8.498555
  8.546462  8.555409  8.621427 ]
 [8.1938925 8.211952  8.347012  8.446964  8.4529915 8.454856  8.473566
  8.504776  8.51364   8.530683 ]
 [8.369622  8.549448  8.704068  8.736766  8.760081  8.777319  8.831343
  8.835488  8.858268  8.860054 ]
 [8.299072  8.432399  8.434377  8.457372  8.539215  8.562353  8.579033
  8.618742  8.630864  8.643396 ]
 [8.615     8.615165  8.726043  8.7309475 8.762618  8.796934  8.797069
  8.797364  8.813985  8.83472  ]
 [8.377228  8.522775  8.71116   8.724566  8.745737  8.763844  8.768602
  8.772798  8.786855  8.828219 ]
 [8.342917  8.488055  8.655108  8.662769  8.701336  8.741289  8.743606
  8.770508  8.786264  8.84905  ]
 [8.522164  8.575698  8.684619  8.767242  8.782908  8.850493  8.883731
  8.90369   8.909393  8.9176855]]
[[1269 1525 1723 1160 1694   48 1075 1028  544  916]
 [1035  259 1279 1116 1398  879  289  882 1420 1927]
 [ 327  345 1401  389 1904 1992 1612  106  981 1179]
 [1259  112  351  804 1412 1987 1377  250 1624  133]
 [1666  854 1135  616   94  280   30   99 1212    3]
 [ 574 1523  366  766 1046   91  456  649   46  896]
 [1945  944  244  655 1686  981  256 1555 1280 1969]
 [ 879 1025  390  269 1115 1662 1831  610   11  191]
 [ 156  154   99   31 1237  289  769 1524   56  661]
 [ 427  182  375 1826  610 1384 1299  750    2 1430]]
```

通过改变`nprobe（n个桶）`的值，发现在**nprobe值较小的时候，查询可能会出错，但时间开销很小，随着nprobe的值增加，精度逐渐增大，但时间开销也逐渐增加**

**当`nprobe=nlist`时，等效于IndexFlatL2索引类型**

简而言之，**倒排表索引首先将数据库向量通过聚类方法分割成若干子类，每个子类用类中心表示**，当查询向量来临，**选择距离最近的类中心，然后在子类中应用精确查询方法**，通过增加相邻的子类个数提高索引的精确度

### 乘积量化索引

在上述两种索引方式中，**在index中都保存了完整的数据库向量，在数据量非常大的时候会占用太多内存，甚至超出内存限制。**
**在faiss中，当数据量非常大的时候，一般采用乘积量化方法保存原始向量的有损压缩形式,故而查询阶段返回的结果也是近似的**

```python
nlist = 50
m = 8                             # 列方向划分个数，必须能被d整除
k = 10
quantizer = faiss.IndexFlatL2(d)  
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 4)
                                    # 4 表示每个子向量被编码为 4 bits
index.train(data)
index.add(data)
index.nprobe = 50
dis, ind = index.search(query_self, k)  # 查询自身
print(dis)
print(ind)
dis, ind = index.search(query, k)  # 真实查询
print(dis)
print(ind)
```

```
[[4.8332453 4.916275  5.0142426 5.0211687 5.0282335 5.039744  5.063374
  5.0652556 5.065288  5.0683947]
 [4.456933  4.6813188 4.698038  4.709836  4.72171   4.7280436 4.728564
  4.728917  4.7406554 4.752378 ]
 [4.3990726 4.554667  4.622962  4.6567664 4.665245  4.700697  4.7056646
  4.715714  4.7222314 4.7242   ]
 [4.4063187 4.659938  4.719548  4.7234855 4.727058  4.7630377 4.767138
  4.770565  4.7718883 4.7720337]
 [4.5876865 4.702366  4.7323933 4.7387223 4.7550535 4.7652235 4.7820272
  4.788397  4.792813  4.7930083]]
[[   0 1036 1552  517 1686 1666    9 1798  451 1550]
 [   1  725  270 1964  430  511  598   20  583  728]
 [   2  761 1254  928 1913 1886  400  360 1850 1840]
 [   3 1035 1259 1884  584 1802 1337 1244 1472  468]
 [   4 1557  350  233 1545 1084 1979 1537  665 1432]]
 
[[5.184828  5.1985765 5.2006407 5.202751  5.209732  5.2114754 5.2203827
  5.22132   5.2252693 5.2286644]
 [5.478416  5.5195136 5.532296  5.563965  5.564443  5.5696826 5.586555
  5.5897493 5.59312   5.5942397]
 [4.7446747 4.8150816 4.824335  4.834736  4.83847   4.844829  4.850663
  4.853364  4.856619  4.865398 ]
 [4.733185  4.7483554 4.7688575 4.783175  4.785554  4.7890463 4.7939577
  4.797909  4.8015175 4.802591 ]
 [5.1260395 5.1264906 5.134188  5.1386065 5.141901  5.148476  5.1756086
  5.1886897 5.192538  5.1938267]
 [4.882325  4.900981  4.9040375 4.911916  4.916094  4.923492  4.928433
  4.928472  4.937878  4.9518585]
 [4.9729834 4.976016  4.984484  5.0074816 5.015956  5.0174923 5.0200887
  5.0217285 5.028976  5.029479 ]
 [5.064405  5.0903125 5.0971365 5.098599  5.108646  5.113497  5.1155915
  5.1244674 5.1263866 5.129635 ]
 [5.060173  5.0623484 5.075763  5.087064  5.100909  5.1075807 5.109309
  5.110051  5.1323767 5.1330123]
 [5.12455   5.149974  5.151128  5.163775  5.1637926 5.1726117 5.1732545
  5.1762547 5.1780767 5.185327 ]]
[[1264  666   99 1525 1962 1228  366  268  358 1509]
 [ 520  797 1973  365 1545 1032 1077   71  763  753]
 [1632  689 1315  321  459 1486  818 1094  378 1479]
 [ 721 1837  537 1741 1627  154 1557  880  539 1784]
 [1772  750 1166 1799  572  997  340  127  756  375]
 [1738 1978  724  749  816 1046 1402  444 1955  246]
 [1457 1488 1902 1187 1485  986   32  531   56  913]
 [1488 1244  121 1144 1280 1078 1012 1215 1639 1175]
 [ 426   45  122 1239  300 1290  546  505 1687  434]
 [ 263  343 1025  583 1489  356 1570 1282  627 1432]]
```

**实验发现，乘积量化后查询返回的距离值与真实值相比偏小，返回的结果只是近似值**

查询自身时能够返回自身，但真实查询时效果较差，这里只是使用了正态分布的数据集，在真实使用时效果会更好，原因有：

- **正态分布的数据相对更难查询，难以聚类/降维**；
- **自然数据相似的向量与不相似的向量差别更大，更容易查找**；

### 参考

* [https://zhuanlan.zhihu.com/p/33896575](https://zhuanlan.zhihu.com/p/33896575)
* https://github.com/liqima/faiss_note