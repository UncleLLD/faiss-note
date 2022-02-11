### faiss-4 选择合适的index类型

选择index类型并没有一套精准的法则可以依据，需要根据自己的实际情况选取。

下面的几个问题可以作为选取index的参考：

### 是否需要精确的结果

* 如果需要，应选择使用`Flat`， 只有`IndexFlatL2` 能确保返回精确结果。一般将其作为baseline与其他索引方式对比，以便在精度和时间开销之间做权衡。
* 但是Flat方式不支持`add_with_ids`,  如果需要，可以使用`IDMap, Flat` ，支持GPU

```python
# 导入faiss
import sys
import faiss
import numpy as np 

# 构建数据
d = 512       # 维数
n_data = 2000   
np.random.seed(0) 
data = []
mu = 3
sigma = 0.1
for i in range(n_data):
    data.append(np.random.normal(mu, sigma, d))
data = np.array(data).astype('float32')

# ids, 6位随机数
ids = []
start = 100000
for i in range(data.shape[0]):
    ids.append(start)
    start += 100  # id间隔是100， [100000 100100 100200 ... 299700 299800 299900]
ids = np.array(ids)
```



**`Flat`不支持`add_with_ids`**

```python
index = faiss.index_factory(d, "Flat")
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind)
```

```
[[   0  798  879  223  981 1401 1458 1174  919   26]
 [   1  981 1524 1639 1949 1472 1162  923  840  300]
 [   2 1886  375 1351  518 1735 1551 1958  390 1695]
 [   3 1459  331  389  655 1943 1483 1723 1672 1859]
 [   4   13  715 1470  608  459  888  850 1080 1654]]
```

**使用`IDMap, Flat`支持`add_with_ids`**

```python
index = faiss.index_factory(d, "IDMap, Flat")
index.add_with_ids(data, ids)
dis, ind = index.search(data[:5], 10)
print(ind)   # 返回的结果是我们自己定义的id
```

```
[[100000 179800 187900 122300 198100 240100 245800 217400 191900 102600]
 [100100 198100 252400 263900 294900 247200 216200 192300 184000 130000]
 [100200 288600 137500 235100 151800 273500 255100 295800 139000 269500]
 [100300 245900 133100 138900 165500 294300 248300 272300 267200 285900]
 [100400 101300 171500 247000 160800 145900 188800 185000 208000 265400]]
```



### 关心内存开销

**faiss在索引时必须将index读入内存中**，如果不需要精确的结果, 并且内存有限, 那么在有限的内存中，要**在精确与速度之间做出平衡**。

#### 如果不在意内存占用空间，使用“HNSWx”

**如果内存空间很大，数据库很小，HNSW是最好的选择(基于图检索的方式)**，速度快，精度高，一般**4<=x<=64**。**不支持`add_with_ids`，不支持移除向量，不需要训练，不支持GPU。**

```python
index = faiss.index_factory(d, "HNSW8")  # 选择HNSW8
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind) 
```

```
[[ 879  981   26 1132  807 1639 1334 1832 1821  827]
 [   1  981 1524 1639 1949 1472 1162  923  300 1029]
 [   2 1886  375 1351  518  390 1707 1080 1832 1398]
 [   3 1459  331  389  655 1483 1723 1672 1859  650]
 [   4   13  715 1470  608  459  850 1080 1654  665]]
```

可以看到第一个结果没有检索正确。

#### 如果稍微有点在意，使用“..., Flat“

**"..."是聚类操作**，聚类之后将每个向量映射到相应的bucket。**该索引类型并不会保存压缩之后的数据，而是保存原始数据，所以内存开销与原始数据一致**。通过`nprobe`参数控制速度/精度。
**支持GPU,但是要注意，选用的聚类操作必须也支持**

```python
index = faiss.index_factory(d, "IVF100, Flat")
index.train(data)
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind)  
```

```
[[   0  879  981 1401  919  143    2  807 1515 1393]
 [   1  511 1504  987  747  422 1911  638  851 1198]
 [   2  879  807  981 1401 1143  733  441 1324 1280]
 [   3  740  155 1337 1578 1181 1743  290  588 1340]
 [   4 1176  256 1186  574 1459  218  480 1828  942]]
```

#### 如果很在意，使用”PCARx,...,SQ8“

如果保存全部原始数据的开销太大，可以用这个索引方式。包含三个部分，

* **降维**
* **聚类**
* **scalar量化**，每个向量编码为8bit 不支持GPU

```python
index = faiss.index_factory(d, "PCAR16,IVF50,SQ8")  # 每个向量降为16维
index.train(data)
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind)  
```

```
[[   0  289  671  798 1144  916   31 1512 1716  238]
 [   1 1008  698  206  657  294  383  700  574 1968]
 [   2 1594  754 1850  266  559  774  154 1723 1949]
 [   3 1778 1740 1750  593 1174  572 1852  696 1298]
 [   4 1457  466 1604 1951  912  804  736  362  750]]
```

#### 如果非常非常在意，使用"OPQx_y,...,PQx"

`PQx`表示使用x-byte量化向量，一般$x<=64$，`OPQ`是一个线性变换，使得其更容易被压缩。

y需要是x的倍数，一般保持`y<=d，y<=4*x（推荐y=4*x）`，支持GPU

```python
index = faiss.index_factory(d, "OPQ32_128,IVF50,PQ32")  
index.train(data)
index.add(data)
dis, ind = index.search(data[:5], 10)
print(ind)  
```

```
[[   0 1334  807  123   26 1122  769 1966   30  400]
 [   1   20  959  992 1492  911  566  790  145 1794]
 [   2   21 1966 1886  123  769 1171 1883 1850  807]
 [   3   92 1523  271 1934 1793 1090 1067 1299  302]
 [   4 1801  764  753  255 1610 1305  240  734 1031]]
```

### 关心数据集的大小

该问题用于聚类选项（上面的`...`）。将数据集群集到存储桶中，并在搜索时，仅访问桶的一小部分（`nprobe`存储桶）。对数据集向量的代表性样本执行聚类，通常是数据集的样本。

####  如果小于1M， 使用"...,IVFx,..."

N是数据集中向量个数，x一般取值`[4*sqrt(N),16*sqrt(N)]`,需要`30*x ～ 256*x`个向量的数据集去训练，支持GPU



#### 如果大于1M，小于10M， 使用"...,IVF65536_HNSW32,..."

`IVF`与`HNSW`结合使用，使用`HNSW`进行聚类分配。需要在`30*65536`和`256*65536`个向量进行训练，不支持GPU



#### 如果大于10M，小于100M， 使用"...,IVF262144_HNSW32,..."

与上面类似，只是用262144（2^18）替换65536(2^16).

注意，该训练非常慢。支持GPU训练，请参与：[[train_ivf_with_gpu.ipynb](https://gist.github.com/mdouze/46d6bbbaabca0b9778fca37ed2bcccf6)](https://gist.github.com/mdouze/46d6bbbaabca0b9778fca37ed2bcccf6)



#### 如果大于100M，小于1B，使用"...,IVF1048576_HNSW32,..."

与上面类似，只是用1048576（2^20）代替65536，该训练更慢

### 参考

* [http://www.bdata-cap.com/newsinfo/39514.html](http://www.bdata-cap.com/newsinfo/39514.html)
* [https://zhuanlan.zhihu.com/p/357414033](https://zhuanlan.zhihu.com/p/357414033)
* [https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
* [https://github.com/liqima/faiss_note](https://github.com/liqima/faiss_note)
* [https://www.cnblogs.com/houkai/p/9316172.html](https://www.cnblogs.com/houkai/p/9316172.html)