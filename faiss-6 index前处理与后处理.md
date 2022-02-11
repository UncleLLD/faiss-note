### faiss-6: index前处理与后处理

在某些情形下，需要对Index做前处理或后处理



### ID映射

默认情况下，**faiss会为每个输入的向量记录一个次序id(1,2,3...,)，在使用中也可以为向量指定任意需要的id。**
**部分index类型有`add_with_ids`方法，可以为每个向量对应一个64-bit的id，搜索的时候返回这个指定的id。**



```python
# 导入faiss
import sys
import faiss
import numpy as np 

# 生成数据和id
d = 512
n_data = 2000
data = np.random.rand(n_data, d).astype('float32')
ids = np.arange(100000, 102000)  # id设定为6位数整数
print(ids, len(ids))
```

```
[100000 100001 100002 ... 101997 101998 101999] 2000
```

```python
nlist = 10  # 将数据集向量分为10个维诺空间
quantizer = faiss.IndexFlatL2(d)  # 欧式距离
# quantizer = faiss.IndexFlatIP(d)    # 点乘
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
index.train(data)
index.add_with_ids(data, ids)
d, i = index.search(data[:5], 5)  # 搜索与前五个向量相近的向量
print(i)  # 返回的id应该是自己设定的
```

```
[[100000 100563 101646 100741 100421]
 [100001 100727 100786 100269 101902]
 [100002 100800 100362 100835 101783]
 [100003 101986 101340 100803 101233]
 [100004 100902 101084 101562 101006]]
```



但是对有些`Index`类型，并不支持`add_with_ids`，因此需要与其他`Index`类型结合，将默认的id映射到指定id，**用`IndexIDMap`类实现。**
**指定的`ids`不能是字符串，只能是整数。**

**`IndexFlatL2`不支持`add_with_ids`**，下面语句报错

```python
index = faiss.IndexFlatL2(data.shape[1]) 
index.add_with_ids(data, ids)  # error
```

```
add_with_ids not implemented for this type of index
```

**`IndexDMap`支持`add_with_ids`**

```python
index = faiss.IndexFlatL2(data.shape[1]) 
index2 = faiss.IndexIDMap(index)  
index2.add_with_ids(data, ids)  # 将index的id映射到index2的id,会维持一个映射表
```



### 数据转换

有些时候需要在索引之前转换数据。转换类继承了VectorTransform类，将输入向量转换为输出向量。

* 随机旋转，类名`RandomRotationMatri`，用以均衡向量中的元素，一般在`IndexPQ`和`IndexLSH`之前；
* PCA,类名PCAMatrix，降维；
* 改变维度，类名`RemapDimensionsTransform`，可以升高或降低向量维数



#### PCA降维（通过IndexPreTransform）

输入向量是2048维，需要减少到16byte

```python
# 生成数据并转换成格式
data = np.random.rand(n_data, 2048).astype('float32')

# the IndexIVFPQ will be in 256D not 2048
coarse_quantizer = faiss.IndexFlatL2(256) 
sub_index = faiss.IndexIVFPQ (coarse_quantizer, 256, 16, 16, 8)

# PCA 2048->256
# 降维后随机旋转 (第四个参数)
pca_matrix = faiss.PCAMatrix (2048, 256, 0, True) 

# the wrapping index
index = faiss.IndexPreTransform (pca_matrix, sub_index)

# will also train the PCA
index.train(data)  # 数据需要是2048维

# PCA will be applied prior to addition
index.add(data)
```



#### 升维

有时候需要在向量中插入升高维度，一般需要

- d是4的整数倍，有利于举例计算
- d是M的整数倍

```
d = 512
M = 8   # M是在维度方向上分割的子空间个数
d2 = int((d + M - 1) / M) * M
print(d2)
remapper = faiss.RemapDimensionsTransform (d, d2, True)
index_pq = faiss.IndexPQ(d2, M, 8)
index = faiss.IndexPreTransform (remapper, index_pq)  # 后续可以添加数据/索引
```

```
512
```



###  对搜索结果重新排序

**当查询向量时，可以用真实距离值对结果进行重新排序。**
在下面的例子中，搜索阶段会首先选取4*10个结果，然后对这些结果计算真实距离值，再从中选取10个结果返回。`IndexRefineFlat`保存了全部的向量信息，内存开销不容小觑。

```python
data = np.random.rand(n_data, d).astype('float32')
nbits_per_index = 4
q = faiss.IndexPQ (d, M, nbits_per_index)
rq = faiss.IndexRefineFlat(q)
rq.train(data)
rq.add(data)
rq.k_factor = 4
dis, ind = rq.search (data[:5], 10)
print(ind)
```

```
[[   0 1747 1124  120 1625  129  345 1848 1833 1431]
 [   1  614  522 1578 1662 1813  737 1479  181  919]
 [   2 1182 1372 1901  871  523 1807   74  685  335]
 [   3 1130 1127 1426  181 1479 1064 1525 1113  931]
 [   4  696  944  217 1359 1987 1518 1880  755  490]]
```



### 综合多个index返回的结果

当数据集分布在多个`index`中，需要在每个`index`中都执行搜索，然后使用`IndexShards`综合得到结果。同样也适用于`index`分布在不同的GPU的情况





### 参考

* [https://github.com/liqima/faiss_note](https://github.com/liqima/faiss_note)