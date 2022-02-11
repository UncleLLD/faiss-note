### faiss-5： 索引I/O操作

所有的函数都是深复制，不需要关系对象关系

### 读写操作

```python
import faiss

# 写到本地文件
faiss.write_index(index, "index_file.index")  # 将index保存为index_file.index文件

# 读到本地文件
index = faiss.read_index("index_file.index")  # 读入index_file.index文件
```



### 复制

```python
# 完全复制一个index
index_new = faiss.clone_index(index)  

# 复制索引到GPU
index_cpu_to_gpu = faiss.index_cpu_to_gpu()

# 从GPU复制到CPU
index_gpu_to_cpu = faiss.index_gpu_to_cpu(index)

# 使用IndexShards或IndexProxy将索引复制到几个GPU
index_cpu_to_mlp_gpu = faiss.index_cpu_to_gpu_multiple(index)
```



### Index factory

`Index_factory`用一个字符串构建Index，用逗号分割可以分为3部分：

* **前处理部分**
* **倒排表（聚类）**
* **细化后处理部分(编码)**

**前处理部分（preprocessing）**

* `PCA`:`PCA64`表示通过PCA将数据维度降为64，`PCAR64`表示增加了随机旋转（random rotation）
* `OPQ`: `OPQ16`表示用`OPQMatrix`将数组量化为16位

**倒排表部分（inverted file）**

* `IVF`: `IVF4096`表示使用粗量化器`IndexFlatL2`建立一个大小是4096的倒排表，即聚类为4096类
* `IMI`：`IMI2x8`表示通过`Mutil-index`使用2x8个bits（MultiIndexQuantizer）建立2^(2*8)份的倒排索引
* `IDMap`: 如果不使用倒排但需要add_with_ids，可以通过`IndexIDMap`来添加`id`

**细化后处理部分（refinement）**

* `Flat`： 保存完整向量，通过`IndexFlat`或者`IndexIVFFlat`实现
* `PQ16`: 将向量编码为16byte，通过`IndexPQ`或者`IndexIVFPQ`实现
* `PQ8+16`：表示通过8字节来进行`PQ`，16个字节对第一级别量化的误差再做`PQ`，通过`IndexIVFPQR`实现

例如：

`index = index_factory(128, "OPQ16_64,IMI2x8,PQ8+16")`

 上述语句是处理128维的向量

* `OPQ16_64`：使用`OPQ`来预处理数据,16是OPQ内部处理的blocks大小，64为OPQ后的输出维度
* `IMI2x8`: 使用multi-index建立65536（2^16）和倒排列表
* `PQ8+16`: 编码采用8字节`PQ`和16字节refine的Re-rank方案



```python
index = faiss.index_factory(128, "PCA80,Flat")  # 原始向量128维，用PCA降为80维，然后应用精确搜索
index = faiss.index_factory(128, "OPQ16_64,IMI2x8,PQ8+16")  # 原始向量128维，用OPQ降为64维，分为16类，用2*8bit的倒排多索引，用PQ编码为8byte保存，检索时使用16byte
```



### 参考

* [https://github.com/liqima/faiss_note](https://github.com/liqima/faiss_note)
* [https://www.cnblogs.com/houkai/p/9316172.html](https://www.cnblogs.com/houkai/p/9316172.html)

