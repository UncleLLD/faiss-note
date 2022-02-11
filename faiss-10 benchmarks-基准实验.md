### faiss-10: benchmarks-基准实验

### 最基本功能的实验基准 Low level benchmarks

仅使用faiss的最基本功能获得一些实验基准数据。  

* 使用k-means方法将1百万个256维向量的数据集聚类为20000个类；  
* 使用k-means方法将95百万个128维向量的数据集聚类85000个类；  
* 实验中关注时间和精度的变化；

faiss wiki:[https://github.com/facebookresearch/faiss/wiki/Low-level-benchmarks](https://github.com/facebookresearch/faiss/wiki/Low-level-benchmarks)

### 检索 Indexing 1M vectors

当数据集中的向量个数在百万级别，暴力精确搜索的时间开销太大，比较好的选择是使用IndexIVFFlat索引类型。IndexIVFFlat也会返回精确的距离值，但返回的结果并不是完全正确的，可能会漏掉某个结果。  
facebook官方通过一些实验，通过不同的检索类型在1百万向量的数据集上做检索，其中主要关注速度和精度的变化。实验结果展示在faiss wiki中。   
实验中使用特征提取器提取1百万张图片的特征表达，对每张图片提取4096维特征向量，然后使用PCA将4096维向量降维到256维。

faiss wiki:[https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)

### 1G级别向量检索 Indexing 1G vectors

对这一量级的数据集，必须使用向量的压缩编码形式，主要的方法有乘积量化（PQ）。 
使用Bigann和Deep1B分别进行实验。实验结果在faiss wiki。  
faiss wiki:[https://github.com/facebookresearch/faiss/wiki/Indexing-1G-vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)

### 1T级别向量检索 Indexing 1T vectors

对这一量级的数据集，向量个数在兆级别，不能像之前那样进行测试，矢量需要用量化器量化为1 unit8的组件。构建索引的时候需要跨数据库分区拆分。

faiss wiki:[https://github.com/facebookresearch/faiss/wiki/Indexing-1T-vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1T-vectors)



### 向量编码标准 Vector codec benchmarks

在SIFT1M、Resnet50、Sentence embedding数据集上进行编码测试以及恢复误差测试

faiss wiki:[https://github.com/facebookresearch/faiss/wiki/Vector-codec-benchmarks](https://github.com/facebookresearch/faiss/wiki/Vector-codec-benchmarks)