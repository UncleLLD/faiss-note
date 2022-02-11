### faiss-9: index进阶操作

下面介绍的方法只支持部分index类型

### 从index中恢复出原始数据

给定id，可以使用`reconstruct`或者`reconstruct_n`方法从`index`中回复出原始向量。支持下述几类`index`：

* `IndexFlat`
*  `IndexIVFFlat `(需要与make_direct_map结合)
*  `IndexIVFPQ`
*  `IndexPreTransform`

```python
import sys
import numpy as np 
import faiss

# 生成数据
d = 16
n_data = 500
data = np.random.rand(n_data, d).astype('float32')

index = faiss.IndexFlatL2(d)
index.add(data)

# reconstruct 恢复一个
re_data = index.reconstruct(0)  # 指定需要恢复的向量的id,每次只能恢复一个向量
print(re_data)
# reconstruct_n 恢复指定区间
re_data_n = index.reconstruct_n(0, 10)  # 从第0个向量开始，连续取10个
print(re_data_n.shape)
```

```
[0.83482474 0.6438837  0.24734788 0.6537704  0.61774486 0.9384649
 0.12293569 0.80372137 0.6918489  0.6864977  0.6374077  0.52337897
 0.09799734 0.14641039 0.5328222  0.88016546]
 
(10, 16)
```



### 从index中移除向量

使用`remove_ids`方法可以移除`index`中的部分向量，调用了`IDSelector`对象（或`IDSelectorBatch`批量操作）标识每个向量是否应该被移除。**移除操作因为要遍历标识数据库中的每一个向量，所以只有在需要移除大部分向量时才建议使用。**支持下述几类`index`：

* `IndexFlat`
* `IndexIVFFlat`
* `IndexIVFPQ`
* `IDMap`



```python
index = faiss.IndexFlatL2(d)
index.add(data)
print(index.ntotal)

# remove, 指定要删除的向量id，是一个np的array
index.remove_ids(np.arrange(5))  # 删除id为0,1,2,3,4的向量
# index.remove_ids(np.array([0]))
print(index.ntotal)
```

```
500
495
```

可以看到，总的减少了5个



### 搜索距离范围内的向量

以查询向量为中心，返回距离在一定范围内的结果，如返回数据库中与查询向量距离小于0.3的结果。
支持以下几类`index`：

* `IndexFlat`
* `IndexIVFFlat`
* 只支持在CPU使用



```python
index = faiss.IndexFlatL2(d)
index.add(data)

dist = float(np.linalg.norm(data[3] - data[0])) * 0.99  # 定义一个半径/阈值
print(dist)

num, dis, ind = index.range_search(data[[49], :], dist)  # 用第50个向量查询
print(dis)  # 返回结果是一个三元组，分别是limit(返回的结果的数量), distance, index

num, dis, ind = index.range_search(data[[9], :], dist)  # 用第10个向量查询
print(dis)  # 返回结果是一个三元组，分别是limit(返回的结果的数量), distance, index
```

```
1.9466463339328766

[1.5384469 1.4711096 1.6992308 1.3599523 1.8639128 1.8728075 1.816922
 0.        1.6398396 1.6408762 1.5378789 1.6585189 1.590842  1.8358855
 1.7690482 1.6529568 1.6374359 1.9222212 1.7712178 1.7988551 1.6408085
 1.3410411 1.6889251 1.9464017 1.7024078 1.6522535 1.8238969 1.8631847
 1.5418944 1.670974  1.6715947 1.5081348 1.9099058 1.8395643 1.4895053
 1.8079453 1.934546  0.6599543 1.6656785 1.8234447 1.8075504 1.548338
 1.6937177 1.6249175 1.729368  1.3271422 1.7416501 1.3548261 1.0927346
 1.7518137 1.7552993 1.896307  1.9200239 1.6671427 1.2523265 1.1999724
 1.8288059 1.6901855 1.3587701 1.9093046 1.6236575 1.6918842 1.9014039
 1.8057953 1.9295995 1.8126392]
 
 [1.6781944  1.6698971  0.         1.905075   1.2732563  1.8204238
 0.7628526  1.7816186  1.301338   1.6780018  1.6545635  1.6984217
 1.4493203  1.6713542  1.537207   1.5003004  1.6203446  0.824362
 1.4711096  1.5239989  1.6029797  1.6971319  1.337048   1.7034856
 1.2767963  1.7624524  1.901829   1.3511816  1.7108729  0.7751142
 1.7164278  1.7713003  0.90857565 1.4278122  1.8474922  1.4491313
 1.4352927  1.7064259  1.6955849  1.824855   1.2134202  1.2110215
 1.459719   1.3473287  1.603931   1.6431534  1.9166925  1.297124
 1.6608323  0.954386   1.3093655  1.2556585  1.4246114  1.9071138
 1.6585203  1.181331   1.3955002  1.8366412  1.4554479  1.8914598
 1.9038694  1.5533493  1.4605153  1.3316728  1.1256341  1.4339149
 0.98044544 1.5410831  0.9028618  1.2111411  1.8140744  1.6053402
 1.42471    1.9080161  1.619217   1.7909881  1.371387   1.5482172
 1.6865985  1.8430685  1.7086227  1.1592953  1.9445957  1.86112
 1.9262257  1.1428056  1.9131099  1.1875193  1.5940969  1.888087
 1.9143637  1.8768654  1.1045247  1.9368241  1.60408    0.9170128
 1.8888283  0.98666286 1.2195413  1.4605683  1.7673925  1.635049
 1.9208401  1.681262   1.1764605  1.4240292  1.4972473  1.9276001
 1.4129896  1.4303442  0.73805165 1.710277   1.3814571  1.7112839
 1.2163604  1.8131944  1.0360564  1.7986903  1.4174187  1.8276937
 1.5105457  1.909564   1.699083   1.7194629  1.5950986  1.329985
 1.3965563  1.2130578  1.8877728  0.84847486 1.3567262  1.8569231
 1.6550059  1.6447781  1.105884   1.9460237  1.443865   1.5296811
 1.3227925  1.526395   1.6378742  1.5647403  1.839086   1.7195833
 1.2111061  1.6579268  1.4656825  1.8722528  1.6359022  1.8060739
 0.8323817  1.7267382  1.388557   0.8495468  1.5946512  1.7971337
 1.4035878  1.5542881  1.9240556  1.9365174  1.578675   1.1193347
 1.8796601  1.9358166  1.9066932  1.6251965  1.8400447  1.2772393
 1.9200763  1.8264031  1.8106583  1.5018897  1.5844185  1.5264275
 1.7154095  1.3320069  1.7229189  1.9042165  1.793323   1.9336419
 1.899312   1.6293273  0.99493396 1.5397438  1.5304623 ]
```

可以看到上述得到的距离都在设定的阈值范围内

### 拆分/合并index

**可以将多个`index`合并，需要注意的是，多个`index`的数据应该满足同一分布，并且用同一分布的数据训练`index`**；

**如果多个`index`的数据分布不同，合并时并不会报错，但在理论上会降低索引的精度，应该用与合并后的数据集同分布的训练集再次训练；**



```python
# index1使用前250个向量
nlist = 10
quantizer = faiss.IndexFlatL2(d)
index1 = faiss.IndexIVFFlat(quantizer, d, nlist)
index1.train(data)  # 使用总的数据集进行训练
index1.add(data[:250])

# index2使用后250个向量
index2 = faiss.IndexIVFFlat(quantizer, d, nlist)
index2.train(data)
index2.add(data[250:])

index1.merge_from(index2, 250)  # merge_from(index2, add_id), add_id控制新增元素的id，id为index1长度+add_id作为新增元素的第一个下标，后面依次类推，最好是与index2的长度保持一致
print(index1.ntotal) # 将index2合并到index1， 合并后应该包含500个向量
dis, ind = index1.search(data[:5], 10)
print(ind)
```

```
500

[[  0 454  12 345 278  95 306 296 161 322]
 [  1  53 477 455  36  63 337 140 287 138]
 [  2  99 113  63 479 229  36 337 173   1]
 [  3 492 197 478 484   5 152 272 271  85]
 [  4 346   8 194 312 215  69 342 126 260]]
```



**源码**

```cpp
void IndexIVF::merge_from (IndexIVF &other, idx_t add_id)
{
    // minimal sanity checks
    FAISS_THROW_IF_NOT (other.d == d);
    FAISS_THROW_IF_NOT (other.nlist == nlist);
    FAISS_THROW_IF_NOT_MSG ((!maintain_direct_map &&
                             !other.maintain_direct_map),
                  "direct map copy not implemented");
    FAISS_THROW_IF_NOT_MSG (typeid (*this) == typeid (other),
                  "can only merge indexes of the same type");
    for (long i = 0; i < nlist; i++) {
        std::vector<idx_t> & src = other.ids[i];
        std::vector<idx_t> & dest = ids[i];
        for (long j = 0; j < src.size(); j++)
            dest.push_back (src[j] + add_id);  // dest控制的是新增id的数值
        src.clear();
        codes[i].insert (codes[i].end(),
                         other.codes[i].begin(),
                         other.codes[i].end());
        other.codes[i].clear();
    }

    ntotal += other.ntotal;
    other.ntotal = 0;
}
```



### 参考

* [https://www.cnblogs.com/houkai/p/9316172.html](https://www.cnblogs.com/houkai/p/9316172.html)
* [https://github.com/liqima/faiss_note](https://github.com/liqima/faiss_note)
* [https://zhuanlan.zhihu.com/p/40220119](https://zhuanlan.zhihu.com/p/40220119)

