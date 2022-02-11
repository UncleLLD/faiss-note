# faiss安装

git项目的安装教程：

* [https://github.com/facebookresearch/faiss/blob/main/INSTALL.md](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)

## 使用Anaconda安装
使用Anaconda安装使用faiss是最方便快速的方式，facebook会及时推出faiss的新版本conda安装包，在conda安装时会自行安装所需的libgcc, mkl, numpy模块。  

faiss的cpu版本目前仅支持Linux和MacOS以及Windows操作系统，gpu版本提供可在Linux操作系统下用CUDA编译的版本。



先更新conda以及mkl

```shell
#更新conda
conda update conda

#安装mkl
conda install mkl
```



### 安装cpu版本

```
# CPU-only version
$ conda install -c pytorch faiss-cpu

# GPU(+CPU) version
$ conda install -c pytorch faiss-gpu

# or for a specific CUDA version
$ conda install -c pytorch faiss-gpu cudatoolkit=10.2 # for CUDA 10.2
```

按如下方式安装夜间预发布软件包：

```
# CPU-only version
$ conda install -c pytorch/label/nightly faiss-cpu

# GPU(+CPU) version
$ conda install -c pytorch/label/nightly faiss-gpu
```



### 从conda-forge安装

Faiss也由[conda forge](https://conda-forge.org/)进行包装，conda forge是conda的社区驱动包装生态系统。包装工作正在与Faiss团队合作，以确保高质量的包构建。

由于conda forge的全面基础设施，甚至可能会出现某些构建组合在conda forge中得到支持，而pytorch渠道不支持这些构建组合。要安装，请使用

```shell
# CPU version
$ conda install -c conda-forge faiss-cpu

# GPU version
$ conda install -c conda-forge faiss-gpu
```



## 从源码安装

Faiss可以使用CMake来源地构建。

Faiss在Linux，OSX和Windows上支持X86_64机器。它也可以在其他平台上运行，请参阅[其他平台](https://github.com/facebookresearch/faiss/wiki/Related-projects#bindings-to-other-languages-and-porting-to-other-platforms)。



基本要求是：

* C ++ 11编译器（支持OpenMP支持版本2或更高版本）
* Blas实现（强烈建议使用Intel MKL以获得最佳性能）

可以选择的要求：

* 对于GPU设备：
  - nvcc
  - CUDA toolkit

* python要求：
  - python 3
  - numpy
  - swig



如果有问题的话，请参考该页面：

* [Troubleshooting](https://github.com/facebookresearch/faiss/wiki/Troubleshooting)



### 步骤1：调用cmake

```shell
$ cmake -B build .
```

这在构建/子目录中生成了系统相关的配置/构建文件。

几个选项可以传递给cmake，其中：

* 常规选项：

  - `-DFAISS_ENABLE_GPU=OFF` 禁用GPU指数（可能的值 `ON`和`OFF`）

  - `-DFAISS_ENABLE_PYTHON=OFF` 禁用构建python绑定（可能的值`ON`和`OFF`）
  - `-DBUILD_TESTING=OFF`禁用C ++测试
  - `-DBUILD_SHARED_LIBS=ON`构建共享库（可能的值 `ON`和`OFF`）



* 优化相关选项：

  * `-DCMAKE_BUILD_TYPE=Release`以启用通用编译器优化选项（例如，在GCC上启用 `-03`）

  * `-DFAISS_OPT_LEVEL=avx2`以便使用优化的SIMD指令（可能的值是`generic`, `sse4`, `avx2`,）生成代码的所需编译器标志（可能的值）

* BLAS相关的选项：
  - `-DBLA_VENDOR=Intel10_64_dyn -DMKL_LIBRARIES=/path/to/mkl/libs`使用Intel Mkl Blas实现，这比OpenBlas更快（更多信息可以在[CMake文档](https://cmake.org/cmake/help/latest/module/FindBLAS.html)中找到Bla_Vendor选项的值），

* GPU相关的选项：
  * `-DCUDAToolkit_ROOT=/path/to/cuda-10.1`为了提示CUDA工具包的路径（有关更多信息，请参阅[CMake文档](https://cmake.org/cmake/help/latest/module/FindBLAS.html)），
  * `-DCMAKE_CUDA_ARCHITECTURES="75;72”`，用于指定要构建的GPU架构（参见[CUDA文档](https://developer.nvidia.com/cuda-gpus)以确定您应该选择哪个架构）

* Python相关的选项：
  * `-DPython_EXECUTABLE=/path/to/python3.7`才能为不同的Python构建Python接口而不是默认值（请参阅[CMake文档](https://cmake.org/cmake/help/latest/module/FindBLAS.html)）



### 步骤2：调用Make

```shell
$ make -C build -j faiss
```

这构建了C ++库（默认情况下，libfaiss.so如果`-DBUILD_SHARED_LIBS=ON`被传递给CMake）。

`-j`选项启用了多个单元的并行编译，导致更快的构建，但增加内存不足的机会，在这种情况下建议将-j选项设置为固定值（例如-j4）。



### 步骤3： 构建python依赖（可选）

```shell
$ make -C build -j swigfaiss
$ (cd build/faiss/python && python setup.py install)
```

第一个命令构建faiss的python绑定，而第二个命令生成并安装python包。



### 步骤4： 安装C ++库和header（可选）

```shell
$ make -C build install
```

这将使编译的库（`Libfaiss.a`或`libfaiss.so`上的linux）提供系统范围，以及C ++ header，仅需要此步骤才能仅安装Python包。



### 步骤5： 测试（可选）测试

#### 测试c++套件

要运行整个测试套件，请确保使用`-DBUILD_TESTING=ON`调用CMake，并运行：

```shell
$ make -C build test
```

#### 测试python套件

```shell
$ (cd build/faiss/python && python setup.py build)
$ PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py
```



#### 简单例子

一个使用案列可以在[`demos/demo_ivfpq_indexing.cpp`](https://github.com/facebookresearch/faiss/blob/main/demos/demo_ivfpq_indexing.cpp)参考

它创建了一个小索引，存储它并执行一些搜索。正常的运行时大约20s。使用快速机器和英特尔MKL的BLAS，大约耗时2.5s。

* 构建

```shell
$ make -C build demo_ivfpq_indexing
```

*  运行

```shell
$ ./build/demos/demo_ivfpq_indexing
```



#### 简单的GPU例子

```shell
$ make -C build demo_ivfpq_indexing_gpu
$ ./build/demos/demo_ivfpq_indexing_gpu
```

这会产生相当于CPU `demo_ivfpq_indexing`的GPU代码，它还展示了如何将索引转换为GPU。

#### benchmark

 更长时间的示例在Sift1M数据集上运行和评估Faiss。在运行之前，请从http://corpus-texmex.irisa.fr/下载ANN_SIFT1M并将其解压缩到此存储库源目录的根目录的子目录`sift1M`。

然后编译并运行以下（确保已安装Faiss）：

```shell
$ make -c build demo_sift1m
$ ./build/demos/demo_sift1m
```

这是高级自动调谐API的演示， 可以尝试设置不同的`index_key`以查找提供最佳性能的索引结构。

#### 真实测试

以下脚本将demo_sift1M`测试扩展到几种类型的索引。这必须从此存储库的源目录的根目录运行：

```shell
$ mkdir tmp  # graphs of the output will be written here
$ python demos/demo_auto_tune.py
```

它将循环通过几种类型的索引并找到最佳操作点，可以使用索引类型进行游戏。



#### 真实GPU测试

上面的示例也在GPU上运行。编辑`demos/demo_auto_tune.py`，修改第100行

```python
keys_to_test = keys_gpu
use_gpu = True
```

运行以下代码测试GPU代码

```shell
$ python demos/demo_auto_tune.py
```

