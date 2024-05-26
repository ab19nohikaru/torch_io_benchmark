# 小结

## 测试环境
OS：windows10 

单显卡 NVIDIA GeForce GTX 1050 

选取两类数据集 

1. torchvision.datasets.FashionMNIST：图片数据，读取时需要一定预处理时间
2. datasets_preprocess.MyDataSet：.npz格式的numpy数组，读取时基本不需要预处理时间

## 数据文件生成

使用下列指令生成测试使用的数据集

```
python dataset_download.py --path PATH
```

该脚本执行下列操作

1. 将FashionMNIST数据集下载到 PATH 目录下子目录FashionMNIST
2. 在 PATH 目录下生成MyDataSet数据集mydataset.npz
3. 加载FashionMNIST数据集后，memmap到mnist子目录下文件，并保存
4. 加载mydataset数据集后，memmap到mydataset子目录下文件，并保存

## 单机单卡

### 数据集预处理

实验对数据集进行了不同预处理，得到四类DataLoader

1. raw: 不做预处理，每次从DataLoader时获取batch时都需要进行图片解码
2. preload: 将所有图片预解码后载入内存
3. tensorclass：将所有图片预解码后，按照tensordict的索引式结构保存在内存中
4. tensorclass_memmap：通过TensorDict.load_memmap直接memmap与disk上文件建立内存映射
5. np_memmap：通过numpy.memmap直接与disk上文件建立内存映射

## DataLoader速度测试

<font color=red>由于Windows版本torch.profiler的strack_trace对GPU不生效[Issue 93855](https://github.com/pytorch/pytorch/issues/93855), 下面仅对CPU耗时情况测试和分析</font>

利用下列代码获取遍历DataLoader耗时

```python
# 完整代码见 test_funcs.py
def get_dataloader_tranverse_time(dataloader:DataLoader, epochs:int):
    t0 = time.time()
    for t in range(epochs):
        for batch in dataloader:
            source, targets = batch
            # avoid memmap lazy load
            source.to(0)
            targets.to(0)
    return time.time() - t0
```

为了减小误差 **epoch 取 5， 重复20次**， 使用脚本test_funcs.py

```
python test_funcs.py --path data/ --dataset=mnist --test_tranverse_time
python test_funcs.py --path data/ --dataset=mydataset --test_tranverse_time
```

得到用时的均值标准差（单位：秒）如下

|                    | FashionMNIST     | MyDataSet       |
| ------------------ | ---------------- | --------------- |
| raw                | 32.976 （0.526） | 3.193 （0.575） |
| preload            | 2.076 （0.023）  | 3.774 （0.164） |
| tensorclass        | 3.645 （0.103）  | 3.898 （0.335） |
| tensorclass_memmap | 4.233 （0.132）  | 3.765 （0.423） |
| np_memmap          | 4.408 （0.046）  | 4.317 （0.112） |

用一个简单模型ToyNet，比较训练用时

为了减小误差 **epoch 取 5， 重复20次**， 使用脚本test_funcs.py

```
python test_funcs.py --path data/ --dataset=mnist --test_train_time
python test_funcs.py --path data/ --dataset=mydataset --test_train_time
```

得到用时的均值标准差（单位：秒）如下

|                    | FashionMNIST    | MyDataSet        |
| ------------------ | --------------- | ---------------- |
| raw                | 51.462（0.657） | 10.646 （0.247） |
| preload            | 8.680（0.041）  | 11.019 （0.394） |
| tensorclass        | 12.411（0.177） | 14.461 （0.859） |
| tensorclass_memmap | 12.963（0.285） | 14.807 （0.781） |
| np_memmap          | 10.988（0.034） | 12.088 （0.601） |

**分析**

FashionMNIST数据集

1. 比较raw和其他DataLoader，对于需要解码的图片数据集，提前预处理能带来很大的速度提升
2. preload与tensorclass都保存在内存中，tensorclass略慢，可能是tensordict数据结构带来的overhead
3. tensorclass_memmap经过内存映射，因此比tensorclass慢
4. np_memmap遍历时间比tensorclass慢，但训练时间反而更快，暂未定位到原因

MyDataSet数据集

1. 比较raw和其他DataLoader，对于不需要预处理的数据集，提前预处理并不能带来速度提升
2. 可以看出tensorclass和tensorclass_memmap的std较大，因此二者之间及与preload的对比没有意义
3. std较大的原因，通过torch.profiler工具发现tensorclass和tensorclass_memmap的每个batch用时相差很大，后来发现cpu占用率超过80%，而其他DataLoader只有40%作用，推测是cpu占用过高造成的。

**结论**

尽管受限于测试环境，还是能够得到以下结论

**tensorclass相较于DataLoader的提速来自于数据集的预处理，使用其它方式预处理也能达到类似效果**

 [RFC](https://github.com/Lightning-AI/pytorch-lightning/issues/17851) 中强调preprocessing带来speed-up，与我们的结论一致

> One of the goals of tensordict is to get an efficient representation of data on disk, by tracking file name and location, while preserving a fast memmap-based indexing. This allows us to represent **huge datasets** composed of multiple tensors on disk. If part of the **preprocessing** can be done anticipatively, this can bring a tremendous speed-up on dataloading:

## 单机多卡和多机多卡

对于多卡和多节点的情况，一般使用大数据集和共享文件系统。因此仅对比raw，tensorclass_memmap和np_memmap三类DataLoader

测试代码已完成，数据待补充。
