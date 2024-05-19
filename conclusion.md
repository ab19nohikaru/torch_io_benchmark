# 小结

## 单卡测试
### 2024-5-14

测试环境 windows10 显卡 NVIDIA GeForce GTX 1050 

数据集 torchvision.datasets.FashionMNIST

训练和遍历数据集 **epoch 取 5， batch_size 取 64**

| dataset预处理  | 不做处理  | 预载入内存 | 使用tensorclass构造 | numpy.memmap |
| -------------- | --------- | ---------- | ------------------- | ------------ |
| 预处理时间     | 0         | 6.3731 s   | 19.9147 s           | 7.3651 s     |
| 遍历数据集时间 | 31.5313 s | 0.8992 s   | 2.7449 s            | 171.7400 s   |
| 整个训练时间   | 49.4222 s | 9.5569 s   | 9.7066 s            | 195.1260 s   |

 **分析**：

1. 不做处理，直接DataLoader时，每次读数据集都需要进行图片解码并拷贝进显存
2. 将解码后的图片数据预载入内存，每次读数据只需从内存拷进显存
3. tensorclass 会预先将解码后的图片数据mmap到disk, 每次读数据集可直接读进显存，因此遍历数据集耗时比预载入内存慢，但整体训练时长只是略慢
4. 尝试图片解码后使用numpy.memmap映射到disk，由于实现简单效果不好

### 2024-5-18

使用torch.profiler模块，评估训练过程中数据加载的耗时

由于笔记本性能限制，**epoch 取 1， batch_size 取 64**

下表中除training time之外，都是profiler输出中enumerate(DataLoader)#_SingleProcessDataLoaderIter 行的对应的数值

|               | 不做处理  | 预载入内存 | 使用tensorclass构造 | numpy.memmap |
| ------------- | --------- | ---------- | ------------------- | ------------ |
| training time | 19.005 s  | 3.833 s    | 4.212 s             | 40.495 s     |
| CPU time      | 15.489 s  | 346.387 ms | 475.061 ms          | 36.069 s     |
| CPU Mem       | 180.09 Mb | 180.09 Mb  | 180.09 Mb           | 180.09 Mb    |
| CUDA Mem      | 0 b       | 0 b        | 0 b                 | 0 b          |

<font color=red>由于Windows版本torch.profiler的strack_trace对GPU不生效[Issue 93855](https://github.com/pytorch/pytorch/issues/93855), 未对GPU耗时情况分析</font>

**利用strack_trace对tensorclass的优势来源进行分析**:

1. 不做处理和numpy.memmap每次都进行系统调用**open**，开销很大，而 tensorclass 能不调用**open**直接拷贝
2. 除了tensorclass外的三种情况，对于batch中的每条数据都要调用\_\_getitem\_\_去获取单条数据，tensorclass可以单次获取整个batch，不需要多次调用\_\_getitem\_\_再将数据整合

