# 小结

## 测试环境
OS：windows10 

单显卡 NVIDIA GeForce GTX 1050 

数据集 torchvision.datasets.FashionMNIST

## 数据集预处理

实验对数据集FashionMNIST 进行了不同预处理，得到四类DataLoader

1. raw_dataloader: 不做预处理，每次从DataLoader时获取batch时都需要进行图片解码
2. preload_dataloader: 将所有图片预解码后载入内存
3. tensorclass_dataloader：将所有图片预解码后，按照tensordict的索引式结构保存在内存中
4. memmap_dataloader: 将所有图片预解码后保存到硬盘上的file，通过numpy.memmap访问memory-map file来获取batch

## 单机单卡

训练和遍历数据集 **epoch 取 1， batch_size 取 64**

| DataLoader  | raw      | preload  | tensorclass | memmap   |
| ----------- | -------- | -------- | ----------- | -------- |
| 预处理时间  | 0        | 6.1872 s | 15.4617 s   | 7.3240 s |
| 训练时间    | 8.5375 s | 1.8220 s | 2.2306 s    | 2.4898 s |
| 预处理+训练 | 8.5375 s | 8.0092 s | 17.6923     | 9.8138 s |

 打开选项--with_profiler --export_json，借助torch.profiler工具，**分析结果如下**：

1. 对于epoch=1的训练，preload与raw预处理+训练总时间相近；当多个epoch时preload能节省预处理时间
2. tensorclass由于采用了索引式的数据结构组织方式，预处理和获取batch都有额外的开销，比preload慢
3. 相较于preload，memmap预处理还需将解码后数据写入file；获取batch时也需要mmap访问file，因此比tensorclass慢

## 单机多卡和多机多卡

根据官方文档的描述，通过memmap建立共享文件系统上file到tensordict对象的映射，不同进程/节点能直接访问同一个tensordict对象。

从单机单卡的结果来看，tensorclass组织数据结构的方式，会在预处理和获取batch时带来一定的overhead。想要体现官方文档描述的优势，则要构建完整数据集预处理后无法全部加载进内存的情形，即大数据集。这时tensorclass能将预处理后的完整数据集保存到磁盘上，每次获取batch都能直接访问对象，从而消除额外的预处理开销。

而由于最开始的理解错误，目前的单机多卡和多机多卡实验中，每个进程都会实例化自己的tensorclass_dataloader而不是共用一个对象，与单机单卡实验没有本质区别，因此不能验证其IO优势。

# 后续的实验

可以从两个方向进行

1. 参考[官方测试](https://github.com/pytorch/tensordict/blob/main/benchmarks/distributed/dataloading.py)，构建多节点共享tensorclass_dataloader的情形，与一般的DataLoader性能进行对比；
2. 当数据集以一种比较高效的方式如hdf5存储时，验证tensorclass是否还有优势。

