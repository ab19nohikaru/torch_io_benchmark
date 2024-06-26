# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [v0.0.1] - 2024-5-14

完成单机单卡测试demo

## [v0.0.2] - 2024-5-18

单机单卡测试demo，增加torch.profiler相关代码和结果分析

## [v0.0.3] - 2024-5-19

1. 修正之前tensorclass构造时直接拷进gpu的问题，更正相关conclusion
2. 废弃单机单卡demo cmp_tensordict.py，重构为 singlegpu.py
3. 完成多机多卡的简单移植 ddp_multigpu.py

## [v0.0.4] - 2024-5-20

将dataset download与training拆分。使用dataset_download.py脚本下载dataset到指定路径，training脚本运行时需要指定dataset路径。

## [v0.0.5] - 2024-5-21

1. 补充ddp_multigpu.py的slurm启动脚本范例
2. 实现基于pytorch-lightning版本的多机多卡 lightning_multinode.py，并附上slurm启动脚本范例

## [v0.0.6] - 2024-5-23

1. 修复MemmappedDataSet读取慢的问题
2. 修复singlegpu.py使用选项--with_profiler --export_json报错的bug
3. 修正和更新conclusion.md

## [v0.1.0] - 2024-5-26

1. 新增数据集MyDataSet，更新相关测试代码

2. 更新dataset_download.py，测试前需要重新运行生成数据集文件

3. 新增数据集遍历时间和训练时间的统计测试，更新conclusion.md

## [v0.1.2] - 2024-5-28

1. 增加脚本选项--repeats重复次数， --num_workers Dataloader进程数
2. lambda函数改为正常函数，使得脚本适配多进程Dataloader
3. 增加实验参数列表