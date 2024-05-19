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
