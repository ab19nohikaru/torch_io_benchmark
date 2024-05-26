# 数据集下载

执行训练任务前需要执行下列指令，下载FashionMNIST数据集并生成MyDataSet数据集到指定路径
```
python dataset_download.py --path PATH
```

# 单机单卡

```python
python singlegpu.py 5 gpu --path data/ --dataset=mnist
python singlegpu.py 5 gpu --path data/ --dataset=mydataset
```

**会在log目录下生成log**, 若使用--with_profiler --export_json 会在data目录下生成json文件

```she
usage: singlegpu.py [-h] --path PATH --dataset {mnist,mydataset} [--batch_size BATCH_SIZE] [--with_profiler] [--export_json] total_epochs {cpu,gpu}

Benchmark IO for single GPU

positional arguments:
  total_epochs          Total epochs to train the model
  {cpu,gpu}             use CPU or single GPU to train

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path of dataset
  --dataset {mnist,mydataset}
                        Select dataset for test
  --batch_size BATCH_SIZE
                        Input batch size on each device (default: 64)
  --with_profiler       Use torch.profile to get a verbose output
  --export_json         Export result by export_chrome_trace method

example:
python singlegpu.py 5 gpu --path data/ --dataset=mnist
python singlegpu.py 5 gpu --path data/ --dataset=mydataset
```

脚本test_funcs.py，测试对比数据集不同预处理后遍历/训练时间 **epoch 取 5， 重复20次**

测试结果以.npz格式保存在log目录

```
usage: test_funcs.py [-h] --path PATH [--dataset {mnist,mydataset}] [--test_train_time] [--test_tranverse_time]

Test dataset train/tranverse time for single gpu

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path of dataset
  --dataset {mnist,mydataset}
                        Select dataset for test
  --test_train_time     Test dataset train time
  --test_tranverse_time
                        Test dataset tranverse time
                        
example:
python test_funcs.py --path data/ --dataset=mydataset --test_tranverse_time
python test_funcs.py --path data/ --dataset=mnist --test_train_time
```

# 多机多卡(DDP)

单卡版本的简单移植

## 单机多卡

ddp_multigpu.py 使用[torchrun](https://pytorch.org/docs/stable/elastic/run.html)启动,，可参考[源码](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py)

**运行结果保持在log目录下**

```
torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    ddp_multigpu.py [--batch_size BATCH_SIZE] total_epochs --path DATASETPATH
    
example:
  torchrun --standalone  --nnodes=1 --nproc-per-node=1 ddp_multigpu.py 5 --path data
```

## 多机多卡

### 1. ddp_multigpu.py

利用 [torchrun](https://pytorch.org/docs/stable/elastic/run.html)启动， slurm脚本参考srun_ddp.sh，**脚本需指定DATASETPATH**

**运行结果保持在log目录下**

```
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    ddp_multigpu.py [--batch_size BATCH_SIZE] total_epochs --path DATASETPATH
```



### 2. lightning_multinode.py

slurm脚本参考srun_lightning.sh，**脚本需指定DATASETPATH**

**运行结果保持在log目录下**

```
usage: lightning_multinode.py [-h] [--batch_size BATCH_SIZE] --path PATH --gpus GPUS --nnodes NNODES total_epochs

simple distributed training job

positional arguments:
  total_epochs          Total epochs to train the model

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Input batch size on each device (default: 32)
  --path PATH           Path of dataset
  --gpus GPUS           GPUs per node
  --nnodes NNODES       Number of nodes
  
example:
  python lightning_multinode.py 5 --path data/ --gpus 8 --nnodes 3
```



# Conclusion

测试数据和小结参见[conclusion.md](https://github.com/ab19nohikaru/torch_io_benchmark/blob/main/conclusion.md)

