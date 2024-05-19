# 单机单卡

```python
python singlegpu.py 5 gpu
```

会在log目录下生成log, 若使用--with_profiler --export_json 会在data目录下生成json文件

```she
usage: singlegpu.py [-h] [--batch_size BATCH_SIZE] [--with_profiler] [--export_json] total_epochs {cpu,gpu}

Benchmark IO for single GPU

positional arguments:
  total_epochs          Total epochs to train the model
  {cpu,gpu}             use CPU or single GPU to train

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Input batch size on each device (default: 64)
  --with_profiler       Use torch.profile to get a verbose output
  --export_json         Export result by export_chrome_trace method

example:
python singlegpu.py 1 gpu
```



# 多机多卡(DDP)

单卡版本的简单移植，使用[torchrun](https://pytorch.org/docs/stable/elastic/run.html)启动,，可参考[源码](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py)

## 单机多卡

```
torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    ddp_multigpu.py [--batch_size BATCH_SIZE] total_epochs
```

## 多机多卡

```
torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    ddp_multigpu.py [--batch_size BATCH_SIZE] total_epochs
```

