import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import time, os, platform
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from torch.profiler import profile, record_function, ProfilerActivity
from test_datasets import *

logging.basicConfig(filename="data/log_"+ time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) + ".log",
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def get_available_devices():
    devices = [(torch.device("cpu"), platform.processor())]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [(torch.device(f"cuda:{i}"),  torch.cuda.get_device_name(i))]
    return devices

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Available cpus: {os.cpu_count()}")
logger.info(f"Available devices: {get_available_devices()}")
logger.info(f"Using device: {device}")

# various dataset preprocessing method
dataset_preprocess = {"raw":lambda x:x,
                      "preload":PreloadDataSet,
                      "tensorclass":lambda dataset:FashionMNISTData.from_dataset(dataset, device=device),
                      "memmap":MemmappedDataSet
                      }
dl_types = dataset_preprocess.keys()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            logger.debug(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def tranverse_dataloader(dataloader:DataLoader, epochs:int)->float:
    t0 = time.time()
    for t in range(epochs):
        for data in dataloader:
            pass
    return time.time() - t0

def train_tc(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, data in enumerate(dataloader):
        X, y = data.images.contiguous(), data.targets.contiguous()

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            logger.debug(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    logger.debug(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

def test_tc(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch.images.contiguous(), batch.targets.contiguous()

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    logger.debug(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def cmp_tensorclass_demo(raw_dataset:Dataset, shuffle:bool):
    batch_size = 64
    epochs = 1

    loss_fn = nn.CrossEntropyLoss()

    for dl_index, preprocess_type in enumerate(dl_types):
        if preprocess_type == "tensorclass":
            collate_fn=lambda x: x
        else:
            collate_fn = None
        training_data = dataset_preprocess[preprocess_type](raw_dataset)
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        logger.info(f"{preprocess_type.capitalize()} dataloader {'random' if shuffle else 'sequential'} tranverse! time: {tranverse_dataloader(dataloader, epochs): 4.4f} s")

        model = Net().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    #on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{preprocess_type}'),
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=True,
                    #experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
                    ) as prof:
            with record_function(f"{preprocess_type} dataset train {epochs} epochs"):
                t0 = time.time()
                for t in range(epochs):
                    logger.debug(f"Epoch {t + 1}\n-------------------------")
                    if preprocess_type == "tensorclass":
                        collate_fn=lambda x: x
                        train_tc(dataloader, model, loss_fn, optimizer)
                    else:
                        train(dataloader, model, loss_fn, optimizer)
                logger.info(f"{preprocess_type.capitalize()} training done! time: {time.time() - t0: 4.4f} s")
        logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        #logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace(f"data/{preprocess_type}_trace.json")
        #prof.export_stacks(f"data/{preprocess_type}_cuda_profiler_stacks.txt", "self_cpu_time_total")
        #prof.export_stacks(f"data/{preprocess_type}_cpu_profiler_stacks.txt", "self_cuda_time_total")

def cmp_tranverse_time_for_batch(raw_dataset:Dataset, shuffle:bool):
    epochs = 1
    batch_range = range(1, 100)
    tranverse_time = np.zeros((len(batch_range), len(dl_types)))
    tranverse_time_dict = {"batch_size": batch_range}

    for dl_index, preprocess_type in enumerate(dl_types):
        if preprocess_type == "tensorclass":
            collate_fn=lambda x: x
        else:
            collate_fn = None
        training_data = dataset_preprocess[preprocess_type](raw_dataset)
        for bs_index, batch_size in enumerate(batch_range):
            dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
            tranverse_time[bs_index, dl_index] = tranverse_dataloader(dataloader, epochs)
        tranverse_time_dict[preprocess_type] = tranverse_time[:, dl_index]

    pd.DataFrame(tranverse_time_dict).to_csv(
                      f"data/{'random' if shuffle else 'sequential'}_tranverse_time_for_batchsize_" + 
                      f"{int(time.time())}" + ".csv")
    plt.plot(batch_range, tranverse_time)
    plt.legend(dl_types)
    #plt.show()


def cmp_tranverse_time_for_epoch(raw_dataset:Dataset, shuffle:bool):
    batch_size = 64
    epochs_range = range(1, 11)
    tranverse_time = np.zeros((len(epochs_range), len(dl_types)))
    tranverse_time_dict = {"epoch": epochs_range}

    for dl_index, preprocess_type in enumerate(dl_types):
        if preprocess_type == "tensorclass":
            collate_fn=lambda x: x
        else:
            collate_fn = None
        training_data = dataset_preprocess[preprocess_type](raw_dataset)
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        for ep_index, _ in enumerate(epochs_range):
            tranverse_time[ep_index, dl_index] = tranverse_dataloader(dataloader, 1)
        tranverse_time_dict[preprocess_type] = tranverse_time[:, dl_index]

    plt.plot(epochs_range, tranverse_time)
    plt.legend(["raw","preload","tc"])
    plt.show()

# tested: profiler stack trace not work with gpu in Win10
def test_cuda_trace_enable():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/test'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True) as prof:
        with record_function(f"test"):
            model = Net().to(device)

def trace_stack_tensorclass_preload(raw_dataset:Dataset):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                ) as prof:
        with record_function(f"Preload dataset as tensorclass"):
            FashionMNISTData.from_dataset(raw_dataset, device=device)
    prof.export_chrome_trace(f"data/tensorclass_preload.json")

if __name__ == "__main__":
    raw_training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    logger.info(f"dataset len {len(raw_training_data)}")
    #trace_stack_tensorclass_preload(raw_training_data)
    cmp_tensorclass_demo(raw_training_data, shuffle=True)
    #cmp_tensorclass_demo(raw_training_data, shuffle=False)
    #cmp_tranverse_time_for_batch(raw_training_data, shuffle=True)
logging.shutdown()