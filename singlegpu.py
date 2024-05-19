import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import time, os, platform
import logging
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity
from datasets_preprocess import dataset_preprocess_dict
from mytrainer import Trainer, ToyNet

log_timestr = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())

if not os.path.exists("log"):
    os.makedirs("log")
logging.basicConfig(handlers=(logging.FileHandler(filename="log/singlegpu_"+ log_timestr + ".log"),
                              ),
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

logger.info(f"Available cpus: {os.cpu_count()}")
logger.info(f"Available devices: {get_available_devices()}")

def get_dataloader_tranverse_time(dataloader:DataLoader, epochs:int)->float:
    t0 = time.time()
    for t in range(epochs):
        for data in dataloader:
            pass
    return time.time() - t0

def test_tensorclass_singlegpu(raw_dataset:Dataset, shuffle:bool, batch_size:int, epochs:int,
                               device, with_profiler:bool, export_josn:bool):

    dl_types = dataset_preprocess_dict.keys()

    for dl_index, preprocess_type in enumerate(dl_types):
        if preprocess_type == "tensorclass":
            collate_fn=lambda x: x
        else:
            collate_fn = None
        training_data = dataset_preprocess_dict[preprocess_type](raw_dataset)
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        logger.info(f"{preprocess_type.capitalize()} dataloader {'random' if shuffle else 'sequential'} tranverse! time: {get_dataloader_tranverse_time(dataloader, epochs): 4.4f} s")

        model = ToyNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = Trainer(model, dataloader, optimizer, device, preprocess_type)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    profile_memory=True,
                    with_stack=True,
                    ) if with_profiler else nullcontext() as prof:
            with record_function(f"{preprocess_type} dataset train {epochs} epochs"):
                t0 = time.time()
                trainer.train(epochs)
                logger.info(f"{preprocess_type.capitalize()} training done! time: {time.time() - t0: 4.4f} s")
        if with_profiler:
            logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            if export_josn:
                prof.export_chrome_trace(f"data/singlegpu_{log_timestr}/{preprocess_type}_trace.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark IO for single GPU')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('device', choices=["cpu", "gpu"], help='use CPU or single GPU to train')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 64)')
    parser.add_argument('--with_profiler', action="store_true", help='Use torch.profile to get a verbose output')
    parser.add_argument('--export_json', action="store_true", help='Export result by export_chrome_trace method')
    args = parser.parse_args()
    if args.device == "gpu":
        if torch.cuda.is_available():
            device = 0 
        else:
            raise ValueError("no available gpu")
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    raw_training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    logger.info(f"dataset len {len(raw_training_data)}")

    test_tensorclass_singlegpu(raw_training_data, True, args.batch_size,
                            args.total_epochs, device, args.with_profiler,
                            args.export_json)