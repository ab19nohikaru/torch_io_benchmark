import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import time, os, platform
import logging
from contextlib import nullcontext
from torch.profiler import profile, record_function, ProfilerActivity
from datasets_preprocess import get_preprocessed_dataset, MyDataSet, dataset_preprocess_list
from mytrainer import Trainer, ToyNet, Collate

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

def test_tensorclass_singlegpu(raw_dataset:Dataset, shuffle:bool, batch_size:int, epochs:int,
                               device, data_dir:str, with_profiler:bool, export_josn:bool,
                               num_workers:int):

    dl_types = dataset_preprocess_list
    for dl_index, preprocess_type in enumerate(dl_types):
        if "tensorclass" in preprocess_type:
            collate_fn=Collate()
        else:
            collate_fn = None
        training_data = get_preprocessed_dataset(preprocess_type, raw_dataset, data_dir)
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                                num_workers=num_workers)
        #logger.info(f"{preprocess_type.capitalize()} dataloader {'random' if shuffle else 'sequential'} tranverse! time: {get_dataloader_tranverse_time(dataloader, epochs): 4.4f} s")
        # logger.info(f"{preprocess_type.capitalize()} dataloader {len(dataloader)}")
    
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
            time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
            if export_josn:
                if not os.path.exists(f"data/singlegpu_{time_str}"):
                    os.makedirs(f"data/singlegpu_{time_str}")
                prof.export_chrome_trace(f"data/singlegpu_{time_str}/{preprocess_type}_trace.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark IO for single GPU')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('device', choices=["cpu", "gpu"], help='use CPU or single GPU to train')
    parser.add_argument('--path', required=True, type=str, help='Path of dataset')
    parser.add_argument('--dataset', required=True, choices=["mnist", "mydataset"], help='Select dataset for test')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 64)')
    parser.add_argument('--with_profiler', action="store_true", help='Use torch.profile to get a verbose output')
    parser.add_argument('--export_json', action="store_true", help='Export result by export_chrome_trace method')
    parser.add_argument('--repeats', default=1, type=int, help='Number of repeat runs')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of DataLoader workers')
    args = parser.parse_args()
    if args.device == "gpu":
        if torch.cuda.is_available():
            device = 0 
        else:
            raise ValueError("no available gpu")
    else:
        device = "cpu"
    logger.info(f"Available cpus: {os.cpu_count()}")
    logger.info(f"Available devices: {get_available_devices()}")
    logger.info(f"Using device: {device}")
    logger.info(args)
    if args.dataset == "mnist":
        raw_training_data = datasets.FashionMNIST(
            root=args.path,
            train=True,
            download=False,
            transform=ToTensor(),
        )
        data_path = os.path.join(args.path, "mnist")
    else:
        raw_training_data = MyDataSet.from_pyz(args.path)
        data_path = os.path.join(args.path, "mydataset")
    logger.info(f"loading {args.dataset} from {data_path} len {len(raw_training_data)}")

    split_line = "*"*65
    for i in range(args.repeats):
        logger.info(split_line + f"Loop {i+1} Start" + split_line)
        test_tensorclass_singlegpu(raw_training_data, True, args.batch_size,
                                args.total_epochs, device, data_path, args.with_profiler,
                                args.export_json, args.num_workers)
        logger.info(split_line + f"Loop {i+1} End" + split_line)