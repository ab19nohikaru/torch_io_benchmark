from datasets_preprocess import dataset_preprocess_list, MyDataSet, get_preprocessed_dataset
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from mytrainer import Trainer, ToyNet, Collate
import torch
import os, logging

log_timestr = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())

if not os.path.exists("log"):
    os.makedirs("log")
logging.basicConfig(handlers=(logging.FileHandler(filename="log/test_script_"+ log_timestr + ".log"),
                              ),
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def get_dataloader_tranverse_time(dataloader:DataLoader, epochs:int, dataloader_type:str):
    t0 = time.time()
    #logger.info(f"{dataloader_type}:{epochs} {len(dataloader)}")
    for t in range(epochs):
        for batch in dataloader:
            if "tensorclass" in dataloader_type:
                source = batch.images.contiguous()
                targets = batch.targets.contiguous()
            else:
                source, targets = batch
            # avoid memmap lazy load
            source.to(0)
            targets.to(0)
    return time.time() - t0

# def get_dataloader_tranverse_time(dataloader:DataLoader, epochs:int, dataloader_type:str):
#     t0 = time.time()
#     for t in range(epochs):
#         for batch in dataloader:
#             pass
#     return time.time() - t0

def test_dataloader_traverse(raw_dataset:Dataset, batch_size:int, epochs:int,
                             data_dir:str):
    repeats = 20
    shuffle = True

    dl_types = dataset_preprocess_list
    # dl_types = [#"raw",
    #                 "preload",
    #                 # "tensorclass",
    #                 # "tensorclass_memmap",
    #                 # "np_memmap"
    #                 ]
    tranverse_time = {preprocess_type:np.zeros((repeats,)) for preprocess_type in dl_types}

    for dl_index, preprocess_type in enumerate(dl_types):
        if "tensorclass" in preprocess_type:
            collate_fn=Collate()
        else:
            collate_fn = None
        training_data = get_preprocessed_dataset(preprocess_type, raw_dataset, data_dir)
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        for index in range(repeats):
            tranverse_time[preprocess_type][index] = get_dataloader_tranverse_time(dataloader, epochs, preprocess_type)
        logger.info(f"{preprocess_type} tranverse time {np.mean(tranverse_time[preprocess_type])} {np.std(tranverse_time[preprocess_type])}")
    return tranverse_time

def test_dataloader_train(raw_dataset:Dataset, batch_size:int, epochs:int,
                             data_dir:str):
    repeats = 20
    shuffle = True
    device = 0

    dl_types = dataset_preprocess_list
    train_time_dict = {preprocess_type:np.zeros((repeats,)) for preprocess_type in dl_types}

    for dl_index, preprocess_type in enumerate(dl_types):
        if "tensorclass" in preprocess_type:
            collate_fn=Collate()
        else:
            collate_fn = None
        training_data = get_preprocessed_dataset(preprocess_type, raw_dataset, data_dir)
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                                pin_memory=False)
        model = ToyNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = Trainer(model, dataloader, optimizer, device, preprocess_type)
        for index in range(repeats):
            t0 = time.time()
            trainer.train(epochs)
            train_time_dict[preprocess_type][index] = time.time() - t0
        logger.info(f"{preprocess_type} train time {np.mean(train_time_dict[preprocess_type])} {np.std(train_time_dict[preprocess_type])}")
    return train_time_dict

def profile_dataset_preprocess(raw_dataset:Dataset, batch_size:int, epochs:int,
                             data_dir:str):
    from torch.profiler import profile, record_function, ProfilerActivity

    shuffle = True

    save_fir = "log/tranverse_profile"
    if not os.path.exists(save_fir):
        os.makedirs(save_fir)
    preprocess_type = "np_memmap"
    if "tensorclass" in preprocess_type:
        collate_fn=Collate()
    else:
        collate_fn = None
    training_data = get_preprocessed_dataset(preprocess_type, raw_dataset, data_dir)
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    logger.info(f"{preprocess_type} len {len(training_data)}")
    with profile(activities=[ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=True,
        with_stack=True,
        ) as prof:
        with record_function(f"{preprocess_type} tranverse_time"):
            tranverse_time = get_dataloader_tranverse_time(dataloader, epochs, preprocess_type)
    logger.info(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    prof.export_chrome_trace(f"{save_fir}/preprocess_{preprocess_type}_tranverse_trace.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test dataset train/tranverse time for single gpu')
    parser.add_argument('--path', required=True, type=str, help='Path of dataset')
    parser.add_argument('--dataset', choices=["mnist", "mydataset"], help='Select dataset for test')
    parser.add_argument('--test_train_time', action="store_true", help='Test dataset train time')
    parser.add_argument('--test_tranverse_time', action="store_true", help='Test dataset tranverse time')
    args = parser.parse_args()

    data_path = args.path
    epochs_once = 5
    batch = 64
    import os
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    if args.dataset == "mnist":
        raw_training_data = datasets.FashionMNIST(
            root=data_path,
            train=True,
            download=False,
            transform=ToTensor(),
        )
        tag = "FashionMNIST"
        data_path = os.path.join(data_path, "mnist")
    else:
        raw_training_data = MyDataSet.from_pyz(data_path)
        tag = "MyDataSet"
        data_path = os.path.join(data_path, "mydataset")
    logger.info(f"{tag} len {len(raw_training_data)}")
    time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())

    if args.test_tranverse_time:
        tranverse_time_dict = test_dataloader_traverse(raw_training_data, batch, epochs_once, data_path)
        save_dir = f"log/tranverse_{tag}_{epochs_once}_time_{time_str}"
        np.savez(save_dir, **tranverse_time_dict)
    #profile_dataset_preprocess(raw_training_data, batch, epochs_once, data_path)
    if args.test_train_time:
        train_time_dict = test_dataloader_train(raw_training_data, batch, epochs_once, data_path)
        save_dir = f"log/train_{tag}_{epochs_once}_time_{time_str}"
        np.savez(save_dir, **train_time_dict)