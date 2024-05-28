import torch
from torch.utils.data import Dataset

from tensordict import MemoryMappedTensor, TensorDict
from tensordict.prototype import tensorclass

import time
import logging

logger = logging.getLogger(__name__)

import numpy as np
from tempfile import mkdtemp
import os

class PreloadDataSet(Dataset):
    def __init__(self, raw_dataset:Dataset) -> None:
        super().__init__()
        self.data = []
        t0 = time.time()
        for i in range(len(raw_dataset)):
            self.data.append(raw_dataset[i])
        logger.info(f"DataSet preload into memory done! time: {time.time() - t0: 4.4f} s")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class MemmappedDataSet(Dataset):
    DATAFILENAME = 'pydatafile'
    TARGETFILENAME = 'pytargetfile'
    def __init__(self, raw_dataset:Dataset, path:str=None, load:bool=True) -> None:
        super().__init__()
        if path is None:
            path = mkdtemp()
        if load:
            mmap_mode = 'w+'
        else:
            mmap_mode = 'r+'
        self.datafilename = os.path.join(path, self.DATAFILENAME)
        self.targetfilename = os.path.join(path, self.TARGETFILENAME)
        self.shape = ((len(raw_dataset), *raw_dataset[0][0].squeeze().shape))

        self.data_fp = np.memmap(self.datafilename, dtype='float32', mode=mmap_mode, shape=self.shape)
        self.target_fp = np.memmap(self.targetfilename, dtype='int64', mode=mmap_mode, shape=(self.shape[0],))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data_fp[index]), self.target_fp[index]

    @classmethod
    def from_dataset(cls, raw_dataset:Dataset):
        data = cls(raw_dataset)
        t0 = time.time()
        for i, (image, target) in enumerate(raw_dataset):
            data.data_fp[i] = image
            data.target_fp[i] = target
        logger.info(f"load DataSet done! time: {time.time() - t0: 4.4f} s")
        return data

    @classmethod
    def generate_mmap(cls, raw_dataset:Dataset, path:str):
        if not os.path.exists(path):
            os.makedirs(path)
        data = cls(raw_dataset, path, load=True)
        t0 = time.time()
        for i, (image, target) in enumerate(raw_dataset):
            data.data_fp[i] = image
            data.target_fp[i] = target
        logger.info(f"load DataSet done! time: {time.time() - t0: 4.4f} s")
        return data

    @classmethod
    def load_mmap(cls, raw_dataset:Dataset, path:str):
        t0 = time.time()
        data = cls(raw_dataset, path, load=False)
        logger.info(f"Load DataSet from numpy.memmap file done! time: {time.time() - t0: 4.4f} s")
        return data

@tensorclass
class ImageData:
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset, device=None):
        t0 = time.time()
        data = cls(
            images=MemoryMappedTensor.empty(
                (len(dataset), *dataset[0][0].squeeze().shape), dtype=torch.float32
            ),
            targets=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
            batch_size=[len(dataset)],
            device=device,
        )
        for i, (image, target) in enumerate(dataset):
            data[i] = cls(images=image, targets=torch.tensor(target), batch_size=[])
        logger.info(f"DataSet preload as tensorclass done! time: {time.time() - t0: 4.4f} s")
        return data

    @classmethod
    def load_mmap(cls, path:str):
        t0 = time.time()
        data = TensorDict.load_memmap(path)
        logger.info(f"Load DataSet from tensordict.memmap file done! time: {time.time() - t0: 4.4f} s")
        return data

class MyDataSet(Dataset):
    PYZFILENAME = 'mydataset.npz'
    def __init__(self):
        self.data = None
        self.targets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def save(self, save_dir:str):
        save_file = os.path.join(save_dir, self.PYZFILENAME)
        np.savez_compressed(save_file,
                            data=self.data, targets=self.targets)

    @classmethod
    def generate_random(cls, batch_size:int):
        datasets = cls()
        datasets.data = torch.rand([batch_size,28,28])
        datasets.targets = torch.randint(0, 10, (len(datasets.data),))
        return datasets

    @classmethod
    def from_pyz(cls, data_dir:str):
        load_file = os.path.join(data_dir, cls.PYZFILENAME)
        loaded_pack = np.load(load_file, mmap_mode="r")
        datasets = cls()
        datasets.data = loaded_pack["data"]
        datasets.targets = loaded_pack["targets"]
        return datasets

dataset_preprocess_list = ["raw", "preload", "tensorclass", "tensorclass_memmap", "np_memmap"]
multigpu_dataset_preprocess_list = ["raw", "tensorclass_memmap", "np_memmap"]

def get_preprocessed_dataset(preprocess_type:str, dataset:Dataset, path:str):
    if preprocess_type not in dataset_preprocess_list:
        raise ValueError(f"Invalid preprocess type {preprocess_type}")
    if "raw" == preprocess_type:
        return dataset
    elif "preload" == preprocess_type:
        return PreloadDataSet(dataset)
    elif "tensorclass" == preprocess_type:
        return ImageData.from_dataset(dataset)
    elif "tensorclass_memmap" == preprocess_type:
        return ImageData.load_mmap(path)
    elif "np_memmap" == preprocess_type:
        return MemmappedDataSet.load_mmap(dataset, path)

def profile_dataset_preprocess(raw_training_data: Dataset):
    from torch.profiler import profile, record_function, ProfilerActivity
    import os

    save_fir = "log/preprocess_profile"
    if not os.path.exists(save_fir):
        os.makedirs(save_fir)
    logging.basicConfig(handlers=(logging.FileHandler(filename=save_fir + "/self_cpu_time_total.log", mode="w"),),
                    level=logging.INFO)
    for preprocess_type in dataset_preprocess_list:
        with profile(activities=[ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=True,
            with_stack=True,
            ) as prof:
            with record_function(f"{preprocess_type} dataset"):
                get_preprocessed_dataset(preprocess_type, raw_training_data)
        logging.info(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        prof.export_chrome_trace(f"{save_fir}/preprocess_{preprocess_type}_trace.json")

if __name__ == "__main__":
    pass