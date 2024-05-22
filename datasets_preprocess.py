import torch
from torch.utils.data import Dataset
from tensordict import MemoryMappedTensor
from tensordict.prototype import tensorclass

import time
import logging

logger = logging.getLogger(__name__)

import numpy as np
from tempfile import mkdtemp
import os.path as path

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
    def __init__(self, raw_dataset:Dataset) -> None:
        super().__init__()
        self.datafilename = path.join(mkdtemp(), 'pydatafile.dat')
        self.targetfilename = path.join(mkdtemp(), 'pytargetfile.dat')
        self.shape = ((len(raw_dataset), *raw_dataset[0][0].squeeze().shape))

        t0 = time.time()
        self.data_fp = np.memmap(self.datafilename, dtype='float32', mode='w+', shape=self.shape)
        self.target_fp = np.memmap(self.targetfilename, dtype='int64', mode='w+', shape=(self.shape[0],))
        for i, (image, target) in enumerate(raw_dataset):
            self.data_fp[i] = image
            self.target_fp[i] = target
        logger.info(f"DataSet memmap done! time: {time.time() - t0: 4.4f} s")

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data_fp[index]), self.target_fp[index]

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

class MyDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = torch.rand([20,28,28])
        self.targets = torch.randint(0, 10, (len(self.data),))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

# various dataset preprocessing method
dataset_preprocess_dict = {"raw":lambda x:x,
                      "preload":PreloadDataSet,
                      "tensorclass":lambda dataset:ImageData.from_dataset(dataset),
                      "memmap":MemmappedDataSet
                      }

if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
    import os
    raw_training_data = MyDataSet()
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    raw_training_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )
    save_fir = "log/preprocess_profile"
    if not os.path.exists(save_fir):
        os.makedirs(save_fir)
    logging.basicConfig(handlers=(logging.FileHandler(filename=save_fir + "/self_cpu_time_total.log", mode="w"),),
                    level=logging.INFO)
    for preprocess_type, preprocess_method in dataset_preprocess_dict.items():
        with profile(activities=[ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=True,
            with_stack=True,
            ) as prof:
            with record_function(f"{preprocess_type} dataset"):
                preprocess_method(raw_training_data)
        logging.info(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
        prof.export_chrome_trace(f"{save_fir}/preprocess_{preprocess_type}_trace.json")