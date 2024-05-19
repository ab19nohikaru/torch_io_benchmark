import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os, time, logging

from datasets_preprocess import dataset_preprocess_dict
from mytrainer import Trainer, ToyNet

log_timestr = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())

logging.basicConfig(handlers=(logging.FileHandler(filename="log/ddp_"+ log_timestr + ".log"),
                              ),
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class DDPTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        dataloader_type: str, 
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        super().__init__(model, train_data, optimizer, self.local_rank, dataloader_type)
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _run_epoch(self, epoch):
        logger.info(f"[GPU{self.local_rank}:{self.global_rank}] Epoch {epoch} Start")
        self.train_data.sampler.set_epoch(epoch)
        super()._run_epoch(epoch)


def load_train_objs():
    train_set = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    model = ToyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, preprocess_type:str):
    if preprocess_type == "tensorclass":
        collate_fn=lambda x: x
    else:
        collate_fn = None
    training_data = dataset_preprocess_dict[preprocess_type](dataset)
    return DataLoader(
        training_data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(training_data),
        collate_fn=collate_fn
    )


def main(total_epochs: int, batch_size: int):
    ddp_setup()
    raw_dataset, model, optimizer = load_train_objs()
    dl_types = dataset_preprocess_dict.keys()

    for preprocess_type in dl_types:
        dataloader = prepare_dataloader(raw_dataset, batch_size, preprocess_type)
        trainer = DDPTrainer(model, dataloader, optimizer, preprocess_type)
        trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    main(args.total_epochs, args.batch_size)