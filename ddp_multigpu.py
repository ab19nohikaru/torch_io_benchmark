import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os, time, logging

from datasets_preprocess import MyDataSet, multigpu_dataset_preprocess_list, get_preprocessed_dataset
from mytrainer import Trainer, ToyNet, Collate

log_timestr = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())

if not os.path.exists("log"):
    os.makedirs("log")
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
        logger.info(f"[GPU{self.local_rank}:{self.global_rank}] {self.dataloader_type} | Epoch {epoch} Start")
        self.train_data.sampler.set_epoch(epoch)
        super()._run_epoch(epoch)


def load_train_objs(path: str, dataset_name:str):
    if dataset_name == "mnist":
        raw_training_data = datasets.FashionMNIST(
            root=path,
            train=True,
            download=False,
            transform=ToTensor(),
        )
        data_path = os.path.join(path, "mnist")
    else:
        raw_training_data = MyDataSet.from_pyz(path)
        data_path = os.path.join(path, "mydataset")

    logger.info(f"loading {dataset_name} from {data_path} len {len(raw_training_data)}")
    model = ToyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return raw_training_data, model, optimizer, data_path


def prepare_dataloader(dataset: Dataset, batch_size: int, preprocess_type:str, data_path:str, num_workers:int):
    if "tensorclass" in preprocess_type:
        collate_fn=Collate()
    else:
        collate_fn = None
    training_data = get_preprocessed_dataset(preprocess_type, dataset, data_path)
    return DataLoader(
        training_data,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
        sampler=DistributedSampler(training_data),
        collate_fn=collate_fn
    )


def main(total_epochs: int, batch_size: int, path: str, dataset_name:str, num_workers:int):
    ddp_setup()
    raw_dataset, model, optimizer, data_path = load_train_objs(path, dataset_name)
    dl_types = multigpu_dataset_preprocess_list

    for preprocess_type in dl_types:
        dataloader = prepare_dataloader(raw_dataset, batch_size, preprocess_type, data_path, num_workers)
        trainer = DDPTrainer(model, dataloader, optimizer, preprocess_type)
        logger.info(f"{preprocess_type} Start Train")
        trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--path', required=True, type=str, help='Path of dataset')
    #parser.add_argument('--dataset', choices=["mnist", "mydataset"], help='Select dataset for test')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of DataLoader workers')
    parser.add_argument('--repeats', default=1, type=int, help='Number of repeat runs')
    args = parser.parse_args()

    logger.info(args)
    #main(args.total_epochs, args.batch_size, args.path, args.dataset)
    split_line = "*"*65
    for i in range(args.repeats):
        logger.info(split_line + f"Loop {i+1} Start" + split_line)
        main(args.total_epochs, args.batch_size, args.path, "mnist", args.num_workers)
        logger.info(split_line)
        logger.info(split_line)
        main(args.total_epochs, args.batch_size, args.path, "mydataset", args.num_workers)
        logger.info(split_line + f"Loop {i+1} End" + split_line)
    