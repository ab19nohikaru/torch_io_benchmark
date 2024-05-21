from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import time, os, logging

from datasets_preprocess import dataset_preprocess_dict
from mytrainer import ToyNet

log_timestr = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())

if not os.path.exists("log"):
    os.makedirs("log")
logging.basicConfig(handlers=(logging.FileHandler(filename="log/lightning_"+ log_timestr + ".log"),
                              ),
                    level=logging.INFO)

class MyLitModule(L.LightningModule):
    def __init__(self, model, dataloader_type):
        super().__init__()
        self.model = model
        self.dataloader_type = dataloader_type
        self.epoch_start_time = 0

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        if self.dataloader_type == "tensorclass":
            source = batch.images.contiguous()
            targets = batch.targets.contiguous()
        else:
            source, targets = batch
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        self.log("Epoch Time:", time.time() - self.epoch_start_time)


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
        collate_fn=collate_fn
    )

def main(total_epochs: int, batch_size: int, gpus: int, nnodes:int, path: str):
    raw_dataset = datasets.FashionMNIST(
        root=path,
        train=True,
        download=False,
        transform=ToTensor(),
    )
    dl_types = dataset_preprocess_dict.keys()

    for preprocess_type in dl_types:
        dataloader = prepare_dataloader(raw_dataset, batch_size, preprocess_type)
        model = MyLitModule(ToyNet(), preprocess_type)

        trainer = L.Trainer(
            accelerator="gpu",
            devices=gpus,
            num_nodes=nnodes,
            max_epochs=total_epochs,
            logger=CSVLogger("log", name=preprocess_type, version=log_timestr),
            enable_checkpointing=False,
        )
        trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=64, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--path', required=True, type=str, help='Path of dataset')
    parser.add_argument('--gpus', required=True, type=int, help='GPUs per node')
    parser.add_argument('--nnodes', required=True, type=int, help='Number of nodes')
    args = parser.parse_args()

    main(args.total_epochs, args.batch_size, args.gpus, args.nnodes,  args.path)