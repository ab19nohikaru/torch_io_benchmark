from torchvision import datasets
from torch.utils.data import Dataset
from datasets_preprocess import MemmappedDataSet, ImageData

def generate_mmap_datasets(raw_training_data:Dataset, save_dir:str):
    MemmappedDataSet.generate_mmap(raw_training_data, save_dir)
    tcdata = ImageData.from_dataset(raw_training_data).memmap(save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download and prepare datasets for subsequent training')
    parser.add_argument('--path', required=True, type=str, help='Save path of dataset')
    args = parser.parse_args()

    from torchvision.transforms import ToTensor
    import os
    save_dir = args.path
    raw_training_data = datasets.FashionMNIST(
        root=save_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )
    MNIST_save_dir = os.path.join(save_dir, "mnist")
    generate_mmap_datasets(raw_training_data, MNIST_save_dir)
    print(f"FashionMNIST dataset is ready in {MNIST_save_dir}")

    from datasets_preprocess import MyDataSet
    my_training_data = MyDataSet.generate_random(60000)
    mydataset_save_dir = os.path.join(save_dir, "mydataset")
    generate_mmap_datasets(my_training_data, mydataset_save_dir)
    my_training_data.save(save_dir)
    print(f"MyDataSet dataset is ready in {mydataset_save_dir}")