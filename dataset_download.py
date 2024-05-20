from torchvision import datasets

def download_FashionMNIST(path: str):
    train_set = datasets.FashionMNIST(
        root=path,
        train=True,
        download=True,
    )
    print(f"dataset is ready in {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download dataset for subsequent training')
    parser.add_argument('--path', required=True, type=str, help='Save path of dataset')
    args = parser.parse_args()

    download_FashionMNIST(args.path)