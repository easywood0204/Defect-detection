import os
import numpy as np
from matplotlib import pyplot as pil
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler


class MyDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        imgs = []
        for classes in range(0, 2):
            list_path = os.listdir(self.path + '/' + str(classes))
            for file in list_path:
                image_location = self.path + '/' + str(classes) + '/' + file
                img = pil.imread(image_location).copy()
                if self.transform is not None:
                    img = self.transform(img)
                imgs.append((img, classes))
                self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        return img, label

    def __len__(self):
        return len(self.imgs)


def get_data(path, transform, batch_size, train_percent):
    dataset = MyDataset(path=path, transform=transform)
    shuffled_indices = np.random.permutation(len(dataset))
    print(shuffled_indices)
    train_idx = shuffled_indices[:int(train_percent * len(dataset))]
    test_idx = shuffled_indices[int(train_percent * len(dataset)):]
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx))
    return train_loader, test_loader


def random_seed():
    np.random.seed(1)


def main():
    random_seed()
    train_loader, test_loader = get_data(path='./data',
                                         transform=transforms.ToTensor(),
                                         batch_size=64,
                                         train_percent=0.7)
    
    # for idx, (data, label) in enumerate(train_loader)


if __name__ == '__main__':
    main()
