import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CovidDataset(Dataset):
    def __init__(self, data, mode):
        self.class_dict = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2, 'positive': 2}
        self.data_PATH = "./data/{}/{}".format(data, mode)
        with open(self.data_PATH + ".txt", "r") as f:
            self.data = f.readlines()
        self.mode = mode

    def __getitem__(self, index):
        img_name = self.data[index].split(' ')[1]
        img_label = self.data[index].split(' ')[2]
        img_path = self.data_PATH + "/" + img_name
        if self.mode == "train":
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.mode == "val":
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.mode == "test":
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        else:
            raise (Exception("mode error"))

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        label_tensor = torch.tensor(self.class_dict[img_label])
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.data)


def Creat_DataSet(data, mode, batch, shuffle):
    COVID_DataLoader = DataLoader(dataset=CovidDataset(data, mode), batch_size=batch, shuffle=shuffle, num_workers=4,
                                  pin_memory=True)
    return COVID_DataLoader


def Creat_OneData(data_path):
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(data_path).convert("RGB")
    img_tensor = transform(img)
    return img_tensor
