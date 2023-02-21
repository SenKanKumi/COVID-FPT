import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CovidDataset(Dataset):
    def __init__(self, mode, dim=(224, 224)):
        self.class_dict = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2, 'positive': 2}
        self.dim = dim
        self.mode = mode
        if self.mode == "train":
            with open("./data/small/train.txt", 'r')as f:
                self.info = f.readlines()
        elif self.mode == "val":
            with open("./data/small/val.txt", 'r')as f:
                self.info = f.readlines()
        elif self.mode == "test":
            with open("./data/small/test.txt", 'r')as f:
                self.info = f.readlines()
        else:
            raise (Exception("mode error"))

    def __getitem__(self, index):
        img_name = self.info[index].split(' ')[1]
        img_label = self.info[index].split(' ')[2]
        if self.mode == "train":
            img_path = "./data/small/train/" + img_name
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.mode == "val":
            img_path = "./data/small/val/" + img_name
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.mode == "test":
            img_path = "./data/small/test/" + img_name
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
        return len(self.info)


if __name__ == "__main__":
    Train_Dataset = CovidDataset("train")
