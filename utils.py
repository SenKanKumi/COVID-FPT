from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.header import Header
import time
import os
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Grayscale, ColorJitter
from Models.unet import UNet
import numpy as np


import cv2
import math
import random
import shutil



# Service Inform
class ServiceInform:
    def __init__(self, info):
        smtp = SMTP_SSL('smtp.qq.com', 465)
        smtp.login("2463515960@qq.com", "zpmoukavpmxldiig")
        mail_msg = """<p>{}</p>""".format(info)
        message = MIMEText(mail_msg, 'html', 'utf-8')
        message['From'] = Header("MiaoMiao", 'utf-8')
        message['To'] = Header("Chikann", 'utf-8')
        message['Subject'] = Header('Service Inform', 'utf-8')
        smtp.sendmail("2463515960@qq.com", "2463515960@qq.com", message.as_string())
        smtp.quit()


def CreateLog(epoch, batch, model, optim, sche, info):
    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    with open("./Log/{}.txt".format(now), "w") as f:
        f.write("All Epoch:{}\n".format(epoch))
        f.write("BatchSize:{}\n".format(batch))
        f.write("Model and Dataset:{}\n".format(model))
        f.write("optim:{}\n".format(optim))
        f.write("scheduler:{}\n".format(sche))
        f.write("info:{}\n".format(info))
    return "./Log/{}.txt".format(now)


def Log(file, message):
    with open(file, "a") as f:
        f.write(message + "\n")


def Confusion_matrix(pred, target, matrix):
    for (t, p) in zip(target, pred):
        matrix[t, p] += 1
    return matrix


def benchmark(matrix):
    TP = np.diag(matrix)
    FP = np.sum(np.array(matrix), axis=0) - TP
    FN = np.sum(np.array(matrix), axis=1) - TP
    TN = np.sum(np.sum(np.array(matrix), axis=0)) - (TP + FP + FN)
    TP = TP.astype(float)
    TN = TN.astype(float)
    FP = FP.astype(float)
    FN = FN.astype(float)
    Accuracy = TP / (TP + FP + FN + TN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return Accuracy, Precision, Recall


def Evaluate(epoch, matrix, train_loss, val_loss, file):
    Accuracy, Precision, Recall = benchmark(matrix)
    print("\ntrain loss:{}  val loss:{}".format(train_loss, val_loss))
    print("{0:^15}\t{1:^15}\t{2:^15}".format("Kind", "Precision", "Recall"))
    print("{0:^15}\t{1:^15.2f}\t{2:^15.2f}".format("pneumonia", Precision[0], Recall[0]))
    print("{0:^15}\t{1:^15.2f}\t{2:^15.2f}".format("normal", Precision[1], Recall[1]))
    print("{0:^15}\t{1:^15.2f}\t{2:^15.2f}".format("covid-19", Precision[2], Recall[2]))
    Accuracy = np.sum(Accuracy)
    Precision = np.sum(Precision) / 3.
    Recall = np.sum(Recall) / 3.
    print("Accuracy:{:.4f};Precision;{:.4f};Recall:{:.4f}".format(Accuracy, Precision, Recall))
    Log(file, "Epoch:{}\ntrain_loss:{} val_loss:{}".format(epoch, train_loss, val_loss))
    Log(file, "Accuracy:{:.4f}; Precision:{:.4f}; Recall:{:.4f}".format(Accuracy, Precision, Recall))
    Log(file, "Confusion Matrix:{}\n".format(list(matrix)))
    return Accuracy, Precision, Recall


def lossCurve(filename):
    # small train dataset has 2799 pictures
    # small val dataset has 310 pictures

    # base train dataset has 21618
    # base val dataset has 6176
    with open("./Log/{}".format(filename), "r") as f:
        info = f.readlines()
    Epoch = [i + 1 for i in range(int(info[0].split(":")[1]) - 1)]
    batchsize = int(info[1].split(":")[1])
    train_loss = []
    val_loss = []
    Acc = []
    Pre = []
    Rec = []
    for index in range(len(info)):
        if info[index].split(":")[0] == "Epoch":
            train_loss.append(float(info[index + 1].split(" ")[0].split(":")[1]))
            val_loss.append(float(info[index + 1].split(" ")[1].split(":")[1]))
            Acc.append(float(info[index + 2].split(";")[0].split(":")[1]))
            Pre.append(float(info[index + 2].split(";")[1].split(":")[1]))
            Rec.append(float(info[index + 2].split(";")[2].split(":")[1]))
    plt.plot(Epoch, train_loss, Epoch, val_loss)
    # F2 = []
    # for i in range(len(Pre)):
    #     F2.append(float((5 * Pre[i] * Rec[i]) / (4 * Pre[i] + Rec[i])))
    # plt.plot(Epoch, F2)
    # plt.xlabel('Epoch')
    # plt.ylabel('F2')
    plt.show()


def splitData():
    trainData = os.listdir("D://ChikannDatabase/COVID/train")
    valData = os.listdir("D://ChikannDatabase/COVID/val")
    with open("D://ChikannDatabase/COVID/all.txt", "r") as f:
        traininfo = f.readlines()
    for line in traininfo:
        imgname = line.split(' ')[1]
        if imgname in trainData:
            with open("D://ChikannDatabase/COVID/train.txt", "a") as f1:
                f1.write(line)
        elif imgname in valData:
            with open("D://ChikannDatabase/COVID/val.txt", "a") as f2:
                f2.write(line)
        else:
            print("can not find " + imgname)


class dataset(Dataset):
    def __init__(self, mode, dim=(224, 224)):
        self.dim = dim
        self.mode = mode
        self.info = os.listdir("./data/base/{}_enhance".format(mode))

    def __getitem__(self, index):
        img_name = self.info[index]
        # D:\ChikannDatabase\Lung Segmentation\CXR_png
        img_path = "./data/base/{}_enhance/".format(self.mode) + img_name
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            Grayscale(),
            transforms.ToTensor()])

        try:
            img = Image.open(img_path).convert('LA')
            img_tensor = transform(img)
            return img_tensor, img_name
        except:
            print(img_path)
            with open("./aaaaaa.txt", "a") as f:
                f.write(img_path)
            img_tensor = []
            return img_tensor, ""

    def __len__(self):
        return len(self.info)


def Lung_Seg(mode):
    net = UNet(n_channels=1, n_classes=1)
    net.eval()
    checkpoint = torch.load("./Models/pth/Unet224.pt")
    net.load_state_dict(checkpoint['model_state_dict'])

    Train_Dataset = dataset(mode)
    Train_Dataloader = DataLoader(dataset=Train_Dataset, batch_size=1)

    for idx, (x, name) in enumerate(Train_Dataloader):
        print(idx, name)
        if name[0] == "":
            continue
        x0 = x.squeeze(0).numpy()
        x0 = x0 * 255 / x0.max()
        x1 = np.uint8(x0)
        x1 = x1.transpose(1, 2, 0)

        mask = net(x)
        mask = torch.sigmoid(mask) > 0.5
        mask = mask.squeeze(0)
        each = np.array(mask, dtype=np.uint8)
        each = each.transpose(1, 2, 0)

        # sobel = cv2.Sobel(each * 255, cvint2.CV_8U, 1, 0, ksize=3)
        # ret, binary = cv2.threshold(each * 255, 255, 0, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary = cv2.dilate(each * 255, k1, iterations=1)
        binary = cv2.erode(binary, k2, iterations=1)
        binary = np.uint8(binary[:, :, np.newaxis] / 255)
        mix = x1 * binary
        # mix = np.hstack((x1, binary * 255, mix))
        # cv2.imshow("a", mix)
        # cv2.waitKey(0)
        cv2.imwrite("./data/base/{}_seg/".format(mode) + name[0], mix)


def Enhance_Image(mode):
    imgs = os.listdir("./data/base/{}/".format(mode))
    for index, each in zip(range(len(imgs)), imgs):
        if os.path.exists("./data/base/{}_enhance/{}".format(mode, each)):
            continue
        print(index, each)
        try:
            img = cv2.imread("./data/base/{}/{}".format(mode, each))
            # Histogram equalization
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # Laplace enhanced sharpening
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)
            # img = np.hstack((img0, img3))
            # D:\ChikannDatabase\COVID\img_enahance
            cv2.imwrite("./data/base/{}_enhance/{}".format(mode, each), img)
        except:
            with open("./data/Exception.txt", "a") as f:
                f.write(each)


def DataCheck():
    class_dict = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2, 'positive': 2}
    path = ["./data/base/train.txt",
            "./data/base/test.txt",
            "./data/base/val.txt"]
    dir = ["./data/base/train/",
           "./data/base/test/",
           "./data/base/val/"]
    for p, d in zip(path, dir):
        with open(p, 'r') as f:
            info = f.readlines()
        kind = [0, 0, 0]
        total = 0
        for img in info:
            img_name = img.split(' ')[1]
            img_label = img.split(' ')[2]
            if os.path.exists(d + img_name):
                total += 1
                kind[class_dict[img_label]] += 1
            else:
                print("Can not find the {}".format(img_name))
        print("-----------------------------------------------")
        print("find {} imgs in {}".format(total, d))
        print("pneumonia:{}\tnormal:{}\tcovid:{}".format(kind[0], kind[1], kind[2]))


def visualize(pth_path):
    model = SwinT("tiny", load=False)
    state_dict = torch.load('./checkpoint/tinydata/{}'.format(pth_path), map_location=lambda storage, loc: storage)['model']
    model.load_state_dict(state_dict)
    img = Creat_OneData("./data/tiny/test/COVID19(139).jpg").unsqueeze(0)
    output, feature = model(img)
    map1 = feature[0].transpose(1, 2).view(1, 96, 56, 56).squeeze(0)
    channel_num = map1.size(0)
    Feature = map1[0,:,:].detach().numpy()
    for i in range(1,channel_num):
        Feature += map1[i,:,:].detach().numpy()
    Feature = np.expand_dims(Feature,axis=2)
    # Feature = np.asarray(Feature * 255, dtype=np.uint8)
    # Feature = cv2.resize(Feature, (224, 224), interpolation=cv2.INTER_NEAREST)
    plt.imshow(Feature)
    plt.savefig("./record/picture/channel.png")
    # Feature = cv2.applyColorMap(Feature, cv2.COLORMAP_JET)
    # cv2.imwrite("./record/picture/channel_{}.png".format(i), Feature)


if __name__ == "__main__":
    for file,name in zip(["FPT on small.txt","SwinT on small.txt","TNT on small.txt"],["COVID-FPT","SwinT-T","TNT-S"]):
        with open("./Log/{}".format(file), "r") as f:
            info = f.readlines()
        Epoch = [i + 1 for i in range(int(info[0].split(":")[1]) - 1)]
        batchsize = int(info[1].split(":")[1])
        train_loss = []
        val_loss = []
        Acc = []
        Pre = []
        Rec = []
        for index in range(len(info)):
            if info[index].split(":")[0] == "Epoch":
                train_loss.append(float(info[index + 1].split(" ")[0].split(":")[1]))
                val_loss.append(float(info[index + 1].split(" ")[1].split(":")[1]))
                Acc.append(float(info[index + 2].split(";")[0].split(":")[1]))
                Pre.append(float(info[index + 2].split(";")[1].split(":")[1]))
                Rec.append(float(info[index + 2].split(";")[2].split(":")[1]))
        F2 = []
        for i in range(len(Pre)):
            F2.append(float((5 * Pre[i] * Rec[i]) / (4 * Pre[i] + Rec[i])))
        plt.plot(Epoch, F2,label=name)
        plt.legend(loc=4)
    plt.xlabel('Epoch')
    plt.ylabel('F2')
    plt.savefig("F2-score of small.svg",dpi=300,format="svg")
