import os
import cv2
from Models.MyModel import SwinT
from Models.MyModel import FPN
from Models.MyModel import CovidNET
import torch
from build.DataLoader import Creat_OneData
import numpy as np
from matplotlib import pyplot as plt


def SwinT_visualize(pth_path):
    model = SwinT("tiny", load=False)
    state_dict = torch.load('./checkpoint/tinydata/{}'.format(pth_path), map_location=lambda storage, loc: storage)[
        'model']
    model.load_state_dict(state_dict)
    for img_name in os.listdir("./data/feature"):
        print(img_name)
        img = Creat_OneData("./data/feature/{}".format(img_name)).unsqueeze(0)
        output, feature = model(img)
        map1 = feature[3].transpose(1, 2).view(1, 768, 7, 7)
        #map1 = cv2.resize(map1, (224, 224), interpolation=cv2.INTER_NEAREST)
        UP = torch.nn.UpsamplingNearest2d(size=(224,224))
        map1 = UP(map1).squeeze(0)
        channel_num = map1.size(0)
        Feature = map1[0, :, :].detach().numpy()
        for i in range(1, channel_num):
            Feature += map1[i, :, :].detach().numpy()
        Feature = np.expand_dims(Feature, axis=2)
        # Feature = np.asarray(Feature * 255, dtype=np.uint8)
        #
        plt.imshow(Feature)
        plt.savefig("./record/picture/SwinT_tiny77/{}".format(img_name))
        # Feature = cv2.applyColorMap(Feature, cv2.COLORMAP_JET)
        # cv2.imwrite("./record/picture/channel_{}.png".format(i), Feature)

def FPSwinT_visualize(pth_path):
    model = FPN(load=False)
    state_dict = torch.load('./checkpoint/tinydata/{}'.format(pth_path), map_location=lambda storage, loc: storage)[
        'model']
    model.load_state_dict(state_dict)
    for img_name in os.listdir("./data/feature"):
        print(img_name)
        img = Creat_OneData("./data/feature/{}".format(img_name)).unsqueeze(0)
        output, feature = model(img)
        UP = torch.nn.UpsamplingNearest2d(size=(224, 224))
        map1 = UP(feature[0]).squeeze(0)
        # map2 = UP(feature[1]).squeeze(0)
        # map3 = UP(feature[2]).squeeze(0)
        # map4 = UP(feature[3]).squeeze(0)
        #map = map1* 0.4+map2*0.3+map3*0.2+map4*0.1
        channel_num = map1.size(0)
        Feature = map1[0, :, :].detach().numpy()
        for i in range(1, channel_num):
            Feature += map1[i, :, :].detach().numpy()
        Feature = np.expand_dims(Feature, axis=2)
        # Feature = np.asarray(Feature * 255, dtype=np.uint8)
        # Feature = cv2.resize(Feature, (224, 224), interpolation=cv2.INTER_NEAREST)
        plt.imshow(Feature)
        plt.savefig("./record/picture/FPSwinT77/{}".format(img_name))
        # Feature = cv2.applyColorMap(Feature, cv2.COLORMAP_JET)
        # cv2.imwrite("./record/picture/channel_{}.png".format(i), Feature)

def CovidNet_visualize(pth_path):
    model = CovidNET()
    state_dict = torch.load('./checkpoint/tinydata/{}'.format(pth_path), map_location=lambda storage, loc: storage)[
        'model']
    model.load_state_dict(state_dict)
    for img_name in os.listdir("./data/feature"):
        print(img_name)
        img = Creat_OneData("./data/feature/{}".format(img_name)).unsqueeze(0)
        output, feature = model(img)
        map1 = feature
        #map1 = cv2.resize(map1, (224, 224), interpolation=cv2.INTER_NEAREST)
        UP = torch.nn.UpsamplingNearest2d(size=(224,224))
        map1 = UP(map1).squeeze(0)
        channel_num = map1.size(0)
        Feature = map1[0, :, :].detach().numpy()
        for i in range(1, channel_num):
            Feature += map1[i, :, :].detach().numpy()
        Feature = np.expand_dims(Feature, axis=2)
        # Feature = np.asarray(Feature * 255, dtype=np.uint8)
        #
        plt.imshow(Feature)
        plt.savefig("./record/picture/CovidNet77/{}".format(img_name))
        # Feature = cv2.applyColorMap(Feature, cv2.COLORMAP_JET)
        # cv2.imwrite("./record/picture/channel_{}.png".format(i), Feature)


if __name__ == "__main__":
    # CovidNet_visualize("1_22_0.9635_COVIDNET_tiny.pth")
    #FPSwinT_visualize("3_17_0.979_FPSIT_tiny.pth")
    SwinT_visualize("1_37_0.9806_SwinT_tiny.pth")
