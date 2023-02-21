import torch
import torch.nn as nn

#from Models.SwinT import SwinTransformer
from Models.SwinT_feature import SwinTransformer

from Models.FPSwinT import FPSwinT
from Models.Reverse_FPSwinT import Reverse_FPSwinT
from Models.CAT1 import CATransformer_1
from Models.CAT2 import CATransformer_2
from Models.CAT import CATransformer
from Models.TNT import TINT
from Models.poolformer import PoolFormer
from Models.covid_net import CovidNet
from Models.Resnet import ResNet18


# total = sum([param.nelement() for param in model.parameters()])
# print("Numberofparameter: % .2fM" % (total / 1e6))
class SwinT(nn.Module):
    def __init__(self, size, load=True):
        super().__init__()
        if size == "tiny":
            self.SIT = SwinTransformer(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], drop_path_rate=0.1)
            if load:
                state_dict = torch.load("./Models/pth/swin_tiny_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Linear(768, 3)
        elif size == "small":
            self.SIT = SwinTransformer(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], drop_path_rate=0.2)
            if load:
                state_dict = torch.load("./Models/pth/swin_small_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Linear(768, 3)
        elif size == "base":
            self.SIT = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                       drop_path_rate=0.3)
            if load:
                state_dict = torch.load("./Models/pth/swin_base_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Linear(1024, 3)
        elif size == "teacher":
            self.SIT = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                                       drop_path_rate=0.3)
            self.Classify = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3)
            )
        else:
            raise (Exception("SwinT size is error, please check your size"))

    def forward(self, x):
        x,feature= self.SIT(x)
        x = self.Classify(x)
        return x,feature


class CAT1(nn.Module):
    def __init__(self, size, load):
        super().__init__()
        # the number of Params:28.5M
        if size == "tiny":
            self.SIT = CATransformer_1(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
            if load:
                state_dict = torch.load("./Models/pth/swin_tiny_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(768,3)
            )
        elif size == "small":
            self.SIT = CATransformer_1(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24])
            if load:
                state_dict = torch.load("./Models/swin_small_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(768, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )
        elif size == "base":
            self.SIT = CATransformer_1(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            if load:
                state_dict = torch.load("./Models/pth/swin_base_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )

    def forward(self, x):
        x = self.Classify(self.SIT(x))
        return x


class CAT2(nn.Module):
    def __init__(self, size, load):
        super().__init__()
        # the number of Params:29.07M
        if size == "tiny":
            self.SIT = CATransformer_2(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
            if load:
                state_dict = torch.load("./Models/swin_tiny_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(768, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )
        elif size == "small":
            self.SIT = CATransformer_2(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24])
            if load:
                state_dict = torch.load("./Models/swin_small_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(768, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )
        elif size == "base":
            self.SIT = CATransformer_2(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            if load:
                state_dict = torch.load("./Models/swin_base_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )

    def forward(self, x):
        x = self.Classify(self.SIT(x))
        return x


class CAT(nn.Module):
    def __init__(self, size, load):
        super().__init__()
        # the number of Params:28.5M
        if size == "tiny":
            self.SIT = CATransformer(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
            if load:
                state_dict = torch.load("./Models/swin_tiny_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(768, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )
        elif size == "small":
            self.SIT = CATransformer(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24])
            if load:
                state_dict = torch.load("./Models/swin_small_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(768, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )
        elif size == "base":
            self.SIT = CATransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            if load:
                state_dict = torch.load("./Models/swin_base_patch4_window7_224.pth")['model']
                self.SIT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(64, 3),
            )

    def forward(self, x):
        x = self.Classify(self.SIT(x))
        return x


class TNT(nn.Module):
    def __init__(self, size, load):
        super().__init__()
        # the number of Params:28.5M
        if size == "small":
            self.TNT = TINT()
            if load:
                state_dict = torch.load("./Models/pth/tnt_s_81.5.pth")
                self.TNT.load_state_dict(state_dict, strict=True)
            self.Classify = nn.Linear(1000, 3)
        elif size == "base":
            self.SIT = TINT()
            if load:
                state_dict = torch.load("./Models/swin_small_patch4_window7_224.pth")['model']
                self.TNT.load_state_dict(state_dict, strict=False)
            self.Classify = nn.Sequential(
                nn.Linear(384, 128),
                nn.Mish(),
                nn.Dropout(0.1),
                nn.Linear(128, 3)
            )

    def forward(self, x):
        x = self.Classify(self.TNT(x))
        return x


class FPN(nn.Module):
    def __init__(self, load):
        super().__init__()
        self.FPSIT = FPSwinT(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
        if load:
            state_dict = torch.load("./pth/swin_tiny_patch4_window7_224.pth")['model']
            self.FPSIT.load_state_dict(state_dict, strict=False)
        self.R_FPSIT = Reverse_FPSwinT()

    def forward(self, x):
        x = self.FPSIT(x)
        x,feature = self.R_FPSIT(x)
        return x,feature


class poolFormer(nn.Module):
    def __init__(self, size, load):
        super().__init__()
        layers = [2, 2, 6, 2]
        embed_dims = [64, 128, 320, 512]
        mlp_ratios = [4, 4, 4, 4]
        downsamples = [True, True, True, True]
        self.SIT = PoolFormer(layers, embed_dims=embed_dims, mlp_ratios=mlp_ratios, downsamples=downsamples)
        if load:
            state_dict = torch.load("./Models/pth/poolformer_s12.pth")
            self.SIT.load_state_dict(state_dict, strict=False)
        self.Classify = nn.Linear(1000, 3)

    def forward(self, x):
        x = self.Classify(self.SIT(x))
        return x


class CovidNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = CovidNet()

    def forward(self, x):
        return self.net(x)

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ResNet18()

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 224)
    model = Resnet()
    # state_dict = torch.load("./1_18_0.9708_FPSIT_tiny.pth")
    # net.load_state_dict(state_dict, strict=False)

    score = model(img)
    print(score.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberofparameter: % .2fM" % (total / 1e6))
