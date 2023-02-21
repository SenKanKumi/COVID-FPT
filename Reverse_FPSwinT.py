import torch
import torch.nn as nn
from torch.nn import functional as F


class Eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size):
        super(Eca_layer, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, l, c]
        b, l, c = x.size()
        # feature descriptor on the global spatial information
        y = self.ap(x.transpose(1, 2).view(b, c, int(l ** 0.5), int(l ** 0.5)))
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(1, 2))
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Reverse_FPSwinT(nn.Module):
    def __init__(self):
        super().__init__()

        self.latlayer1 = nn.Conv2d(384, 768, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(192, 768, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(96, 768, kernel_size=1, stride=1, padding=0)

        # self.Eca1 = Eca_layer(7)
        # self.Eca2 = Eca_layer(5)
        # self.Eca3 = Eca_layer(3)
        # self.Eca = Eca_layer(17)

        self.smooth1 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1)

        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.norm4 = nn.LayerNorm(768)
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.avgpool3 = nn.AdaptiveAvgPool1d(1)
        self.avgpool4 = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Linear(768, 3)

        self.v1 = torch.tensor(0.4)
        self.v2 = torch.tensor(0.3)
        self.v3 = torch.tensor(0.2)
        # self.v4 = torch.tensor(0.1)

        self.v1 = nn.Parameter(self.v1, requires_grad=True)
        self.v2 = nn.Parameter(self.v2, requires_grad=True)
        self.v3 = nn.Parameter(self.v3, requires_grad=True)
        # self.v4 = nn.Parameter(self.v4, requires_grad=True)

    def forward(self, x):
        # c2 = self.Eca3(x[0]).transpose(1, 2).view(-1, 96, 56, 56)
        # c3 = self.Eca2(x[1]).transpose(1, 2).view(-1, 192, 28, 28)
        # c4 = self.Eca1(x[2]).transpose(1, 2).view(-1, 384, 14, 14)
        c2 = x[0].transpose(1, 2).view(-1, 96, 56, 56)
        c3 = x[1].transpose(1, 2).view(-1, 192, 28, 28)
        c4 = x[2].transpose(1, 2).view(-1, 384, 14, 14)
        c5 = x[3].transpose(1, 2).view(-1, 768, 7, 7)

        p4 = self._upsample_add(c5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        feature = [c5, p4, p3, p2]

        p5 = self.avgpool1(self.norm1(x[3]).transpose(1, 2))
        p4 = self.avgpool2(self.norm2(self.smooth1(p4).flatten(2).transpose(1, 2)).transpose(1, 2))
        p3 = self.avgpool3(self.norm3(self.smooth2(p3).flatten(2).transpose(1, 2)).transpose(1, 2))
        p2 = self.avgpool4(self.norm4(self.smooth3(p2).flatten(2).transpose(1, 2)).transpose(1, 2))

        self.v4 = torch.tensor(1.0) - self.v1 - self.v2 - self.v3
        p = self.v1 * p5 + self.v2 * p4 + self.v3 * p3 + self.v4 * p2
        # p = torch.cat([p5,p4,p3,p2],1)
        # p = self.Eca(p.transpose(1,2)).transpose(1,2)
        p = self.head(torch.flatten(p, 1))
        return p,feature

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y


if __name__ == "__main__":
    img1 = torch.randn(1, 3, 224, 224)
    net = Reverse_FPSwinT()
    logits= net(img1)