import torch
from torch import nn
from torch.onnx.symbolic_opset9 import tensor


class U_NET_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32,num_channels=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.down_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=512, out_channels=512, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.up_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=256, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.up_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.last_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1)

    def copy_and_crop(self, need_crop, up_data): # 进行中心裁剪，need_crop是需要裁剪的
        need_crop_size = need_crop.size()[2]  # size 有4维，为batch_size, channel, height, width ,取高度
        up_data_size = up_data.size()[2]
        delta = need_crop_size - up_data_size
        delta = delta // 2
        return tensor[:, :, delta:need_crop_size - delta, delta:need_crop_size - delta]

    def forward(self, input):
        x1 = self.down_1(input)
        x_pool1 = self.maxpool2(x1)
        x2 = self.down_2(x_pool1)
        x_pool2 = self.maxpool2(x2)
        x3 = self.down_3(x_pool2)
        x_pool3 = self.maxpool2(x3)
        x4 = self.down_4(x_pool3)

        up_1 = self.up_conv_1(x4)
        crop_1 = self.copy_and_crop(x3, up_1)
        x5 = torch.cat([crop_1, up_1], dim=1)
        x6 = self.up_1(x5)

        up_2 = self.up_conv_2(x6)
        crop_2 = self.copy_and_crop(x2, up_2)
        x7 = torch.cat([crop_2, up_2], dim=1)
        x8 = self.up_2(x7)

        up_3 = self.up_conv_2(x8)
        crop_3 = self.copy_and_crop(x1, up_3)
        x9 = torch.cat([crop_3, up_3], dim=1)
        x10 = self.up_2(x9)

        output = self.last_conv(x10)
        return output





