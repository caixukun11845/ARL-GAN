import torch
from torch import nn, tensor
import tensorboard


class Gr_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.up = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_conv = nn.ConvTranspose2d(kernel_size=3, stride=1)
        self.last_conv = nn.Conv2d(kernel_size=1, stride=1)

    def copy_and_crop(self, need_crop_data, up_data):
        need_crop_data_size = need_crop_data.size()
        up_data_size = need_crop_data_size()
        delta = need_crop_data_size - up_data_size
        delta = delta // 2
        return tensor[:, :, delta:need_crop_data_size - delta, delta:need_crop_data_size]

    def forward(self, input):
        down_x1 = self.down(input)
        pool1 = self.maxpool(down_x1)

        down_x2 = self.down(pool1)
        pool2 = self.maxpool(down_x2)

        down_x3 = self.down(pool2)
        pool3 = self.maxpool(down_x3)

        down_x4 = self.down(pool3)

        up_conv_x1 = self.up_conv(down_x4)

        c_and_c_1 = self.copy_and_crop(down_x3, up_conv_x1)
        cat_x1 = torch.cat(c_and_c_1, up_conv_x1)
        up_1 = self.up(cat_x1)

        up_conv_x2 = self.up_conv(up_1)

        c_and_c_2 = self.copy_and_crop(down_x2, up_conv_x2)
        cat_x2 = torch.cat(c_and_c_2, up_conv_x2)
        up_2 = self.up(cat_x2)

        up_conv_x3 = self.up_conv(up_2)
        c_and_c_3 = self.copy_and_crop(down_x1, up_conv_x3)
        cat_x3 = torch.cat(c_and_c_3, up_conv_x3)

        output = self.last_conv(cat_x3)

        return output