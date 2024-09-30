from torch import nn

class Dr_Dirsc(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


    def forward(self, input):
        x1 = self.down(input)
        x2 = self.down(input)
        x3 = self.down(input)