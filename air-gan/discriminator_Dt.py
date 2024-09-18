from torch import nn


class Dt_Dirsc(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=0.2),
            nn.Conv2d(kernel_size=3, stride=1),
            nn.GroupNorm(num_groups=32),
            nn.LeakyReLU(negative_slope=0.2, inplace=0.2),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )


    def forward(self, input):
        x1 = self.down(input)
        x2 = self.down(input)
        x3 = self.down(input)