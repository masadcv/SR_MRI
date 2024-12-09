"""
Following code is adapted from: https://github.com/yjn870/IDN-pytorch

Changes by Muhammad Asad (09/12/2024):
- Added support for proper padding in output deconvolution
- Added support for multiple input channels / output channels (previously expected only RGB==3 channels)
- Changed input arguments to kwargs for IDN (instead of args struct)
"""

import torch
from torch import nn
import torch.nn.functional as F


class FBlock(nn.Module):
    def __init__(self, image_features, num_features):
        super(FBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(image_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
        )

    def forward(self, x):
        return self.module(x)


class DBlock(nn.Module):
    def __init__(self, num_features, d, s):
        super(DBlock, self).__init__()
        self.num_features = num_features
        self.s = s
        self.enhancement_top = nn.Sequential(
            nn.Conv2d(num_features, num_features - d, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(
                num_features - d,
                num_features - 2 * d,
                kernel_size=3,
                padding=1,
                groups=4,
            ),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features - 2 * d, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
        )
        self.enhancement_bottom = nn.Sequential(
            nn.Conv2d(num_features - d, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(
                num_features, num_features - d, kernel_size=3, padding=1, groups=4
            ),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_features - d, num_features + d, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
        )
        self.compression = nn.Conv2d(num_features + d, num_features, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.enhancement_top(x)
        slice_1 = x[:, : int((self.num_features - self.num_features / self.s)), :, :]
        slice_2 = x[:, int((self.num_features - self.num_features / self.s)) :, :, :]
        x = self.enhancement_bottom(slice_1)
        x = x + torch.cat((residual, slice_2), 1)
        x = self.compression(x)
        return x


class IDN(nn.Module):
    def __init__(
        self,
        scale,
        image_features=16,
        fblock_num_features=16,
        num_features=64,
        d=16,
        s=1,
    ):
        super(IDN, self).__init__()
        self.scale = scale

        self.upsampling_layer = torch.nn.Upsample(
            scale_factor=scale, mode="bicubic", align_corners=False
        )

        self.fblock1 = FBlock(image_features, fblock_num_features)
        self.fblock2 = FBlock(fblock_num_features, num_features)
        self.dblocks = nn.Sequential(*[DBlock(num_features, d, s) for _ in range(4)])
        # self.deconv = nn.ConvTranspose2d(num_features, out_features, kernel_size=17, stride=self.scale, padding=8, output_padding=self.scale-1)
        self.deconv = nn.Conv2d(
            num_features,
            out_channels=image_features,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.upsampling_layer = self.upsampling_layer.to(*args, **kwargs)
        return self

    def forward(self, x):
        upresolved = self.upsampling_layer(x)
        x = self.fblock1(x)
        x = self.upsampling_layer(x)
        x = self.fblock2(x)
        x = self.dblocks(x)
        x = self.deconv(x)
        return upresolved + x


if __name__ == "__main__":
    model = IDN(
        scale=2, image_features=1, fblock_num_features=16, num_features=64, d=16, s=4
    )
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    print(y.shape)

    # x4
    model = IDN(
        scale=4, image_features=1, fblock_num_features=16, num_features=64, d=16, s=4
    )
    x = torch.randn(1, 1, 32, 32)
    y = model(x)
    print(y.shape)
