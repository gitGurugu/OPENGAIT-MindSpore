import mindspore.nn as nn
import mindspore.ops as ops


class ConvBlock(nn.Cell):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        ])

    def construct(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Cell):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.SequentialCell([
            nn.ResizeBilinear(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, pad_mode='pad', padding=1, has_bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        ])
        # For upsampling, we'll use ResizeBilinear with scale factor
        self.upsample = nn.ResizeBilinear()

    def construct(self, x):
        # Upsample first
        x = self.upsample(x, size=(x.shape[2]*2, x.shape[3]*2))
        # Then apply conv
        x = self.up[1](x)
        x = self.up[2](x)
        x = self.up[3](x)
        return x


class U_Net(nn.Cell):
    def __init__(self, in_channels=3, freeze_half=True):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=in_channels, ch_out=16)
        self.Conv2 = ConvBlock(ch_in=16, ch_out=32)
        self.Conv3 = ConvBlock(ch_in=32, ch_out=64)
        self.Conv4 = ConvBlock(ch_in=64, ch_out=128)
        self.freeze = freeze_half
        
        # In MindSpore, we set requires_grad=False for parameters
        if freeze_half:
            for param in self.Conv1.get_parameters():
                param.requires_grad = False
            for param in self.Conv2.get_parameters():
                param.requires_grad = False
            for param in self.Conv3.get_parameters():
                param.requires_grad = False
            for param in self.Conv4.get_parameters():
                param.requires_grad = False

        self.Up4 = UpConv(ch_in=128, ch_out=64)
        self.Up_conv4 = ConvBlock(ch_in=128, ch_out=64)

        self.Up3 = UpConv(ch_in=64, ch_out=32)
        self.Up_conv3 = ConvBlock(ch_in=64, ch_out=32)

        self.Up2 = UpConv(ch_in=32, ch_out=16)
        self.Up_conv2 = ConvBlock(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(
            16, 1, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)
        
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        d4 = self.Up4(x4)
        d4 = self.concat((x3, d4))
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = self.concat((x2, d3))
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.concat((x1, d2))
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1

