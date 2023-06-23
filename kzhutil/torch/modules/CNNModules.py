import torch
from torch import nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x):
        x = self.c1(x) + self.c2(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.t(x)
        x = self.c(x)
        return x


class Conv2dMaxPool(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, pooling=2):
        super(Conv2dMaxPool, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class ResConv2dMaxPool(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(ResConv2dMaxPool, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = x + out
        out = self.mp(self.relu(out))
        return out


class ResidualConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(ResidualConv2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1),
            dilation=dilation,
            **kwargs)

    def forward(self, x):
        out = super(CausalConv1d, self).forward(x)
        return out[:, :, :-self.padding[0]]


class WaveNetLayer(nn.Module):
    def __init__(self, residual_channels, skip_channels, cond_channels, kernel_size=2, dilation=1, causal_input=False):
        super(WaveNetLayer, self).__init__()
        if causal_input:
            self.conv = CausalConv1d(residual_channels, 2 * residual_channels, kernel_size, dilation=dilation, bias=True)
        else:
            self.conv = nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size, dilation=dilation, bias=True)
        self.condition = nn.Conv1d(cond_channels, 2 * residual_channels, kernel_size=1, bias=True)
        self.residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1, bias=True)
        self.skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1, bias=True)

    def _condition(self, x, c):
        c = self.condition(c)
        x = x + c
        return x

    def forward(self, x, c=None):
        x = self.conv(x)
        if c is not None:
            x = self._condition(x, c)

        assert x.size(1) % 2 == 0
        gate, output = x.chunk(2, 1)
        gate = torch.sigmoid(gate)
        output = torch.tanh(output)
        x = gate * output

        residual = self.residual(x)
        skip = self.skip(x)

        return residual, skip


class WaveNet(nn.Module):
    def __init__(self, blocks, layers, in_channel, out_channel, residual_channels, skip_channels=None,
                 cond_channels=None, kernel_size=2, create_layers=True, causal_input=False):
        super().__init__()

        self.blocks = blocks
        self.layer_num = layers
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.classes = out_channel
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels if skip_channels is not None else residual_channels
        self.cond_channels = cond_channels if cond_channels is not None else residual_channels
        self.causal_input = causal_input

        if create_layers:
            layers = []
            for _ in range(self.blocks):
                for i in range(self.layer_num):
                    dilation = 2 ** i
                    layers.append(WaveNetLayer(self.residual_channels, self.skip_channels, self.cond_channels,
                                               self.kernel_size, dilation, causal_input))
            self.layers = nn.ModuleList(layers)

        self.first_conv = CausalConv1d(self.in_channel, self.residual_channels, kernel_size=self.kernel_size)
        self.skip_conv = nn.Conv1d(self.residual_channels, self.skip_channels, kernel_size=1)
        self.condition = nn.Conv1d(self.cond_channels, self.skip_channels, kernel_size=1)
        self.fc = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1)
        self.logits = nn.Conv1d(self.skip_channels, self.classes, kernel_size=1)

    def _condition(self, x, c):
        c = self.condition(c)
        x = x + c
        return x

    @staticmethod
    def _upsample_cond(x, c):
        bsz, channels, length = x.size()
        cond_bsz, cond_channels, cond_length = c.size()
        assert bsz == cond_bsz

        if c.size(2) != 1:
            c = c.unsqueeze(3).repeat(1, 1, 1, length // cond_length + 1)
            c = c.view(bsz, cond_channels, -1)[:, :, :length]

        return c

    @staticmethod
    def shift_right(x):
        x = F.pad(x, (1, 0))
        return x[:, :, :-1]

    def forward(self, x, c=None):
        if self.causal_input:
            x = self.shift_right(x)

        if c is not None:
            c = self._upsample_cond(x, c)

        residual = self.first_conv(x)
        skip = self.skip_conv(residual)

        for layer in self.layers:
            r, s = layer(residual, c)
            residual = residual + r
            skip = skip + s

        skip = F.relu(skip)
        skip = self.fc(skip)
        if c is not None:
            skip = self._condition(skip, c)
        skip = F.relu(skip)
        skip = self.logits(skip)

        return skip
