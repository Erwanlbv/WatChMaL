"""
Width-scalable ResNet, for CI-sized runs and local smoke tests.

`watchmal.model.resnet.ResNet` hardcodes the torchvision stage widths (64/128/256/512)
and `nn.Linear(512 * expansion, ...)`, so the smallest model it can build is resnet18 at
~11.2M parameters - too heavy for a GitHub-hosted runner, where the point of the run is
to prove the pipeline executes, not to fit anything.

`LightResNet` is the same architecture with the stage widths exposed: stages are
`base_width`, 2x, 4x and 8x, and the classifier head follows. It reuses `BasicBlock`,
`Bottleneck` and `conv1x1` from `resnet.py` unchanged, so a run through this model
exercises the same residual-block, padding-mode and normalisation code as the real one -
it is a smaller instance of the same network, not a different one. `resnet.py` itself is
untouched.

Parameter count scales close to quadratically in `base_width`. For the resnet18 layer
pattern [2, 2, 2, 2] with 2 input channels and 3 outputs:

    base_width   64 -> 11.17 M   (identical to resnet.resnet18)
                 16 ->  0.70 M
                 12 ->  0.40 M
                  8 ->  0.18 M
                  6 ->  0.10 M

Use `light_resnet` and pass `layers` explicitly, or the `light_resnet18` /
`light_resnet34` helpers which fix the layer pattern and leave the width free.
"""

import torch.nn as nn
import torch.nn.functional as F

from watchmal.model.resnet import BasicBlock, Bottleneck, ShiftInvariantConv2d, conv1x1


class LightResNet(nn.Module):
    """ResNet whose stage widths derive from `base_width` instead of being fixed at 64.

    Parameters
    ----------
    block
        Residual block class, `BasicBlock` or `Bottleneck` from `watchmal.model.resnet`.
    layers : sequence of int
        Number of blocks in each of the four stages, e.g. [2, 2, 2, 2] for resnet18.
    num_input_channels : int
        Channels of the input image (e.g. 2 for a time+charge event display).
    num_output_channels : int
        Size of the network output (number of classes, or number of regressed values).
    base_width : int
        Channel count of the first stage; stages are `base_width`, 2x, 4x, 8x. The
        default 64 reproduces the standard ResNet widths exactly.
    zero_init_residual : bool
        Zero-initialise the last norm layer of each residual branch, so each block starts
        as an identity (https://arxiv.org/abs/1706.02677).
    first_kernel_size, first_stride : int
        Kernel size and stride of the stem convolution.
    shift_inv_channels : list of int, optional
        Input channels to make shift-invariant, via `ShiftInvariantConv2d`.
    conv_pad_mode : str
        Padding mode for the convolutions, e.g. 'circular' for a wrapped event display.
    group_norm : bool
        Use GroupNorm instead of BatchNorm. `n_groups` must then divide every stage
        width, so with a small `base_width` it must be lowered to match.
    n_groups : int
        Number of groups when `group_norm` is set.
    """

    def __init__(self, block, layers, num_input_channels, num_output_channels, base_width=64,
                 zero_init_residual=False, first_kernel_size=1, first_stride=1, shift_inv_channels=None,
                 conv_pad_mode='zeros', group_norm=False, n_groups=32):
        if group_norm:
            if base_width % n_groups:
                raise ValueError(f"group_norm needs n_groups ({n_groups}) to divide base_width ({base_width}); "
                                 f"lower n_groups for a narrow model.")

            class GroupNorm(nn.GroupNorm):
                def __init__(self, num_channels):
                    super().__init__(n_groups, num_channels)
            self.norm = GroupNorm
        else:
            self.norm = nn.BatchNorm2d

        super().__init__()

        self.base_width = base_width
        self.inplanes = base_width
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]

        pad_l = first_kernel_size // 2
        pad_r = first_kernel_size - 1 - pad_l
        self.pad1 = lambda x: F.pad(x, (pad_l, pad_r, pad_l, pad_r),
                                    mode="constant" if conv_pad_mode == "zeros" else conv_pad_mode)
        if shift_inv_channels is None:
            self.conv1 = nn.Conv2d(num_input_channels, base_width, kernel_size=first_kernel_size,
                                   stride=first_stride, padding=0, bias=False)
        else:
            if conv_pad_mode == "zeros":
                pad1 = self.pad1

                def replace_inv_channel_pad(x):
                    values = x[:, shift_inv_channels, 0:1, 0:1]
                    x = pad1(x)
                    x[:, shift_inv_channels, :pad_l, :] = values
                    x[:, shift_inv_channels, -pad_r:, :] = values
                    x[:, shift_inv_channels, :, :pad_l] = values
                    x[:, shift_inv_channels, :, -pad_r:] = values
                    return x
                self.pad1 = replace_inv_channel_pad
            self.conv1 = ShiftInvariantConv2d(shift_inv_channels, num_input_channels, base_width,
                                              kernel_size=first_kernel_size, stride=first_stride,
                                              padding=0, bias=False)
        self.bn1 = self.norm(base_width)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool: `resnet.py` builds an `nn.MaxPool2d` in __init__ but its forward()
        # never applies it, so the stem does not downsample. Omitted here rather than
        # carried as dead state, so this network stays output-equivalent to that one.

        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=1, conv_pad_mode=conv_pad_mode)
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2, conv_pad_mode=conv_pad_mode)
        self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2, conv_pad_mode=conv_pad_mode)
        self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2, conv_pad_mode=conv_pad_mode)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3] * block.expansion, num_output_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, conv_pad_mode='zeros'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.norm(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, conv_pad_mode, self.norm)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_pad_mode=conv_pad_mode, norm=self.norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


def light_resnet(layers=(2, 2, 2, 2), bottleneck=False, **kwargs):
    """LightResNet with an explicit layer pattern. `bottleneck` selects `Bottleneck` over
    `BasicBlock`; everything else is forwarded to `LightResNet`."""
    return LightResNet(Bottleneck if bottleneck else BasicBlock, list(layers), **kwargs)


def light_resnet18(**kwargs):
    """resnet18 layer pattern [2, 2, 2, 2] at a configurable `base_width`."""
    return LightResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def light_resnet34(**kwargs):
    """resnet34 layer pattern [3, 4, 6, 3] at a configurable `base_width`."""
    return LightResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
