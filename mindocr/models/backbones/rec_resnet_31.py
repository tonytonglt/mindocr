from mindspore import nn

from ._registry import register_backbone_class

__all__ = ['ResNet31']

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        pad_mode="pad",
        padding=1)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        # 与pytorch中的momentum是1-momentum的关系
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        if downsample:
            self.downsample = nn.SequentialCell(
                nn.Conv2d(
                    in_channels,
                    channels * self.expansion,
                    1,
                    stride),
                nn.BatchNorm2d(channels * self.expansion))
        else:
            self.downsample = nn.SequentialCell()
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


@register_backbone_class
class ResNet31(nn.Cell):
    '''
    Args:
        in_channels (int): Number of channels of input image tensor.
        layers (list[int]): List of BasicBlock number for each stage.
        channels (list[int]): List of out_channels of Conv2d layer.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    '''

    def __init__(self,
                 in_channels=3,
                 layers=[1, 2, 5, 3],
                 channels=[64, 128, 256, 256, 512, 512, 512],
                 last_stage_pool=False):
        super(ResNet31, self).__init__()
        assert isinstance(in_channels, int)
        assert isinstance(last_stage_pool, bool)

        self.last_stage_pool = last_stage_pool

        # conv 1 (Conv Conv)
        self.conv1_1 = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn1_1 = nn.BatchNorm2d(channels[0])
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(
            channels[0], channels[1], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn1_2 = nn.BatchNorm2d(channels[1])
        self.relu1_2 = nn.ReLU()

        # conv 2 (Max-pooling, Residual block, Conv)
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.block2 = self._make_layer(channels[1], channels[2], layers[0])
        self.conv2 = nn.Conv2d(
            channels[2], channels[2], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu2 = nn.ReLU()

        # conv 3 (Max-pooling, Residual block, Conv)
        self.pool3 = nn.MaxPool2d(
            kernel_size=2, stride=2)
        self.block3 = self._make_layer(channels[2], channels[3], layers[1])
        self.conv3 = nn.Conv2d(
            channels[3], channels[3], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.relu3 = nn.ReLU()

        # conv 4 (Max-pooling, Residual block, Conv)
        self.pool4 = nn.MaxPool2d(
            kernel_size=(2, 1), stride=(2, 1), pad_mode="same")
        self.block4 = self._make_layer(channels[3], channels[4], layers[2])
        self.conv4 = nn.Conv2d(
            channels[4], channels[4], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn4 = nn.BatchNorm2d(channels[4])
        self.relu4 = nn.ReLU()

        # conv 5 ((Max-pooling), Residual block, Conv)
        self.pool5 = None
        if self.last_stage_pool:
            self.pool5 = nn.MaxPool2d(
                kernel_size=2, stride=2)
        self.block5 = self._make_layer(channels[4], channels[5], layers[3])
        self.conv5 = nn.Conv2d(
            channels[5], channels[5], kernel_size=3, stride=1, pad_mode="pad", padding=1)
        self.bn5 = nn.BatchNorm2d(channels[5])
        self.relu5 = nn.ReLU()

        self.out_channels = [channels[-1]]

    def _make_layer(self, input_channels, output_channels, blocks):
        layers = []
        for _ in range(blocks):
            downsample = None
            if input_channels != output_channels:
                downsample = nn.SequentialCell(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=1,
                        stride=1),
                    nn.BatchNorm2d(output_channels))

            layers.append(
                BasicBlock(
                    input_channels, output_channels, downsample=downsample))
            input_channels = output_channels

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        outs = []

        x = self.pool2(x)
        x = self.block2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        outs.append(x)

        x = self.pool3(x)
        x = self.block3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        outs.append(x)

        x = self.pool4(x)
        x = self.block4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        outs.append(x)

        if self.pool5 is not None:
            x = self.pool5(x)
        x = self.block5(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        # for i in range(4):
        #     layer_index = i + 2

        #     pool_layer = getattr(self, f'pool{layer_index}')
        #     block_layer = getattr(self, f'block{layer_index}')
        #     conv_layer = getattr(self, f'conv{layer_index}')
        #     bn_layer = getattr(self, f'bn{layer_index}')
        #     relu_layer = getattr(self, f'relu{layer_index}')

        #     x = pool_layer(x)
        #     x = block_layer(x)
        #     x = conv_layer(x)
        #     x = bn_layer(x)
        #     x = relu_layer(x)

        #     outs.append(x)

        outs.append(x)
        return outs