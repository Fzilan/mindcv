from typing import List

import mindspore.nn as nn
import mindspore.ops as ops


__all__ = [
    "DeepLabV3",
    "DeepLabV3WithLoss",
]


class ASPP(nn.Cell):
    def __init__(
        self,
        atrous_rates: List[int],
        is_training: bool = True,
        # phase='train',
        in_channels: int = 2048,
        out_channels: int = 256,
        num_classes: int = 21,
        # use_batch_statistics: bool = True,
        weight_init:str = 'xavier_uniform',
    ) -> 'ASPP':
        super(ASPP, self).__init__()

        self.is_training = is_training

        self.aspp_convs = nn.CellList()
        for rate in atrous_rates:
            self.aspp_convs.append(
                ASPPConv(
                    in_channels,
                    out_channels,
                    rate,
                    # use_batch_statistics=use_batch_statistics,
                )
            )
        self.aspp_convs.append(
            ASPPPooling(
                in_channels, out_channels #, use_batch_statistics=use_batch_statistics
            )
        )
        
        self.conv1 = nn.Conv2d(
            out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1,
            weight_init=weight_init
        )
        self.bn1 = nn.BatchNorm2d(
            out_channels #, use_batch_statistics=use_batch_statistics
        )
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.7)
        self.conv2 = nn.Conv2d(out_channels, num_classes, kernel_size=1, 
                               weight_init=weight_init, has_bias=True)

    def construct(self, x):
        _out = []
        for conv in self.aspp_convs:
            _out.append(conv(x))
        x = ops.cat(_out, axis=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.is_training:
            x = self.drop(x)
        x = self.conv2(x)
        return x


class ASPPPooling(nn.Cell):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        # use_batch_statistics: bool = True,
        weight_init:str = 'xavier_uniform',
    ) -> None:
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init=weight_init),
                nn.BatchNorm2d(out_channels), #, use_batch_statistics=use_batch_statistics),
                nn.ReLU(),
            ]
        )

    def construct(self, x):
        size = ops.shape(x)
        out = nn.AvgPool2d(size[2])(x)
        out = self.conv(out)
        out = ops.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        return out


class ASPPConv(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rate: int = 1,
        # use_batch_statistics: bool = True,
        weight_init: str = 'xavier_uniform'
    ) -> None:
        super(ASPPConv, self).__init__()

        self._aspp_conv = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init=weight_init)
            if atrous_rate == 1
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                pad_mode="pad",
                padding=atrous_rate,
                dilation=atrous_rate,
                weight_init=weight_init,
            ),
            nn.BatchNorm2d(out_channels), #, use_batch_statistics=use_batch_statistics),
            nn.ReLU(),
        ])
        # for i in self._aspp_conv[0].get_parameters():
        #     print(i.name)

    def construct(self, x):
        out = self._aspp_conv(x)
        return out


class DeepLabV3(nn.Cell):
    def __init__(
        self,
        backbone,
        args,
        is_training: bool = True,
        # freeze_bn: bool = False,
        # phase='train',
        # num_classes=21,
        # 
    ):
        super(DeepLabV3, self).__init__()
        # use_batch_statistics = not freeze_bn
        self.is_training = is_training
        self.backbone = backbone
        self.aspp = ASPP(
            atrous_rates=[1, 6, 12, 18],
            is_training=is_training,
            in_channels=2048,
            num_classes=args.num_classes,
            # use_batch_statistics = use_batch_statistics,
        )

    def construct(self, x):
        size = ops.shape(x)
        features = self.backbone(x)[-1] # TODO: 返回来一个List[Tensor[...]], 取最后一个？
        out = self.aspp(features)
        out = ops.interpolate(
            out, size=(size[2], size[3]), mode="bilinear", align_corners=True
        )
        return out


class DeepLabV3WithLoss(nn.Cell):
    """ 
    Provide DeeplabV3 training loss through network.

    Args:
        network (Cell): The training network.
        config (dict): DeeplabV3 config.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, network, criterion):
        super(DeepLabV3WithLoss, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


class DeepLabV3InferNetwork(nn.Cell):
    """ 
    Provide DeeplabV3 infer network.

    """    
    def __init__(self, network, input_format="NCHW"):
        super(DeepLabV3InferNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)
        self.format = input_format

    def construct(self, input_data):
        if self.format == "NHWC":
            input_data = ops.transpose(input_data, (0, 3, 1, 2))
        output = self.network(input_data)
        output = self.softmax(output)
        return output
