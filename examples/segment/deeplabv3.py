from typing import List

import mindspore.nn as nn
from mindspore.ops import operations as P
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
        use_batch_statistics: bool = False,
    ) -> None:
        super(ASPP, self).__init__()

        self.is_training = is_training

        self.aspp_convs = []
        for rate in atrous_rates:
            self.aspp_convs.append(
                ASPPConv(in_channels, out_channels, rate, use_batch_statistics=use_batch_statistics)
            )
        self.aspp_convs.append(
            ASPPPooling(in_channels, out_channels, use_batch_statistics=use_batch_statistics)
        )

        # self.aspp1 = ASPPConv(in_channels, out_channels, atrous_rates[0], use_batch_statistics=use_batch_statistics)
        # self.aspp2 = ASPPConv(in_channels, out_channels, atrous_rates[1], use_batch_statistics=use_batch_statistics)
        # self.aspp3 = ASPPConv(in_channels, out_channels, atrous_rates[2], use_batch_statistics=use_batch_statistics)
        # self.aspp4 = ASPPConv(in_channels, out_channels, atrous_rates[3], use_batch_statistics=use_batch_statistics)
        # self.aspp_pooling = ASPPPooling(in_channels, out_channels, use_batch_statistics=use_batch_statistics)

        # modelzoo 所有conv set weight_init='xavier_uniform'
        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1)
                            #    weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.7)
        self.conv2 = nn.Conv2d(out_channels, num_classes, kernel_size=1, has_bias=True)

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
        use_batch_statistics: bool = False
    ) -> None:
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU()
        ])

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
        use_batch_statistics: bool = False
    ) -> None:
        super(ASPPConv, self).__init__()
        
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', 
                                     padding=atrous_rate, dilation=atrous_rate)
        
        self.aspp_conv = nn.SequentialCell(
            conv,
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU()
        )
        
    def construct(self, x):
        out = self.aspp_conv(x)
        return out



class DeepLabV3(nn.Cell):
    def __init__(
        self, 
        backbone,
        args,
        is_training: bool = True,
        # phase='train', 
        # num_classes=21, 
        # freeze_bn: bool = False
    ):
        super(DeepLabV3, self).__init__()
        # use_batch_statistics = not args.freeze_bn
        self.is_training = is_training
        self.backbone = backbone
        self.aspp = ASPP(atrous_rates=[1, 6, 12, 18], is_training=is_training,
                         in_channels=2048, num_classes=args.num_classes)

    def construct(self, x):
        size = ops.shape(x)
        features = self.backbone(x)
        out = self.aspp(features)
        out = ops.interpolate(out, size=(size[2],size[3]), mode='bilinear', align_corners=True)
        # torchvison设置了align_corners flase
        return out


class DeepLabV3WithLoss(nn.Cell):
    """ "
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