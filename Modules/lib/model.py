import torch
import torch.nn as nn
from torch.nn import functional as F
from models.resnet import resnet34
from Modules.lib.ResNet_models_Custom import BasicConv2d
import math


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



###############################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)




class CAAF(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CAAF, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)

        self.CBR_branch = nn.Sequential(nn.Conv2d(out_dim // 4, out_dim // 4, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_dim // 4), act_fn, )
        self.CBR = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(out_dim), act_fn, )
        self.adp = nn.AdaptiveMaxPool2d((None, None))

        self.layer_ful12 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(out_dim), act_fn, )

        self.rgb_branch1 = BasicConv2d(out_dim, out_dim // 4, 3, padding=1, dilation=1)
        self.rgb_branch2 = BasicConv2d(out_dim, out_dim // 4, 3, padding=3, dilation=3)
        self.rgb_branch3 = BasicConv2d(out_dim, out_dim // 4, 3, padding=5, dilation=5)
        self.rgb_branch4 = BasicConv2d(out_dim, out_dim // 4, 3, padding=7, dilation=7)

        self.rgb_branch1_sa = SpatialAttention()
        self.rgb_branch2_sa = SpatialAttention()
        self.rgb_branch3_sa = SpatialAttention()
        self.rgb_branch4_sa = SpatialAttention()

        self.rgb_branch1_ca = ChannelAttention(out_dim // 4)
        self.rgb_branch2_ca = ChannelAttention(out_dim // 4)
        self.rgb_branch3_ca = ChannelAttention(out_dim // 4)
        self.rgb_branch4_ca = ChannelAttention(out_dim // 4)

    def forward(self, rgb, depth):
        ################################
        x_rgb = self.reduc_1(rgb)
        x_dep = self.reduc_2(depth)

        x1_rgb = self.rgb_branch1(x_rgb)
        x2_rgb = self.rgb_branch2(x_rgb)
        x3_rgb = self.rgb_branch3(x_rgb)
        x4_rgb = self.rgb_branch4(x_rgb)

        # x1_dep = self.d_branch1(x_dep)
        # x2_dep = self.d_branch2(x_dep)
        # x3_dep = self.d_branch3(x_dep)
        # x4_dep = self.d_branch4(x_dep)

        x1_rgb_ca = x1_rgb.mul(self.rgb_branch1_ca(x1_rgb))
        x1_rgb_sa = self.rgb_branch1_sa(x1_rgb_ca)
        x1_rgb_w = x1_rgb.mul(x1_rgb_sa)
        ful_out1 = self.CBR_branch(x1_rgb_w)

        x2_rgb_ca = x2_rgb.mul(self.rgb_branch2_ca(x2_rgb))
        x2_rgb_sa = self.rgb_branch2_sa(x2_rgb_ca)
        x2_rgb_w = x2_rgb.mul(x2_rgb_sa)
        ful_out2 = self.CBR_branch(x2_rgb_w)

        x3_rgb_ca = x3_rgb.mul(self.rgb_branch3_ca(x3_rgb))
        x3_rgb_sa = self.rgb_branch3_sa(x3_rgb_ca)
        x3_rgb_w = x3_rgb.mul(x3_rgb_sa)
        ful_out3 = self.CBR_branch(x3_rgb_w)

        x4_rgb_ca = x4_rgb.mul(self.rgb_branch4_ca(x4_rgb))
        x4_rgb_sa = self.rgb_branch4_sa(x4_rgb_ca)
        x4_rgb_w = x4_rgb.mul(x4_rgb_sa)
        ful_out4 = self.CBR_branch(x4_rgb_w)

        rgb_cat = torch.cat((ful_out1, ful_out2, ful_out3, ful_out4), 1)
        rgb_ful = self.layer_ful12(rgb_cat)

        x_cat = torch.cat((x1_rgb, x2_rgb, x3_rgb, x4_rgb), 1)
        x_ful = self.CBR(x_cat)
        dep_adp = self.adp(x_dep)
        x_ful.mul(dep_adp)
        out = x_ful.mul(rgb_ful)
        out = out + rgb_ful

        return out


class MFIB(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MFIB, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_cat1 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_dim), )
        self.d_linear = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        self.cbr1 = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), )
        self.cbr2 = nn.Sequential(nn.Conv2d(out_dim * 3, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True), )

    def forward(self, x_ful, x_rgb, x_dep):
        x1 = self.layer_10(self.d_linear(x_dep.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=3))
        x_ful1 = x_ful.mul(torch.sigmoid(x1)) + x_ful
        x2 = self.layer_20(self.d_linear(x_rgb.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=3))
        x_ful2 = x_ful.mul(torch.sigmoid(x2)) + x_ful
        ful = self.cbr1(torch.cat((x_ful1, x_ful2), 1))
        out = self.cbr2(torch.cat((ful, x_rgb, x_dep), 1))

        return out



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel=32):
        super(Decoder, self).__init__()
        # self.reduce_conv=nn.Sequential(
        #     #nn.Conv2d(side_channel, in_channel, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(side_channel, in_channel, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(inplace=True)  ###
        # )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)  ###
        )
        init_weight(self)

    def forward(self, x):
        # x=F.interpolate(x, size=side.size()[2:], mode='bilinear', align_corners=True)
        # side=self.reduce_conv(side)
        # x=torch.cat((x, side), 1)
        x = self.decoder(x)
        return x


class PredLayer(nn.Module):
    def __init__(self, in_channel=32):
        super(PredLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        init_weight(self)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x


class SAINet(nn.Module):
    def __init__(self, channel=32, ind=50):
        super(SAINet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        # Backbone model
        self.layer_rgb = resnet34()
        self.layer_dep = resnet34()
        self.rgb_inplanes = 2048
        self.dep_inplanes = 2048
        self.base_width = 64

        # CAAF #
        ###############################################
        self.caaf_0 = CAAF(64, 64)

        self.caaf_1 = CAAF(64, 64)

        self.caaf_2 = CAAF(128, 64)

        self.caaf_3 = CAAF(256, 128)

        self.caaf_4 = CAAF(512, 256)

        # MFIB #
        ###############################################
        self.mfib_layer4 = MFIB(256, 256)
        self.mfib_layer3 = MFIB(128, 128)
        self.mfib_layer2 = MFIB(64, 64)
        self.mfib_layer1 = MFIB(64, 64)
        self.mfib_layer0 = MFIB(64, 64)

        # Decoder #
        ###############################################
        self.ful_gcm_4 = Decoder(256, 32)

        # self.ful_conv_3 = nn.Sequential(BasicConv2d(1024 + 32 * 3, 256, 3, padding=1), self.relu)
        self.ful_gcm_3 = Decoder(128 + 32, channel)

        # self.ful_conv_2 = nn.Sequential(BasicConv2d(512 + 32 * 3, 128, 3, padding=1), self.relu)
        self.ful_gcm_2 = Decoder(64 + 32, channel)

        # self.ful_conv_1 = nn.Sequential(BasicConv2d(256 + 32 * 3, 128, 3, padding=1), self.relu)
        self.ful_gcm_1 = Decoder(64 + 32, channel)

        # self.ful_conv_0 = nn.Sequential(BasicConv2d(128 + 32 * 3, 64, 3, padding=1), self.relu)
        self.ful_gcm_0 = Decoder(64 + 32, channel)

        self.rgb_gcm_4 = Decoder(512, 256)
        self.rgb_gcm_3 = Decoder(256 + 256, 128)
        self.rgb_gcm_2 = Decoder(128 + 128, 64)
        self.rgb_gcm_1 = Decoder(64 + 64, 64)
        self.rgb_gcm_0 = Decoder(64 + 64, 64)

        self.dep_gcm_4 = Decoder(512, 256)
        self.dep_gcm_3 = Decoder(256 + 256, 128)
        self.dep_gcm_2 = Decoder(128 + 128, 64)
        self.dep_gcm_1 = Decoder(64 + 64, 64)
        self.dep_gcm_0 = Decoder(64 + 64, 64)

        # Pred #
        ###############################################
        self.S0 = PredLayer()
        self.S1 = PredLayer()
        self.S2 = PredLayer()
        self.S3 = PredLayer()
        self.S4 = PredLayer()
        self.rgb = PredLayer(in_channel=64)
        self.dep = PredLayer(in_channel=64)


    def forward(self, imgs, depths):
        img_0, img_1, img_2, img_3, img_4 = self.layer_rgb(imgs)
        dep_0, dep_1, dep_2, dep_3, dep_4 = self.layer_dep(depths)

        # CAAF #
        ###############################################
        ful_0 = self.caaf_0(img_0, dep_0)
        ful_1 = self.caaf_1(img_1, dep_1)
        ful_2 = self.caaf_2(img_2, dep_2)
        ful_3 = self.caaf_3(img_3, dep_3)
        ful_4 = self.caaf_4(img_4, dep_4)

        # RGB/D Decoder #
        ###############################################
        x_rgb_42 = self.rgb_gcm_4(img_4)
        x_rgb_32 = self.rgb_gcm_3(torch.cat((img_3, self.upsample_2(x_rgb_42)), dim=1))
        x_rgb_22 = self.rgb_gcm_2(torch.cat((img_2, self.upsample_2(x_rgb_32)), dim=1))
        x_rgb_12 = self.rgb_gcm_1(torch.cat((img_1, self.upsample_2(x_rgb_22)), dim=1))
        x_rgb_02 = self.rgb_gcm_0(torch.cat((img_0, self.upsample_2(x_rgb_12)), dim=1))

        x_dep_42 = self.dep_gcm_4(img_4)
        x_dep_32 = self.dep_gcm_3(torch.cat((img_3, self.upsample_2(x_dep_42)), dim=1))
        x_dep_22 = self.dep_gcm_2(torch.cat((img_2, self.upsample_2(x_dep_32)), dim=1))
        x_dep_12 = self.dep_gcm_1(torch.cat((img_1, self.upsample_2(x_dep_22)), dim=1))
        x_dep_02 = self.dep_gcm_0(torch.cat((img_0, self.upsample_2(x_dep_12)), dim=1))


        # MFIB+Decoder+Pre #
        ###############################################
        x_ful_42 = self.mfib_layer4(ful_4, x_rgb_42, x_dep_42)
        x_ful_42 = self.upsample_2(self.ful_gcm_4(x_ful_42))

        x_ful_32 = self.mfib_layer3(ful_3, x_rgb_32, x_dep_32)
        x_ful_32 = self.upsample_2(self.ful_gcm_3(torch.cat((x_ful_42, x_ful_32), 1)))

        x_ful_22 = self.mfib_layer2(ful_2, x_rgb_22, x_dep_22)
        x_ful_22 = self.upsample_2(self.ful_gcm_2(torch.cat((x_ful_32, x_ful_22), 1)))

        x_ful_12 = self.mfib_layer1(ful_1, x_rgb_12, x_dep_12)
        x_ful_12 = self.upsample_2(self.ful_gcm_1(torch.cat((x_ful_22, x_ful_12), 1)))

        x_ful_02 = self.mfib_layer0(ful_0, x_rgb_02, x_dep_02)
        x_ful_02 = self.ful_gcm_0(torch.cat((x_ful_12, x_ful_02), 1))

        size = imgs.size()[2:]
        s0 = self.S0(x_ful_02, size)
        s1 = self.S1(x_ful_12, size)
        s2 = self.S2(x_ful_22, size)
        s3 = self.S3(x_ful_32, size)
        s4 = self.S4(x_ful_42, size)
        rgb = self.rgb(x_rgb_02, size)
        dep = self.dep(x_dep_02, size)

        return (s0, s1, s2, s3, s4, rgb, dep)


    def _make_deplayer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        groups = 1
        expansion = 4
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.dep_inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.dep_inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = []
        layers.append(Bottleneck(self.dep_inplanes, planes, stride, downsample, groups,
                                 self.base_width, previous_dilation, norm_layer))
        self.dep_inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.dep_inplanes, planes, groups=groups,
                                     base_width=self.base_width, dilation=1,
                                     norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.rgbinplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.rgbinplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.rgbinplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.rgbinplanes, self.rgbinplanes))

        layers.append(block(self.rgbinplanes, planes, stride, upsample))
        self.rgbinplanes = planes

        return nn.Sequential(*layers)



