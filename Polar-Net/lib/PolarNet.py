import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s


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


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class RFB_modified_large(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified_large, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel, out_channel, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=9, dilation=9)
        )
        self.conv_cat = BasicConv2d(5*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3, x4), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual


class PD(nn.Module):
    # Partial Decoder module
    def __init__(self, channel):
        super(PD, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        # print x1.shape, x2.shape, x3.shape, x4.shape
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2)), x4), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class Network(nn.Module):
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.padding_right_down = nn.ZeroPad2d((0, 4, 0, 4))
        self.PIE_6x6 = nn.Sequential(
            nn.Conv2d(3, 32, 6, 4, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.downSample = nn.MaxPool2d(2, stride=2)
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # self.rfb0_1 = RFB_modified_large(320, channel)
        # self.rfb2_1 = RFB_modified_large(512, channel)
        # self.rfb3_1 = RFB_modified_large(1024, channel)
        # self.rfb4_1 = RFB_modified_large(2048, channel)
        self.rfb0_1 = RFB_modified(320, channel)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.PD = PD(channel)
        self.refine_module = RefUnet(1, 64)
        if self.training:
            self.initialize_weights()

    def forward(self, x):
        x = self.padding_right_down(x)  # [bs, 3, 704, 704] -> [bs, 3, 708, 708]
        x = self.PIE_6x6(x)             # [bs, 3, 708, 708] -> [bs, 64, 176, 176]
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # [bs, 64, 176, 176] -> [bs, 64, 88, 88]
        x1 = self.resnet.layer1(x)      # [bs, 64, 88, 88] -> [bs, 256, 88, 88]

        x0_1 = torch.cat((x, x1), 1)       # channel -> 320
        x0_1_down = self.downSample(x0_1)  # [bs, 320, 88, 88] -> [bs, 320, 44, 44]

        x2 = self.resnet.layer2(x1)     # [bs, 320, 88, 88] -> [bs, 512, 44, 44]
        x3 = self.resnet.layer3(x2)     # [bs, 512, 44, 44] -> [bs, 1024, 22, 22]
        x4 = self.resnet.layer4(x3)     # [bs, 1024, 22, 22] -> [bs, 2048, 11, 11]

        x0_1_rfb = self.rfb0_1(x0_1_down)    # channel -> 32
        x2_rfb = self.rfb2_1(x2)             # channel -> 32
        x3_rfb = self.rfb3_1(x3)             # channel -> 32
        x4_rfb = self.rfb4_1(x4)             # channel -> 32
        S_g = self.PD(x4_rfb, x3_rfb, x2_rfb, x0_1_rfb)  # [bs, 1, 44, 44]
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')  # [bs, 1, 44, 44] -> [bs, 1, 352, 352]
        out_refine = self.refine_module(S_g_pred)
        return S_g_pred, out_refine

    def initialize_weights(self):
        # initialize PIE's param using the conv1 from Res2Net
        res2net = res2net50_v1b_26w_4s(pretrained=True)
        pretrained_dict = res2net.state_dict()
        all_params = {}
        for k, v in self.PIE_6x6.state_dict().items():
            if k == '0.weight':
                name = 'conv1.' + k
                new_weights_6x6 = torch.empty(32, 3, 6, 6)
                v1 = pretrained_dict[name]
                for n in range(32):
                    w_0_0 = v1[n, :, 0, 0]
                    w_0_1 = v1[n, :, 0, 1]
                    w_0_2 = v1[n, :, 0, 2]
                    w_1_0 = v1[n, :, 1, 0]
                    w_1_1 = v1[n, :, 1, 1]
                    w_1_2 = v1[n, :, 1, 2]
                    w_2_0 = v1[n, :, 2, 0]
                    w_2_1 = v1[n, :, 2, 1]
                    w_2_2 = v1[n, :, 2, 2]
                    new_weights_6x6[n, :, 0, 0] = new_weights_6x6[n, :, 0, 1] = new_weights_6x6[n, :, 1, 0] = new_weights_6x6[n, :, 1, 1] = w_0_0
                    new_weights_6x6[n, :, 0, 2] = new_weights_6x6[n, :, 0, 3] = new_weights_6x6[n, :, 1, 2] = new_weights_6x6[n, :, 1, 3] = w_0_1
                    new_weights_6x6[n, :, 0, 4] = new_weights_6x6[n, :, 0, 5] = new_weights_6x6[n, :, 1, 4] = new_weights_6x6[n, :, 1, 5] = w_0_2
                    new_weights_6x6[n, :, 2, 0] = new_weights_6x6[n, :, 2, 1] = new_weights_6x6[n, :, 3, 0] = new_weights_6x6[n, :, 3, 1] = w_1_0
                    new_weights_6x6[n, :, 2, 2] = new_weights_6x6[n, :, 2, 3] = new_weights_6x6[n, :, 3, 2] = new_weights_6x6[n, :, 3, 3] = w_1_1
                    new_weights_6x6[n, :, 2, 4] = new_weights_6x6[n, :, 2, 5] = new_weights_6x6[n, :, 3, 4] = new_weights_6x6[n, :, 3, 5] = w_1_2
                    new_weights_6x6[n, :, 4, 0] = new_weights_6x6[n, :, 4, 1] = new_weights_6x6[n, :, 5, 0] = new_weights_6x6[n, :, 5, 1] = w_2_0
                    new_weights_6x6[n, :, 4, 2] = new_weights_6x6[n, :, 4, 3] = new_weights_6x6[n, :, 5, 2] = new_weights_6x6[n, :, 5, 3] = w_2_1
                    new_weights_6x6[n, :, 4, 4] = new_weights_6x6[n, :, 4, 5] = new_weights_6x6[n, :, 5, 4] = new_weights_6x6[n, :, 5, 5] = w_2_2
                v = new_weights_6x6
                all_params[k] = v
            else:
                name = 'conv1.' + k
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.conv1.state_dict().keys())
        self.conv1.load_state_dict(all_params)
        print('Initialized and recombined weights from pretrained model')

# from torchstat import stat
if __name__ == "__main__":
    net = Network().cpu()
    input = torch.randn(1, 3, 704, 704)
    out = net(input)
    # stat(net, (3, 704, 704))