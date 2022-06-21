import torch
from torch import nn
import torch.nn.functional as F

class Res_ViT_decode(nn.Module):
    def __init__(self, image_height, image_width, batch_size, classes_num):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.classes_num = classes_num

        def block(input_dims, out_dims):
            upsamples = nn.Sequential(
                nn.Conv2d(input_dims, out_dims, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dims),
                nn.ReLU(inplace=True),
            )
            return upsamples

        self.tconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=4)
        self.tconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.block1 = block(512+256+128, 512)

        self.dconv1 = nn.Conv2d(512, 512, 3, stride=1, padding=2, dilation=2)
        self.dconv2 = nn.Conv2d(512, 512, 3, stride=1, padding=4, dilation=4)
        self.dconv3 = nn.Conv2d(512, 512, 3, stride=1, padding=8, dilation=8)
        self.block2 = block(512*5, 512)

        self.tconv3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.block3 = block(512+64, 256)
        self.tconv4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.tconv5 = nn.ConvTranspose2d(64, self.classes_num, kernel_size=2, stride=2)


    def forward(self, inputs):
        x, x_1, x_2, x_3 = inputs
        x_3 = self.tconv1(x_3)
        x_2 = self.tconv2(x_2)
        x_1 = torch.cat((x_1, x_2, x_3), 1)
        x_1 = self.block1(x_1)

        xd = self.dconv1(x_1)
        xd1 = self.dconv2(x_1)
        xd2 = self.dconv3(x_1)
        xmean = torch.mean(x_1, 1, keepdim=True).repeat(1, 256, 1, 1)
        xmax, indexes = torch.max(x_1, 1, keepdim=True)
        xmax = xmax.repeat(1, 256, 1, 1)
        x_1 = torch.cat((x_1, xd, xd1, xd2, xmean, xmax), 1)
        x_1 = self.block2(x_1)

        x_1 = self.tconv3(x_1)
        x = torch.cat((x, x_1), 1)
        x = self.block3(x)
        x = F.relu(self.bn1(self.tconv4(x)))
        return F.relu(self.tconv5(x))


