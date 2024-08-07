import numpy as np
import torch
import torch.nn as nn
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
if module_path not in sys.path:
    sys.path.append(module_path)
from net_utils import *


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(
        np.arange(1, sz[0] + 1) - np.floor(float(sz[0]) / 2),
        np.arange(1, sz[1] + 1) - np.floor(float(sz[1]) / 2),
    )
    d = x**2 + y**2
    g = np.exp(-0.5 / (sigma**2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.0) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.0) + 1), axis=1)
    return g.astype(np.float32)


class TrackerConfig(object):
    crop_sz = 128
    output_sz = 124
    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])
    yf = torch.fft.rfft2(torch.Tensor(y).view(1, 1, output_sz, output_sz).cuda())
    yf = torch.view_as_real(yf)
    best_loss = 1e6


class DCFNetFeature(nn.Module):
    def __init__(self, in_channel):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3),  # , padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class EventEnc(nn.Module):
    def __init__(self, in_channel):
        super(EventEnc, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


class GreyEnc(nn.Module):
    def __init__(self, in_channel):
        super(GreyEnc, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.convT1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.convT2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=1)
        self.lclres = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)

    def forward(self, x):
        x = self.convT1(x)
        x = self.relu(x)
        x = self.convT2(x)
        x = self.lclres(x)
        return x


class DCFNet(nn.Module):
    def __init__(self, num_bins, config=None):
        super(DCFNet, self).__init__()
        self.enc_e = EventEnc(num_bins)
        self.enc_f = GreyEnc(1)
        self.dec = Decoder()
        self.event_feat = DCFNetFeature(num_bins)
        self.frame_feat = DCFNetFeature(1)

        if config != None:
            self.yf = config.yf.clone()
            self.lambda0 = config.lambda0

    def forward(self, z, x, label):
        zf = torch.torch.fft.rfft2(z)
        xf = torch.torch.fft.rfft2(x)

        kzzf = torch.sum(
            torch.sum(torch.view_as_real(zf) ** 2, dim=4, keepdim=True),
            dim=1,
            keepdim=True,
        )
        kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
        alphaf = label.to(device=z.device) / (kzzf + self.lambda0)
        response = torch.fft.irfft2(torch.view_as_complex(complex_mul(kxzf, alphaf)))
        return response
