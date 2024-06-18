import torch.nn as nn
import torch
import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils'))
if module_path not in sys.path:
    sys.path.append(module_path)
from net_utils import *

class DCFNetFeature(nn.Module):
    def __init__(self,in_channel=3):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)

class EventEnc(nn.Module):
    def __init__(self, num_bins=3):
        super(EventEnc,self).__init__()
        self.conv1 = nn.Conv2d(num_bins, 32, kernel_size=3, stride=1, padding=1)
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
    def __init__(self):
        super(GreyEnc,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
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
        super(Decoder,self).__init__()
        self.convT1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1)
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
    def __init__(self, num_bins,config=None):
        super(DCFNet, self).__init__()
        self.event_feat = DCFNetFeature(num_bins)
        self.enc_e = EventEnc(num_bins)
        self.enc_f = GreyEnc()
        self.dec = Decoder()
        self.frame_feat = DCFNetFeature(1)
        self.model_alphaf = []
        self.model_xf = []
        self.config = config

    def forward(self, x, events = False):
        if(not events):
            x_enc = self.enc_f(x)
            feat = self.frame_feat(x)
        else:
            x_enc = self.enc_e(x)
            feat = self.event_feat(x)
        x = (self.dec(x_enc) + feat) * self.config.cos_window
        xf = torch.torch.fft.rfft2(x)
        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response = torch.fft.irfft2(torch.view_as_complex(complex_mul(kxzf, self.model_alphaf)))
        return response

    def update(self,  x , events = False, lr=1.):
        if(not events):
            x_enc = self.enc_f(x)
            feat = self.frame_feat(x)
        else:
            x_enc = self.enc_e(x)
            feat = self.event_feat(x)
        z = (self.dec(x_enc) + feat) * self.config.cos_window
        zf = torch.torch.fft.rfft2(z)
        kzzf = torch.sum(torch.sum(torch.view_as_real(zf) ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0)
        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:
            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict'] 
            if 'module' in state_dict.keys():  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)


class DCFNet_add(nn.Module):
    def __init__(self, num_bins,config=None,p=None):
        super(DCFNet_add, self).__init__()
        self.event_feat = DCFNetFeature(num_bins)
        self.enc_e = EventEnc(num_bins)
        self.enc_f = GreyEnc()
        self.dec = Decoder()
        self.frame_feat = DCFNetFeature(1)
        self.model_alphaf = []
        self.model_xf = []
        self.config = config
        if(p == None):
            self.p = 1
        else:
            self.p = p
            
    def forward(self, x, y):
        x_enc = self.enc_f(x)
        x_feat = self.frame_feat(x)*self.p
        y_enc = self.enc_e(y)
        y_feat = self.event_feat(y)
        x = (self.dec(x_enc) +self.dec(y_enc)+ x_feat + y_feat) * self.config.cos_window
        xf = torch.torch.fft.rfft2(x)

        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response = torch.fft.irfft2(torch.view_as_complex(complex_mul(kxzf, self.model_alphaf)))
        return response

    def update(self,  x, y, lr=1.):
        x_enc = self.enc_f(x)
        x_feat = self.frame_feat(x)*self.p
        y_enc = self.enc_e(y)
        y_feat = self.event_feat(y)
        z = (self.dec(x_enc) +self.dec(y_enc)+ x_feat + y_feat) * self.config.cos_window
        zf = torch.torch.fft.rfft2(z)
        kzzf = torch.sum(torch.sum(torch.view_as_real(zf) ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0)
        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:
            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict'] 
            if 'module' in state_dict.keys():  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)
