import torch
from torch import nn

import scipy.misc
import scipy.io

from config import DefaultConfig
from models.basic_module import BasicModule

opt = DefaultConfig()
MODEL_DIR = opt.vgg_path

class ImgModule(BasicModule):
    def __init__(self, bit):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.bit = bit
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        # fc8
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self.mean = torch.zeros(3, 224, 224)
        
        self._init()

    def _init(self):
        # classifier init
        self.classifier.weight.data = torch.randn(self.bit, 4096) * 0.01
        self.classifier.bias.data = torch.randn(self.bit) * 0.01

        data = scipy.io.loadmat(MODEL_DIR)
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        # [batch_size, 224, 224, 3]
        x = x.permute(0, 3, 2, 1) 
        # [batch_size, 3, 224, 224]
        if x.is_cuda:
            x = x - self.mean.cuda()
        else:
            x = x - self.mean
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        x = torch.tanh(x)
        
        return x
