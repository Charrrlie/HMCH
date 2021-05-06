import torch
from torch import nn

from models.basic_module import BasicModule


class ClassifierModule(BasicModule):
    def __init__(self, bit, n_class):
        super(ClassifierModule, self).__init__()
        self.module_name = "classifier_model"
        self.bit = bit

        self.fc = nn.Linear(in_features=bit, out_features=n_class)
        
    
    def forward(self, x):
        # [batch_size, bit]
        x = self.fc(x)
     
        return x
