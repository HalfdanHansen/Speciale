#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:34:40 2021

@author: s152576
"""

from PackagesAndModels.pack import *

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv_3D_block(in_channels, out_channels, pool=False):
    layers = [nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = out_f, kernel_size=1,
                      padding=0, bias=False, groups = 1),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(3,1),
                      padding=(1,0), bias=False, groups = out_f),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(1,3),
                      padding=(0,1), bias=False, groups = out_f)
            ),
               nn.BatchNorm2d(out_f), 
               nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet152(nn.Module):
    def __init__(self, block, in_channels, num_classes):
    #         super().__init__()
    #         # Use a pretrained model
    #         self.network = models.resnet34(pretrained=True)
    #         # Replace last layer
    #         num_ftrs = self.network.fc.in_features
    #         self.network.fc = nn.Linear(num_ftrs, num_classes)
            super().__init__()
            
            self.conv1 = block(in_channels, 64)
            self.conv2 = block(64, 128, pool=True)
            self.res1 = nn.Sequential(block(128, 128), block(128, 128))
            
            self.conv3 = block(128, 256, pool=True)
            self.conv4 = block(256, 512, pool=True)
            self.res2 = nn.Sequential(block(512, 512), block(512, 512))
            self.conv5 = block(512, 1028, pool=True)
            self.res3 = nn.Sequential(block(1028, 1028), block(1028, 1028))
            
            self.classifier = nn.Sequential(nn.MaxPool2d(2), 
                                            nn.Flatten(), 
                                            nn.Linear(1028, num_classes))
            
        
    #     def forward(self, xb):
            
    #         return torch.relu(self.network(xb))
    def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.conv5(out)
            out = self.res3(out) + out
            out = self.classifier(out)
            return out
    
resnet152 = ResNet152(conv_3D_block, 3, 100)
    
#     def freeze(self):
#         # To freeze the residual layers
#         for param in self.network.parameters():
#             param.require_grad = False
#         for param in self.network.fc.parameters():
#             param.require_grad = True
    
#     def unfreeze(self):
#         # Unfreeze all layers
#         for param in self.network.parameters():
#             param.require_grad = True