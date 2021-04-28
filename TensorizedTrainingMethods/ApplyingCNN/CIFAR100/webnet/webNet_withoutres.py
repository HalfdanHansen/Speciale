#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:34:40 2021

@author: s152576
"""

from PackagesAndModels.pack import *

def conv_block(in_channels, out_channels, pool=False, drop1=False, drop2=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    elif drop1: layers.append(nn.Dropout2d(0.4))
    elif drop2: layers.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layers)


def conv_3D_block(in_channels, out_channels, pool=False, drop1=False, drop2=False):
    layers = [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1,
                      padding=0, bias=False, groups = 1),
                nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=(3,1),
                      padding=(1,0), bias=False, groups = out_channels),
                nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=(1,3),
                      padding=(0,1), bias=False, groups = out_channels),
               nn.BatchNorm2d(out_channels), 
               nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    elif drop1: layers.append(nn.Dropout2d(0.4))
    elif drop2: layers.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layers)

def conv_4D_block(in_channels, out_channels, pool=False, drop1=False, drop2=False):
    rank =1
    layers = [
              nn.Conv2d(in_channels = in_channels, out_channels = rank, kernel_size=(1,1), 
                        padding=0, bias=False),
              nn.Conv2d(in_channels = rank, out_channels = rank, kernel_size=(3,1),
                        padding=(1,0), bias=False, groups = rank),
              nn.Conv2d(in_channels = rank, out_channels = rank, kernel_size=(1,3),
                        padding=(0,1), bias=False, groups = rank),
              nn.Conv2d(in_channels = rank, out_channels = out_channels, kernel_size=(1,1),
                        padding=(0,0), bias=False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    elif drop1: layers.append(nn.Dropout2d(0.4))
    elif drop2: layers.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layers)

def conv_Tucker2_block(in_channels, out_channels, pool=False, drop1=False, drop2=False):
    rank = [1,1]
    layers = [
            nn.Conv2d(in_channels = in_channels, out_channels = rank[0], kernel_size=(1,1),
                      padding=0, bias=False),
            nn.Conv2d(in_channels = rank[0], out_channels = rank[1], kernel_size=(3,3),
                      padding=(1,1), bias=False),
            nn.Conv2d(in_channels = rank[1], out_channels = out_channels, kernel_size=(1,1),
                      padding=(0,0), bias=False),
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    elif drop1: layers.append(nn.Dropout2d(0.4))
    elif drop2: layers.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layers)

class WebNet_noRes(nn.Module):
    def __init__(self, block, in_channels, num_classes):
            super().__init__()
            
            self.conv1 = block(in_channels, 64, drop2=True)
            self.conv2 = block(64, 128, pool=True)
            self.conv3 = block(128, 128, drop1=True)
            self.conv4 = block(128, 128, pool=True)
            self.conv5 = block(128, 256, drop1=True)
            self.conv6 = block(256, 512, drop1=True)
            self.conv7 = block(512, 512, pool=True)
            self.conv8 = block(512, 512, drop1=True)
            self.conv9 = block(512, 1028, drop1=True)
            self.conv10 = block(1028, 1028, pool=True)
            self.conv11 = block(1028, 1028, drop1=True)
            self.classifier = nn.Sequential(nn.MaxPool2d(2), 
                                            nn.Flatten(), 
                                            nn.Linear(1028, num_classes))
            
    def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.conv5(out)
            out = self.conv6(out)
            out = self.conv7(out)
            out = self.conv8(out)
            out = self.conv9(out)
            out = self.conv10(out)
            out = self.conv11(out)
            out = self.classifier(out)
            return out
    
webNet_noRes = WebNet_noRes(conv_block, 3, 100)

webNet_noRes3D = WebNet_noRes(conv_3D_block, 3, 100)

webNet_noRes4D = WebNet_noRes(conv_4D_block, 3, 100)

webNet_noResTucker2 = WebNet_noRes(conv_Tucker2_block, 3, 100)