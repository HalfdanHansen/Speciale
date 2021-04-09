import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes,affine=False, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes,affine=False, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine=False, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
def conv_3D_block(in_f, out_f, kernelsize,stride,pad = 1):
    return  nn.Sequential(
            nn.Conv2d(in_channels = in_f, out_channels = out_f, kernel_size=1,
                      stride=1, padding=0, bias=False, groups = 1),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(kernelsize,1),
                      stride=(stride,1), padding=(pad,0), bias=False, groups = out_f),
            nn.Conv2d(in_channels = out_f, out_channels = out_f, kernel_size=(1,kernelsize),
                      stride=(1,stride), padding=(0,pad), bias=False, groups = out_f)
        )

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        skip = stride != 1 or in_planes != self.expansion*planes

        super(BasicBlock3D, self).__init__()
        self.conv1 = conv_3D_block(in_planes,planes,3,1+skip)
        self.bn1 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)
        self.conv2 = conv_3D_block(planes,planes,3,1)
        self.bn2 = nn.BatchNorm2d(planes, affine=False, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if skip:
            self.shortcut = nn.Sequential(
                conv_3D_block(in_planes,planes,1,1+skip,0),
                nn.BatchNorm2d(self.expansion*planes, affine=False, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        self.conv1 = conv_3D_block(3,64,3,1)
        self.bn1 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
        
def ATDC_get_grads_one_filter_3D_short_tensor(gr, de):

  # gr is the full gradient for a single filter (w)
  # de is the set of decomposed elements [p,q,t] that is [inputchannel, 3, 3]

  # each step has the number of elements of the decomposed elements, and each
  # decomposed element step depends on a double sum (see eq 19-21 in ATDC paper)

  dLdt = torch.einsum('i,ij->j',de[1].reshape(len(de[1])),torch.einsum('i,ijk->jk',de[2].reshape(len(de[2])),gr.permute(2,1,0)))
  dLdp = torch.einsum('i,ij->j',de[0].reshape(len(de[0])),torch.einsum('i,ijk->jk',de[2].reshape(len(de[2])),gr.permute(2,0,1)))
  dLdq = torch.einsum('i,ij->j',de[0].reshape(len(de[0])),torch.einsum('i,ijk->jk',de[1].reshape(len(de[1])),gr.permute(1,0,2)))
  
  return [dLdt, dLdp, dLdq]

def ATDC_update_step_one_filter_3D_tensor(grad, alpha, data):
  return [torch.sub(data[0],grad[0], alpha=alpha),
          torch.sub(data[1],grad[1], alpha=alpha),
          torch.sub(data[2],grad[2], alpha=alpha)]


def ATDC_update_step_one_filter_3D_adam(grad, alpha, data, v, m, beta1, beta2, eps=1e-8):

  minusbeta1 = 1-beta1
  minusbeta2 = 1-beta2
  
  for i, derivatives in enumerate(grad):
    m[i] = torch.add(torch.mul(m[i],beta1), derivatives, alpha=minusbeta1)
    v[i] =  torch.add(torch.mul(v[i],beta2), torch.mul(derivatives,derivatives), alpha=minusbeta2)
    alpha_new = alpha * np.sqrt(minusbeta2) / (minusbeta1)
    data[i]= torch.sub(data[i], torch.div(m[i], torch.add(torch.sqrt(v[i]), eps)), alpha = alpha_new)
  return (m,v,data)

def adam_step(grad, alpha, data, v, m, beta1, beta2, eps=1e-8):

  minusbeta1 = 1-beta1
  minusbeta2 = 1-beta2

  m = torch.add(torch.mul(m,beta1), grad, alpha=minusbeta1)
  v =  torch.add(torch.mul(v,beta2), torch.mul(grad,grad), alpha=minusbeta2)
  alpha_new = alpha * np.sqrt(minusbeta2) / (minusbeta1)
  data = torch.sub(data, torch.div(m, torch.add(torch.sqrt(v), eps)), alpha = alpha_new)

  return (m,v,data)
  
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
        

def evaluate_cifar(loader,model):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  correct = 0
  total = 0
  with torch.no_grad():
    for data in loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().detach() #item()

  return (correct / total)