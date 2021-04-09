from pack import *

def apply_decomp(inputchannel, outputchannel, spatial, padding, dimx, dimy, FD, batch, RANK):
  FilterList = []
  for i in range(4):
      FilterList.append(FD[1][i])

  # Applying the filters to the layers
  Layer1List = []
  for i in range(RANK):
      Layer1List.append(F.conv2d(batch,torch.tensor(FilterList[0][:,i].reshape(1,inputchannel,1,1))))

  Layer2List = []
  for i in range(RANK):
      Layer2List.append(F.conv2d(Layer1List[i],torch.tensor(FilterList[1][:,i].reshape(1,1,spatial,1)), padding = padding))

  Layer3List = []
  for i in range(RANK):
      Layer3List.append(F.conv2d(Layer2List[i],torch.tensor(FilterList[2][:,i].reshape(1,1,1,spatial))))
      if i == 0:
        Layer3 = Layer3List[i]
      else:
        Layer3 = torch.cat((Layer3,Layer3List[-1]),1)

  Layer4List = []
  for i in range(outputchannel):
      Layer4List.append(F.conv2d(Layer3,torch.tensor(FilterList[3][i].reshape(1,RANK,1,1))))
      if i == 0:
        Layer4 = Layer4List[i]
      else:
        Layer4 = torch.cat((Layer4,Layer4List[-1]),1)
  
  return (Layer4)

def conv_to_PARAFAC_firsttry(layer, rank):
    """
    Takes a convolutional layer and decomposes is using PARAFAC with the given rank.
    """
    # (Estimating the rank using VBMF) Fra Tobbes
    #rank = estimate_ranks(weights, [0, 1]) if rank is None else rank

    # The bias from the original layer is added to the last convolution Fra tobbes
    # last_layer.bias.data = layer.bias.data

    # Making the decomposition of the weights
    weights = layer.weight.data

    # Decomposing
    decomp = tl.decomposition.parafac(tl.tensor(torch.tensor(weights).permute(1,2,3,0)), rank=rank)
    decomp = decomp[1]

    # Making the layer into 4 sequential layers using the decomposition
    first_layer = Conv2d(in_channels = int(decomp[0].shape[0]), out_channels = rank, 
                         kernel_size=1, stride=1, padding=0, bias=False)

    second_layer = Conv2d(in_channels = rank, out_channels = rank,
                         kernel_size=(3,1), stride=1, padding=(1,0), bias=False)
    
    third_layer = Conv2d(in_channels = rank, out_channels = rank,
                         kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
    
    fourth_layer = Conv2d(in_channels = rank, out_channels = decomp[3].shape[0],
                         kernel_size=1, stride=1, padding=0, bias=False)

    # The decomposition is chosen as weights in the network (output, input, height, width)
    first_layer.weight.data = torch.transpose(torch.tensor(decomp[0]), 1, 0).unsqueeze(-1).unsqueeze(-1)
    second_layer.weight.data = torch.transpose(torch.tensor(decomp[1]), 1, 0).reshape(decomp[1].shape[1],decomp[1].shape[0],1,1).permute(0,3,1,2)
    third_layer.weight.data = torch.transpose(torch.tensor(decomp[2]), 1, 0).reshape(decomp[2].shape[1],decomp[2].shape[0],1,1).permute(0,3,2,1)
    fourth_layer.weight.data = torch.tensor(decomp[3]).unsqueeze(-1).unsqueeze(-1)
    
    new_layers = [first_layer, second_layer, third_layer, fourth_layer]
    return nn.Sequential(*new_layers)
    
def apply_decomp_3D_test(inputchannel, outputchannel, spatial, padding, dimx, dimy, pqt, batch):
  
  # Takes a 3D decomposed layer of filters and applies each of them and concatenates them.

  for i,filters in enumerate(pqt):
    if i == 0:
      Layer1 = F.conv2d(batch,torch.tensor(pqt[i][0]).reshape(1,inputchannel,1,1))
      Layer2 = F.conv2d(Layer1,torch.tensor(pqt[i][1]).reshape(1,1,spatial,1), padding = padding)
      Layer3 = F.conv2d(Layer2,torch.tensor(pqt[i][2]).reshape(1,1,1,spatial))
      Layer4 = Layer3
    else:
      Layer1 = F.conv2d(batch,torch.tensor(pqt[i][0]).reshape(1,inputchannel,1,1))
      Layer2 = F.conv2d(Layer1,torch.tensor(pqt[i][1]).reshape(1,1,spatial,1), padding = padding)
      Layer3 = F.conv2d(Layer2,torch.tensor(pqt[i][2]).reshape(1,1,1,spatial))
      Layer4 = torch.cat((Layer4, Layer3),1)
  
  return (Layer4)

def apply_decomp_ATDC_4D(inputchannel, outputchannel, spatial, padding, dimx, dimy, FD, batch, RANK):
  '''
  FilterList = []
  for i in range(4):
      FilterList.append(FD[1][i])
  '''
  FilterList = FD
  # Applying the filters to the layers

  Layer1 = F.conv2d(batch,torch.tensor(FilterList[0].reshape(1,inputchannel,1,1)))

  Layer2 = F.conv2d(Layer1,torch.tensor(FilterList[1].reshape(1,1,spatial,1)), padding = padding)

  Layer3 = F.conv2d(Layer2,torch.tensor(FilterList[2].reshape(1,1,1,spatial)))

  Layer4List = []
  for i in range(outputchannel):
      Layer4List.append(F.conv2d(Layer3,torch.tensor(FilterList[3][i].reshape(1,RANK,1,1))))
      if i == 0:
        Layer4 = Layer4List[i]
      else:
        Layer4 = torch.cat((Layer4,Layer4List[-1]),1)
  
  return (Layer4)


def conv_to_PARAFAC_last_conv_layer(layer, rank):
    """
    Takes a convolutional layer and decomposes is using PARAFAC with the given rank.
    """
    # (Estimating the rank using VBMF) Fra Tobbes
    #rank = estimate_ranks(weights, [0, 1]) if rank is None else rank

    # The bias from the original layer is added to the last convolution Fra tobbes
    # last_layer.bias.data = layer.bias.data

    # Making the decomposition of the weights
    weights = layer.weight.data

    # Decomposing
    decomp = tl.decomposition.parafac(tl.tensor(torch.tensor(weights).permute(1,2,3,0)), rank=rank)
    decomp = decomp[1]

    # Making the layer into 4 sequential layers using the decomposition
    first_layer = Conv2d(in_channels = int(decomp[0].shape[0]), out_channels = rank, 
                         kernel_size=1, stride=1, padding=0, bias=False)

    second_layer = Conv2d(in_channels = rank, out_channels = rank,
                         kernel_size=(7,1), stride=1, padding=0, bias=False)
    
    third_layer = Conv2d(in_channels = rank, out_channels = rank,
                         kernel_size=(1,7), stride=1, padding=0, bias=False)
    
    fourth_layer = Conv2d(in_channels = rank, out_channels = decomp[3].shape[0],
                         kernel_size=1, stride=1, padding=0, bias=False)

    # The decomposition is chosen as weights in the network (output, input, height, width)
    first_layer.weight.data = torch.transpose(torch.tensor(decomp[0]), 1, 0).unsqueeze(-1).unsqueeze(-1)
    second_layer.weight.data = torch.transpose(torch.tensor(decomp[1]), 1, 0).reshape(decomp[1].shape[1],decomp[1].shape[0],1,1).permute(0,3,1,2)
    third_layer.weight.data = torch.transpose(torch.tensor(decomp[2]), 1, 0).reshape(decomp[2].shape[1],decomp[2].shape[0],1,1).permute(0,3,2,1)
    fourth_layer.weight.data = torch.tensor(decomp[3]).unsqueeze(-1).unsqueeze(-1)
    
    new_layers = [first_layer, second_layer, third_layer, fourth_layer]
    return nn.Sequential(*new_layers)
