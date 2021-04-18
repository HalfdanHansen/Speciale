from .pack import *

## ATDC FUNCTIONS ##

def ATDC_get_grads_one_filter_3D(gr, de):

  # gr is the full gradient for a single filter (w)
  # de is the set of decomposed elements [p,q,t] that is [inputchannel, 3, 3]

  # the direction in which we want to take a step
  dLdp = []
  dLdq = []
  dLdt = []

  # each step has the number of elements of the decomposed elements, and each
  # decomposed element step depends on a double sum (see eq 19-21 in ATDC paper)

  for i in range(len(de[0])):
    temp = 0
    for j,q in enumerate(de[1]):
      for k,t in enumerate(de[2]):
        temp += gr[i,j,k]*q*t
    dLdp.append(float(temp))

  for j in range(len(de[1])):
    temp = 0
    for i,p in enumerate(de[0]):
      for k,t in enumerate(de[2]):
        temp += gr[i,j,k]*p*t
    dLdq.append(float(temp))

  for k in range(len(de[2])):
    temp = 0
    for i,p in enumerate(de[0]):
      for j,q in enumerate(de[1]):
        temp += gr[i,j,k]*p*q
    dLdt.append(float(temp))


  dL = [dLdp, dLdq, dLdt]

  return dL
  
def ATDC_get_grads_one_filter_3D_short(gr, de):

  # gr is the full gradient for a single filter (w)
  # de is the set of decomposed elements [p,q,t] that is [inputchannel, 3, 3]

  # the direction in which we want to take a step
  dLdp = []
  dLdq = []
  dLdt = []

  # each step has the number of elements of the decomposed elements, and each
  # decomposed element step depends on a double sum (see eq 19-21 in ATDC paper)

  dLdp = np.einsum('i,ij->j',de[1].reshape(len(de[1])),np.einsum('i,ijk->jk',de[2].reshape(len(de[2])),np.transpose(gr.permute(0,1,2).numpy()))).tolist()
  dLdq = np.einsum('i,ij->j',de[2].reshape(len(de[2])),np.einsum('i,ijk->jk',de[0].reshape(len(de[0])),np.transpose(gr.permute(1,2,0).numpy    ()))).tolist()
  dLdt = np.einsum('i,ij->j',de[1].reshape(len(de[1])),np.einsum('i,ijk->jk',de[0].reshape(len(de[0])),np.transpose(gr.permute(2,1,0).numpy()))).tolist()

  dL = [dLdp, dLdq, dLdt]

  return dL

def ATDC_update_step_one_filter_3D(dL, alpha, de):
  pqt_steps = []
  for e,d in enumerate(dL):
    steps = []
    for l in range(len(d)):
      step = de[e][l]-alpha*dL[e][l]
      steps.append(step.reshape(1,1))
    pqt_steps.append(np.concatenate(steps))
  return pqt_steps
 
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

def ATDC_get_grads_one_filter_4D(gr, de):

  # gr is the full gradient for a single filter (w)
  # de is the set of decomposed elements [u,t,p,q] that is [outputchannel,inputchannel, 3, 3]

  # the direction in which we want to take a step
  dLdp = []
  dLdq = []
  dLdt = []
  dLdu = []

  # each step has the number of elements of the decomposed elements, and each
  # decomposed element step depends on a double sum (see eq 19-21 in ATDC paper)

  for i in range(len(de[0])):
    temp = 0
    for j,q in enumerate(de[1]):
      for k,t in enumerate(de[2]):
        for h,u in enumerate(de[3]):
          temp += gr[i,j,k,h]*q*t*u
    dLdp.append(float(temp))

  for j in range(len(de[1])):
    temp = 0
    for i,p in enumerate(de[0]):
      for k,t in enumerate(de[2]):
        for h,u in enumerate(de[3]):
          temp += gr[i,j,k,h]*p*t*u
    dLdq.append(float(temp))

  for k in range(len(de[2])):
    temp = 0
    for i,p in enumerate(de[0]):
      for j,q in enumerate(de[1]):
        for h,u in enumerate(de[3]):
          temp += gr[i,j,k,h]*p*q*u
    dLdt.append(float(temp))

  for h in range(len(de[3])):
    temp = 0
    for i,p in enumerate(de[0]):
      for j,q in enumerate(de[1]):
        for k,t in enumerate(de[2]):
          temp += gr[i,j,k,h]*p*q*t
    dLdu.append(float(temp))

  dL = [dLdp, dLdq, dLdt, dLdu]

  return dL
  
def ATDC_get_grads_one_filter_4D_short(gr, de):

  # gr is the full gradient for a single filter (w)
  # de is the set of decomposed elements [u,t,p,q] that is [outputchannel,inputchannel, 3, 3]

  # the direction in which we want to take a step
  dLdp = []
  dLdq = []
  dLdt = []
  dLdu = []

  # each step has the number of elements of the decomposed elements, and each
  # decomposed element step depends on a double sum (see eq 19-21 in ATDC paper)

  dLdp = np.einsum('i,ij->j',de[1].reshape(len(de[1])),np.einsum('i,ijk->jk',de[2].reshape(len(de[2])),np.einsum('i,ijkl->jkl',de[3].reshape(len(de[3])),np.transpose(gr.permute(0,1,2,3).numpy())))).tolist()
  dLdq = np.einsum('i,ij->j',de[0].reshape(len(de[0])),np.einsum('i,ijk->jk',de[2].reshape(len(de[2])),np.einsum('i,ijkl->jkl',de[3].reshape(len(de[3])),np.transpose(gr.permute(1,0,2,3).numpy())))).tolist()
  dLdt = np.einsum('i,ij->j',de[0].reshape(len(de[0])),np.einsum('i,ijk->jk',de[1].reshape(len(de[1])),np.einsum('i,ijkl->jkl',de[3].reshape(len(de[3])),np.transpose(gr.permute(2,0,1,3).numpy())))).tolist()
  dLdu = np.einsum('i,ij->j',de[0].reshape(len(de[0])),np.einsum('i,ijk->jk',de[1].reshape(len(de[1])),np.einsum('i,ijkl->jkl',de[2].reshape(len(de[2])),np.transpose(gr.permute(3,0,1,2).numpy())))).tolist()

  dL = [dLdp, dLdq, dLdt, dLdu]

  return dL

def ATDC_update_step_one_filter_4D(dL, alpha, de):
  pqtu_steps = []
  for e in range(len(dL)):
    steps = []
    for l in range(len(dL[e])):
      step = de[e][l]-alpha*dL[e][l]
      steps.append(step.reshape(1,1))
    pqtu_steps.append(np.concatenate(steps))
  return pqtu_steps

def ATDC_get_grads_one_filter_4D_short_tensor(gr, de):

  # gr is the full gradient for a layer
  # de is the set of decomposed elements [u,t,p,q] that is [outputchannel,inputchannel, 3, 3]

  # each step has the number of elements of the decomposed elements, and each
  # decomposed element step depends on a triple sum (see eq 19-21 in ATDC paper)

  # each einsum shaves a dimension off of gr in order of the permutation
  dLdu = torch.einsum('i,ij->j',de[1].reshape(len(de[1])),torch.einsum('i,ijk->jk',de[2].reshape(len(de[2])),torch.einsum('i,ijkl->jkl',de[3].reshape(len(de[3])),gr.permute(3,2,1,0))))
  dLdt = torch.einsum('i,ij->j',de[0].reshape(len(de[0])),torch.einsum('i,ijk->jk',de[2].reshape(len(de[2])),torch.einsum('i,ijkl->jkl',de[3].reshape(len(de[3])),gr.permute(3,2,0,1))))
  dLdp = torch.einsum('i,ij->j',de[0].reshape(len(de[0])),torch.einsum('i,ijk->jk',de[1].reshape(len(de[1])),torch.einsum('i,ijkl->jkl',de[3].reshape(len(de[3])),gr.permute(3,1,0,2))))
  dLdq = torch.einsum('i,ij->j',de[0].reshape(len(de[0])),torch.einsum('i,ijk->jk',de[1].reshape(len(de[1])),torch.einsum('i,ijkl->jkl',de[2].reshape(len(de[2])),gr.permute(2,1,0,3))))

  return [dLdu, dLdt, dLdp, dLdq]

def ATDC_update_step_one_filter_4D_tensor(dL, alpha, de):
  return [torch.sub(de[0],dL[0], alpha=alpha),
          torch.sub(de[1],dL[1], alpha=alpha),
          torch.sub(de[2],dL[2], alpha=alpha),
          torch.sub(de[3],dL[3], alpha=alpha)]

def ATDC_update_step_one_filter_4D_adam(dL, alpha, de, v, m, beta1, beta2, eps=1e-8):
  '''
  initially
  alpha = 0.001
  beta1 = 0.9      reduced each epoch
  beta2 = 0.999    reduced each epoch
  v,m = zeroes as pqtu 
  '''
  minusbeta1 = 1-beta1
  minusbeta2 = 1-beta2
  
  for i,derivatives in enumerate(dL):
    m[i] = torch.add(torch.mul(m[i],beta1), derivatives, alpha=minusbeta1)
    v[i] =  torch.add(torch.mul(v[i],beta2), torch.mul(derivatives,derivatives), alpha=minusbeta2)
    alpha_new = alpha * np.sqrt(minusbeta2) / (minusbeta1)
    de[i]= torch.sub(de[i], torch.div(m[i], torch.add(torch.sqrt(v[i]), eps)), alpha = alpha_new)
  return (m,v,de)

def adam_step(grad, alpha, data, v, m, beta1, beta2, eps=1e-8):

  minusbeta1 = 1-beta1
  minusbeta2 = 1-beta2

  m = torch.add(torch.mul(m,beta1), grad, alpha=minusbeta1)
  v =  torch.add(torch.mul(v,beta2), torch.mul(grad,grad), alpha=minusbeta2)
  alpha_new = alpha * np.sqrt(minusbeta2) / (minusbeta1)
  data = torch.sub(data, torch.div(m, torch.add(torch.sqrt(v), eps)), alpha = alpha_new)

  return (m,v,data)

## BAF FUNCTIONS ##

# Decompose filter and return estimated filter
def BAF_4D(filter, rank):
  #Decompose filter with parafac
  decomp = tl.decomposition.parafac(tl.tensor(filter), rank = rank)
  DF = 0
  for i in range(rank):
    #Outer product of decompositions
    DF += np.einsum('i,j,k,l->ijkl',decomp[1][0][:,i],decomp[1][1][:,i],decomp[1][2][:,i],decomp[1][3][:,i])
  return DF, decomp


def BAF_3D(filter, rank):
  #Decompose filter with parafac
  decomp = tl.decomposition.parafac(tl.tensor(filter), rank = rank)
  DF = 0
  for i in range(rank):
    #Outer product of decompositions
    DF += np.einsum('i,j,k->ijk',decomp[1][0][:,i],decomp[1][1][:,i],decomp[1][2][:,i])
  return DF, decomp

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


### TUCKER2 METHOD FUNCTIONS ###

def initialize_model_weights_from_Tucker2(convName,net,netname,rank1,rank2,kdims):

  # convName are the names of the layers. In strings
  # netname is the name og the network. In string

  # Make weights from initialization of random weights centered around 0, with std of 0.33.

  utc_convs = []
  for k1,c in enumerate(convName):
    convData = eval(netname+"."+c+".weight.data")
    layer =  ([torch.mul(torch.randn(convData.shape[0],rank2),0.333).cuda(),     #u
              torch.mul(torch.randn(convData.shape[1],rank1),0.333).cuda(),      #t
              torch.mul(torch.randn(rank1,rank2,kdims[k1],kdims[k1]),0.333).cuda()])             #c
    utc_convs.append(layer)

  for k1,utc in enumerate(utc_convs):
    convData = eval(netname+"."+convName[k1]+".weight.data")
    convData[:] = torch.einsum('hq,sw,wqij->hsij',utc[0],utc[1],utc[2])

  return utc_convs

def ATDC_get_grads_Tucker2_ATDC(gr, de, rank1, rank2):

  # gr is the full gradient for a layer in dimensions: [outputchannel (H),inputchannel (S), kdims (i), kdims (j)] (hsij)
  # de is the set of decomposed elements 
  # [u, t, c] that is in dimensions: [[outputchannel,rank2], [inputchannel,rank1], [rank1,rank2,3,3]]

  dLdu = torch.zeros_like(de[0])
  dLdt = torch.zeros_like(de[1])
  dLdc = torch.zeros_like(de[2])

  # each einsum shaves a dimension off of gr in order of the permutation
  for r2 in range(rank2):
    temp = torch.zeros_like(de[0][:,r2])
    for r1 in range(rank1):
      temp += torch.einsum('ij,hij->h',de[2][r1,r2,:,:],torch.einsum('s,hsij->hij',de[1][:,r1].reshape(len(de[1][:,r1])),gr))
    dLdu[:,r2] = temp

  for r1 in range(rank1):
    temp = torch.zeros_like(de[1][:,r1])
    for r2 in range(rank2):
      temp += torch.einsum('ij,sij->s',de[2][r1,r2,:,:],torch.einsum('h,hsij->sij',de[0][:,r2].reshape(len(de[0][:,r2])),gr))
    dLdt[:,r1] = temp

  for r1 in range(rank1):
    for r2 in range(rank2):
      dLdc[r1,r2,:,:] = torch.einsum('s,sij->ij',de[1][:,r1].reshape(len(de[1][:,r1])),torch.einsum('h,hsij->sij',de[0][:,r2].reshape(len(de[0][:,r2])),gr))

  return [dLdu,dLdt,dLdc]

def ATDC_update_step_Tucker2_ATDC(dL, alpha, de):
  return [torch.sub(de[0],dL[0], alpha=alpha),
          torch.sub(de[1],dL[1], alpha=alpha),
          torch.sub(de[2],dL[2], alpha=alpha)]

def ATDC_update_step_adam_Tucker2_ATDC(dL, alpha, de, v, m, beta1, beta2, eps=1e-8):
  '''
  initially
  alpha = 0.001
  beta1 = 0.9      reduced each epoch
  beta2 = 0.999    reduced each epoch
  v,m = zeroes as pqtu 
  '''
  minusbeta1 = 1-beta1
  minusbeta2 = 1-beta2
  
  for i,derivatives in enumerate(dL):
    m[i] = torch.add(torch.mul(m[i],beta1), derivatives, alpha=minusbeta1)
    v[i] =  torch.add(torch.mul(v[i],beta2), torch.mul(derivatives,derivatives), alpha=minusbeta2)
    alpha_new = alpha * np.sqrt(minusbeta2) / (minusbeta1)
    de[i]= torch.sub(de[i], torch.div(m[i], torch.add(torch.sqrt(v[i]), eps)), alpha = alpha_new)
  return (m,v,de)

def adam_step_tucker2(grad, alpha, data, v, m, beta1, beta2, eps=1e-8):

  minusbeta1 = 1-beta1
  minusbeta2 = 1-beta2

  m = torch.add(torch.mul(m,beta1), grad, alpha=minusbeta1)
  v =  torch.add(torch.mul(v,beta2), torch.mul(grad,grad), alpha=minusbeta2)
  alpha_new = alpha * np.sqrt(minusbeta2) / (minusbeta1)
  data = torch.sub(data, torch.div(m, torch.add(torch.sqrt(v), eps)), alpha = alpha_new)

  return (m,v,data)






### PARAFAC RANK R METHODS ###

def initialize_model_weights_from_PARAFAC_rank(convName,net,netname,rank):

  # convName are the names of the layers. In strings
  # netname is the name og the network. In string
  # Make weights from initialization of random weights centered around 0, with std of 0.33.

  pqtu_convs = []
  for k1,c in enumerate(convName):
    convData = eval(netname+"."+c+".weight.data")
    layer =  ([torch.mul(torch.randn(convData.shape[0],rank),0.333).cuda(),
               torch.mul(torch.randn(convData.shape[1],rank),0.333).cuda(),
               torch.mul(torch.randn(convData.shape[2],rank),0.333).cuda(),
               torch.mul(torch.randn(convData.shape[3],rank),0.333).cuda()])

    pqtu_convs.append(layer)

  for k1,pqtu in enumerate(pqtu_convs):
    convData = eval(netname+"."+convName[k1]+".weight.data")
    '''
    temp = []
    for R in range(rank):
        temp += torch.einsum('h,s,i,j->hsij',pqtu[0][R,:],pqtu[1][R,:],pqtu[2][R,:],pqtu[3][R,:])
    convData[:] = temp
    '''
    convData[:] = torch.einsum('hsijr->hsij',torch.einsum('hr,sr,ir,jr->hsijr',pqtu[0],pqtu[1],pqtu[2],pqtu[3]))

  return pqtu_convs

def ATDC_get_grads_one_filter_4D_PARAFAC_rank(gr, de, rank):

  # gr is the full gradient for a layer
  # de is the set of decomposed elements [u,t,p,q] that is [outputchannel,inputchannel, 3, 3]

  # each einsum shaves a dimension off of gr in order of the permutation

  dLdu = torch.einsum('ir,ijr->jr',de[1],torch.einsum('ir,ijkr->jkr',de[2],torch.einsum('ir,ijkl->jklr',de[3],gr.permute(3,2,1,0))))
  dLdt = torch.einsum('ir,ijr->jr',de[0],torch.einsum('ir,ijkr->jkr',de[2],torch.einsum('ir,ijkl->jklr',de[3],gr.permute(3,2,0,1))))
  dLdp = torch.einsum('ir,ijr->jr',de[0],torch.einsum('ir,ijkr->jkr',de[1],torch.einsum('ir,ijkl->jklr',de[3],gr.permute(3,1,0,2))))
  dLdq = torch.einsum('ir,ijr->jr',de[0],torch.einsum('ir,ijkr->jkr',de[1],torch.einsum('ir,ijkl->jklr',de[2],gr.permute(2,1,0,3))))
  
  return [dLdu, dLdt, dLdp, dLdq]

def ATDC_update_step_one_filter_4D_PARAFAC_rank(dL, alpha, de):
  return [torch.sub(de[0],dL[0], alpha=alpha),
          torch.sub(de[1],dL[1], alpha=alpha),
          torch.sub(de[2],dL[2], alpha=alpha),
          torch.sub(de[3],dL[3], alpha=alpha)]

