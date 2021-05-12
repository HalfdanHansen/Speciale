if __name__ == '__main__':

    import os
    import sys
    from copy import deepcopy
    
    from PackagesAndModels.pack import *
    from Papernet4model import *
    
    def ATDCTRUE_get_grads(gr, p, q, t, pindex, qindex):
      # gr is the full gradient for a single filter (w) --- [hsij] = [output, input, kernel, kernel]
      # p, q, t is the set of decomposed elements [p,q,t] that is [inputchannel, 3, 3] last 2 dim are shared through root
      
      for k1 in range(len(t)):
          dLdt = torch.einsum('j,js->s',q[k1].reshape(len(q[k1])),torch.einsum('i,ijs->js',p[k1].reshape(len(p[k1])),gr[:,k1,:,:].permute(2,3,0)))
      
      for k1 in range(len(p)):
          dLdp = torch.einsum('h,hi->i',pindex[k1],torch.einsum('j,jhi->hi',q[k1].reshape(len(q[k1])),torch.einsum('s,sjhi->jhi',t[k1].reshape(len(t[k1])),gr.permute(1,3,0,2))))
          dLdq = torch.einsum('h,hj->j',qindex[k1],torch.einsum('i,ihj->hj',p[k1].reshape(len(p[k1])),torch.einsum('s,sihj->ihj',t[k1].reshape(len(t[k1])),gr.permute(1,2,0,3))))
          
      return dp,dq,dt
    
    def ATDCTRUE_update_step(dp,dq,dt, alpha, p,q,t):
      return torch.sub(p,dp, alpha=alpha),torch.sub(q,dq, alpha=alpha),torch.sub(t,dt, alpha=alpha)
    
    def train_net_PARAFAC3D_ATDCTRUE(losses, net, netname, trainloader, criterion, optimizer, convName, p, q, t, alpha, rank, lName):
      running_loss = 0
      net.train()
    
      for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for k1 in range(len(convName)):
          convGrad = eval(netname+"."+convName[k1]+".weight.grad")
          convData = eval(netname+"."+convName[k1]+".weight.data")
          
          p,q,t = ATDCTRUE_update_step(
                  ATDCTRUE_get_grads(convGrad, p, q, t, pindex, qindex), 
                  alpha, p, q, t)
          
          root = int(np.sqrt(len(convGrad)))
          for k2 in range(root):
              for k3 in range(root):
                  convData[k2+k3] = torch.einsum('s,i,j->sij',t[k2+k3],p[k2],q[k3])
    
        for name in lName:
            a = eval('net.'+name+'.weight.data[:]')
            b = eval('torch.sub(net.'+name+'.weight.data,net.'+name+'.weight.grad, alpha = alpha)')
            a[:] = b
            c = eval('net.'+name+'.bias.data[:]')
            d = eval('torch.sub(net.'+name+'.bias.data,net.'+name+'.bias.grad,alpha = alpha)')
            c[:] = d
            
        running_loss += loss.item()
    
      return running_loss


    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    batchsize = 100

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                shuffle=True, num_workers = 2, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                            shuffle=False, num_workers =  2, pin_memory=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    alpha = 0.1
    epochs = 50
    
    filternumbers = [64, 64, 144, 144, 256, 256, 256, 484, 484, 484, 484]
    
    
    # creating the indexes where the elements of p and q are present in each filter
    qindex = []
    pindex = []
    
    for i,filternumber in enumerate(filternumbers):
        root = int(np.sqrt(filternumber))
        
        pin = np.zeros([root*root,root], dtype = int)
        l = 0
        k = root
        
        for j in range(root):
            if j == 0:
                qin = np.identity(root, dtype = int)
            else: 
                qin = np.concatenate((qin,np.identity(root, dtype = int)))
                
            pin[l:k,j] = 1
            l += root
            k += root
            
        qindex.append(torch.tensor(qin))
        pindex.append(torch.tensor(pin))
    
    
    p = []
    q = []
    t = [torch.randn(3,64)]
    
    for i,filternumber in enumerate(filternumbers):
        root = int(np.sqrt(filternumber))
        p.append(torch.randn(root,3))
        q.append(torch.randn(root,3))
        if i > 0:
            t.append(torch.randn(filternumbers[i-1],filternumbers[i]))

    convNames = ['conv1[0].conv',
                 'conv2[0].conv',
                 'conv3[0].conv',
                 'conv4[0].conv',
                 'conv5[0].conv',
                 'conv6[0].conv',
                 'conv7[0].conv',
                 'conv8[0].conv',
                 'conv9[0].conv',
                 'conv10[0].conv',
                 'conv11[0].conv']
    
    lNames = ['classifier[1]', 'classifier[5]', 'classifier[9]', 'classifier[13]']
    
    net = deepcopy(papernet4True)
    net.to(device)
    
    netname = "net"
    
    for k1,convName in enumerate(convNames):
        convData = eval(netname+"."+convName+".weight.data")
        root = int(np.sqrt(filternumbers[k1]))
         
        for k2 in range(root):
            for k3 in range(root):
                convData[k2+k3] = torch.einsum('s,i,j->sij', t[k1][:,k2+k3], p[k1][k2], q[k1][k3]) # tænk over det her
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=alpha)

    train_acc = []
    test_acc = []
    losses = []

    for epoch in range(epochs):
        running_loss = train_net_PARAFAC3D_ATDCTRUE(losses, net, "net", trainloader, criterion, optimizer, convNames, p, q, t, alpha, 1, lNames)

        net.eval()
        train_acc.append(evaluate_cifar(trainloader, net).cpu().item())
        test_acc.append(evaluate_cifar(testloader, net).cpu().item())
        losses.append(running_loss)

    save_train = pd.DataFrame(train_acc)
    save_test = pd.DataFrame(test_acc)
    save_loss = pd.DataFrame(losses)
    pd.concat([save_train,save_test,save_loss],axis = 0).to_csv('0305_webNet_noRes_ATCD3D_CIFAR100.csv',index=False,header=False)

    
    