from pack import *
from method_functions import *

import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')

data = np.load('mnist.npz')
num_classes = 10
nchannels, rows, cols = 1, 28, 28

x_train = data['X_train'][:10000].astype('float32')
x_train = x_train.reshape((-1, nchannels, rows, cols))
targets_train = data['y_train'][:10000].astype('int32')

x_valid = data['X_valid'][:1000].astype('float32')
x_valid = x_valid.reshape((-1, nchannels, rows, cols))
targets_valid = data['y_valid'][:1000].astype('int32')

x_test = data['X_test'][:1000].astype('float32')
x_test = x_test.reshape((-1, nchannels, rows, cols))
targets_test = data['y_test'][:1000].astype('int32')

print("Information on dataset")
print("x_train", x_train.shape)
print("targets_train", targets_train.shape)
print("x_valid", x_valid.shape)
print("targets_valid", targets_valid.shape)
print("x_test", x_test.shape)
print("targets_test", targets_test.shape)

get_slice = lambda i, size: range(i * size, (i + 1) * size)

## MNIST TRAINING ##

# Normal
def train_net(losses, num_batches_train, x_train, batch_size, targets_train, net, criterion, optimizer):
    #losses = []
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        x_batch = Variable(torch.from_numpy(x_train[slce]))
        output = net(x_batch)
        
        # compute gradients given loss
        target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
        batch_loss = criterion(output, target_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        cur_loss += batch_loss
    losses.append(cur_loss / batch_size)

    net.eval()
    return losses

# BAF 4D
def train_net_BAF_4D(losses, num_batches_train, x_train, batch_size, targets_train, net, criterion, optimizer, convName, R):
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        x_batch = Variable(torch.from_numpy(x_train[slce]))
        output = net(x_batch)

        # compute gradients given loss
        target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
        batch_loss = criterion(output, target_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # BAF STEP
        # Extract weights for convolutions
        scope = locals()
        convData = [eval("net."+convName[i]+".weight.data",scope) for i in range(len(convName))]
        TD = []
        # Decompose all weights and use as new weights
        for c in convData:
          FIL,D = BAF_4D(c, R)
          c[:,:,:] = torch.from_numpy(FIL) #torch.tensor
          TD.append(D)
        
        
        cur_loss += batch_loss
    losses.append(cur_loss / batch_size)

    net.eval()
    return losses


# BAF 3D
def train_net_BAF_3D(losses, num_batches_train, x_train, batch_size, targets_train, net, criterion, optimizer, convName, R):
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        x_batch = Variable(torch.from_numpy(x_train[slce]))
        output = net(x_batch)

        # compute gradients given loss
        target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
        batch_loss = criterion(output, target_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # BAF STEP
        # Extract weights for convolutions
        scope = locals()
        convData = [eval("net."+convName[i]+".weight.data",scope) for i in range(len(convName))]
        # Decompose all weights and use as new weights
        for c in convData:
          for FF in c:
            FIL,D = BAF_3D(FF,R)
            FF[:,:,:] = torch.tensor(FIL)
        
        cur_loss += batch_loss
    losses.append(cur_loss / batch_size)

    net.eval()
    return losses

# ATDC 3D 
def train_net_ATDC_3D(losses, num_batches_train, x_train, batch_size, targets_train, net, criterion, optimizer, convName, pqt_convs, alpha):
    cur_loss = 0
    net.train()
    scope = locals()
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        x_batch = Variable(torch.from_numpy(x_train[slce]))
        output = net(x_batch)                               # Forward
        
        # compute gradients given loss
        target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
        batch_loss = criterion(output, target_batch)

        optimizer.zero_grad()
        batch_loss.backward()                               # Backward
        #optimizer.step()
        
        #ATDC step for convolutional layers
        for k1,pqt in enumerate(pqt_convs):
          convGrad = eval("net."+convName[k1]+".weight.grad", scope)
          convData = eval("net."+convName[k1]+".weight.data", scope)
          for k2,pqt_filter in enumerate(pqt):
            DL = ATDC_get_grads_one_filter_3D_short(convGrad[k2], pqt_filter)
            step = ATDC_update_step_one_filter_3D(DL, alpha, pqt_filter)
            pqt_convs[k1][k2] = step                                                            # new update params
            convData[k2] = torch.tensor(np.einsum('i,j,k->ijk',np.concatenate(step[0]),np.concatenate(step[1]),np.concatenate(step[2])))

        #Normal step for linear layer
        convGradl1 = net.l_1.weight.grad
        convDatal1 = net.l_1.weight.data

        convDatal1[:] = convDatal1 - alpha*convGradl1

        cur_loss += batch_loss

    losses.append(cur_loss / batch_size)

    net.eval()
    return losses

# ATDC 4D
def train_net_ATDC_4D(losses, num_batches_train, x_train, batch_size, targets_train, net, criterion, optimizer, convName, pqtu_convs, alpha):
    cur_loss = 0
    net.train()
    scope = locals()
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        x_batch = Variable(torch.from_numpy(x_train[slce]))
        output = net(x_batch)                               # Forward
        
        # compute gradients given loss
        target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
        batch_loss = criterion(output, target_batch)

        optimizer.zero_grad()
        batch_loss.backward()                               # Backward
        #optimizer.step()

        #ATDC step for convolutional layers
        for k1,pqtu in enumerate(pqtu_convs):

          convGrad = eval("net."+convName[k1]+".weight.grad", scope)
          convData = eval("net."+convName[k1]+".weight.data", scope)

          DL = ATDC_get_grads_one_filter_4D_short(convGrad, pqtu[1])
          step = ATDC_update_step_one_filter_4D(DL, alpha, pqtu[1])
          pqtu_convs[k1][1] = step                                          # new update params
          convData[:] = torch.tensor(np.einsum('i,j,k,h->ijkh',np.concatenate(step[0]),np.concatenate(step[1]),np.concatenate(step[2]),np.concatenate(step[3])))

        #steepest descent step for linear layer
        convGradl1 = net.l_1.weight.grad
        convDatal1 = net.l_1.weight.data

        convDatal1[:] = convDatal1 - alpha*convGradl1

        cur_loss += batch_loss

    losses.append(cur_loss / batch_size)

    net.eval()
    return losses


  
## CIFAR10 TRAINING ##

# Normal
def train_net_cifar10(losses, num_batches_train, x_train, batch_size, targets_train, net, criterion, optimizer):
    #losses = []
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):
        #print(i)
        slce = get_slice(i, batch_size)
        x_batch = Variable(torch.from_numpy(x_train[slce]))
        output = net(x_batch)
        
        # compute gradients given loss
        target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
        batch_loss = criterion(output, target_batch[:,0])
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        cur_loss += batch_loss
    losses.append(cur_loss / batch_size)

    net.eval()
    return losses

## EVALUATE ##

# Training
def evaluate_train(num_batches_train, x_train, batch_size, targets_train, net):
    train_preds, train_targs = [], []
    #train_acc_cur = []
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        x_batch = Variable(torch.from_numpy(x_train[slce]))
        
        output = net(x_batch)
        preds = torch.max(output, 1)[1]
        
        train_targs += list(targets_train[slce])
        train_preds += list(preds.data.numpy())
    #train_acc_cur = accuracy_score(train_targs, train_preds)

    return train_targs, train_preds

#Validation
def evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net):
    val_preds, val_targs = [], []
    #valid_acc_cur = []
    for i in range(num_batches_valid):
        slce = get_slice(i, batch_size)
        x_batch = Variable(torch.from_numpy(x_valid[slce]))
        
        output = net(x_batch)
        preds = torch.max(output, 1)[1]
        val_preds += list(preds.data.numpy())
        val_targs += list(targets_valid[slce])
        
    #valid_acc_cur = accuracy_score(val_targs, val_preds)

    return val_targs, val_preds

#Test
def evaluate_test(x_test, targets_test, net):
    x_batch = Variable(torch.from_numpy(x_test))
    output = net(x_batch)
    preds = torch.max(output, 1)[1]
    test_acc_full = accuracy_score(list(targets_test), list(preds.data.numpy())) 
    return preds, test_acc_full