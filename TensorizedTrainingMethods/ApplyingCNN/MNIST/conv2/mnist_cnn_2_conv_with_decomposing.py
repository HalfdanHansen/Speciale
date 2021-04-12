from sklearn.metrics import accuracy_score
from copy import deepcopy

import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')

from pack import *
from MNIST_MODELS import *
from train_val_test_MNIST import *
from method_functions import *

net = deepcopy(convNet2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# we could have done this ourselves,
# but we should be aware of sklearn and it's tools
from sklearn.metrics import accuracy_score

batch_size = 100
num_epochs = 50
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
losses = []

for epoch in range(num_epochs):
    # Forward -> Backprob -> Update params
    ## Train
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, net, criterion, optimizer)
    #Evaluate training and validation
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, net)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net)

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

    if epoch % 5 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

# Plot training and validation 
epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')

#Evaluate test
preds, test_acc_full = evaluate_test(x_test, targets_test, net)
print("\nTest set Acc:  %f" % (accuracy_score(list(targets_test), list(preds.data.numpy()))))

total_params = sum(p.numel() for p in net.parameters())
total_params

"""#Decomposition of filters"""

# to start with we print the names of the weights in our network
names_and_vars = {}

for t in net.named_parameters(): 
  names_and_vars[t[0]] = t[1]

#names_and_vars = {​​​​x[0]: x[1] for x in net.named_parameters()}​​​​

print(names_and_vars.keys())

if not 'conv_1.weight' in names_and_vars:
    print("You need to go back and define a convolutional layer in the network.")
else:
    np_W = names_and_vars['conv_1.weight'].data.numpy() # get the filter values from the first conv layer
    print(np_W.shape, "i.e. the shape is (channels_out, channels_in, filter_height, filter_width)")
    channels_out, channels_in, filter_size, _ = np_W.shape
    n = int(channels_out**0.5)

    # reshaping the last dimension to be n by n
    np_W_res = np_W.reshape(filter_size, filter_size, channels_in, n, n)
    fig, ax = plt.subplots(n,n)
    print("learned filter values")
    for i in range(n):
        for j in range(n):
            ax[i,j].imshow(np_W_res[:,:,0,i,j], cmap='gray',interpolation='none')
            ax[i,j].xaxis.set_major_formatter(plt.NullFormatter())
            ax[i,j].yaxis.set_major_formatter(plt.NullFormatter())

conv1_filters = names_and_vars['conv_1.weight'].detach()
conv2_filters = names_and_vars['conv_2.weight'].detach()
l_1_filters = names_and_vars['l_1.weight'].detach()
l_out_filters = names_and_vars['l_out.weight'].detach()
l_1_bias = names_and_vars['l_1.bias'].detach()

conv1_filters_p = conv1_filters.permute(1,2,3,0)
conv2_filters_p = conv2_filters.permute(1,2,3,0)

"""# Apply decomposition"""

num_batches_valid = num_samples_valid // batch_size
valid_acc, valid_loss = [], []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

inputchannel = [1,16]
outputchannel = [16,8]
spatial = 5
padding = 1
dimx = 28
dimy = 28

RANKLIST = [6,8,10,12,14,16]

### Evaluate validation

m = nn.ReLU()
s = nn.Softmax()

test_acc_list = []
param_list = []

for r,R in enumerate(RANKLIST):
  print("Calculating for rank: " + str(R))

  decomp_conv1 = tl.decomposition.parafac(tl.tensor(conv1_filters_p), rank = R)
  decomp_conv2 = tl.decomposition.parafac(tl.tensor(conv2_filters_p), rank = R)

  output_c_conv1 = outputchannel[0]
  input_c_conv1 = inputchannel[0]

  output_c_conv2 = outputchannel[1]
  input_c_conv2 = inputchannel[1]
  #FD_conv2 = decomps[1]

  test_acc, test_loss = [], []
  x_batch = Variable(torch.from_numpy(x_test))
  #for i in range(num_batches_valid):
    #slce = get_slice(i, batch_size)
    #x_batch = Variable(torch.from_numpy(x_valid[slce]))
  output_conv1 = m(apply_decomp(input_c_conv1, output_c_conv1, spatial, padding, dimx, dimy, decomp_conv1, x_batch, R))
  conv1_param = R*(input_c_conv1+spatial*2+output_c_conv1)

  output_conv2 = m(apply_decomp(input_c_conv2, output_c_conv2, spatial, padding, dimx, dimy, decomp_conv2, output_conv1, R))
  conv2_param = R*(input_c_conv2+spatial*2+output_c_conv2)

  output_l1 = m(F.linear(output_conv2.reshape(500,24*24*8), l_1_filters, bias = l_1_bias))
  l1_param = 100*24*24*8+100
  
  output = s(F.linear(output_l1, l_out_filters))
  out_param = 100*10

  preds = torch.max(output, 1)[1]
  test_preds = list(preds.data.numpy())
  test_targs = list(targets_test)#[slce])

  test_acc_cur = accuracy_score(test_targs, test_preds)
  test_acc.append(test_acc_cur)
  param_list.append(conv1_param+conv2_param+l1_param+out_param)
  test_acc_list.append(np.average(test_acc))

fig,ax = plt.subplots()

ax.plot(RANKLIST, test_acc_list,color="red", marker="o")
ax.plot(RANKLIST,np.ones(len(RANKLIST))*test_acc_full,color="red")
plt.xticks(RANKLIST, RANKLIST)
ax2=ax.twinx()
ax2.plot(RANKLIST,param_list,color="blue", marker="o")
ax2.plot(RANKLIST,np.ones(len(RANKLIST))*total_params,color="blue")
ax2.set_ylabel("Number of Parameters",color = "blue")
ax.set_xlabel("Rank")
ax.set_ylabel("Accuracy",color = "red")
ax.set_title("Test accuracy for decomposition")
ax.legend(["Decompositions","Original"])

plt.show()

"""replace for loop attempt

"""