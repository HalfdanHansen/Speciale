from sklearn.metrics import accuracy_score
from copy import deepcopy

import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')

from pack import *
from MNIST_MODELS import *
from train_val_test_MNIST import *
from method_functions import *

net = deepcopy(convNet4)

"""# Make CNN"""

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

names_and_vars = {}
for t in net.named_parameters(): 
  names_and_vars[t[0]] = t[1]

conv1_filters = names_and_vars['conv_1.weight'].detach()
conv2_filters = names_and_vars['conv_2.weight'].detach()
conv3_filters = names_and_vars['conv_3.weight'].detach()
conv4_filters = names_and_vars['conv_4.weight'].detach()
l_1_weights = names_and_vars['l_1.weight'].detach()

conv1_filters_p = conv1_filters.permute(1,2,3,0)
conv2_filters_p = conv2_filters.permute(1,2,3,0)
conv3_filters_p = conv3_filters.permute(1,2,3,0)
conv4_filters_p = conv4_filters.permute(1,2,3,0)

"""# Apply decomposition"""

num_batches_valid = num_samples_valid // batch_size
valid_acc, valid_loss = [], []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

channel = [1,3,6,12,3,10]
spatial = 3
padding = 1
dimlist = [28,14,7,7]

RANKLIST = [1,2,3,4,5,6,7,8,9,10]

### Evaluate validation

m = nn.ReLU()
bn1 = nn.BatchNorm2d(3)
bn2 = nn.BatchNorm2d(6)
bn3 = nn.BatchNorm2d(12)
bn4 = nn.BatchNorm2d(3)
s = nn.Softmax()
mp = nn.MaxPool2d(2, stride=2)

repeats = 20 # takes a while if high

accuracymatrix = []

for e in range(repeats):
  test_acc_list = []
  param_list = []
  for r,R in enumerate(RANKLIST):
    #print("Calculating for rank: " + str(R))

    decomp_conv1 = tl.decomposition.parafac(tl.tensor(conv1_filters_p), rank = R)
    decomp_conv2 = tl.decomposition.parafac(tl.tensor(conv2_filters_p), rank = R)
    decomp_conv3 = tl.decomposition.parafac(tl.tensor(conv3_filters_p), rank = R)
    decomp_conv4 = tl.decomposition.parafac(tl.tensor(conv4_filters_p), rank = R)

    test_acc, test_loss = [], []
    x_batch = Variable(torch.from_numpy(x_test))

    output_conv1 = mp(bn1(m(apply_decomp(channel[0], channel[1], spatial, padding, dimlist[0], dimlist[0], decomp_conv1, x_batch, R))))

    output_conv2 = mp(bn2(m(apply_decomp(channel[1], channel[2], spatial, padding, dimlist[1], dimlist[1], decomp_conv2, output_conv1, R))))

    output_conv3 = bn3(m(apply_decomp(channel[2], channel[3], spatial, padding, dimlist[2], dimlist[2], decomp_conv3, output_conv2, R)))

    output_conv4 = bn4(m(apply_decomp(channel[3], channel[4], spatial, padding, dimlist[3], dimlist[3], decomp_conv4, output_conv3, R)))
    
    output = s(F.linear(output_conv4.reshape(500,7*7*3), l_1_weights))

    preds = torch.max(output, 1)[1]

    test_preds = list(preds.data.numpy())
    test_targs = list(targets_test)
    test_acc_cur = accuracy_score(test_targs, test_preds)
    test_acc.append(test_acc_cur)
    test_acc_list.append(np.average(test_acc))

  accuracymatrix.append(test_acc_list)

param_list = []
for i in range(len(RANKLIST)):
  param_list.append(RANKLIST[i]*((3+3+1+3)+(3+3+3+6)+(3+3+6+12)+(3+3+12+3)))

test_acc_list = []
for i in np.transpose(np.array(accuracymatrix)):
  test_acc_list.append(np.mean(i))

fig,ax = plt.subplots()

param_list = np.add(param_list,3*7*7*10)
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
ax.legend(["Decomps","Original"],loc = 'lower center')
ax2.legend(["Decomps","Original"],loc = 'lower right')
ax.set(ylim = (0.1, 0.9))
ax2.set(ylim = (1500, 2800))

plt.show()

plt.savefig('accuracyofdecomps.png')