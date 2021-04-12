#Import packages
import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')
from pack import *
from sklearn.metrics import accuracy_score
from copy import deepcopy
from MNIST_MODELS import *
from train_val_test_MNIST import *
from method_functions import *

criterion = nn.CrossEntropyLoss()

convName =  ["conv_1", "conv_2", "conv_3", "conv_4"]
R = 1

batch_size = 100
num_epochs = 50
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
losses = []

#netNormal = deepcopy(net)
#netNormal.apply(weight_reset)
net = deepcopy(convNet500)
optimizerNormal = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    losses = train_net(losses, num_batches_train, x_train, batch_size,targets_train, net, criterion, optimizerNormal)
    train_targs, train_preds = evaluate_train(num_batches_train, x_train, batch_size, targets_train, net)
    val_targs, val_preds = evaluate_validation(num_batches_valid, x_valid, batch_size, targets_valid, net)
    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)
    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

    if epoch % 1 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

preds, test_acc_full = evaluate_test(x_test, targets_test, net)
normalvalid_acc = valid_acc
normaltest_acc = accuracy_score(list(targets_test), list(preds.data.numpy()))
print("\nTest set Acc:  %f" % normaltest_acc)

train_test_acc_array = np.array([train_acc, valid_acc])

np.savetxt("MNIST_conv500_normal.txt", train_test_acc_array, delimiter = ",")
