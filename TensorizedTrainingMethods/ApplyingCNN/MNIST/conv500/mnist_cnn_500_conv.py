from sklearn.metrics import accuracy_score
from copy import deepcopy

import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')

from pack import *
from MNIST_MODELS import *
from train_val_test_MNIST import *
from method_functions import *

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

from sklearn.metrics import accuracy_score

batch_size = 100
num_epochs = 1
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
total_params/640