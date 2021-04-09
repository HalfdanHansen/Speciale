from torchvision.datasets import MNIST

mnist_trainset = MNIST("./temp/", train=True, download=True)

mnist_testset = MNIST("./temp/", train=False, download=True)

# To speed up training we'll only work on a subset of the data

x_train = mnist_trainset.data[:1000].view(-1, 784).float()

targets_train = mnist_trainset.targets[:1000]



x_valid = mnist_trainset.data[1000:1500].view(-1, 784).float()

targets_valid = mnist_trainset.targets[1000:1500]



x_test = mnist_testset.data[:500].view(-1, 784).float()

targets_test = mnist_testset.targets[:500]



print("Information on dataset")

print("x_train", x_train.shape)

print("targets_train", targets_train.shape)

print("x_valid", x_valid.shape)

print("targets_valid", targets_valid.shape)

print("x_test", x_test.shape)

print("targets_test", targets_test.shape)
