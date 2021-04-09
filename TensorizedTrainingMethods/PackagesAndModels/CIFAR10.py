from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

(x_train, targets_train), (x_test, targets_test) = cifar10.load_data()
x_valid = x_train[45001:50000]
targets_valid = targets_train[45001:50000]
x_train = x_train[0:45000]
targets_train = targets_train[0:45000]


