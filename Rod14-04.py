import sys
sys.path.insert(1,'TensorizedTrainingMethods/PackagesAndModels')

from CIFAR_MODELS import *
from method_functions import *

convNet500.cuda()
c = convNet500.conv_2.weight.data
de = initialize_model_weights_from_Tucker2(["conv_2"],"convNet500",4,4)
print(de[0][0].shape)
print(ATDC_get_grads_Tucker2(c, de[0], 4, 4)[0].shape)