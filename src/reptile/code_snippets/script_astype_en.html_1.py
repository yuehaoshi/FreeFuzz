import paddle
import numpy as np

original_tensor = paddle.ones([2, 2])
print("original tensor's dtype is: {}".format(original_tensor.dtype))
new_tensor = original_tensor.astype('float32')
print("new tensor's dtype is: {}".format(new_tensor.dtype))