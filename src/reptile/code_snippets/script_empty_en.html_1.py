import paddle
import numpy as np

paddle.set_device("cpu")  # and use cpu device

# example 1: argument ``shape`` is a list which doesn't contain Tensor.
data1 = paddle.empty(shape=[2,3], dtype='float32')
#[[4.3612203e+27 1.8176809e+31 1.3555911e-19]     # uninitialized
# [1.1699684e-19 1.3563156e-19 3.6408321e-11]]    # uninitialized

# example 2: argument ``shape`` is a Tensor, the data type must be int64 or int32.
shape_data = np.array([2, 3]).astype('int32')
shape = paddle.to_tensor(shape_data)
data2 = paddle.empty(shape=shape, dtype='float32')
#[[1.7192326e-37 4.8125365e-38 1.9866003e-36]     # uninitialized
# [1.3284029e-40 7.1117408e-37 2.5353012e+30]]    # uninitialized

# example 3: argument ``shape`` is a list which contains Tensor.
dim2_data = np.array([3]).astype('int32')
dim2 = paddle.to_tensor(dim2_data)
data3 = paddle.empty(shape=[2, dim2], dtype='float32')
#[[1.1024214e+24 7.0379409e+22 6.5737699e-34]     # uninitialized
# [7.5563101e+31 7.7130405e+31 2.8020654e+20]]    # uninitialized