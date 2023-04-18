import paddle

# example 1: attr shape is a list which doesn't contain Tensor.
out1 = paddle.rand(shape=[2, 3])
# [[0.451152  , 0.55825245, 0.403311  ],  # random
#  [0.22550228, 0.22106001, 0.7877319 ]]  # random

# example 2: attr shape is a list which contains Tensor.
dim1 = paddle.to_tensor([2], 'int64')
dim2 = paddle.to_tensor([3], 'int32')
out2 = paddle.rand(shape=[dim1, dim2, 2])
# [[[0.8879919 , 0.25788337],  # random
#   [0.28826773, 0.9712097 ],  # random
#   [0.26438272, 0.01796806]],  # random
#  [[0.33633623, 0.28654453],  # random
#   [0.79109055, 0.7305809 ],  # random
#   [0.870881  , 0.2984597 ]]]  # random

# example 3: attr shape is a Tensor, the data type must be int64 or int32.
shape_tensor = paddle.to_tensor([2, 3])
out3 = paddle.rand(shape_tensor)
# [[0.22920267, 0.841956  , 0.05981819],  # random
#  [0.4836288 , 0.24573246, 0.7516129 ]]  # random