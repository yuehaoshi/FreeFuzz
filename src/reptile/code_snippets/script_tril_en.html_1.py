import numpy as np
import paddle

data = np.arange(1, 13, dtype="int64").reshape(3,-1)
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])


x = paddle.to_tensor(data)

tril1 = paddle.tensor.tril(x)
# array([[ 1,  0,  0,  0],
#        [ 5,  6,  0,  0],
#        [ 9, 10, 11,  0]])

# example 2, positive diagonal value
tril2 = paddle.tensor.tril(x, diagonal=2)
# array([[ 1,  2,  3,  0],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

# example 3, negative diagonal value
tril3 = paddle.tensor.tril(x, diagonal=-1)
# array([[ 0,  0,  0,  0],
#        [ 5,  0,  0,  0],
#        [ 9, 10,  0,  0]])