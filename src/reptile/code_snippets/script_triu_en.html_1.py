import numpy as np
import paddle

data = np.arange(1, 13, dtype="int64").reshape(3,-1)
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])


# example 1, default diagonal
x = paddle.to_tensor(data)
triu1 = paddle.tensor.triu(x)
# array([[ 1,  2,  3,  4],
#        [ 0,  6,  7,  8],
#        [ 0,  0, 11, 12]])

# example 2, positive diagonal value
triu2 = paddle.tensor.triu(x, diagonal=2)
# array([[0, 0, 3, 4],
#        [0, 0, 0, 8],
#        [0, 0, 0, 0]])

# example 3, negative diagonal value
triu3 = paddle.tensor.triu(x, diagonal=-1)
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 0, 10, 11, 12]])