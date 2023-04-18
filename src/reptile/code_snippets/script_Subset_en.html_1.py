import paddle
from paddle.io import Subset

# example 1:
a = paddle.io.Subset(dataset=range(1, 4), indices=[0, 2])
print(list(a))
# [1, 3]

# example 2:
b = paddle.io.Subset(dataset=range(1, 4), indices=[1, 1])
print(list(b))
# [2, 2]