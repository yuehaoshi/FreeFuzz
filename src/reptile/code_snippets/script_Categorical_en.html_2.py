import paddle
from paddle.distribution import Categorical

paddle.seed(100) # on CPU device
x = paddle.rand([6])
print(x)
# [0.5535528  0.20714243 0.01162981
#  0.51577556 0.36369765 0.2609165 ]

cat = Categorical(x)

paddle.seed(1000) # on CPU device
cat.sample([2,3])
# [[0, 0, 5],
#  [3, 4, 5]]